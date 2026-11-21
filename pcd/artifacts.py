"""Small serialization and implementation-identity helpers for run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ARTIFACT_SEGMENT_MAX = 32
_CONTENT_PATH_KEY_LENGTH = 32
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def runtime_environment() -> dict[str, Any]:
    """Record the numerical environment used to interpret solver evidence."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {name: version(name) for name in ("numpy", "pandas", "scipy")},
    }


@lru_cache(maxsize=1)
def package_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str | None:
    """Hash one file without loading a potentially large dataset into memory."""

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_digest(record: Mapping[str, Any], name: str, path: Path) -> str:
    """Compare a published artifact with its recorded digest."""
    digest = file_sha256(path.resolve())
    saved = record.get("artifact_sha256")
    if not isinstance(saved, Mapping) or not digest or saved.get(name) != digest:
        raise ValueError(f"saved evidence differs from its recorded {name}: {path}")
    return digest


def capture_evidence(root: Path, artifacts: Mapping[str, str], *, within: Path | None = None) -> dict[str, str]:
    """Record publication digests for the files explicitly chosen by the writer."""
    evidence = {}
    for name, declared in artifacts.items():
        path = resolve_artifact_path(root, declared).resolve()
        if within is not None and not path.is_relative_to(within):
            raise ValueError(f"evidence artifact {name!r} is outside its generation: {path}")
        if (digest := file_sha256(path)) is None:
            raise FileNotFoundError(f"cannot publish without evidence artifact {name!r}: {path}")
        evidence[name] = digest
    return evidence


def resolve_artifact_path(root: Path, declared: str) -> Path:
    """Resolve a saved artifact reference relative to its owning result."""

    path = Path(declared)
    return path if path.is_absolute() else root / path


def artifact_path_segment(value: str, max_length: int = _ARTIFACT_SEGMENT_MAX) -> str:
    """Return one bounded, deterministic directory or file-name segment.

    Full user-facing IDs remain in manifests.  Only internal path segments are
    shortened, with a hash suffix preserving distinct long names.  The result
    is also safe for Windows device-name rules.
    """

    if max_length < 16:
        raise ValueError("artifact path segments need at least 16 characters")
    raw = str(value)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "unnamed"
    reserved = name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    if len(name) <= max_length and not reserved and name == raw:
        return name
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    prefix_length = max_length - len(digest) - 1
    prefix = name[: max(0, prefix_length)].rstrip("._-")
    return f"{prefix}-{digest}" if prefix else digest


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace one text artifact using a short sibling temp name."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            descriptor = -1
            handle.write(text)
        move_artifact(temporary_path, target, replace=True)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def move_artifact(source: Path, target: Path, *, replace: bool = False) -> None:
    """Publish by same-filesystem rename, with bounded retries for Windows sharing errors."""
    move = source.replace if replace else source.rename
    for attempt in range(5):
        try:
            move(target)
            return
        except PermissionError as exc:
            # A reader without delete sharing can block moving a file or
            # directory even when its destination has yet to be created.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.01 * 3**attempt)


class FileLockUnavailable(OSError):
    """Another process holds an exclusive artifact lock."""


@contextmanager
def exclusive_file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold a portable advisory lock released automatically on process exit."""

    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(lock.fileno(), mode)
        except OSError as exc:
            raise FileLockUnavailable(f"could not acquire artifact lock: {path}") from exc
        yield


def create_run_directory(root: str | Path, name: str) -> Path:
    """Allocate a bounded run name atomically, preserving every previous run."""

    root = Path(root).resolve()
    for index in range(1000):
        suffix = f"_{index:03d}" if index else ""
        path = root / artifact_path_segment(name + suffix)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"cannot create a unique run directory for {name!r} under {root}")


def archive_data_files(
    data: Mapping[str, Any],
    base_dir: str | Path,
    bundle_root: str | Path,
    *,
    extra_references: Iterable[tuple[str, str | Path]] = (),
    captured: Mapping[Path, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Snapshot runtime data dependencies into one content-addressed bundle.

    Declared data files include the built-in inputs and plugin settings ending
    in ``_file``. Executable plugins retain separate implementation provenance.
    """

    base = Path(base_dir)
    root = Path(bundle_root)
    inputs = root / "inputs"
    entries: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    archives: dict[Path, tuple[str, int, Path] | None] = {}
    captured = captured or {}
    references = [*data_file_references(data), *((field, str(path)) for field, path in extra_references)]
    for field, declared in references:
        source = Path(declared)
        source = (source if source.is_absolute() else base / source).resolve()
        entry: dict[str, Any] = {
            "field": field,
            "declared_path": declared,
            "source_path": str(source),
            "original_name": source.name,
            "was_absolute": Path(declared).is_absolute(),
        }
        if source not in archives:
            try:
                archives[source] = _copy_content_addressed(source, inputs, captured.get(source))
            except FileNotFoundError:
                archives[source] = None
        archived = archives[source]
        if archived is None:
            entry.update({"status": "missing", "sha256": None, "size_bytes": None, "artifact": None})
            entries.append(entry)
            absent = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
            replacements[str(source)] = f"inputs/missing_{absent}"
            continue

        digest, size, target = archived
        artifact = target.relative_to(root).as_posix()
        entry.update(
            {
                "status": "archived",
                "sha256": digest,
                "size_bytes": size,
                "artifact": artifact,
            }
        )
        entries.append(entry)
        replacements[str(source)] = artifact
    return {"schema": "input_manifest.v1", "inputs": entries}, replacements


def rewrite_data_file_paths(
    data: Mapping[str, Any], base_dir: str | Path, replacements: Mapping[str, str]
) -> dict[str, Any]:
    """Point an archived case at its bundled data while retaining its shape."""

    base = Path(base_dir)

    fields = {field for field, _declared in data_file_references(data)}

    def visit(value: Any, path: str = "$") -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for raw_name, item in value.items():
                name = str(raw_name)
                item_path = f"{path}.{name}"
                if item_path in fields and isinstance(item, str):
                    source = Path(item)
                    source = (source if source.is_absolute() else base / source).resolve()
                    out[name] = replacements.get(str(source), item)
                else:
                    out[name] = visit(item, item_path)
            return out
        if isinstance(value, list):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return deepcopy(value)

    return visit(data)


def data_file_references(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return declared built-in and plugin data files by their case fields."""
    references = _component_profile_references(data)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            impedance_table = value.get("type") == "impedance_table"
            for raw_name, item in value.items():
                name = str(raw_name)
                field = f"{path}.{name}"
                is_file = name.endswith("_file") or (name == "file" and impedance_table)
                if is_file and isinstance(item, str) and item.strip():
                    references.append((field, item))
                else:
                    visit(item, field)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(data, "$")
    return references


def _component_profile_references(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    for section_name in ("circuit", "load"):
        section = data.get(section_name)
        if not isinstance(section, Mapping):
            continue
        components = section.get("components")
        if not isinstance(components, list):
            continue
        for index, component in enumerate(components):
            if not isinstance(component, Mapping):
                continue
            value = component.get("value")
            if not isinstance(value, Mapping):
                continue
            profile = value.get("profile")
            if isinstance(profile, str) and profile.strip():
                references.append((f"$.{section_name}.components[{index}].value.profile", profile))
    return references


def _copy_content_addressed(source: Path, inputs: Path, captured: bytes | None = None) -> tuple[str, int, Path]:
    inputs.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary_path: Path | None = None
    try:
        with (
            source.open("rb") if captured is None else BytesIO(captured) as source_handle,
            tempfile.NamedTemporaryFile(mode="wb", dir=inputs, prefix=".input-", delete=False) as temporary,
        ):
            temporary_path = Path(temporary.name)
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                temporary.write(chunk)
                size += len(chunk)
        full_digest = digest.hexdigest()
        target = inputs / full_digest[:_CONTENT_PATH_KEY_LENGTH]
        if target.exists():
            if file_sha256(target) != full_digest:
                raise RuntimeError(f"content-addressed input path collision: {target}")
            temporary_path.unlink()
        else:
            move_artifact(temporary_path, target, replace=True)
        return full_digest, size, target
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
    )


def yaml_dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(type(obj).__name__)
