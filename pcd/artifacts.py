"""Small serialization and implementation-identity helpers for run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_DATA_FILE_FIELDS = (
    ("study", "scenario_table", "table_file"),
    ("target", "waveform_file"),
    ("circuit", "netlist_file"),
)

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
        temporary_path.replace(target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def archive_data_files(
    data: Mapping[str, Any], base_dir: str | Path, bundle_root: str | Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Snapshot runtime data dependencies into one content-addressed bundle.

    Only files read by the built-in execution path are included.  Plugins are
    executable code and retain their separate implementation provenance.
    """

    base = Path(base_dir)
    root = Path(bundle_root)
    inputs = root / "inputs"
    entries: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    for field, declared in _data_file_references(data):
        source = Path(declared)
        source = (source if source.is_absolute() else base / source).resolve()
        entry: dict[str, Any] = {
            "field": field,
            "declared_path": declared,
            "original_name": source.name,
            "was_absolute": Path(declared).is_absolute(),
        }
        if not source.is_file():
            entry.update({"status": "missing", "sha256": None, "size_bytes": None, "artifact": None})
            entries.append(entry)
            continue

        digest, size, target = _copy_content_addressed(source, inputs)
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

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            is_impedance_table = str(value.get("type", "")) == "impedance_table"
            out: dict[str, Any] = {}
            for raw_name, item in value.items():
                name = str(raw_name)
                is_file = name.endswith("_file") or (name == "file" and is_impedance_table)
                if is_file and isinstance(item, str):
                    source = Path(item)
                    source = (source if source.is_absolute() else base / source).resolve()
                    out[name] = replacements.get(str(source), item)
                else:
                    out[name] = visit(item)
            return out
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return [visit(item) for item in value]
        return deepcopy(value)

    return visit(data)


def _data_file_references(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    for parts in _DATA_FILE_FIELDS:
        value: Any = data
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            if isinstance(value, str) and value.strip():
                references.append(("$." + ".".join(parts), value))
    return references


def _copy_content_addressed(source: Path, inputs: Path) -> tuple[str, int, Path]:
    inputs.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary_path: Path | None = None
    try:
        with (
            source.open("rb") as source_handle,
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
            temporary_path.replace(target)
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
