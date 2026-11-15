"""Explicit maintenance operations for study generations and cache indices."""

from __future__ import annotations

import re
import shutil
import stat
import sys
import uuid
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from pcd.artifacts import FileLockUnavailable, exclusive_file_lock, read_json, resolve_artifact_path

from .store import (
    _CACHE_PATH_KEY_LENGTH,
    _GENERATION_ID,
    _PUBLICATION_LOCK,
    _RAW_CACHE_DIRECTORY,
    _RAW_CACHE_INDEX,
    _RAW_CACHE_LOCK,
    _STUDY_ID_FILE,
    PUBLISHED_GENERATION_FILE,
    _store_lock,
)

_RAW_CACHE_KEY = re.compile(rf"[a-f0-9]{{{_CACHE_PATH_KEY_LENGTH}}}")


def prune_generations(study_root: str | Path, *, keep: int = 3, apply: bool = False) -> dict[str, Any]:
    """Plan or remove old generations while always preserving the active one.

    Retain unmarked generations, which may be in progress or contain legacy
    replay inputs. The default is a dry run.
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    root = Path(study_root).resolve()
    result_path = root / "study_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"study_result.json not found under {root}")
    with _store_lock(root, _PUBLICATION_LOCK) if apply else nullcontext():
        return _prune_generations(root, result_path, keep, apply)


def _prune_generations(root: Path, result_path: Path, keep: int, apply: bool) -> dict[str, Any]:
    if result_path.resolve() != result_path:
        raise ValueError("study_result.json must be a file directly under the study root")
    generation_dir = (root / "generations").resolve()
    if generation_dir != root / "generations":
        raise ValueError("generations/ must be a directory directly under the study root")
    # A preview scans before reading the active pointer. Apply plans and
    # removes under the same lock used by publication.
    generations = []
    for path in generation_dir.glob("g_*"):
        if not _GENERATION_ID.fullmatch(path.name) or path.resolve() != path or not path.is_dir():
            continue
        marker = path / PUBLISHED_GENERATION_FILE
        with suppress(FileNotFoundError):
            published = marker.stat(follow_symlinks=False)
            if stat.S_ISREG(published.st_mode):
                generations.append((published.st_mtime_ns, path))
    payload = read_json(result_path)
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    declared = artifacts.get("generation") if isinstance(artifacts, dict) else None
    active = resolve_artifact_path(root, declared).resolve() if isinstance(declared, str) and declared else None
    if active is None or active.parent != generation_dir or not active.is_dir():
        raise ValueError("study_result.json does not declare an active generation under generations/")

    ordered = [path for _published, path in sorted(generations, reverse=True)]
    if active not in ordered:
        ordered.insert(0, active)
    retained = set(ordered[:keep]) | {active}
    removed = []
    busy = []
    for path in ordered:
        if path in retained:
            continue
        if not apply or _remove_generation(path):
            removed.append(path)
        else:
            busy.append(path)
            retained.add(path)
    return {
        "study_root": str(root),
        "applied": apply,
        "active_generation": active.relative_to(root).as_posix(),
        "kept": [path.relative_to(root).as_posix() for path in ordered if path in retained],
        "removed": [path.relative_to(root).as_posix() for path in removed],
        "busy": [path.relative_to(root).as_posix() for path in busy],
    }


def _remove_generation(path: Path) -> bool:
    if path.resolve() != path:
        raise ValueError(f"generation path changed before pruning: {path}")
    detached = path.with_name(f".prune-{uuid.uuid4().hex[:16]}")
    checkpoint_lock = path / "checkpoint" / ".lock"
    # Windows itself blocks a directory rename when a checkpoint file is
    # open. On POSIX, use the existing lock without waiting on a reader that
    # might also need the publication lock.
    lock = (
        exclusive_file_lock(checkpoint_lock, blocking=False)
        if sys.platform != "win32" and checkpoint_lock.is_file() and checkpoint_lock.resolve() == checkpoint_lock
        else nullcontext()
    )
    try:
        with lock:
            path.rename(detached)
    except FileLockUnavailable:
        return False
    except PermissionError as exc:
        if getattr(exc, "winerror", None) in {5, 32, 33}:
            return False
        raise
    except FileNotFoundError:
        if not path.exists():
            return True
        raise

    try:
        shutil.rmtree(detached)
    except OSError as exc:
        raise OSError(f"could not finish deleting detached generation {detached}: {exc}") from exc
    return True


def _has_study_identity(root: Path) -> bool:
    identity = root / _STUDY_ID_FILE
    if identity.is_file() and identity.resolve() == identity:
        return bool(identity.read_text(encoding="utf-8"))
    pointer = root / "study_result.json"
    if not pointer.is_file() or pointer.resolve() != pointer:
        return False
    result = read_json(pointer)
    study = result.get("study") if isinstance(result, dict) and result.get("schema") == "study_result.v1" else None
    return isinstance(study, dict) and isinstance(study.get("study_id"), str) and bool(study["study_id"])


def _raw_cache_entries(root: Path) -> list[tuple[int, Path, int]]:
    cache = root / _RAW_CACHE_DIRECTORY
    if cache.resolve() != cache:
        raise ValueError(f"raw cache must be directly under the study root: {cache}")
    entries = []
    for path in cache.glob(f"*/{_RAW_CACHE_INDEX}"):
        if not _RAW_CACHE_KEY.fullmatch(path.parent.name) or path.resolve() != path:
            continue
        with suppress(FileNotFoundError):
            info = path.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                entries.append((info.st_mtime_ns, path, info.st_size))
    return sorted(entries, reverse=True)


def prune_raw_cache(
    study_root: str | Path, *, keep: int | None = None, max_bytes: int | None = None, apply: bool = False
) -> dict[str, Any]:
    """Keep the newest cache indices that fit both requested limits."""
    if keep is None and max_bytes is None:
        raise ValueError("raw cache retention requires keep or max_bytes")
    for name, value in (("keep", keep), ("max_bytes", max_bytes)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"raw cache {name} must be a nonnegative integer")
    root = Path(study_root).resolve()
    if not _has_study_identity(root):
        raise ValueError(f"raw cache retention requires a recognized study root: {root}")

    kept: list[str] = []
    removed: list[str] = []
    kept_bytes = removed_bytes = 0
    with _store_lock(root, _RAW_CACHE_LOCK) if apply else nullcontext():
        for _modified, path, size in _raw_cache_entries(root):
            name = path.relative_to(root).as_posix()
            if (keep is None or len(kept) < keep) and (max_bytes is None or kept_bytes + size <= max_bytes):
                kept.append(name)
                kept_bytes += size
            else:
                if apply:
                    if path.resolve() != path:
                        raise ValueError(f"raw cache index path changed before pruning: {path}")
                    path.unlink(missing_ok=True)
                removed.append(name)
                removed_bytes += size
    return {
        "study_root": str(root),
        "scope": "raw_cache_indices",
        "applied": apply,
        "keep": keep,
        "max_bytes": max_bytes,
        "kept": kept,
        "removed": removed,
        "kept_bytes": kept_bytes,
        "removed_bytes": removed_bytes,
    }
