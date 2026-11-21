"""Locate published study and identification result records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcd.artifacts import evidence_digest, file_sha256, resolve_artifact_path

from .store import GENERATION_RESULT_FILE, PUBLISHED_GENERATION_FILE

_STUDY_RESULT = "study_result.json"
_RESULT_SCHEMAS = {
    _STUDY_RESULT: "study_result.v1",
    GENERATION_RESULT_FILE: "study_result.v1",
    "identification_result.json": "identification_result.v1",
}


@dataclass(frozen=True, slots=True)
class SavedResult:
    path: Path
    root: Path
    payload: dict[str, Any]
    sha256: str
    generation: Path | None = None

    def artifact_path(self, name: str) -> Path | None:
        declared = (self.payload.get("artifacts") or {}).get(name)
        if not isinstance(declared, str) or not declared.strip():
            return None
        path = resolve_artifact_path(self.root, declared)
        if self.generation is not None and name != "generation":
            if not path.resolve().is_relative_to(self.generation):
                raise ValueError(f"saved study artifact {name!r} is outside its declared generation")
            evidence_digest(self.payload, name, path)
        return path


def read_saved_result(source: str | Path, *, study_only: bool = False) -> SavedResult | None:
    """Read one aggregate record and pin its published generation if present."""

    path = Path(source)
    names = (_STUDY_RESULT, GENERATION_RESULT_FILE) if study_only else tuple(_RESULT_SCHEMAS)
    if path.is_dir():
        found = [path / name for name in names if (path / name).is_file()]
        if len(found) > 1:
            raise ValueError("directory contains multiple result records; choose a result file")
        if not found:
            return None
        path = found[0]
    if path.name not in names:
        return None

    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"saved result must be a JSON object: {path}")
    if payload.get("schema") != _RESULT_SCHEMAS[path.name]:
        raise ValueError(f"unsupported saved result record: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"saved result artifacts must be a JSON object: {path}")
    digest = hashlib.sha256(data).hexdigest()
    if path.name == "identification_result.json":
        return SavedResult(path, path.parent, payload, digest)
    return _study_record(path, payload, digest)


def _study_record(path: Path, payload: dict[str, Any], digest: str) -> SavedResult:
    immutable = path.name == GENERATION_RESULT_FILE
    if immutable and path.parent.parent.name != "generations":
        raise ValueError(f"saved generation record must be under a study's generations directory: {path}")
    root = path.parents[2] if immutable else path.parent
    record = SavedResult(path, root, payload, digest)
    declared = record.artifact_path("generation")
    generation = declared.resolve() if declared is not None else None
    if generation is None or generation.parent != root.resolve() / "generations" or not generation.is_dir():
        raise ValueError(f"saved study result does not point to an available generation under {root}")

    pinned = generation / GENERATION_RESULT_FILE
    if pinned.is_symlink():
        raise ValueError(f"saved generation record must be a file within its generation: {pinned}")
    marker = generation / PUBLISHED_GENERATION_FILE
    marker_hash = marker.read_text(encoding="utf-8").strip() if marker.is_file() and not marker.is_symlink() else None
    if marker_hash and marker_hash != digest:
        raise ValueError(f"saved generation record differs from its publication: {pinned}")
    if immutable:
        if generation != path.parent.resolve() or (not marker_hash and file_sha256(root / _STUDY_RESULT) != digest):
            raise ValueError(f"saved study result does not match its published generation: {path}")
    else:
        if file_sha256(pinned) != digest:
            raise ValueError("active study result does not match its generation record")
        path = pinned
    return SavedResult(path, root, payload, digest, generation)
