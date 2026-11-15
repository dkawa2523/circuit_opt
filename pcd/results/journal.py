"""Commit complete candidate evidence for deterministic optimizer replay."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from pcd.artifacts import FileLockUnavailable, exclusive_file_lock, read_json, write_json
from pcd.core.models import Candidate, CandidateResult

from .record import read_saved_result
from .store import (
    GENERATION_EVIDENCE,
    GENERATION_RESULT_FILE,
    PUBLISHED_GENERATION_FILE,
    FileResultStore,
    _mark_published,
    _stable_json,
)

_DIRECTORY = "checkpoint"
_HEADER = "attempt.json"
_SCHEMA = "study_trial_checkpoint.v1"
_ARCHIVE_NAMES = {
    "case": "case.yaml",
    "input_manifest": "input_manifest.json",
    "input_case": "input_case.yaml",
    "resolved_plan": "resolved_plan.yaml",
    "imported_netlist": "imported_netlist.cir",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read study checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"study checkpoint must contain an object: {path}")
    return value


def _active_result(store: FileResultStore) -> dict[str, Any]:
    path = store.root / "study_result.json"
    return _read_object(path) if path.is_file() else {}


def _is_active(store: FileResultStore, generation: Path, result: Mapping[str, Any]) -> bool:
    return (result.get("artifacts") or {}).get("generation") == generation.relative_to(store.root).as_posix()


def select_checkpoint(store: FileResultStore, identity: Mapping[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    active = _active_result(store)
    compatible = []
    incompatible = []
    completed = None
    for generation in sorted((store.root / "generations").glob("g_*")):
        header = generation / _DIRECTORY / _HEADER
        if generation.resolve() != generation or not header.is_file():
            continue
        try:
            matches = _read_object(header).get("identity") == identity
        except ValueError:
            matches = False
        if _is_active(store, generation, active):
            if matches:
                completed = generation.name
        elif not (generation / PUBLISHED_GENERATION_FILE).exists():
            (compatible if matches else incompatible).append(generation.name)
    if len(compatible) == 1:
        return compatible[0]
    if compatible:
        raise ValueError(f"multiple matching interrupted studies; use --resume GENERATION with one of {compatible}")
    if completed is not None:
        return completed
    if incompatible:
        raise ValueError(
            f"no checkpoint matches the current study inputs or runtime; interrupted generations: {incompatible}"
        )
    raise ValueError(f"no restartable study checkpoint found under {store.root}")


class TrialJournal:
    def __init__(self, store: FileResultStore, generation: Path, identity: Mapping[str, Any]) -> None:
        self.store = store
        self.generation = generation
        self.root = generation / _DIRECTORY
        self.header_path = self.root / _HEADER
        self.identity = dict(identity)
        self.header: dict[str, Any] = {}
        self.completed_trials = 0

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Prevent two live processes from mutating the same generation."""

        if self.root.resolve() != self.root:
            raise ValueError(f"study checkpoint must be inside its generation: {self.root}")
        self.root.mkdir(exist_ok=True)
        with ExitStack() as stack:
            try:
                stack.enter_context(exclusive_file_lock(self.root / ".lock", blocking=False))
            except FileLockUnavailable as exc:
                raise ValueError(f"study generation {self.generation.name} is already running") from exc
            yield

    def start(self, files: Mapping[str, str], archive: Path) -> None:
        self.header = {
            "schema": _SCHEMA,
            "identity": self.identity,
            "generation": self.generation.name,
            "archive_files": dict(files),
            "archive": archive.relative_to(self.store.root).as_posix(),
            "invocations": 1,
        }
        write_json(self.header_path, self.header)

    def load(self, budget: int) -> None:
        if not self.header_path.is_file():
            raise ValueError(f"study generation {self.generation.name} has no trial checkpoint")
        header = _read_object(self.header_path)
        if header.get("schema") != _SCHEMA or header.get("generation") != self.generation.name:
            raise ValueError(f"invalid study checkpoint header: {self.header_path}")
        if header.get("identity") != self.identity:
            raise ValueError(
                "study checkpoint identity does not match the effective case, optimizer, inputs or runtime"
            )
        files = header.get("archive_files")
        if (
            not isinstance(files, dict)
            or any(_ARCHIVE_NAMES.get(key) != value for key, value in files.items())
            or not {"case", "input_manifest"} <= set(files)
        ):
            raise ValueError(f"invalid study checkpoint archive manifest: {self.header_path}")
        if type(header.get("invocations")) is not int or header["invocations"] < 1:
            raise ValueError(f"invalid study checkpoint invocation count: {self.header_path}")
        self.header = header
        entries = set(self.root.glob("trial_*.json"))
        if len(entries) > budget or entries != {self._trial_path(index) for index in range(len(entries))}:
            raise ValueError(f"study checkpoint trials must form a contiguous prefix of the budget: {self.root}")
        self.completed_trials = len(entries)

    def published_result(self) -> dict[str, Any] | None:
        active = _active_result(self.store)
        is_active = _is_active(self.store, self.generation, active)
        marker = self.generation / PUBLISHED_GENERATION_FILE
        if not is_active and not marker.is_file():
            return None
        path = self.generation / GENERATION_RESULT_FILE
        saved = read_saved_result(self.store.root if is_active else path, study_only=True)
        if saved is None or saved.generation != self.generation or saved.path != path:
            raise ValueError(f"published result does not identify its checkpoint generation: {path}")
        for name in GENERATION_EVIDENCE:
            artifact = saved.artifact_path(name)
            if artifact is None or not artifact.is_file():
                raise ValueError(f"published study result is missing {name} evidence: {path}")
        if is_active and not marker.is_file():
            _mark_published(path)
        return {**saved.payload, "run_root": str(self.store.root)}

    def resume_archive(self) -> tuple[dict[str, str], Path]:
        files = self.header["archive_files"]
        archive = self.store.retain_case_archive(files, require_existing=True)
        if archive.relative_to(self.store.root).as_posix() != self.header.get("archive"):
            raise ValueError(f"study checkpoint archive identity changed: {archive}")
        self.header["invocations"] += 1
        write_json(self.header_path, self.header)
        return files, archive

    def _trial_path(self, index: int) -> Path:
        return self.root / f"trial_{index:08d}.json"

    def replay(self, index: int, candidate: Candidate) -> CandidateResult:
        path = self._trial_path(index)
        record = _read_object(path)
        data = record.get("result")
        hashes = record.get("artifact_sha256")
        if not isinstance(data, dict) or record.get("sha256") != _digest([data, hashes]):
            raise ValueError(f"study trial checkpoint checksum mismatch: {path}")
        result = CandidateResult.from_dict(data)
        if _stable_json(result.candidate.to_dict()) != _stable_json(candidate.to_dict()):
            raise ValueError(f"optimizer proposal does not match completed checkpoint trial {index}: {path}")
        if self.store.candidate_artifact_hashes(result) != hashes:
            raise ValueError(f"saved evaluation artifacts changed for checkpoint trial {index}: {path}")
        return result

    def save(self, index: int, result: CandidateResult) -> None:
        path = self._trial_path(index)
        if path.exists():
            raise RuntimeError(f"cannot replace a completed study trial checkpoint: {path}")
        data = result.to_dict()
        hashes = self.store.candidate_artifact_hashes(result)
        write_json(path, {"result": data, "artifact_sha256": hashes, "sha256": _digest([data, hashes])})

    def summary(self, **counts: int) -> dict[str, Any]:
        return {
            "scope": "deterministic_candidate_replay",
            "generation": self.generation.name,
            "checkpoint": self.header_path.relative_to(self.store.root).as_posix(),
            "invocations": self.header["invocations"],
            "previous_incomplete_invocations": self.header["invocations"] - 1,
            **counts,
        }
