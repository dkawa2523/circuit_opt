"""Reusable raw-result cache and selected study evidence storage."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from pcd.artifacts import (
    artifact_path_segment,
    atomic_write_text,
    capture_evidence,
    exclusive_file_lock,
    file_sha256,
    move_artifact,
    read_json,
    write_json,
)
from pcd.core.models import CandidateResult, EvaluationRequest, RawResult, to_plain

_CACHE_PATH_KEY_LENGTH = 24
_RAW_CACHE_DIRECTORY = "raw"
_RAW_CACHE_INDEX = "raw_result.json"
_RAW_CACHE_LOCK = ".raw-cache.lock"
_STUDY_ID_FILE = ".study_id"
_PUBLICATION_LOCK = ".publication.lock"
_STUDY_SEGMENT_MAX = 32
_STUDY_SEGMENT_MIN = 16
_WINDOWS_PATH_BUDGET = 248
_INTERNAL_PATH_RESERVE = 64
PUBLISHED_GENERATION_FILE = ".published"
GENERATION_RESULT_FILE = "generation_result.json"
GENERATION_EVIDENCE = ("case", "best_candidate", "history", "evaluation_table")
_GENERATION_ID = re.compile(r"g_[a-f0-9]{16}")


def _stable_json(value: Any) -> str:
    return json.dumps(to_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _mark_published(result: Path) -> None:
    if (digest := file_sha256(result)) is None:
        raise FileNotFoundError(result)
    atomic_write_text(result.parent / PUBLISHED_GENERATION_FILE, digest + "\n")


def raw_evaluation_identity(
    request: EvaluationRequest,
    runtime_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The inputs that can change expensive evaluator output.

    IDs, scenario weights, objectives, and constraints are intentionally
    absent. They change attribution or interpretation, not the evaluator run.
    Every physical value is present exactly once through ``merged_inputs``.
    """

    return {
        "inputs": request.merged_inputs(),
        "runtime": to_plain(runtime_fingerprint or {}),
    }


def raw_evaluation_key(
    request: EvaluationRequest,
    runtime_fingerprint: Mapping[str, Any] | None = None,
) -> str:
    return hashlib.sha256(
        _stable_json(raw_evaluation_identity(request, runtime_fingerprint)).encode("utf-8")
    ).hexdigest()


def _study_segment(base: Path, study_id: str) -> str:
    limit = _STUDY_SEGMENT_MAX
    if sys.platform == "win32":
        limit = min(limit, _WINDOWS_PATH_BUDGET - len(str(base)) - 1 - _INTERNAL_PATH_RESERVE)
        if limit < _STUDY_SEGMENT_MIN:
            raise ValueError(
                f"output directory is too deep for Windows-safe artifacts ({base}); choose a shorter --output path"
            )
    return artifact_path_segment(study_id, limit)


def _store_lock(root: Path, name: str) -> AbstractContextManager[None]:
    lock = root / name
    if lock.resolve() != lock:
        raise ValueError(f"study lock must be directly under the study root: {lock}")
    return exclusive_file_lock(lock)


def _claim_study_root(path: Path, study_id: str) -> Path | None:
    path.mkdir(parents=True, exist_ok=True)
    root = path.resolve()
    if root.parent != path.parent or root.name != path.name:
        return None
    identity = root / _STUDY_ID_FILE
    with _store_lock(root, _PUBLICATION_LOCK):
        if identity.is_file():
            return root if identity.read_bytes() == study_id.encode("utf-8") else None
        if (root / "study_result.json").is_file():
            return None
        atomic_write_text(identity, study_id)
    return root


def _study_root(base: Path, study_id: str) -> Path:
    name = _study_segment(base, study_id)
    proposed = base / name
    # Mixed-case names receive a digest on every platform.
    needs_digest = name == study_id and name != name.casefold()
    if not needs_digest and (root := _claim_study_root(proposed, study_id)) is not None:
        return root
    alternate = base / _study_segment(base, f"{study_id} [study identity]")
    if (root := _claim_study_root(alternate, study_id)) is not None:
        return root
    raise ValueError(f"cannot allocate a distinct study directory for {study_id!r} under {base}")


def _cache_path_key(full_key: str) -> str:
    return full_key[:_CACHE_PATH_KEY_LENGTH]


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class FileResultStore:
    """Reuse physical evaluator output and commit one selected candidate."""

    def __init__(
        self,
        root: str | Path,
        study_id: str,
        runtime_fingerprint: Mapping[str, Any] | None = None,
        raw_runtime_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        base = Path(root).resolve()
        self.root = _study_root(base, study_id)
        self.study_id = study_id
        self.runtime_fingerprint = dict(runtime_fingerprint or {})
        self.raw_runtime_fingerprint = dict(
            self.runtime_fingerprint if raw_runtime_fingerprint is None else raw_runtime_fingerprint
        )
        self.raw_cache_unavailable_reason: str | None = None
        self._generation_root: Path | None = None

    @property
    def generation_root(self) -> Path | None:
        return self._generation_root

    def begin_generation(self, existing: str | None = None) -> Path:
        """Create a generation, or bind this writer to a selected checkpoint."""

        if self._generation_root is not None:
            raise RuntimeError("a result generation is already active")
        if existing is None:
            generation = self.root / "generations" / f"g_{uuid.uuid4().hex[:16]}"
            generation.mkdir(parents=True, exist_ok=False)
        else:
            generation = self.generation_path(existing)
        self._generation_root = generation
        return self._generation_root

    def generation_path(self, name: str) -> Path:
        if not _GENERATION_ID.fullmatch(name):
            raise ValueError(f"invalid generation ID {name!r}; expected g_ followed by 16 hexadecimal digits")
        path = self.root / "generations" / name
        if path.resolve() != path or not path.is_dir():
            raise ValueError(f"study generation does not exist: {path}")
        return path

    def retain_case_archive(self, files: Mapping[str, str], *, require_existing: bool = False) -> Path:
        """Share one immutable replay bundle across generations with identical inputs."""

        generation = self._generation_root
        if generation is None:
            raise RuntimeError("case archiving requires an active result generation")
        # The input manifest already commits to the hashes of bundled data files.
        hashes = {name: file_sha256(generation / name) for name in files.values()}
        digest = hashlib.sha256(_stable_json(hashes).encode("utf-8")).hexdigest()
        archives = self.root / "archives"
        destination = archives / _cache_path_key(digest)
        marker = "bundle_sha256"
        if not destination.is_dir():
            if require_existing:
                raise ValueError(f"saved case archive is missing: {destination}")
            archives.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=archives, prefix=".tmp-") as temporary:
                staged = Path(temporary) / "bundle"
                shutil.copytree(generation, staged)
                (staged / marker).write_text(digest, encoding="utf-8")
                try:
                    move_artifact(staged, destination)
                except OSError:
                    if not destination.is_dir():
                        raise
                else:
                    return destination
        checksum = destination / marker
        if not checksum.is_file() or checksum.read_text(encoding="utf-8") != digest:
            raise ValueError(f"case archive identity does not match its shortened path: {destination}")
        manifest = read_json(generation / files["input_manifest"])
        hashes.update({entry["artifact"]: entry["sha256"] for entry in manifest["inputs"] if entry["artifact"]})
        for name, expected in hashes.items():
            if file_sha256(destination / name) != expected:
                raise ValueError(f"case archive content does not match its snapshot: {destination / name}")
        return destination

    def capture_generation_evidence(self, payload: dict[str, Any]) -> None:
        """Record the generation's file evidence before rendering or publication."""
        generation = self._generation_root
        if generation is None:
            raise RuntimeError("evidence capture requires an active result generation")
        payload["artifact_sha256"] = capture_evidence(
            self.root,
            {name: path for name, path in payload["artifacts"].items() if name != "generation"},
            within=generation,
        )

    def publish_generation(self, payload: dict[str, Any]) -> None:
        """Commit a result once and make the generation eligible for retention."""

        generation = self._generation_root
        if generation is None:
            raise RuntimeError("publication requires an active result generation")
        self.capture_generation_evidence(payload)
        root_result = self.root / "study_result.json"
        immutable = generation / GENERATION_RESULT_FILE
        with _store_lock(self.root, _PUBLICATION_LOCK):
            self._seal_active_generation(root_result)
            write_json(immutable, payload)
            write_json(root_result, payload)
            # A subsequent publisher also seals the old active pointer under
            # this lock if this process dies before writing the marker.
            _mark_published(immutable)
        self._generation_root = None

    def _seal_active_generation(self, root_result: Path) -> None:
        if not root_result.is_file():
            return
        active = read_json(root_result)
        declared = (active.get("artifacts") or {}).get("generation") if isinstance(active, dict) else None
        if not isinstance(declared, str):
            return
        relative = Path(declared)
        if relative.parent != Path("generations") or not _GENERATION_ID.fullmatch(relative.name):
            return
        generation = self.root / relative
        if generation.resolve() != generation:
            return
        immutable = generation / GENERATION_RESULT_FILE
        marker = generation / PUBLISHED_GENERATION_FILE
        if not immutable.is_file() or marker.is_file():
            return
        if read_json(immutable) != active:
            raise ValueError(f"active study result differs from its published generation: {immutable}")
        _mark_published(immutable)

    def raw_key(self, request: EvaluationRequest) -> str:
        return raw_evaluation_key(request, self.raw_runtime_fingerprint)

    def raw_dir(self, request: EvaluationRequest) -> Path:
        return self.root / _RAW_CACHE_DIRECTORY / _cache_path_key(self.raw_key(request))

    def load_raw(self, request: EvaluationRequest) -> RawResult | None:
        if self.raw_cache_unavailable_reason:
            return None
        path = self.raw_dir(request) / _RAW_CACHE_INDEX
        if not path.is_file():
            return None
        with _store_lock(self.root, _RAW_CACHE_LOCK):
            payload = _read_cache(path)
        if payload is None:
            return None
        if payload.get("schema") != "raw_evaluation_cache.v2":
            raise ValueError(f"raw result cache must use raw_evaluation_cache.v2: {path}")
        expected_key = self.raw_key(request)
        if payload.get("cache_key") != expected_key:
            raise ValueError(f"cached raw key does not match its shortened path: {path}")
        expected = raw_evaluation_identity(request, self.raw_runtime_fingerprint)
        if payload.get("identity") != expected:
            raise ValueError(f"cached raw identity does not match its content key: {path}")
        try:
            raw = RawResult.from_dict(payload["raw"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        recorded_hashes = payload.get("artifact_sha256")
        if not isinstance(recorded_hashes, dict) or not self._artifacts_match(raw, recorded_hashes):
            return None
        return raw if raw.ok else None

    def save_raw(self, request: EvaluationRequest, raw: RawResult) -> None:
        if not raw.ok or self.raw_cache_unavailable_reason:
            return
        path = self.raw_dir(request) / _RAW_CACHE_INDEX
        key = self.raw_key(request)
        identity = raw_evaluation_identity(request, self.raw_runtime_fingerprint)
        payload = {
            "schema": "raw_evaluation_cache.v2",
            "cache_key": key,
            "identity": identity,
            "artifact_sha256": self._artifact_hashes(raw),
            "raw": raw.to_dict(),
        }
        with _store_lock(self.root, _RAW_CACHE_LOCK):
            existing = _read_cache(path) or {}
            if existing.get("cache_key", key) != key or existing.get("identity", identity) != identity:
                raise ValueError(f"shortened raw cache path collision: {path}")
            self._write_json(path, payload)

    def _artifact_path(self, declared: object) -> Path | None:
        if not isinstance(declared, str) or not declared.strip():
            return None
        path = Path(declared)
        return path if path.is_absolute() else self.root / path

    def _artifact_hashes(self, raw: RawResult) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for name, declared in raw.artifacts.items():
            path = self._artifact_path(declared)
            if path is None or path.is_dir():
                continue
            # The empty marker deliberately makes an otherwise-successful raw
            # result non-reusable if one of its declared files was not saved.
            hashes[str(name)] = file_sha256(path) or ""
        return hashes

    def candidate_artifact_hashes(self, result: CandidateResult) -> list[dict[str, str]]:
        return [self._artifact_hashes(evaluation.raw) for evaluation in result.control_evaluations]

    def _artifacts_match(self, raw: RawResult, recorded: Mapping[str, Any]) -> bool:
        expected_files = {
            str(name): path
            for name, declared in raw.artifacts.items()
            if (path := self._artifact_path(declared)) is not None and not path.is_dir()
        }
        if set(recorded) != set(expected_files):
            return False
        for name, path in expected_files.items():
            if not recorded.get(name) or file_sha256(path) != recorded[name]:
                return False
        return True

    def save_selected_candidate(self, result: CandidateResult) -> Path:
        """Persist the selected structured evidence after ranking completes."""

        if self._generation_root is None:
            raise RuntimeError("selected candidate requires an active result generation")
        path = self._generation_root / "best_candidate.json"
        self._write_json(path, result.to_dict())
        return path

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        atomic_write_text(
            path,
            json.dumps(to_plain(payload), indent=2, ensure_ascii=False, allow_nan=False),
        )
