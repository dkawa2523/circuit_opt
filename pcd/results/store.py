"""Content-addressed storage for generic evaluation results."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcd.artifacts import artifact_path_segment, atomic_write_text
from pcd.core.models import CandidateResult, EvaluationRequest, EvaluationResult, RawResult, to_plain

_CACHE_PATH_KEY_LENGTH = 24
_STUDY_SEGMENT_MAX = 32
_STUDY_SEGMENT_MIN = 16
_WINDOWS_PATH_BUDGET = 248
_INTERNAL_PATH_RESERVE = 64


def _stable_json(value: Any) -> str:
    return json.dumps(to_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def evaluation_key(request: EvaluationRequest, runtime_fingerprint: Mapping[str, Any] | None = None) -> str:
    payload = {"request": request.to_dict(), "runtime": to_plain(runtime_fingerprint or {})}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


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


def _cache_path_key(full_key: str) -> str:
    return full_key[:_CACHE_PATH_KEY_LENGTH]


class FileResultStore:
    """Persist attributed evaluations and reuse physical evaluator output."""

    def __init__(
        self,
        root: str | Path,
        study_id: str,
        runtime_fingerprint: Mapping[str, Any] | None = None,
        raw_runtime_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        base = Path(root).resolve()
        self.root = base / _study_segment(base, study_id)
        self.study_id = study_id
        self.runtime_fingerprint = dict(runtime_fingerprint or {})
        self.raw_runtime_fingerprint = dict(
            self.runtime_fingerprint if raw_runtime_fingerprint is None else raw_runtime_fingerprint
        )

    def key(self, request: EvaluationRequest) -> str:
        return evaluation_key(request, self.runtime_fingerprint)

    def evaluation_dir(self, request: EvaluationRequest) -> Path:
        return self.root / "evaluations" / _cache_path_key(self.key(request))

    def raw_key(self, request: EvaluationRequest) -> str:
        return raw_evaluation_key(request, self.raw_runtime_fingerprint)

    def raw_dir(self, request: EvaluationRequest) -> Path:
        return self.root / "raw" / _cache_path_key(self.raw_key(request))

    def load_raw(self, request: EvaluationRequest) -> RawResult | None:
        path = self.raw_dir(request) / "raw_result.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_key = self.raw_key(request)
        if payload.get("cache_key", expected_key) != expected_key:
            raise ValueError(f"cached raw key does not match its shortened path: {path}")
        expected = raw_evaluation_identity(request, self.raw_runtime_fingerprint)
        if payload.get("identity") != expected:
            raise ValueError(f"cached raw identity does not match its content key: {path}")
        raw = RawResult.from_dict(payload["raw"])
        return raw if raw.ok else None

    def save_raw(self, request: EvaluationRequest, raw: RawResult) -> None:
        if not raw.ok:
            return
        path = self.raw_dir(request) / "raw_result.json"
        key = self.raw_key(request)
        identity = raw_evaluation_identity(request, self.raw_runtime_fingerprint)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_key = existing.get("cache_key")
            if (existing_key is not None and existing_key != key) or existing.get("identity") != identity:
                raise ValueError(f"shortened raw cache path collision: {path}")
        self._write_json(
            path,
            {"schema": "raw_evaluation_cache.v1", "cache_key": key, "identity": identity, "raw": raw.to_dict()},
        )

    def save(self, result: EvaluationResult) -> None:
        path = self.evaluation_dir(result.request) / "result.json"
        key = self.key(result.request)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("cache_key") != key:
                raise ValueError(f"shortened evaluation cache path collision: {path}")
        payload = result.to_dict()
        payload["cache_key"] = key
        self._write_json(path, payload)

    def save_candidate(self, result: CandidateResult) -> None:
        path = self.root / "candidates" / f"{artifact_path_segment(result.candidate.candidate_id)}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_id = str((existing.get("candidate") or {}).get("candidate_id", ""))
            if existing_id != result.candidate.candidate_id:
                raise ValueError(f"shortened candidate path collision: {path}")
        self._write_json(path, result.to_dict())

    def candidate_results(self) -> list[dict[str, Any]]:
        directory = self.root / "candidates"
        if not directory.exists():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        atomic_write_text(
            path,
            json.dumps(to_plain(payload), indent=2, ensure_ascii=False, allow_nan=False),
        )
