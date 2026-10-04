"""One electrical evaluation boundary shared by every study workflow.

The backend translates a role-separated evaluation request into one canonical
solver run and measures that response.  Study orchestration owns scenarios,
constraints, and ranking; this module owns none of those policies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .case import Case
from .core.models import EvaluationRequest, MetricSet, RawResult
from .metrics import measure_response
from .records import load_simulation_result, read_sim_record
from .sim_core import DEBUG_MANIFEST_FILE, SimRecord, execute_case
from .simulation import SimulationResult


class CaseEvaluationBackend:
    """Execute and measure one Case through the canonical response type."""

    def __init__(
        self,
        case: Case,
        study_root: Path,
        solver_override: str | None = None,
        *,
        case_archive_root: Path | None = None,
        artifact_namespace: str = "default",
    ) -> None:
        self.case = case
        self.study_root = study_root
        self.solver_override = solver_override
        self.case_archive_root = case_archive_root or study_root
        self.artifact_namespace = artifact_namespace
        self._fresh_responses: dict[str, tuple[SimulationResult, dict[str, Any]]] = {}

    def evaluate(self, request: EvaluationRequest) -> RawResult:
        """Run one request and retain its in-memory response until measurement."""

        digest = hashlib.sha256(
            json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        run = execute_case(
            self.case,
            params=request.merged_inputs(),
            run_root=self.study_root / "artifacts",
            run_id=f"e_{self.artifact_namespace}_{digest}",
            solver_override=self.solver_override,
            case_archive_root=self.case_archive_root,
        )
        raw = self._raw_result(run.record)
        if raw.ok:
            key = str(raw.artifacts["manifest"])
            self._fresh_responses[key] = (run.response, dict(run.record.params))
        return raw

    def compute(self, request: EvaluationRequest, raw: RawResult) -> MetricSet:
        """Measure a fresh response in memory or restore the same cached shape."""

        del request
        key = str(raw.artifacts["manifest"])
        fresh = self._fresh_responses.pop(key, None)
        if fresh is None:
            record = read_sim_record(self.study_root / key)
            response = load_simulation_result(record)
            params = dict(record.get("params") or {})
        else:
            response, params = fresh
        return MetricSet(measure_response(self.case, response, params))

    def _raw_result(self, record: SimRecord) -> RawResult:
        run_dir = record.run_dir.resolve()
        relative_run = run_dir.relative_to(self.study_root)
        manifest = record.manifest()
        names = manifest["artifacts"]
        artifacts = {
            "manifest": str(relative_run / DEBUG_MANIFEST_FILE),
            "waveform": str(relative_run / str(names["waveform"])),
            "netlist": str(relative_run / str(names["netlist"])),
            "solver_log": str(relative_run / str(names["solver_log"])),
        }
        if names.get("frequency_response"):
            artifacts["frequency_response"] = str(relative_run / str(names["frequency_response"]))
        return RawResult(
            status="ok" if record.status == "ok" else "failed",
            artifacts=artifacts,
            diagnostics=dict(record.diagnostics),
            error=record.error,
        )
