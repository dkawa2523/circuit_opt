"""Run declarative electrical cases through the generic design-study engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .artifacts import file_sha256, package_source_sha256, write_json
from .case import Case, default_params
from .core.aggregation import candidate_rank_key
from .core.models import (
    Candidate,
    CandidateResult,
    EvaluationRequest,
    MetricSet,
    RawResult,
    StudySpec,
)
from .core.pipeline import StudyRunner
from .metrics import constraints_from_case, measure_record
from .netlist_import import flatten_netlist_file
from .results import FileResultStore, best_decision_summary
from .search import create_optimizer, validate_proposal
from .sim_core import archive_case_bundle, simulate_case
from .solver import solver_identity
from .study_config import CaseControlPolicy, candidate_case, mapping, study_spec_from_case


class CaseEvaluator:
    """Run one declared electrical circuit evaluation."""

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

    def evaluate(self, request: EvaluationRequest) -> RawResult:
        digest = hashlib.sha256(
            json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        solver_name = self.solver_override
        record = simulate_case(
            self.case,
            params=request.merged_inputs(),
            run_root=self.study_root / "artifacts",
            run_id=f"e_{self.artifact_namespace}_{digest}",
            solver_override=solver_name,
            case_archive_root=self.case_archive_root,
        )
        return self._raw_from_manifest(record.manifest(), request)

    def _raw_from_manifest(
        self,
        manifest: Mapping[str, Any],
        request: EvaluationRequest,
        *,
        status: str | None = None,
        extra_artifacts: Mapping[str, str] | None = None,
    ) -> RawResult:
        run_dir = Path(str(manifest["run_dir"])).resolve()
        relative_run = run_dir.relative_to(self.study_root)
        names = mapping(manifest.get("artifacts"), "simulation_record.artifacts")
        artifacts = {
            "run_dir": str(relative_run),
            "manifest": str(relative_run / "sim_manifest.json"),
            "waveform": str(relative_run / str(names.get("waveform", "waveform.csv"))),
            "netlist": str(relative_run / str(names.get("netlist", "netlist.cir"))),
            "solver_log": str(relative_run / str(names.get("solver_log", "solver.log"))),
        }
        if names.get("frequency_response"):
            artifacts["frequency_response"] = str(relative_run / str(names["frequency_response"]))
        artifacts.update(extra_artifacts or {})
        effective_status = status or ("ok" if manifest.get("status") == "ok" else "failed")
        diagnostics = dict(manifest.get("diagnostics") or {})
        return RawResult(
            status=effective_status,
            observations={
                "case_id": manifest.get("case_id"),
                "params": manifest.get("params", {}),
                "circuit": manifest.get("circuit"),
                "load": manifest.get("load"),
                "solver": manifest.get("solver"),
                "evaluator": "circuit",
            },
            artifacts=artifacts,
            diagnostics=diagnostics,
            error=manifest.get("error"),
        )


class CaseMetrics:
    """Measure immutable simulation artifacts through the selected case metric."""

    def __init__(self, case: Case, study_root: Path) -> None:
        self.case = case
        self.study_root = study_root

    def compute(self, request: EvaluationRequest, raw: RawResult) -> MetricSet:
        del request
        manifest = self.study_root / str(raw.artifacts["manifest"])
        return MetricSet(measure_record(self.case, manifest))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plugin_fingerprints(case: Case) -> dict[str, str | None]:
    plugins: dict[str, str | None] = {}
    for raw in case.data.get("plugins") or []:
        path = Path(raw)
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        plugins[str(path)] = file_sha256(path)
    return plugins


def _referenced_file_fingerprints(case: Case, data: Mapping[str, Any]) -> dict[str, str | None]:
    """Hash declarative file inputs such as a netlist, table, or target trace."""

    files: dict[str, str | None] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_name, item in value.items():
                name = str(raw_name)
                if name.endswith("_file") and isinstance(item, str):
                    path = Path(item)
                    path = (path if path.is_absolute() else case.base_dir / path).resolve()
                    files[str(path)] = file_sha256(path)
                else:
                    visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(data)
    circuit = data.get("circuit")
    if isinstance(circuit, Mapping) and isinstance(circuit.get("netlist_file"), str):
        path = Path(circuit["netlist_file"])
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        if path.is_file():
            _flattened, dependencies = flatten_netlist_file(path)
            for dependency in dependencies:
                files[str(dependency)] = file_sha256(dependency)
    return files


def _simulation_case_data(case: Case) -> dict[str, Any]:
    """Project a case onto settings that can change evaluator output.

    Study axes, objectives, constraints, optimizer settings, and output paths
    interpret or schedule a solve; they do not change the solve. Variable
    search ranges are likewise removed, while defaults remain because they are
    applied when a request omits a value.
    """

    data = deepcopy(case.data)
    for name in ("study", "optimizer", "benchmark", "run"):
        data.pop(name, None)

    target = data.pop("target", None)
    if isinstance(target, Mapping) and "fundamental_Hz" in target:
        data["target"] = {"fundamental_Hz": target["fundamental_Hz"]}

    data.pop("variables", None)
    for section in ("source", "circuit", "load"):
        if isinstance(data.get(section), dict):
            data[section].pop("variables", None)
    for source in data.get("sources") or []:
        if isinstance(source, dict):
            source.pop("variables", None)

    defaults = default_params(case)
    if defaults:
        data["parameter_defaults"] = defaults
    return data


def _runtime_fingerprint(case: Case, solver_override: str | None) -> dict[str, Any]:
    return {
        "pcd_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "case_sha256": _sha256_json(case.data),
        "plugins": _plugin_fingerprints(case),
        "referenced_files": _referenced_file_fingerprints(case, case.data),
        "solver": solver_identity(case, solver_override),
    }


def _simulation_fingerprint(case: Case, solver_override: str | None) -> dict[str, Any]:
    data = _simulation_case_data(case)
    return {
        "pcd_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "simulation_case_sha256": _sha256_json(data),
        "plugins": _plugin_fingerprints(case),
        "referenced_files": _referenced_file_fingerprints(case, data),
        "solver": solver_identity(case, solver_override),
    }


def build_case_runner(
    case: Case,
    run_root: str | Path,
    solver_override: str | None = None,
    *,
    transactional: bool = False,
) -> tuple[StudySpec, StudyRunner, FileResultStore]:
    runtime_fingerprint = _runtime_fingerprint(case, solver_override)
    simulation_fingerprint = _simulation_fingerprint(case, solver_override)
    store = FileResultStore(
        run_root,
        case.case_id,
        runtime_fingerprint,
        raw_runtime_fingerprint=simulation_fingerprint,
    )
    archive_root = store.begin_generation() if transactional else store.root
    snapshot, _case_files = archive_case_bundle(case, archive_root)
    spec = study_spec_from_case(snapshot)
    runner = StudyRunner(
        study=spec,
        evaluator=CaseEvaluator(
            snapshot,
            store.root,
            solver_override,
            case_archive_root=archive_root,
            artifact_namespace=_sha256_json(simulation_fingerprint)[:16],
        ),
        metrics=(CaseMetrics(snapshot, store.root),),
        constraints=constraints_from_case(snapshot),
        control_policy=CaseControlPolicy(snapshot),
        store=store,
    )
    return spec, runner, store


def _feasibility_first_loss(rank: tuple[float, ...]) -> float:
    """Map the lexicographic design rank to one bounded optimizer signal.

    Complete feasible candidates always occupy ``[0, 1)``.  All other
    candidates occupy ``[1, 2)``.  Within the latter band, missing solver
    evidence, feasibility coverage, and normalized constraint violation all
    provide progress signals.  Final design selection still uses the complete
    lexicographic rank because no scalar can preserve that order exactly.
    """

    feasible = bool(rank) and rank[0] == 0.0
    if not feasible:
        success_shortfall = min(max(0.0, rank[1] if len(rank) > 1 else 1.0), 1.0)
        feasible_shortfall = min(max(0.0, rank[2] if len(rank) > 2 else 1.0), 1.0)
        violation = max(0.0, rank[3] if len(rank) > 3 else 0.0)
        normalized_violation = 2.0 * math.atan(violation) / math.pi
        progress = 0.5 * success_shortfall + 0.4 * feasible_shortfall + 0.1 * normalized_violation
        return 1.0 + min(progress, 1.0 - 1e-12)
    primary = rank[4] if len(rank) > 4 else 0.0
    bounded = 0.5 + math.atan(primary) / math.pi
    return min(max(bounded, 0.0), 1.0 - 1e-12)


def _flatten_table_values(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested mappings while retaining non-scalar values as JSON."""

    flattened: dict[str, Any] = {}
    for raw_name, value in values.items():
        name = f"{prefix}.{raw_name}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_table_values(name, value))
        elif isinstance(value, list | tuple):
            flattened[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        else:
            flattened[name] = value
    return flattened


def _evaluation_table_rows(
    trial: int,
    result: CandidateResult,
    dataset_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One analysis-ready row for every candidate/scenario/control solve."""

    rows: list[dict[str, Any]] = []
    design = _flatten_table_values("design", result.candidate.values)
    for scenario_result in result.scenarios:
        scenario = scenario_result.scenario
        scenario_values = _flatten_table_values("scenario", scenario.values)
        for evaluation in scenario_result.trials:
            row: dict[str, Any] = {
                **dataset_identity,
                "trial": trial,
                "candidate_id": result.candidate.candidate_id,
                "scenario_id": scenario.scenario_id,
                "scenario_weight": scenario.weight,
                "selected_control": evaluation is scenario_result.selected,
                "status": evaluation.raw.status,
                "feasible": evaluation.feasible,
                "total_violation": evaluation.total_violation,
                "duration_s": evaluation.duration_s,
                "from_cache": evaluation.from_cache,
                "cache_key": evaluation.cache_key,
                "raw_cache_key": evaluation.raw_cache_key,
                "error": evaluation.raw.error,
                **design,
                **scenario_values,
                **_flatten_table_values("control", evaluation.request.control.values),
                **_flatten_table_values("metric", evaluation.metrics.values),
                **_flatten_table_values("observation", evaluation.raw.observations),
                **_flatten_table_values("artifact", evaluation.raw.artifacts),
            }
            for constraint in evaluation.constraints:
                prefix = f"constraint.{constraint.name}"
                row.update(
                    {
                        f"{prefix}.satisfied": constraint.satisfied,
                        f"{prefix}.violation": constraint.violation,
                        f"{prefix}.value": constraint.value,
                        f"{prefix}.limit": constraint.limit,
                    }
                )
            rows.append(row)
    return rows


def resolve_study_case(
    case: Case,
    n_trials: int,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> Case:
    """Return one case whose archived plan matches the actual run settings."""

    if case.has_exact_candidate_enumeration:
        planned = dict((case.resolved_plan or {}).get("execution") or {})
        trial_value: Any = planned.get("trials")
        if trial_value is None:
            trial_value = (case.data.get("run") or {}).get("trials")
        planned_trials = int(1 if trial_value is None else trial_value)
        planned_optimizer = str(planned.get("optimizer", (case.data.get("optimizer") or {}).get("name", "grid")))
        seed_value: Any = planned.get("seed")
        if seed_value is None:
            seed_value = (case.data.get("optimizer") or {}).get("seed")
        planned_seed = int(0 if seed_value is None else seed_value)
        changes = []
        if int(n_trials) != planned_trials:
            changes.append(f"trials={n_trials} (planned {planned_trials})")
        if optimizer_name is not None and str(optimizer_name) != planned_optimizer:
            changes.append(f"optimizer={optimizer_name} (planned {planned_optimizer})")
        if seed is not None and int(seed) != planned_seed:
            changes.append(f"seed={seed} (planned {planned_seed})")
        if changes:
            raise ValueError(
                "pcd.rf.v1 candidate enumeration is derived from network.search and cannot be overridden: "
                + ", ".join(changes)
            )

    data = deepcopy(case.data)
    solver = mapping(data.get("solver"), "solver")
    optimizer = mapping(data.get("optimizer"), "optimizer")
    run = mapping(data.get("run"), "run")

    effective_solver = str(solver_override or solver.get("name", "ngspice_cli"))
    effective_optimizer = str(optimizer_name or optimizer.get("name", "random"))
    effective_seed = int(seed if seed is not None else optimizer.get("seed", 0))
    data["solver"] = {**solver, "name": effective_solver}
    data["optimizer"] = {**optimizer, "name": effective_optimizer, "seed": effective_seed}
    data["run"] = {**run, "trials": int(n_trials)}

    resolved = deepcopy(case.resolved_plan)
    if resolved is not None:
        effective = {
            "solver": effective_solver,
            "optimizer": effective_optimizer,
            "trials": int(n_trials),
            "seed": effective_seed,
        }
        previous = dict(resolved.get("execution") or {})
        if previous != effective:
            inferences = [
                str(item)
                for item in resolved.get("inferences") or []
                if not str(item).startswith(("run ", "compare all "))
            ]
            inferences.append(
                "apply effective execution settings: "
                + ", ".join(f"{name}={value}" for name, value in effective.items())
            )
            resolved["inferences"] = inferences
        resolved["execution"] = effective
        resolved["case"] = deepcopy(data)
    return Case(
        path=case.path,
        data=data,
        source_data=case.source_data,
        resolved_plan=resolved,
    )


def run_case_study(
    case: Case,
    run_root: str | Path,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Optimize fixed candidates using feasibility-first scenario aggregation."""

    configured_trials = int((case.data.get("run", {}) or {}).get("trials", 1))
    effective_trials = configured_trials if n_trials is None else int(n_trials)
    if effective_trials < 1:
        raise ValueError("n_trials must be positive")
    case = resolve_study_case(case, effective_trials, optimizer_name, solver_override, seed)

    # Validation belongs to the executable study boundary so CLI and Python
    # callers cannot accidentally run different preparation paths.
    from .validation import validate_case

    report = validate_case(case)
    if not report.ok:
        raise ValueError(report.format_text())

    optimizer_case = candidate_case(case)
    optimizer = create_optimizer(optimizer_case)
    grid_size = getattr(optimizer, "n_points", None)
    if grid_size is not None and effective_trials != int(grid_size):
        raise ValueError(f"grid optimizer requires exactly {grid_size} trials, got {effective_trials}")
    spec, runner, store = build_case_runner(case, run_root, transactional=True)
    generation_root = store.generation_root
    if generation_root is None:  # pragma: no cover - an internal construction invariant
        raise RuntimeError("study result generation was not initialized")
    generation = generation_root.relative_to(store.root)
    dataset_identity = {
        "table_schema": "evaluation_table.v1",
        "dataset_id": f"{store.study_id}/{generation_root.name}",
        "study_id": store.study_id,
        "case_schema": str(case.authored_data.get("schema", "case_yaml.v1")),
        "resolved_case_schema": str(case.data.get("schema", "case_yaml.v1")),
        "runtime_fingerprint_sha256": _sha256_json(store.runtime_fingerprint),
        "solver_fingerprint_sha256": _sha256_json(store.raw_runtime_fingerprint.get("solver", {})),
    }
    results = []
    history: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for index in range(effective_trials):
        params = validate_proposal(optimizer_case, optimizer.ask())
        candidate = Candidate(f"trial_{index:04d}", params)
        result = runner.evaluate_candidate(candidate)
        results.append(result)
        evaluation_rows.extend(_evaluation_table_rows(index, result, dataset_identity))
        rank = tuple(value if math.isfinite(value) else 1e30 for value in candidate_rank_key(spec, result))
        optimizer_loss = _feasibility_first_loss(rank)
        feedback = {
            "rank": rank,
            "loss": optimizer_loss,
            "objective_rank": rank[4 : 4 + len(spec.objectives)],
            "aggregates": dict(result.aggregates),
            "feasible_fraction": result.feasible_fraction,
            "success_fraction": result.success_fraction,
            "total_violation": result.total_violation,
            "control_margin": result.control_margin,
            "edge_limited": result.edge_limited,
        }
        optimizer.tell(params, feedback)
        history.append(
            {
                "trial": index,
                "candidate_id": candidate.candidate_id,
                "params": params,
                "aggregates": dict(result.aggregates),
                "feasible_fraction": result.feasible_fraction,
                "success_fraction": result.success_fraction,
                "total_violation": result.total_violation,
                "control_margin": result.control_margin,
                "edge_limited": result.edge_limited,
                "rank": rank,
                "optimizer_loss": optimizer_loss,
            }
        )

    ordered = sorted(results, key=lambda item: candidate_rank_key(spec, item))
    best = ordered[0]
    n_failed = sum(not evaluation.raw.ok for result in results for evaluation in result.control_evaluations)

    def generated(name: str) -> str:
        return str(generation / name).replace("\\", "/")

    payload = {
        "schema": "study_result.v1",
        "study": spec.to_dict(),
        "execution": {
            "solver": str((case.data.get("solver") or {}).get("name")),
            "optimizer": str((case.data.get("optimizer") or {}).get("name")),
            "trials": effective_trials,
            "seed": int((case.data.get("optimizer") or {}).get("seed", 0)),
        },
        "dataset": dataset_identity,
        "run_root": str(store.root),
        "artifacts": {
            "generation": str(generation).replace("\\", "/"),
            "case": generated("case.yaml"),
            "input_manifest": generated("input_manifest.json"),
            "candidate_directory": generated("candidates"),
            "history": generated("study_history.json"),
            "evaluation_table": generated("evaluations.csv"),
            **({"input_case": generated("input_case.yaml")} if (generation_root / "input_case.yaml").is_file() else {}),
            **(
                {"resolved_plan": generated("resolved_plan.yaml")}
                if (generation_root / "resolved_plan.yaml").is_file()
                else {}
            ),
            **(
                {"imported_netlist": generated("imported_netlist.cir")}
                if (generation_root / "imported_netlist.cir").is_file()
                else {}
            ),
        },
        "n_candidates": len(results),
        "n_evaluations": sum(len(item.control_evaluations) for item in results),
        "n_failed_evaluations": n_failed,
        "best": best_decision_summary(spec, best, n_failed_evaluations=n_failed),
        "optimizer_state": optimizer.state(),
    }
    # Everything referenced by the root result exists before this final atomic
    # replacement.  An interrupted rerun therefore leaves the previous complete
    # generation authoritative instead of publishing a mixed candidate set.
    write_json(generation_root / "study_history.json", history)
    pd.DataFrame(evaluation_rows).to_csv(generation_root / "evaluations.csv", index=False)
    write_json(store.root / "study_result.json", payload)
    return payload
