"""Run declarative electrical cases through the generic design-study engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .artifacts import data_file_references, file_sha256, package_source_sha256, write_json
from .case import Case, default_params
from .core.aggregation import candidate_is_pareto_eligible, candidate_rank_key, pareto_front
from .core.models import Candidate, CandidateResult, StudySpec
from .core.pipeline import StudyRunner
from .evaluation import CaseEvaluationBackend
from .metrics import constraints_from_case
from .netlist_import import flatten_netlist_file
from .problem import ParameterRole, ParameterSet, candidate_role, resolve_parameter_set
from .results import FileResultStore, best_decision_summary, pareto_front_table, quasi_static_snapshot_table
from .search import BaseOptimizer, create_optimizer, validate_proposal
from .sim_core import archive_case_bundle
from .simulation_input import resolve_solver_settings
from .solver import solver_identity
from .study_config import CaseControlPolicy, mapping, study_spec_from_case


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

    files = _built_in_file_fingerprints(case, data)

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


def _built_in_file_fingerprints(case: Case, data: Mapping[str, Any]) -> dict[str, str | None]:
    """Hash the same built-in data dependencies that artifact archiving owns."""

    files: dict[str, str | None] = {}
    for _field, declared in data_file_references(data):
        path = Path(declared)
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        files[str(path)] = file_sha256(path)
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
        "solver": solver_identity(resolve_solver_settings(case, solver_override)),
    }


def _simulation_fingerprint(case: Case, solver_override: str | None) -> dict[str, Any]:
    data = _simulation_case_data(case)
    return {
        "pcd_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "simulation_case_sha256": _sha256_json(data),
        "plugins": _plugin_fingerprints(case),
        "referenced_files": _referenced_file_fingerprints(case, data),
        "solver": solver_identity(resolve_solver_settings(case, solver_override)),
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
    evaluation = CaseEvaluationBackend(
        snapshot,
        store.root,
        solver_override,
        case_archive_root=archive_root,
        artifact_namespace=_sha256_json(simulation_fingerprint)[:16],
    )
    runner = StudyRunner(
        study=spec,
        evaluator=evaluation,
        metrics=(evaluation,),
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


def _optimizer_feedback(spec: StudySpec, result: CandidateResult) -> tuple[tuple[float, ...], dict[str, Any]]:
    rank = tuple(value if math.isfinite(value) else 1e30 for value in candidate_rank_key(spec, result))
    loss = _feasibility_first_loss(rank)
    return rank, {
        "rank": rank,
        "loss": loss,
        "objective_rank": rank[4 : 4 + len(spec.objectives)],
        "aggregates": dict(result.aggregates),
        "feasible_fraction": result.feasible_fraction,
        "success_fraction": result.success_fraction,
        "total_violation": result.total_violation,
        "control_margin": result.control_margin,
        "edge_limited": result.edge_limited,
    }


def _failed_constraints(result: CandidateResult) -> dict[str, list[str]]:
    failed: dict[str, list[str]] = {}
    for scenario in result.scenarios:
        names = sorted(item.name for item in scenario.selected.constraints if not item.satisfied)
        if names:
            failed[scenario.scenario.scenario_id] = names
    return failed


def _study_history_entry(
    trial: int,
    result: CandidateResult,
    rank: tuple[float, ...],
    optimizer_name: str,
    seed: int,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    evaluations = result.control_evaluations
    return {
        "trial": trial,
        "candidate_id": result.candidate.candidate_id,
        "optimizer": optimizer_name,
        "seed": seed,
        "proposal": dict(proposal),
        "params": dict(result.candidate.values),
        "aggregates": dict(result.aggregates),
        "scenario_count": len(result.scenarios),
        "evaluation_count": len(evaluations),
        "failed_evaluations": sum(not item.raw.ok for item in evaluations),
        "failed_constraints": _failed_constraints(result),
        "duration_s": sum(item.duration_s for item in evaluations),
        "feasible_fraction": result.feasible_fraction,
        "success_fraction": result.success_fraction,
        "total_violation": result.total_violation,
        "constraint_margins": result.constraint_margins,
        "control_margin": result.control_margin,
        "edge_limited": result.edge_limited,
        "rank": rank,
    }


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
    constant_inputs: Mapping[str, Mapping[str, Any]],
    candidate_label: str,
) -> list[dict[str, Any]]:
    """One analysis-ready row for every candidate/scenario/control solve."""

    rows: list[dict[str, Any]] = []
    constants = {
        name: value
        for role, values in constant_inputs.items()
        for name, value in _flatten_table_values(role, values).items()
    }
    candidate_values = _flatten_table_values(candidate_label, result.candidate.values)
    for scenario_result in result.scenarios:
        scenario = scenario_result.scenario
        scenario_values = _flatten_table_values("scenario", scenario.values)
        for evaluation in scenario_result.trials:
            row: dict[str, Any] = {
                **dataset_identity,
                **constants,
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
                "raw_cache_key": evaluation.raw_cache_key,
                "error": evaluation.raw.error,
                **candidate_values,
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
                        f"{prefix}.margin": constraint.margin,
                    }
                )
            rows.append(row)
    return rows


def _write_selected_projections(
    case: Case,
    result: CandidateResult,
    generation_root: Path,
    generation: Path,
    constant_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    study_mode = str(mapping(case.data.get("study"), "study").get("analysis_mode", ""))
    if study_mode != "quasi_static_snapshot":
        return {}
    filename = "snapshot_response.csv"
    table = quasi_static_snapshot_table(result)
    for role, values in constant_inputs.items():
        for name, value in values.items():
            table[f"{role}.{name}"] = value
    table.to_csv(generation_root / filename, index=False)
    return {"snapshot_response": str(generation / filename).replace("\\", "/")}


def _constant_parameter_inputs(
    parameters: ParameterSet,
    proposed_role: ParameterRole,
) -> dict[str, dict[str, Any]]:
    roles = tuple(
        role for role in ParameterRole if role not in {ParameterRole.OPERATING, ParameterRole.CONTROL, proposed_role}
    )
    return {role.value: values for role in roles if (values := parameters.defaults(role))}


def _write_pareto_projection(
    case: Case,
    study: StudySpec,
    results: list[CandidateResult],
    best: CandidateResult,
    generation_root: Path,
    generation: Path,
    dataset_id: str,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if len(study.objectives) <= 1:
        return {}, None

    eligible = tuple(result for result in results if candidate_is_pareto_eligible(study, result))
    front = pareto_front(study, results)
    filename = "pareto_front.csv"
    pareto_front_table(
        study,
        front,
        selected_candidate_id=best.candidate.candidate_id,
        dataset_id=dataset_id,
    ).to_csv(generation_root / filename, index=False)
    artifacts = {"pareto_front": str(generation / filename).replace("\\", "/")}
    summary = {
        "schema": "pareto_summary.v1",
        "scope": "declared_grid" if case.has_exact_candidate_enumeration else "observed_candidates",
        "eligibility": "complete_solver_evidence_and_full_feasibility",
        "eligible_candidates": len(eligible),
        "front_candidates": len(front),
    }
    return artifacts, summary


def resolve_study_case(
    case: Case,
    n_trials: int,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> Case:
    """Return one case whose archived plan matches the actual run settings."""

    _reject_exact_enumeration_overrides(case, n_trials, optimizer_name, seed)

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

    effective = {
        "solver": effective_solver,
        "optimizer": effective_optimizer,
        "trials": n_trials,
        "seed": effective_seed,
    }
    resolved = _updated_resolved_plan(case.resolved_plan, data, effective)
    return Case(
        path=case.path,
        data=data,
        source_data=case.source_data,
        resolved_plan=resolved,
    )


def _reject_exact_enumeration_overrides(
    case: Case,
    n_trials: int,
    optimizer_name: str | None,
    seed: int | None,
) -> None:
    if not case.has_exact_candidate_enumeration:
        return
    planned_trials, planned_optimizer, planned_seed = _planned_exact_execution(case)
    changes = [
        message
        for changed, message in (
            (n_trials != planned_trials, f"trials={n_trials} (planned {planned_trials})"),
            (
                optimizer_name is not None and optimizer_name != planned_optimizer,
                f"optimizer={optimizer_name} (planned {planned_optimizer})",
            ),
            (seed is not None and seed != planned_seed, f"seed={seed} (planned {planned_seed})"),
        )
        if changed
    ]
    if changes:
        raise ValueError(
            "pcd.rf.v1 candidate enumeration is derived from network.search and cannot be overridden: "
            + ", ".join(changes)
        )


def _planned_exact_execution(case: Case) -> tuple[int, str, int]:
    planned = dict((case.resolved_plan or {}).get("execution") or {})
    run = case.data.get("run") or {}
    optimizer_cfg = case.data.get("optimizer") or {}
    trial_value = planned.get("trials")
    if trial_value is None:
        trial_value = run.get("trials")
    planned_trials = int(1 if trial_value is None else trial_value)
    planned_optimizer = str(planned.get("optimizer", optimizer_cfg.get("name", "grid")))
    seed_value = planned.get("seed")
    if seed_value is None:
        seed_value = optimizer_cfg.get("seed")
    planned_seed = int(0 if seed_value is None else seed_value)
    return planned_trials, planned_optimizer, planned_seed


def _updated_resolved_plan(
    raw_plan: dict[str, Any] | None,
    case_data: dict[str, Any],
    effective: dict[str, Any],
) -> dict[str, Any] | None:
    resolved = deepcopy(raw_plan)
    if resolved is None:
        return None
    if dict(resolved.get("execution") or {}) != effective:
        inferences = [
            str(item) for item in resolved.get("inferences") or [] if not str(item).startswith(("run ", "compare all "))
        ]
        settings = ", ".join(f"{name}={value}" for name, value in effective.items())
        resolved["inferences"] = [*inferences, f"apply effective execution settings: {settings}"]
    resolved["execution"] = dict(effective)
    resolved["case"] = deepcopy(case_data)
    return resolved


def _dataset_identity(case: Case, store: FileResultStore, generation_root: Path) -> dict[str, Any]:
    return {
        "table_schema": "evaluation_table.v2",
        "dataset_id": f"{store.study_id}/{generation_root.name}",
        "study_id": store.study_id,
        "case_schema": str(case.authored_data.get("schema", "case_yaml.v1")),
        "resolved_case_schema": str(case.data.get("schema", "case_yaml.v1")),
        "runtime_fingerprint_sha256": _sha256_json(store.runtime_fingerprint),
        "solver_fingerprint_sha256": _sha256_json(store.raw_runtime_fingerprint.get("solver", {})),
    }


@dataclass(frozen=True, slots=True)
class _TrialBatch:
    results: list[CandidateResult]
    history: list[dict[str, Any]]
    rows: list[dict[str, Any]]


def _run_trials(
    case: Case,
    spec: StudySpec,
    runner: StudyRunner,
    optimizer: BaseOptimizer,
    n_trials: int,
    dataset_identity: Mapping[str, Any],
    constant_inputs: Mapping[str, Mapping[str, Any]],
) -> _TrialBatch:
    results: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    optimizer_name = str((case.data.get("optimizer") or {}).get("name"))
    seed = int((case.data.get("optimizer") or {}).get("seed", 0))
    candidate_label = candidate_role(case).value
    for index in range(n_trials):
        params = validate_proposal(optimizer.case, optimizer.ask())
        result = runner.evaluate_candidate(Candidate(f"trial_{index:04d}", params))
        results.append(result)
        rows.extend(_evaluation_table_rows(index, result, dataset_identity, constant_inputs, candidate_label))
        rank, feedback = _optimizer_feedback(spec, result)
        optimizer.tell(params, feedback)
        history.append(_study_history_entry(index, result, rank, optimizer_name, seed, optimizer.proposal_metadata()))
    return _TrialBatch(results, history, rows)


def _generated_path(generation: Path, name: str) -> str:
    return str(generation / name).replace("\\", "/")


def _generation_artifacts(
    generation_root: Path,
    generation: Path,
    selected_candidate_path: Path,
    selected: Mapping[str, Any],
    pareto: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "generation": str(generation).replace("\\", "/"),
        "case": _generated_path(generation, "case.yaml"),
        "input_manifest": _generated_path(generation, "input_manifest.json"),
        "best_candidate": _generated_path(generation, selected_candidate_path.name),
        "history": _generated_path(generation, "study_history.json"),
        "evaluation_table": _generated_path(generation, "evaluations.csv"),
        **selected,
        **pareto,
    }
    for key, name in (
        ("input_case", "input_case.yaml"),
        ("resolved_plan", "resolved_plan.yaml"),
        ("imported_netlist", "imported_netlist.cir"),
    ):
        if (generation_root / name).is_file():
            artifacts[key] = _generated_path(generation, name)
    return artifacts


def _finalize_study(
    case: Case,
    spec: StudySpec,
    store: FileResultStore,
    generation_root: Path,
    dataset_identity: Mapping[str, Any],
    parameters: ParameterSet,
    constant_inputs: Mapping[str, Mapping[str, Any]],
    trials: _TrialBatch,
    runner: StudyRunner,
    n_trials: int,
) -> dict[str, Any]:
    generation = generation_root.relative_to(store.root)
    search_best = min(trials.results, key=lambda item: candidate_rank_key(spec, item))
    # Candidate selection may use reusable search results, but the published
    # engineering decision must come from a normal solver execution that did
    # not read or write that cache.
    best = runner.verify_candidate(search_best)
    n_failed = sum(not evaluation.raw.ok for result in trials.results for evaluation in result.control_evaluations)
    verification_failed = sum(not evaluation.raw.ok for evaluation in best.control_evaluations)
    selected_candidate_path = store.save_selected_candidate(best)
    selected_artifacts = _write_selected_projections(case, best, generation_root, generation, constant_inputs)
    pareto_artifacts, pareto_summary = _write_pareto_projection(
        case,
        spec,
        trials.results,
        best,
        generation_root,
        generation,
        str(dataset_identity["dataset_id"]),
    )
    payload = {
        "schema": "study_result.v1",
        "study": spec.to_dict(),
        "execution": {
            "solver": str((case.data.get("solver") or {}).get("name")),
            "optimizer": str((case.data.get("optimizer") or {}).get("name")),
            "trials": n_trials,
            "seed": int((case.data.get("optimizer") or {}).get("seed", 0)),
        },
        "dataset": dict(dataset_identity),
        "parameters": {"roles": parameters.role_names(), "constant_values": dict(constant_inputs)},
        "run_root": str(store.root),
        "artifacts": _generation_artifacts(
            generation_root,
            generation,
            selected_candidate_path,
            selected_artifacts,
            pareto_artifacts,
        ),
        "n_candidates": len(trials.results),
        "n_evaluations": sum(len(item.control_evaluations) for item in trials.results),
        "n_failed_evaluations": n_failed,
        "verification": {
            "schema": "final_candidate_verification.v1",
            "cache_reused": False,
            "candidate_id": best.candidate.candidate_id,
            "n_evaluations": len(best.control_evaluations),
            "n_failed_evaluations": verification_failed,
            "status": "verified" if verification_failed == 0 else "failed",
        },
        "best": best_decision_summary(spec, best, n_failed_evaluations=n_failed),
        **({"pareto": pareto_summary} if pareto_summary is not None else {}),
    }
    # Publish only after every referenced file has been committed.
    write_json(generation_root / "study_history.json", trials.history)
    pd.DataFrame(trials.rows).to_csv(generation_root / "evaluations.csv", index=False)
    write_json(store.root / "study_result.json", payload)
    return payload


def run_case_study(
    case: Case,
    run_root: str | Path,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Optimize design candidates using feasibility-first scenario aggregation."""

    if candidate_role(case) is not ParameterRole.DESIGN:
        raise ValueError("run_case_study accepts design candidates; use run_identification for latent parameters")
    return _run_parameter_study(
        case,
        run_root,
        n_trials=n_trials,
        optimizer_name=optimizer_name,
        solver_override=solver_override,
        seed=seed,
        proposed_role=ParameterRole.DESIGN,
    )


def _run_parameter_study(
    case: Case,
    run_root: str | Path,
    *,
    n_trials: int | None,
    optimizer_name: str | None,
    solver_override: str | None,
    seed: int | None,
    proposed_role: ParameterRole,
) -> dict[str, Any]:
    """Run the shared study path for one explicit candidate parameter role."""

    if candidate_role(case) is not proposed_role:
        raise ValueError(f"study candidate role is {candidate_role(case).value!r}; expected {proposed_role.value!r}")

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

    optimizer = create_optimizer(case)
    grid_size = getattr(optimizer, "n_points", None)
    if grid_size is not None and effective_trials != int(grid_size):
        raise ValueError(f"grid optimizer requires exactly {grid_size} trials, got {effective_trials}")
    spec, runner, store = build_case_runner(case, run_root, transactional=True)
    generation_root = store.generation_root
    if generation_root is None:  # pragma: no cover - an internal construction invariant
        raise RuntimeError("study result generation was not initialized")
    dataset_identity = _dataset_identity(case, store, generation_root)
    parameters = resolve_parameter_set(case)
    constant_inputs = _constant_parameter_inputs(parameters, proposed_role)
    trials = _run_trials(
        case,
        spec,
        runner,
        optimizer,
        effective_trials,
        dataset_identity,
        constant_inputs,
    )
    return _finalize_study(
        case,
        spec,
        store,
        generation_root,
        dataset_identity,
        parameters,
        constant_inputs,
        trials,
        runner,
        effective_trials,
    )
