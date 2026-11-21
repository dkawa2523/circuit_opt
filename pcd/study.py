"""Run declarative electrical cases through the generic design-study engine."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    data_file_references,
    file_sha256,
    package_source_sha256,
    read_json,
    runtime_environment,
    write_json,
)
from .case import Case, default_params
from .core.aggregation import candidate_is_pareto_eligible, candidate_rank_key, pareto_front
from .core.models import Candidate, CandidateResult, StudySpec
from .core.pipeline import StudyRunner
from .evaluation import CaseEvaluationBackend
from .inputs import integer, read_input_mapping, reject_unknown
from .metrics import constraints_from_case
from .netlist_import import flatten_netlist_file, stochastic_netlist_function
from .problem import ParameterRole, ParameterSet, candidate_role, resolve_parameter_set
from .registry import declared_plugin_paths, load_plugin_files, loaded_plugin_digest
from .results import FileResultStore, best_decision_summary, pareto_front_table, quasi_static_snapshot_table
from .results.journal import TrialJournal, select_checkpoint
from .results.table import evaluation_table
from .search import BaseOptimizer, GridOptimizer, create_optimizer, validate_proposal
from .sim_core import archive_case_bundle
from .simulation_input import SolverSettings, resolve_solver_settings
from .solver import bind_solver_settings, solver_identity
from .study_config import CaseControlPolicy, mapping, study_spec_from_case

_TRIAL_REPLAY_OPTIMIZERS = {"grid", "random", "differential_evolution"}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plugin_fingerprints(case: Case) -> dict[str, str | None]:
    plugins: dict[str, str | None] = {}
    for path in declared_plugin_paths(case.data.get("plugins"), case.base_dir):
        plugins[str(path)] = loaded_plugin_digest(path) or file_sha256(path)
    return plugins


def _referenced_file_fingerprints(case: Case, data: Mapping[str, Any]) -> dict[str, str | None]:
    """Hash declarative file inputs such as a netlist, table, or target trace."""

    files: dict[str, str | None] = {}
    for _field, declared in data_file_references(data):
        path = Path(declared)
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        files[str(path)] = file_sha256(path)
    circuit = data.get("circuit")
    if isinstance(circuit, Mapping) and isinstance(circuit.get("netlist_file"), str):
        path = Path(circuit["netlist_file"])
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        if path.is_file():
            _flattened, dependencies = flatten_netlist_file(path, mode=circuit.get("netlist_mode", "fragment"))
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
    for name in ("study", "optimizer", "benchmark", "run", "output"):
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


def _captured_file_fingerprints(data: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, str | None]:
    fields = {field for field, _declared in data_file_references(data)}
    return {
        entry["source_path"]: entry["sha256"]
        for entry in manifest["inputs"]
        if entry["field"] in fields
        or ("$.circuit.netlist_file" in fields and entry["field"].startswith("$.circuit.netlist_file.dependencies["))
    }


def _runtime_fingerprint(
    case: Case,
    solver_override: str | None,
    manifest: Mapping[str, Any] | None = None,
    solver_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pcd_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "environment": runtime_environment(),
        "case_sha256": _sha256_json(case.data),
        "plugins": _plugin_fingerprints(case),
        "referenced_files": _referenced_file_fingerprints(case, case.data)
        if manifest is None
        else _captured_file_fingerprints(case.data, manifest),
        "solver": dict(solver_facts)
        if solver_facts is not None
        else solver_identity(resolve_solver_settings(case, solver_override)),
    }


def _simulation_fingerprint(
    case: Case,
    solver_override: str | None,
    manifest: Mapping[str, Any] | None = None,
    solver_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = _simulation_case_data(case)
    return {
        "pcd_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "simulation_case_sha256": _sha256_json(data),
        "plugins": _plugin_fingerprints(case),
        "referenced_files": _referenced_file_fingerprints(case, data)
        if manifest is None
        else _captured_file_fingerprints(data, manifest),
        "solver": dict(solver_facts)
        if solver_facts is not None
        else solver_identity(resolve_solver_settings(case, solver_override)),
    }


def _identify_store_inputs(
    case: Case, store: FileResultStore, archive: Path, files: Mapping[str, str], solver_override: str | None
) -> SolverSettings:
    manifest = read_json(archive / files["input_manifest"])
    load_plugin_files(case.data.get("plugins"), case.base_dir)
    solver = resolve_solver_settings(case, solver_override)
    if solver.name == "ngspice_cli" and (imported := files.get("imported_netlist")):
        mode = (case.data.get("circuit") or {}).get("netlist_mode", "fragment")
        if function := stochastic_netlist_function((archive / imported).read_text(encoding="utf-8"), mode=mode):
            raise ValueError(
                f"deterministic studies do not support ngspice random function {function!r} in imported SPICE; "
                "use explicit operating scenarios or a forward simulation (sim-run)"
            )
    solver = bind_solver_settings(solver)
    solver_facts = solver_identity(solver)
    store.runtime_fingerprint = _runtime_fingerprint(case, solver_override, manifest, solver_facts)
    store.raw_runtime_fingerprint = _simulation_fingerprint(case, solver_override, manifest, solver_facts)
    if (solver.startup_fingerprint or {}).get("unresolved_sources"):
        store.raw_cache_unavailable_reason = "ngspice startup uses a dependency path that could not be resolved"
    return solver


def build_case_runner(
    case: Case,
    run_root: str | Path,
    solver_override: str | None = None,
    *,
    transactional: bool = False,
) -> tuple[StudySpec, StudyRunner, FileResultStore]:
    store = FileResultStore(run_root, case.case_id)
    archive_root = store.begin_generation() if transactional else store.root
    snapshot, case_files = archive_case_bundle(case, archive_root)
    solver = _identify_store_inputs(case, store, archive_root, case_files, solver_override)
    replay_archive = store.retain_case_archive(case_files) if transactional else archive_root
    return _case_runner_from_archive(snapshot, store, solver_override, replay_archive, solver)


def _case_runner_from_archive(
    snapshot: Case, store: FileResultStore, solver_override: str | None, replay_archive: Path, solver: SolverSettings
) -> tuple[StudySpec, StudyRunner, FileResultStore]:
    spec = study_spec_from_case(snapshot)
    evaluation = CaseEvaluationBackend(
        snapshot,
        store.root,
        solver_override,
        case_archive_root=replay_archive,
        artifact_namespace=_sha256_json(store.raw_runtime_fingerprint)[:16],
        solver_settings=solver,
    )
    runner = StudyRunner(
        study=spec,
        evaluator=evaluation,
        metrics=(evaluation,),
        constraints=constraints_from_case(snapshot),
        control_policy=CaseControlPolicy(snapshot, spec.scenarios),
        store=store,
    )
    return spec, runner, store


def _feasibility_first_loss(rank: tuple[float, ...]) -> float:
    """Map the lexicographic design rank to one bounded optimizer signal.

    Accepted, measured rejected, and incomplete candidates occupy ``[0, 1)``,
    ``[1, 1.5)``, and ``[1.5, 2)`` respectively. Within a band the scalar is a
    search heuristic. Final design selection uses the full lexicographic rank.
    """

    tier, evidence_shortfall, feasible_shortfall, violation, primary, *_ = rank
    if tier:
        normalized_violation = 2.0 * math.atan(violation) / math.pi
        progress = 0.5 * evidence_shortfall + 0.4 * feasible_shortfall + 0.1 * normalized_violation
        if tier == 1.0:
            return 1.0 + min(progress, 0.5 - 1e-12)
        return 1.5 + 0.5 * min(progress, 1.0 - 1e-12)
    bounded = 0.5 + math.atan(primary) / math.pi
    return min(max(bounded, 0.0), 1.0 - 1e-12)


def _optimizer_feedback(spec: StudySpec, result: CandidateResult) -> tuple[tuple[float, ...], dict[str, Any]]:
    rank = candidate_rank_key(spec, result)
    plugin_rank = tuple(value if math.isfinite(value) else sys.float_info.max for value in rank)
    loss = _feasibility_first_loss(plugin_rank)
    return rank, {
        "rank": plugin_rank,
        "loss": loss,
        "objective_rank": plugin_rank[4 : 4 + len(spec.objectives)],
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
        "rank": [value if math.isfinite(value) else None for value in rank],
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
                "control": json.dumps(evaluation.request.control.to_dict()["values"], sort_keys=True, allow_nan=False),
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
    return {"snapshot_response": (generation / filename).as_posix()}


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
    eligible_candidates: int,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if len(study.objectives) <= 1:
        return {}, None

    front = pareto_front(study, results)
    filename = "pareto_front.csv"
    pareto_front_table(
        study,
        front,
        selected_candidate_id=best.candidate.candidate_id,
        dataset_id=dataset_id,
    ).to_csv(generation_root / filename, index=False)
    artifacts = {"pareto_front": (generation / filename).as_posix()}
    summary = {
        "schema": "pareto_summary.v1",
        "scope": "declared_grid" if case.has_exact_candidate_enumeration else "observed_candidates",
        "eligibility": "complete_solver_evidence_and_full_feasibility",
        "eligible_candidates": eligible_candidates,
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
    reject_unknown(run, {"trials", "root"}, "run")

    effective_solver = str(solver.get("name", "ngspice_cli") if solver_override is None else solver_override)
    effective_optimizer = str(optimizer.get("name", "random") if optimizer_name is None else optimizer_name)
    effective_seed = integer(seed if seed is not None else optimizer.get("seed", 0), "optimizer.seed", minimum=0)
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
            "exact candidate enumeration is derived from declared finite choices and cannot be overridden: "
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


@dataclass(slots=True)
class _TrialBatch:
    best: CandidateResult | None = None
    best_rank: tuple[float, ...] | None = None
    front: list[CandidateResult] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    n_evaluations: int = 0
    n_failed: int = 0
    missing_acceptance_evidence: bool = False
    eligible_candidates: int = 0
    optimizer_outcome: dict[str, Any] = field(default_factory=dict)
    restart: dict[str, Any] = field(default_factory=dict)

    def include(self, spec: StudySpec, result: CandidateResult, rank: tuple[float, ...]) -> None:
        """Keep detailed trials only for the best candidate and observed front."""

        if self.best_rank is None or rank < self.best_rank:
            self.best = result
            self.best_rank = rank
        self.n_evaluations += len(result.control_evaluations)
        self.n_failed += sum(not evaluation.raw.ok for evaluation in result.control_evaluations)
        self.missing_acceptance_evidence |= any(
            evaluation.raw.ok and not evaluation.evidence_available for evaluation in result.control_evaluations
        )
        if len(spec.objectives) > 1:
            self.eligible_candidates += candidate_is_pareto_eligible(spec, result)
            self.front = list(pareto_front(spec, (*self.front, result)))


def _run_trials(
    case: Case,
    spec: StudySpec,
    runner: StudyRunner,
    optimizer: BaseOptimizer,
    n_trials: int,
    dataset_identity: Mapping[str, Any],
    constant_inputs: Mapping[str, Mapping[str, Any]],
    table_path: Path,
    journal: TrialJournal | None = None,
) -> _TrialBatch:
    batch = _TrialBatch()
    optimizer_name = str((case.data.get("optimizer") or {}).get("name"))
    seed = int((case.data.get("optimizer") or {}).get("seed", 0))
    candidate_label = candidate_role(case).value
    replayed_evaluations = 0
    new_raw_cache_hits = 0
    started_candidates = 0
    with evaluation_table(table_path) as append:

        def evaluate(proposal: dict[str, Any]) -> dict[str, Any]:
            nonlocal replayed_evaluations, new_raw_cache_hits, started_candidates
            if started_candidates >= n_trials:
                raise ValueError(f"optimizer {optimizer_name!r} exceeded the candidate budget of {n_trials}")
            params = validate_proposal(optimizer.case, proposal)
            started_candidates += 1
            index = len(batch.history)
            candidate = Candidate(f"trial_{index:04d}", params)
            if journal is not None and index < journal.completed_trials:
                result = journal.replay(index, candidate)
                replayed_evaluations += len(result.control_evaluations)
            else:
                result = runner.evaluate_candidate(candidate)
                if journal is not None:
                    journal.save(index, result)
                    new_raw_cache_hits += sum(item.from_cache for item in result.control_evaluations)
            rank, feedback = _optimizer_feedback(spec, result)
            batch.include(spec, result, rank)
            append(_evaluation_table_rows(index, result, dataset_identity, constant_inputs, candidate_label))
            feedback.update(optimizer.feedback_details(result))
            optimizer.tell(params, feedback)
            batch.history.append(
                _study_history_entry(index, result, rank, optimizer_name, seed, optimizer.proposal_metadata())
            )
            return feedback

        optimizer.optimize(evaluate, n_trials)
        batch.optimizer_outcome = dict(optimizer.outcome)
    if journal is not None:
        batch.restart = journal.summary(
            replayed_candidates=journal.completed_trials,
            replayed_evaluations=replayed_evaluations,
            new_candidates=len(batch.history) - journal.completed_trials,
            new_evaluations=batch.n_evaluations - replayed_evaluations,
            new_evaluations_from_raw_cache=new_raw_cache_hits,
        )
    elif candidate_role(case) is ParameterRole.DESIGN:
        batch.restart = {"scope": "raw_cache_only", "reason": f"optimizer {optimizer_name!r} does not support replay"}
    return batch


def _generated_path(generation: Path, name: str) -> str:
    return (generation / name).as_posix()


def _generation_artifacts(
    generation_root: Path,
    generation: Path,
    selected_candidate_path: Path,
    selected: Mapping[str, Any],
    pareto: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "generation": generation.as_posix(),
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
    search_best = trials.best
    if search_best is None:
        raise ValueError("optimizer produced no candidate evaluations")
    # Candidate selection may use reusable search results, but the published
    # engineering decision must come from a normal solver execution that did
    # not read or write that cache.
    best = runner.verify_candidate(search_best)
    n_failed = trials.n_failed
    verification_failed = sum(not evaluation.raw.ok for evaluation in best.control_evaluations)
    verification_complete = all(evaluation.evidence_available for evaluation in best.control_evaluations)
    verification_status = (
        "failed" if verification_failed else "verified" if verification_complete else "incomplete_evidence"
    )
    selected_candidate_path = store.save_selected_candidate(best)
    selected_artifacts = _write_selected_projections(case, best, generation_root, generation, constant_inputs)
    pareto_artifacts, pareto_summary = _write_pareto_projection(
        case,
        spec,
        trials.front,
        best,
        generation_root,
        generation,
        str(dataset_identity["dataset_id"]),
        trials.eligible_candidates,
    )
    payload: dict[str, Any] = {
        "schema": "study_result.v1",
        "study": spec.to_dict(),
        "execution": {
            "solver": str((case.data.get("solver") or {}).get("name")),
            "optimizer": str((case.data.get("optimizer") or {}).get("name")),
            "trials": n_trials,
            "optimizer_outcome": trials.optimizer_outcome,
            "seed": int((case.data.get("optimizer") or {}).get("seed", 0)),
            **({"restart": trials.restart} if trials.restart else {}),
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
        "n_candidates": len(trials.history),
        "n_evaluations": trials.n_evaluations,
        "n_failed_evaluations": n_failed,
        "verification": {
            "schema": "final_candidate_verification.v1",
            "cache_reused": False,
            "candidate_id": best.candidate.candidate_id,
            "n_evaluations": len(best.control_evaluations),
            "n_failed_evaluations": verification_failed,
            "status": verification_status,
        },
        "best": best_decision_summary(
            spec,
            best,
            n_failed_evaluations=n_failed + verification_failed,
            missing_acceptance_evidence=trials.missing_acceptance_evidence,
        ),
        **({"pareto": pareto_summary} if pareto_summary is not None else {}),
    }
    # Publish only after every referenced file has been committed.
    write_json(generation_root / "study_history.json", trials.history)
    store.capture_generation_evidence(payload)
    if "output" in case.data and candidate_role(case) is ParameterRole.DESIGN:
        from .reporting import render_requested_output
        from .reporting.study import render_study_outputs

        payload["output"] = render_requested_output(lambda: render_study_outputs(payload, case, generation_root))
        if summary := payload["output"]["summary"]:
            payload["artifacts"]["output_summary"] = summary
    store.publish_generation(payload)
    return payload


def run_case_study(
    case: Case,
    run_root: str | Path,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
    *,
    resume: str | None = None,
) -> dict[str, Any]:
    """Optimize design candidates, or resume an exact built-in search checkpoint.

    Use ``resume="auto"`` to find a compatible attempt in the same study root,
    or pass its generation ID to select it explicitly.
    """

    if case.is_forward_simulation:
        raise ValueError("this case describes a forward simulation; use sim-run, or declare an objective for a study")
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
        resume=resume,
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
    resume: str | None = None,
) -> dict[str, Any]:
    """Run the shared study path for one explicit candidate parameter role."""

    if candidate_role(case) is not proposed_role:
        raise ValueError(f"study candidate role is {candidate_role(case).value!r}; expected {proposed_role.value!r}")

    configured_trials = integer((case.data.get("run", {}) or {}).get("trials", 1), "run.trials", minimum=1)
    effective_trials = configured_trials if n_trials is None else integer(n_trials, "n_trials", minimum=1)
    case = resolve_study_case(case, effective_trials, optimizer_name, solver_override, seed)
    name = str(case.data["optimizer"]["name"])
    # These registered built-ins use the deterministic sequential ask/tell loop.
    # Plugin registration cannot replace their names.
    restartable = proposed_role is ParameterRole.DESIGN and name in _TRIAL_REPLAY_OPTIMIZERS
    if resume is not None and not restartable:
        raise ValueError(f"optimizer {name!r} does not support study checkpoint replay")

    # Validation belongs to the executable study boundary so CLI and Python
    # callers cannot accidentally run different preparation paths.
    from .validation import validate_case

    report = validate_case(case)
    if not report.ok:
        raise ValueError(report.format_text())

    optimizer = create_optimizer(case)
    if isinstance(optimizer, GridOptimizer):
        optimizer.validate_budget(effective_trials)
    if restartable:
        return _run_checkpointed_study(case, run_root, optimizer, effective_trials, resume)
    prepared = build_case_runner(case, run_root, transactional=True)
    return _run_prepared_study(case, optimizer, effective_trials, prepared)


def _run_checkpointed_study(
    case: Case, run_root: str | Path, optimizer: BaseOptimizer, n_trials: int, resume: str | None
) -> dict[str, Any]:
    store = FileResultStore(run_root, case.case_id)
    generation: Path | None = None
    prepared_archive = None
    if resume is None:
        generation = store.begin_generation()
        snapshot, files = archive_case_bundle(case, generation)
        solver = _identify_store_inputs(case, store, generation, files, None)
        prepared_archive = (snapshot, files, store.retain_case_archive(files))
    else:
        with tempfile.TemporaryDirectory(prefix="pcd-resume-") as staging:
            _, files = archive_case_bundle(case, staging)
            solver = _identify_store_inputs(case, store, Path(staging), files, None)
    if reason := store.raw_cache_unavailable_reason:
        if prepared_archive is None:
            raise ValueError(f"checkpoint replay unavailable: {reason}")
        snapshot, _files, archive = prepared_archive
        prepared = _case_runner_from_archive(snapshot, store, None, archive, solver)
        return _run_prepared_study(case, optimizer, n_trials, prepared)
    # Variable order affects enumeration and assigns random draws to axes. The
    # usual provenance digest is order-independent, so also pin declaration order.
    ordered_case = json.dumps(case.data, ensure_ascii=False, default=str).encode("utf-8")
    identity = {"runtime": store.runtime_fingerprint, "ordered_case_sha256": hashlib.sha256(ordered_case).hexdigest()}
    if resume is not None:
        generation = store.begin_generation(select_checkpoint(store, identity, resume))
    if generation is None:
        raise RuntimeError("study result generation was not initialized")
    journal = TrialJournal(store, generation, identity)
    if prepared_archive is None and not journal.header_path.is_file():
        raise ValueError(f"study generation {generation.name} has no trial checkpoint")
    with journal.locked():
        if prepared_archive is not None:
            snapshot, files, archive = prepared_archive
            journal.start(files, archive)
        else:
            journal.load(n_trials)
            if published := journal.published_result():
                return published
            files, archive = journal.resume_archive()
            snapshot = _load_study_snapshot(archive, files)
        prepared = _case_runner_from_archive(snapshot, store, None, archive, solver)
        return _run_prepared_study(case, optimizer, n_trials, prepared, journal)


def _load_study_snapshot(root: Path, files: Mapping[str, str]) -> Case:
    path = root / files["case"]
    return Case(
        path=path,
        data=read_input_mapping(path),
        source_data=read_input_mapping(root / files["input_case"]) if "input_case" in files else None,
        resolved_plan=read_input_mapping(root / files["resolved_plan"]) if "resolved_plan" in files else None,
    )


def _run_prepared_study(
    case: Case,
    optimizer: BaseOptimizer,
    effective_trials: int,
    prepared: tuple[StudySpec, StudyRunner, FileResultStore],
    journal: TrialJournal | None = None,
) -> dict[str, Any]:
    spec, runner, store = prepared
    generation_root = store.generation_root
    if generation_root is None:
        raise RuntimeError("study result generation was not initialized")
    dataset_identity = _dataset_identity(case, store, generation_root)
    parameters = resolve_parameter_set(case)
    constant_inputs = _constant_parameter_inputs(parameters, candidate_role(case))
    trials = _run_trials(
        case,
        spec,
        runner,
        optimizer,
        effective_trials,
        dataset_identity,
        constant_inputs,
        generation_root / "evaluations.csv",
        journal,
    )
    if store.raw_cache_unavailable_reason:
        trials.restart = {"scope": "unavailable", "reason": store.raw_cache_unavailable_reason}
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
