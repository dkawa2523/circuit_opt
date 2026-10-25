"""Identify bounded latent circuit parameters without a second solver path.

Identification is deliberately an orchestration layer.  It splits declared
operating scenarios into fit and holdout sets, reuses the ordinary study and
ngspice path for both, and performs a small local sensitivity check before a
fit may be reported as identified.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import create_run_directory, write_json
from .case import Case, default_params
from .core.models import Candidate, CandidateResult, Scenario
from .core.pipeline import StudyRunner
from .core.spaces import ContinuousAxis
from .inputs import reject_unknown
from .metrics import TERMINAL_VI_PARAMETERS
from .problem import ParameterRole, resolve_parameter_set
from .results import read_best_candidate, selected_evaluation
from .sim_core import archive_case_bundle
from .study import _run_parameter_study, build_case_runner
from .study_config import CaseControlPolicy, mapping, study_spec_from_case


@dataclass(frozen=True, slots=True)
class IdentificationSettings:
    fit_scenarios: tuple[str, ...]
    holdout_scenarios: tuple[str, ...]
    observed_metrics: tuple[str, ...]
    metric_scales: Mapping[str, float]
    fit_loss_max: float
    holdout_loss_max: float
    sensitivity_step: float
    condition_number_max: float
    rank_tolerance: float
    expected: Mapping[str, float]
    recovery_relative_tolerance: float
    holdout_mode: str = "new_conditions"
    condition_parameters: tuple[str, ...] = ()
    observation_parameters: tuple[str, ...] = ()

    @classmethod
    def from_case(cls, case: Case) -> IdentificationSettings:
        cfg = mapping(case.data.get("identification"), "identification")
        reject_unknown(
            cfg,
            {
                "fit_scenarios",
                "holdout_scenarios",
                "observed_metrics",
                "metric_scales",
                "fit_loss_max",
                "holdout_loss_max",
                "sensitivity_step",
                "condition_number_max",
                "rank_tolerance",
                "expected",
                "recovery_relative_tolerance",
                "holdout_mode",
                "condition_parameters",
                "observation_parameters",
            },
            "identification",
        )
        fit = _string_list(cfg.get("fit_scenarios"), "identification.fit_scenarios")
        holdout = _string_list(cfg.get("holdout_scenarios"), "identification.holdout_scenarios")
        observed = _string_list(cfg.get("observed_metrics"), "identification.observed_metrics")
        scales = {
            name: _positive_number(value, f"identification.metric_scales.{name}")
            for name, value in mapping(cfg.get("metric_scales"), "identification.metric_scales").items()
        }
        missing_scales = sorted(set(observed) - set(scales))
        if missing_scales:
            raise ValueError(f"identification.metric_scales is missing {missing_scales}")
        if extra_scales := sorted(set(scales) - set(observed)):
            raise ValueError(f"identification.metric_scales contains unused metrics {extra_scales}")
        if set(fit) & set(holdout):
            raise ValueError("identification fit_scenarios and holdout_scenarios must be disjoint")
        mode = cfg.get("holdout_mode", "new_conditions")
        if mode not in {"new_conditions", "replicates"}:
            raise ValueError("identification.holdout_mode must be new_conditions or replicates")
        step = _positive_number(cfg.get("sensitivity_step", 0.02), "identification.sensitivity_step")
        if step >= 0.5:
            raise ValueError("identification.sensitivity_step must be less than 0.5")
        expected = {
            name: _finite_number(value, f"identification.expected.{name}")
            for name, value in mapping(cfg.get("expected"), "identification.expected").items()
        }
        return cls(
            fit_scenarios=fit,
            holdout_scenarios=holdout,
            observed_metrics=observed,
            metric_scales=scales,
            fit_loss_max=_positive_number(cfg.get("fit_loss_max"), "identification.fit_loss_max"),
            holdout_loss_max=_positive_number(cfg.get("holdout_loss_max"), "identification.holdout_loss_max"),
            sensitivity_step=step,
            condition_number_max=_positive_number(
                cfg.get("condition_number_max", 1e6),
                "identification.condition_number_max",
            ),
            rank_tolerance=_positive_number(cfg.get("rank_tolerance", 1e-6), "identification.rank_tolerance"),
            expected=expected,
            recovery_relative_tolerance=_positive_number(
                cfg.get("recovery_relative_tolerance", 0.1),
                "identification.recovery_relative_tolerance",
            ),
            holdout_mode=mode,
            condition_parameters=_string_list(cfg["condition_parameters"], "identification.condition_parameters")
            if "condition_parameters" in cfg
            else (),
            observation_parameters=_string_list(cfg["observation_parameters"], "identification.observation_parameters")
            if "observation_parameters" in cfg
            else (),
        )


def _string_list(raw: Any, path: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must be a non-empty list")
    values = tuple(str(value).strip() for value in raw)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{path} must contain unique non-empty names")
    return values


def _finite_number(raw: Any, path: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _positive_number(raw: Any, path: str) -> float:
    value = _finite_number(raw, path)
    if value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def _scenario_catalog(case: Case) -> dict[str, Scenario]:
    scenarios = study_spec_from_case(case).scenarios
    return {scenario.scenario_id: scenario for scenario in scenarios}


def _inline_scenario(scenario: Scenario, controls: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": scenario.scenario_id,
        "values": dict(scenario.values),
        "weight": scenario.weight,
        "controls": dict(controls),
    }


def _phase_case(
    case: Case,
    phase: str,
    scenarios: Sequence[Scenario],
    fixed_latent: Mapping[str, Any] | None = None,
) -> Case:
    data = deepcopy(case.data)
    data["case_id"] = phase
    # The split/acceptance contract is owned by this outer workflow.  Each
    # phase is an ordinary executable study and should validate only its own
    # resolved candidate space and scenarios.
    data.pop("identification", None)
    study = mapping(data.get("study"), "study")
    study.pop("scenario_table", None)
    original_study = study_spec_from_case(case)
    policy = CaseControlPolicy(case)
    study["scenarios"] = [
        _inline_scenario(scenario, policy.controls(original_study, Candidate("identification"), scenario)[0].values)
        for scenario in scenarios
    ]
    controls = mapping(study.get("controls"), "study.controls")
    controls.pop("by_scenario", None)
    study["controls"] = controls
    study["candidate_role"] = ParameterRole.LATENT.value
    data["study"] = study
    if fixed_latent is not None:
        _fix_latent_specs(data, fixed_latent)
        data["optimizer"] = {"name": "grid", "seed": 0}
        data["run"] = {**mapping(data.get("run"), "run"), "trials": 1}
    return Case(path=case.path, data=data, source_data=case.source_data)


def _variable_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(data.get("variables"), dict):
        blocks.append(data["variables"])
    for section in ("source", "circuit", "load"):
        block = data.get(section)
        if isinstance(block, dict) and isinstance(block.get("variables"), dict):
            blocks.append(block["variables"])
    for source in data.get("sources") or []:
        if isinstance(source, dict) and isinstance(source.get("variables"), dict):
            blocks.append(source["variables"])
    return blocks


def _fix_latent_specs(data: dict[str, Any], values: Mapping[str, Any]) -> None:
    found: set[str] = set()
    for block in _variable_blocks(data):
        for name in set(block) & set(values):
            raw = block[name]
            spec = dict(raw) if isinstance(raw, Mapping) else {}
            spec.pop("bounds", None)
            spec.pop("scale", None)
            spec.pop("grid", None)
            spec.pop("values", None)
            spec.update({"role": ParameterRole.LATENT.value, "default": values[name], "choices": [values[name]]})
            block[name] = spec
            found.add(name)
    if missing := sorted(set(values) - found):
        raise ValueError(f"identified latent parameters are not declared: {missing}")


def _bounded_latent_specs(case: Case) -> dict[str, dict[str, Any]]:
    specs = resolve_parameter_set(case).specs(ParameterRole.LATENT)
    if not specs:
        raise ValueError("identification requires at least one parameter with role: latent")
    for name, spec in specs.items():
        bounds = spec.get("bounds")
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise ValueError(f"latent parameter {name!r} requires two numeric bounds")
        low, high = (_finite_number(value, f"latent {name} bounds") for value in bounds)
        if low >= high or (spec.get("scale") == "log" and low <= 0):
            raise ValueError(f"latent parameter {name!r} has invalid bounds")
    return specs


def _phase_evidence(
    result: Mapping[str, Any] | None,
    scenario_ids: Sequence[str],
    objective: Mapping[str, Any],
    limit: float,
) -> dict[str, Any]:
    """Represent expected missing solver/measurement evidence in every phase."""

    result = result or {}
    raw = (result.get("best") or {}).get("aggregates", {}).get(objective["metric"])
    value = None if raw is None else _finite_number(raw, f"identified aggregate {objective['metric']}")
    verification = result.get("verification") or {}
    best = result.get("best") or {}
    return {
        "scenario_ids": list(scenario_ids),
        "metric": objective["metric"],
        "aggregation": objective["aggregation"],
        "value": value,
        "limit": limit,
        "status": "not_run" if not result else ("unavailable" if value is None else "available"),
        "passed": value is not None
        and verification.get("status") == "verified"
        and int(verification.get("n_failed_evaluations", 0)) == 0
        and best.get("status") == "meets_declared_acceptance"
        and value <= limit,
    }


def _metric_vector(result: CandidateResult, names: Sequence[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values: list[float] = []
    unavailable = []
    for scenario in result.scenarios:
        selected = scenario.selected
        missing = []
        for name in names:
            try:
                values.append(_finite_number(selected.metrics.values.get(name), f"sensitivity metric {name}"))
            except ValueError:
                missing.append(name)
        if not selected.raw.ok or missing:
            unavailable.append(
                {
                    "candidate_id": result.candidate.candidate_id,
                    "scenario_id": scenario.scenario.scenario_id,
                    "status": selected.raw.status,
                    "missing_metrics": missing,
                    "error": selected.raw.error,
                    "artifacts": {name: f"sensitivity/{path}" for name, path in selected.raw.artifacts.items()},
                }
            )
    return np.asarray(values, dtype=float), unavailable


def _sensitivity_column(
    name: str,
    spec: Mapping[str, Any],
    estimated: Mapping[str, Any],
    runner: StudyRunner,
    settings: IdentificationSettings,
) -> tuple[np.ndarray | None, int, list[dict[str, Any]]]:
    axis = ContinuousAxis(name, float(spec["bounds"][0]), float(spec["bounds"][1]), spec.get("scale") == "log")
    center = axis.encode(estimated[name])
    if center is None:
        raise ValueError(f"estimated latent parameter {name!r} is outside its declared bounds")
    low_unit = max(0.0, center - settings.sensitivity_step)
    high_unit = min(1.0, center + settings.sensitivity_step)
    low = {**estimated, name: axis.decode(low_unit)}
    high = {**estimated, name: axis.decode(high_unit)}
    low_result = runner.evaluate_candidate(Candidate(f"sensitivity_{name}_low", low), reuse_cached=False)
    high_result = runner.evaluate_candidate(Candidate(f"sensitivity_{name}_high", high), reuse_cached=False)
    low_vector, low_missing = _metric_vector(low_result, settings.observed_metrics)
    high_vector, high_missing = _metric_vector(high_result, settings.observed_metrics)
    evaluations = len(low_result.control_evaluations) + len(high_result.control_evaluations)
    missing = low_missing + high_missing
    derivative = None if missing else (high_vector - low_vector) / (high_unit - low_unit)
    return derivative, evaluations, missing


def _sensitivity_diagnosis(
    matrix: np.ndarray,
    parameter_count: int,
    settings: IdentificationSettings,
) -> tuple[np.ndarray, int, float | None, bool]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    # The same declared tolerance has an absolute floor in normalized units.
    # A well-conditioned matrix containing only numerical dust is not evidence.
    cutoff = settings.rank_tolerance * max(1.0, float(singular[0])) if len(singular) else settings.rank_tolerance
    rank = int(np.count_nonzero(singular > cutoff))
    full_rank = rank == parameter_count
    condition = float(singular[0] / singular[-1]) if full_rank and singular[-1] > 0 else None
    identifiable = full_rank and condition is not None and condition <= settings.condition_number_max
    return singular, rank, condition, identifiable


def _local_sensitivity(
    case: Case,
    root: Path,
    estimated: Mapping[str, Any],
    settings: IdentificationSettings,
    solver_override: str | None,
) -> dict[str, Any]:
    specs = _bounded_latent_specs(case)
    _spec, runner, _store = build_case_runner(case, root, solver_override, transactional=False)
    columns: list[np.ndarray] = []
    evaluations = 0
    for name, spec in specs.items():
        column, count, missing = _sensitivity_column(name, spec, estimated, runner, settings)
        evaluations += count
        if column is None:
            return {
                "schema": "local_sensitivity.v1",
                "scope": "local_normalized_sensitivity_only",
                "status": "unavailable",
                "rank": None,
                "required_rank": len(specs),
                "condition_number": None,
                "identifiable": False,
                "n_evaluations": evaluations,
                "unavailable_evidence": missing,
            }
        columns.append(column)

    scales = np.asarray(
        [
            settings.metric_scales[name]
            for _scenario in case.data["study"]["scenarios"]
            for name in settings.observed_metrics
        ],
        dtype=float,
    )
    matrix = np.column_stack(columns) / scales[:, None]
    singular, rank, condition, identifiable = _sensitivity_diagnosis(matrix, len(specs), settings)
    rows = [
        f"{scenario['id']}:{metric}"
        for scenario in case.data["study"]["scenarios"]
        for metric in settings.observed_metrics
    ]
    return {
        "schema": "local_sensitivity.v1",
        "status": "available",
        "scope": "local_normalized_sensitivity_only",
        "parameters": list(specs),
        "observations": rows,
        "normalized_matrix": matrix.tolist(),
        "singular_values": singular.tolist(),
        "rank": rank,
        "required_rank": len(specs),
        "condition_number": condition,
        "condition_number_max": settings.condition_number_max,
        "rank_tolerance": settings.rank_tolerance,
        "identifiable": identifiable,
        "n_evaluations": evaluations,
    }


def _recovery(
    estimated: Mapping[str, Any],
    settings: IdentificationSettings,
) -> dict[str, Any] | None:
    if not settings.expected:
        return None
    parameters: dict[str, Any] = {}
    passed = True
    for name, truth in settings.expected.items():
        value = float(estimated[name])
        relative_error = abs(value - truth) / max(abs(truth), 1e-300)
        accepted = relative_error <= settings.recovery_relative_tolerance
        passed = passed and accepted
        parameters[name] = {
            "estimated": value,
            "expected": truth,
            "relative_error": relative_error,
            "passed": accepted,
        }
    return {
        "relative_tolerance": settings.recovery_relative_tolerance,
        "passed": passed,
        "parameters": parameters,
    }


def _observation_table(study_root: Path, metrics: Sequence[str]) -> pd.DataFrame:
    payload = read_best_candidate(study_root)
    rows: list[dict[str, Any]] = []
    for scenario in payload.get("scenarios") or []:
        selected = selected_evaluation(scenario)
        values = dict((scenario.get("scenario") or {}).get("values") or {})
        measured = dict(selected.get("metrics") or {})
        row: dict[str, Any] = {"scenario_id": (scenario.get("scenario") or {}).get("scenario_id")}
        row.update({f"input.{name}": value for name, value in values.items()})
        for name in metrics:
            row[f"predicted.{name}"] = measured.get(name)
            row[f"target.{name}"] = measured.get(f"target_{name}")
            row[f"residual.{name}"] = measured.get(f"residual_{name}")
        row["normalized_error"] = measured.get("normalized_impedance_error", measured.get("loss"))
        rows.append(row)
    return pd.DataFrame(rows)


def _split_evidence(case: Case, settings: IdentificationSettings, catalog: Mapping[str, Scenario]) -> dict[str, Any]:
    """Compare resolved operating settings, rather than treating new IDs as new conditions."""

    parameters = resolve_parameter_set(case)
    operating = set(parameters.names(ParameterRole.OPERATING)) | set(parameters.names(ParameterRole.CONTROL))
    builtin_observations = (
        TERMINAL_VI_PARAMETERS if (case.data.get("target") or {}).get("objective") == "terminal_vi_fit" else ()
    )
    observations = tuple(dict.fromkeys((*builtin_observations, *settings.observation_parameters)))
    if unknown := sorted(set(observations) - {item.name for item in parameters.definitions}):
        raise ValueError(f"identification.observation_parameters are not declared parameters: {unknown}")
    conditions = settings.condition_parameters or tuple(sorted(operating - set(observations)))
    if invalid := sorted(set(conditions) - (operating - set(observations))):
        raise ValueError(
            f"identification.condition_parameters must be non-observation operating/control parameters: {invalid}"
        )
    defaults = default_params(case)
    study = study_spec_from_case(case)
    policy = CaseControlPolicy(case)
    resolved = {
        name: {**defaults, **scenario.values, **policy.controls(study, Candidate("split"), scenario)[0].values}
        for name, scenario in catalog.items()
    }
    for name, values in resolved.items():
        if missing := sorted((set(conditions) | set(observations)) - set(values)):
            raise ValueError(f"identification scenario {name!r} is missing parameters {missing}")
    overlaps = [
        {"fit": fit, "holdout": holdout}
        for fit in settings.fit_scenarios
        for holdout in settings.holdout_scenarios
        if all(resolved[fit][key] == resolved[holdout][key] for key in conditions)
    ]
    if settings.holdout_mode == "new_conditions" and overlaps:
        raise ValueError(
            f"holdout repeats fit operating conditions: {overlaps}; "
            "use different conditions or declare identification.holdout_mode: replicates for repeatability only"
        )
    if settings.holdout_mode == "replicates" and {pair["holdout"] for pair in overlaps} != set(
        settings.holdout_scenarios
    ):
        raise ValueError("identification.holdout_mode: replicates requires each holdout condition to occur in fit")
    repeated = [
        pair
        for pair in overlaps
        if observations and all(resolved[pair["fit"]][key] == resolved[pair["holdout"]][key] for key in observations)
    ]
    return {
        "mode": settings.holdout_mode,
        "scope": "declared_unseen_operating_conditions"
        if settings.holdout_mode == "new_conditions"
        else "same_condition_repeatability_only",
        "condition_parameters": list(conditions),
        "observation_parameters": list(observations),
        "overlapping_conditions": overlaps,
        "identical_observations": repeated,
        "independent_acquisition_verified": False,
    }


def identification_inputs(case: Case) -> tuple[IdentificationSettings, dict[str, dict[str, Any]], dict[str, Scenario]]:
    """Validate the same fit/holdout problem for preflight and execution."""

    settings = IdentificationSettings.from_case(case)
    latent_specs = _bounded_latent_specs(case)
    if settings.expected and set(settings.expected) != set(latent_specs):
        raise ValueError("identification.expected must name every and only the latent parameters")
    catalog = _scenario_catalog(case)
    selected = set(settings.fit_scenarios) | set(settings.holdout_scenarios)
    if selected != set(catalog):
        raise ValueError(
            f"identification partitions must cover every scenario; missing={sorted(set(catalog) - selected)}, extra={sorted(selected - set(catalog))}"
        )
    study = study_spec_from_case(case)
    if len(study.objectives) != 1 or study.objectives[0].direction != "minimize":
        raise ValueError("identification requires exactly one minimized fit objective")
    policy = CaseControlPolicy(case)
    for scenario in study.scenarios:
        if len(policy.controls(study, Candidate("identification"), scenario)) != 1:
            raise ValueError("identification requires observed controls to be fixed in each scenario")
    _split_evidence(case, settings, catalog)
    return settings, latent_specs, catalog


def run_identification(
    case: Case,
    run_root: str | Path = "runs",
    *,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit latent parameters, verify holdout observations, and check rank."""

    settings, latent_specs, catalog = identification_inputs(case)
    base = create_run_directory(run_root, f"{case.case_id}_identification")
    case, input_files = archive_case_bundle(case, base / "input")
    settings, latent_specs, catalog = identification_inputs(case)
    fit_scenarios = tuple(catalog[name] for name in settings.fit_scenarios)
    holdout_scenarios = tuple(catalog[name] for name in settings.holdout_scenarios)
    fit_case = _phase_case(case, "fit", fit_scenarios)
    fit_result = _run_parameter_study(
        fit_case,
        base,
        n_trials=n_trials,
        optimizer_name=optimizer_name,
        solver_override=solver_override,
        seed=seed,
        proposed_role=ParameterRole.LATENT,
    )
    objective = fit_result["study"]["objectives"][0]
    fit = _phase_evidence(fit_result, settings.fit_scenarios, objective, settings.fit_loss_max)
    holdout = _phase_evidence(None, settings.holdout_scenarios, objective, settings.holdout_loss_max)
    sensitivity: dict[str, Any] = {
        "scope": "local_normalized_sensitivity_only",
        "status": "not_run",
        "rank": None,
        "required_rank": len(latent_specs),
        "condition_number": None,
        "identifiable": False,
    }
    artifacts = {
        "fit_result": "fit/study_result.json",
        "fit_observations": "fit_observations.csv",
        "input_manifest": "input/input_manifest.json",
        "input_case": "input/" + input_files.get("input_case", "case.yaml"),
    }
    _observation_table(base / "fit", settings.observed_metrics).to_csv(base / "fit_observations.csv", index=False)
    estimated = {}
    recovery = None
    if fit["value"] is not None:
        estimated = dict(fit_result["best"]["candidate"]["values"])
        if set(estimated) != set(latent_specs):
            raise ValueError("identified candidate does not contain exactly the declared latent parameters")
        holdout_result = _run_parameter_study(
            _phase_case(case, "holdout", holdout_scenarios, estimated),
            base,
            n_trials=1,
            optimizer_name="grid",
            solver_override=solver_override,
            seed=0,
            proposed_role=ParameterRole.LATENT,
        )
        holdout = _phase_evidence(holdout_result, settings.holdout_scenarios, objective, settings.holdout_loss_max)
        sensitivity = _local_sensitivity(
            _phase_case(case, "sensitivity", fit_scenarios), base, estimated, settings, solver_override
        )
        write_json(base / "sensitivity.json", sensitivity)
        _observation_table(base / "holdout", settings.observed_metrics).to_csv(
            base / "holdout_observations.csv", index=False
        )
        artifacts.update(
            {
                "holdout_result": "holdout/study_result.json",
                "holdout_observations": "holdout_observations.csv",
                "sensitivity": "sensitivity.json",
            }
        )
        recovery = _recovery(estimated, settings)
    unavailable = next(
        (
            name
            for name, evidence in (("fit", fit), ("holdout", holdout), ("sensitivity", sensitivity))
            if evidence["status"] == "unavailable"
        ),
        None,
    )
    passed = (
        fit["passed"]
        and holdout["passed"]
        and sensitivity["identifiable"]
        and (recovery is None or bool(recovery["passed"]))
    )
    payload: dict[str, Any] = {
        "schema": "identification_result.v1",
        "case_id": case.case_id,
        "status": "identified" if passed else "not_identified",
        **({"reason": f"{unavailable}_evidence_unavailable"} if unavailable else {}),
        "physical_scope": "effective_electrical_terminal_model_only",
        "estimated_latent": estimated,
        "fit": fit,
        "holdout": holdout,
        "validation": _split_evidence(case, settings, catalog),
        "identifiability": sensitivity,
        **({"recovery": recovery} if recovery is not None else {}),
        "run_root": str(base),
        "artifacts": artifacts,
    }
    if "output" in case.data:
        from .reporting.study import render_identification_outputs

        payload["output"] = render_identification_outputs(payload, case)
        payload["artifacts"]["output_summary"] = payload["output"]["summary"]
    write_json(base / "identification_result.json", payload)
    return payload
