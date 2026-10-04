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

from .artifacts import write_json
from .case import Case
from .core.models import Candidate, CandidateResult, Scenario
from .core.pipeline import StudyRunner
from .problem import ParameterRole, resolve_parameter_set
from .results import read_best_candidate, selected_evaluation
from .study import _run_parameter_study, build_case_runner
from .study_config import mapping, study_spec_from_case


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

    @classmethod
    def from_case(cls, case: Case) -> IdentificationSettings:
        cfg = mapping(case.data.get("identification"), "identification")
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
        if set(fit) & set(holdout):
            raise ValueError("identification fit_scenarios and holdout_scenarios must be disjoint")
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


def _selected_scenarios(
    catalog: Mapping[str, Scenario],
    names: Sequence[str],
    path: str,
) -> tuple[Scenario, ...]:
    missing = sorted(set(names) - set(catalog))
    if missing:
        raise ValueError(f"{path} names unknown scenarios {missing}")
    return tuple(catalog[name] for name in names)


def _inline_scenario(scenario: Scenario) -> dict[str, Any]:
    return {
        "id": scenario.scenario_id,
        "values": dict(scenario.values),
        "weight": scenario.weight,
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
    study["scenarios"] = [_inline_scenario(scenario) for scenario in scenarios]
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


def _objective_value(result: Mapping[str, Any]) -> tuple[str, float]:
    objectives = (result.get("study") or {}).get("objectives") or []
    if not objectives:
        raise ValueError("identification study has no objective")
    objective = objectives[0]
    if objective.get("direction", "minimize") != "minimize":
        raise ValueError("identification objective must be minimized")
    name = str(objective.get("metric", ""))
    raw = (result.get("best") or {}).get("aggregates", {}).get(name)
    return name, _finite_number(raw, f"identified aggregate {name}")


def _study_passed(result: Mapping[str, Any], loss: float, limit: float) -> bool:
    verification = result.get("verification") or {}
    best = result.get("best") or {}
    return (
        verification.get("status") == "verified"
        and int(result.get("n_failed_evaluations", 0)) == 0
        and int(verification.get("n_failed_evaluations", 0)) == 0
        and best.get("status") == "meets_declared_acceptance"
        and loss <= limit
    )


def _axis_unit_value(value: float, spec: Mapping[str, Any]) -> float:
    low, high = (float(item) for item in spec["bounds"])
    if spec.get("scale") == "log":
        return (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
    return (value - low) / (high - low)


def _axis_value(unit: float, spec: Mapping[str, Any]) -> float:
    low, high = (float(item) for item in spec["bounds"])
    if spec.get("scale") == "log":
        return 10.0 ** (math.log10(low) + unit * (math.log10(high) - math.log10(low)))
    return low + unit * (high - low)


def _metric_vector(result: CandidateResult, names: Sequence[str]) -> np.ndarray:
    values: list[float] = []
    for scenario in result.scenarios:
        if not scenario.selected.raw.ok:
            raise ValueError(f"sensitivity solve failed for scenario {scenario.scenario.scenario_id!r}")
        for name in names:
            values.append(_finite_number(scenario.selected.metrics.values.get(name), f"sensitivity metric {name}"))
    return np.asarray(values, dtype=float)


def _sensitivity_column(
    name: str,
    spec: Mapping[str, Any],
    estimated: Mapping[str, Any],
    runner: StudyRunner,
    settings: IdentificationSettings,
) -> tuple[np.ndarray, int]:
    center = _axis_unit_value(float(estimated[name]), spec)
    low_unit = max(0.0, center - settings.sensitivity_step)
    high_unit = min(1.0, center + settings.sensitivity_step)
    low = {**estimated, name: _axis_value(low_unit, spec)}
    high = {**estimated, name: _axis_value(high_unit, spec)}
    low_result = runner.evaluate_candidate(Candidate(f"sensitivity_{name}_low", low), reuse_cached=False)
    high_result = runner.evaluate_candidate(Candidate(f"sensitivity_{name}_high", high), reuse_cached=False)
    derivative = (
        _metric_vector(high_result, settings.observed_metrics) - _metric_vector(low_result, settings.observed_metrics)
    ) / (high_unit - low_unit)
    evaluations = len(low_result.control_evaluations) + len(high_result.control_evaluations)
    return derivative, evaluations


def _sensitivity_diagnosis(
    matrix: np.ndarray,
    parameter_count: int,
    settings: IdentificationSettings,
) -> tuple[np.ndarray, int, float | None, bool]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    cutoff = settings.rank_tolerance * singular[0] if len(singular) else 0.0
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
        column, count = _sensitivity_column(name, spec, estimated, runner, settings)
        columns.append(column)
        evaluations += count

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
    if set(settings.expected) != set(estimated):
        raise ValueError("identification.expected must name every and only the latent parameters")
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

    settings = IdentificationSettings.from_case(case)
    latent_specs = _bounded_latent_specs(case)
    catalog = _scenario_catalog(case)
    declared_ids = set(catalog)
    selected_ids = set(settings.fit_scenarios) | set(settings.holdout_scenarios)
    if selected_ids != declared_ids:
        missing = sorted(declared_ids - selected_ids)
        extra = sorted(selected_ids - declared_ids)
        raise ValueError(f"identification partitions must cover every scenario; missing={missing}, extra={extra}")

    base = Path(run_root).resolve() / f"{case.case_id}_identification"
    fit_scenarios = _selected_scenarios(catalog, settings.fit_scenarios, "identification.fit_scenarios")
    holdout_scenarios = _selected_scenarios(
        catalog,
        settings.holdout_scenarios,
        "identification.holdout_scenarios",
    )
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
    estimated = dict(fit_result["best"]["candidate"]["values"])
    if set(estimated) != set(latent_specs):
        raise ValueError("identified candidate does not contain exactly the declared latent parameters")

    holdout_case = _phase_case(case, "holdout", holdout_scenarios, estimated)
    holdout_result = _run_parameter_study(
        holdout_case,
        base,
        n_trials=1,
        optimizer_name="grid",
        solver_override=solver_override,
        seed=0,
        proposed_role=ParameterRole.LATENT,
    )
    sensitivity_case = _phase_case(case, "sensitivity", fit_scenarios)
    sensitivity = _local_sensitivity(sensitivity_case, base, estimated, settings, solver_override)
    write_json(base / "sensitivity.json", sensitivity)

    fit_metric, fit_loss = _objective_value(fit_result)
    holdout_metric, holdout_loss = _objective_value(holdout_result)
    if fit_metric != holdout_metric:
        raise ValueError("fit and holdout studies do not use the same objective")
    fit_passed = _study_passed(fit_result, fit_loss, settings.fit_loss_max)
    holdout_passed = _study_passed(holdout_result, holdout_loss, settings.holdout_loss_max)
    recovery = _recovery(estimated, settings)
    passed = (
        fit_passed and holdout_passed and sensitivity["identifiable"] and (recovery is None or bool(recovery["passed"]))
    )

    fit_table = _observation_table(base / "fit", settings.observed_metrics)
    holdout_table = _observation_table(base / "holdout", settings.observed_metrics)
    fit_table.to_csv(base / "fit_observations.csv", index=False)
    holdout_table.to_csv(base / "holdout_observations.csv", index=False)
    payload = {
        "schema": "identification_result.v1",
        "case_id": case.case_id,
        "status": "identified" if passed else "not_identified",
        "physical_scope": "effective_electrical_terminal_model_only",
        "estimated_latent": estimated,
        "fit": {
            "scenario_ids": list(settings.fit_scenarios),
            "metric": fit_metric,
            "value": fit_loss,
            "limit": settings.fit_loss_max,
            "passed": fit_passed,
        },
        "holdout": {
            "scenario_ids": list(settings.holdout_scenarios),
            "metric": holdout_metric,
            "value": holdout_loss,
            "limit": settings.holdout_loss_max,
            "passed": holdout_passed,
        },
        "identifiability": sensitivity,
        **({"recovery": recovery} if recovery is not None else {}),
        "run_root": str(base),
        "artifacts": {
            "fit_result": "fit/study_result.json",
            "holdout_result": "holdout/study_result.json",
            "fit_observations": "fit_observations.csv",
            "holdout_observations": "holdout_observations.csv",
            "sensitivity": "sensitivity.json",
        },
    }
    write_json(base / "identification_result.json", payload)
    return payload
