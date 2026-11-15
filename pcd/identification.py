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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import capture_evidence, create_run_directory, write_json
from .case import Case, default_params, resolve_path
from .core.aggregation import aggregate_objective
from .core.models import Candidate, CandidateResult, ControlState, Objective, Scenario, to_plain
from .core.pipeline import StudyRunner
from .core.spaces import ContinuousAxis
from .inputs import condition_value, observation_metadata, reject_unknown
from .metrics import (
    TERMINAL_VI_PARAMETERS,
    terminal_observation_frequency_hz,
    terminal_observation_frequency_parameter,
    terminal_observation_impedance,
)
from .problem import ParameterRole, resolve_parameter_set
from .results import read_best_candidate, selected_evaluation
from .results.store import GENERATION_RESULT_FILE
from .sim_core import archive_case_bundle
from .simulation import AcSweep, same_frequency_hz
from .study import _run_parameter_study, build_case_runner
from .study_config import CaseControlPolicy, mapping, study_spec_from_case
from .tables import read_csv_rows


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
    acquisition_groups: Mapping[str, str] = field(default_factory=dict)

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
                "acquisition_groups",
                "observation_table_columns",
                "uncertainty",
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
        acquisition_groups = mapping(cfg.get("acquisition_groups"), "identification.acquisition_groups")
        if "acquisition_groups" in cfg and (
            not acquisition_groups
            or any(not isinstance(group, str) or not group.strip() for group in acquisition_groups.values())
        ):
            raise ValueError("identification.acquisition_groups must map scenario IDs to nonempty group labels")
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
            acquisition_groups={name: group.strip() for name, group in acquisition_groups.items()},
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


@dataclass(frozen=True, slots=True)
class _Realization:
    name: str
    calibration: dict[str, float | int]
    observations: dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class _IdentificationInputs:
    settings: IdentificationSettings
    latent_specs: dict[str, dict[str, Any]]
    points: dict[str, tuple[Scenario, ControlState]]
    split_evidence: dict[str, Any]
    condition_values: dict[str, dict[str, Any]]
    frequency_parameter: str | None
    realizations: tuple[_Realization, ...]


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
    points: Sequence[tuple[Scenario, ControlState]],
    fixed_latent: Mapping[str, Any] | None = None,
) -> Case:
    data = deepcopy(case.data)
    data["case_id"] = phase
    # The split/acceptance contract is owned by this outer workflow.  Each
    # phase is an ordinary executable study and should validate only its own
    # resolved candidate space and scenarios.
    data.pop("identification", None)
    if (data.get("target") or {}).get("objective") == "terminal_vi_fit":
        frequency_parameter = terminal_observation_frequency_parameter(case)
        if frequency_parameter is not None:
            data["target"]["observation_frequency_parameter"] = frequency_parameter
    study = mapping(data.get("study"), "study")
    study.pop("scenario_table", None)
    study["scenarios"] = [_inline_scenario(scenario, control.values) for scenario, control in points]
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
        if "bounds" not in spec:
            raise ValueError(f"latent parameter {name!r} requires two numeric bounds")
        low, high = map(float, spec["bounds"])
        if low == high:
            raise ValueError(f"latent parameter {name!r} requires increasing bounds")
    return specs


def _phase_evidence(
    result: Mapping[str, Any] | None,
    scenario_ids: Sequence[str],
    objective: Mapping[str, Any],
    limit: float,
    *,
    available_observations: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Represent expected missing solver/measurement evidence in every phase."""

    result = result or {}
    verification = result.get("verification") or {}
    best = result.get("best") or {}
    replay_complete = verification.get("status") == "verified" and available_observations.issuperset(scenario_ids)
    raw = best.get("aggregates", {}).get(objective["metric"]) if replay_complete else None
    value = None if raw is None else _finite_number(raw, f"identified aggregate {objective['metric']}")
    return {
        "scenario_ids": list(scenario_ids),
        "metric": objective["metric"],
        "aggregation": objective["aggregation"],
        "value": value,
        "limit": limit,
        "status": "not_run" if not result else ("unavailable" if value is None else "available"),
        "passed": value is not None and best.get("status") == "meets_declared_acceptance" and value <= limit,
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
    if low[name] == high[name]:
        return None, 0, [{"parameter": name, "error": "sensitivity step does not change the represented value"}]
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


def _observation_table(
    study_root: Path, metrics: Sequence[str], acquisition_groups: Mapping[str, str]
) -> tuple[pd.DataFrame, frozenset[str]]:
    """Collect comparisons and identify solved rows with finite declared predictions."""

    payload = read_best_candidate(study_root)
    rows: list[dict[str, Any]] = []
    available: set[str] = set()
    for scenario in payload.get("scenarios") or []:
        selected = selected_evaluation(scenario)
        point = scenario["scenario"]
        scenario_id = point["scenario_id"]
        measured = dict(selected.get("metrics") or {})
        row: dict[str, Any] = {"scenario_id": scenario_id}
        if acquisition_groups:
            row["acquisition_group"] = acquisition_groups[scenario_id]
        row.update({f"input.{name}": value for name, value in point["values"].items()})
        # A successful study evaluation already has a finite objective value.
        complete = selected["raw"]["status"] == "ok"
        for name in metrics:
            for column, metric in (("predicted", name), ("target", f"target_{name}"), ("residual", f"residual_{name}")):
                try:
                    row[f"{column}.{name}"] = _finite_number(measured.get(metric), metric)
                except ValueError:
                    row[f"{column}.{name}"] = None
                    if column == "predicted":
                        complete = False
        row["normalized_error"] = measured.get("normalized_impedance_error", measured.get("loss"))
        rows.append(row)
        if complete:
            available.add(scenario_id)
    return pd.DataFrame(rows), frozenset(available)


def _same_conditions(first: Mapping[str, Any], second: Mapping[str, Any], frequency_parameter: str | None) -> bool:
    if frequency_parameter is None:
        return first == second
    return all(
        same_frequency_hz(value, second[name]) if name == frequency_parameter else value == second[name]
        for name, value in first.items()
    )


def _condition_frequency_hz(ac: AcSweep, observed_hz: float) -> float:
    """Identify the intended AC sample without hiding an off-grid observation."""

    if ac.start_hz == ac.stop_hz:
        if not same_frequency_hz(observed_hz, ac.start_hz):
            raise ValueError(f"observed frequency {observed_hz:g} Hz differs from planned AC point {ac.start_hz:g} Hz")
        return ac.start_hz
    if not (
        ac.start_hz <= observed_hz <= ac.stop_hz
        or same_frequency_hz(observed_hz, ac.start_hz)
        or same_frequency_hz(observed_hz, ac.stop_hz)
    ):
        raise ValueError(f"observed frequency {observed_hz:g} Hz is outside the planned AC sweep")
    grid = ac.frequencies_hz()
    nearest_hz = float(grid[np.argmin(np.abs(grid - observed_hz))])
    return nearest_hz if same_frequency_hz(observed_hz, nearest_hz) else observed_hz


def _split_evidence(
    case: Case,
    settings: IdentificationSettings,
    points: Mapping[str, tuple[Scenario, ControlState]],
    frequency_parameter: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare resolved operating settings, rather than treating new IDs as new conditions."""

    acquisition_groups = settings.acquisition_groups
    fit_groups, holdout_groups = set(), set()
    if acquisition_groups:
        if set(acquisition_groups) != set(points):
            raise ValueError("identification.acquisition_groups must cover every declared scenario exactly once")
        fit_groups = {acquisition_groups[name] for name in settings.fit_scenarios}
        holdout_groups = {acquisition_groups[name] for name in settings.holdout_scenarios}
        if shared_groups := sorted(fit_groups & holdout_groups):
            raise ValueError(f"fit and holdout share acquisition groups: {shared_groups}")
    parameters = resolve_parameter_set(case)
    specs = {item.name: item.spec for item in parameters.definitions}
    operating = set(parameters.names(ParameterRole.OPERATING))
    builtin_observations = (
        TERMINAL_VI_PARAMETERS if (case.data.get("target") or {}).get("objective") == "terminal_vi_fit" else ()
    )
    observations = tuple(dict.fromkeys((*builtin_observations, *settings.observation_parameters)))
    if invalid := sorted(set(observations) - operating):
        raise ValueError(f"identification.observation_parameters must be declared operating parameters: {invalid}")
    condition_axes = (operating | set(parameters.names(ParameterRole.CONTROL))) - set(observations)
    conditions = settings.condition_parameters or tuple(sorted(condition_axes))
    if invalid := sorted(set(conditions) - condition_axes):
        raise ValueError(
            f"identification.condition_parameters must be non-observation operating/control parameters: {invalid}"
        )
    required_columns = set(observations)
    if frequency_parameter is not None:
        if frequency_parameter not in operating or frequency_parameter in required_columns:
            raise ValueError("target.observation_frequency_parameter must name an operating condition coordinate")
        if frequency_parameter not in conditions:
            conditions = (*conditions, frequency_parameter)
        required_columns.add(frequency_parameter)
    table = (case.data.get("study") or {}).get("scenario_table")
    if table is not None and (missing := sorted(required_columns - set(table.get("values") or {}))):
        raise ValueError(f"identification scenario table requires observation columns for {missing}")
    ac_config = (case.data.get("solver") or {}).get("ac")
    defaults = default_params(case)
    resolved = {name: {**defaults, **scenario.values, **control.values} for name, (scenario, control) in points.items()}
    condition_values: dict[str, dict[str, Any]] = {}
    for name, (scenario, _control) in points.items():
        if missing := sorted(set(observations) - set(scenario.values)):
            raise ValueError(f"identification scenario {name!r} requires explicit observation values for {missing}")
        values = resolved[name]
        if missing := sorted(set(conditions) - set(values)):
            raise ValueError(f"identification scenario {name!r} is missing parameters {missing}")
        condition = {
            key: condition_value(values[key], key, specs[key], f"identification scenario {name!r}.{key}")
            for key in conditions
        }
        if builtin_observations:
            try:
                terminal_observation_impedance(values)
                if frequency_parameter is not None:
                    frequency = terminal_observation_frequency_hz(frequency_parameter, scenario.values)
                    if ac_config is None:
                        raise ValueError("terminal_vi_fit with observed frequencies requires solver.ac")
                    condition[frequency_parameter] = _condition_frequency_hz(
                        AcSweep.from_config(ac_config, values), frequency
                    )
            except ValueError as exc:
                raise ValueError(f"identification scenario {name!r}: {exc}") from exc
        condition_values[name] = condition
    overlaps = [
        {"fit": fit, "holdout": holdout}
        for fit in settings.fit_scenarios
        for holdout in settings.holdout_scenarios
        if _same_conditions(condition_values[fit], condition_values[holdout], frequency_parameter)
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
    evidence: dict[str, Any] = {
        "mode": settings.holdout_mode,
        "scope": "declared_unseen_operating_conditions"
        if settings.holdout_mode == "new_conditions"
        else "same_condition_repeatability_only",
        "condition_parameters": list(conditions),
        "observation_parameters": list(observations),
        "overlapping_conditions": overlaps,
        "identical_observations": repeated,
        "acquisition_group_separation": "declared_disjoint" if acquisition_groups else "not_declared",
        "independent_acquisition_verified": False,
    }
    if acquisition_groups:
        evidence["acquisition_groups"] = {"fit": sorted(fit_groups), "holdout": sorted(holdout_groups)}
    return evidence, condition_values


def _verify_observation_table_metadata(case: Case, settings: IdentificationSettings) -> None:
    raw = (case.data.get("identification") or {}).get("observation_table_columns")
    if raw is None:
        return
    columns = mapping(raw, "identification.observation_table_columns")
    if set(columns) != {"partition", "acquisition_group"} or any(
        not isinstance(value, str) or not value.strip() for value in columns.values()
    ):
        raise ValueError("identification.observation_table_columns must name partition and acquisition group columns")
    table = mapping((case.data.get("study") or {}).get("scenario_table"), "study.scenario_table")
    if not table.get("table_file"):
        raise ValueError("observation table metadata requires study.scenario_table.table_file")
    partition_column = columns["partition"]
    group_column = columns["acquisition_group"]
    id_column = str(table.get("id_column", "scenario_id"))
    header, rows = read_csv_rows(resolve_path(case, table["table_file"]), (partition_column,))
    scenario_ids = (row.get(id_column, f"point_{index}").strip() for index, row in enumerate(rows, start=1))
    partitions, groups = observation_metadata(
        rows, scenario_ids, partition_column, group_column if group_column in header else None
    )
    if (
        tuple(partitions["fit"]) != settings.fit_scenarios
        or tuple(partitions["holdout"]) != settings.holdout_scenarios
        or groups != settings.acquisition_groups
    ):
        raise ValueError("observation CSV partition or acquisition groups changed after loading; reload the case")


def _numeric_changes(
    raw: Any, specs: Mapping[str, Mapping[str, Any]], nominal: Mapping[str, Any], path: str
) -> dict[str, float | int]:
    values = mapping(raw, path)
    reject_unknown(values, set(specs) & set(nominal), path)
    changes: dict[str, float | int] = {}
    for name, raw_value in values.items():
        field = f"{path}.{name}"
        value = condition_value(raw_value, name, specs[name], field, allow_units=True)
        before = condition_value(nominal[name], name, specs[name], field, allow_units=True)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{field} must be numeric")
        if isinstance(before, bool) or not isinstance(before, (float, int)):
            raise ValueError(f"{field} requires a numeric nominal value")
        _finite_number(value, field)
        _finite_number(before, field)
        if value != before:
            changes[name] = value
    return changes


def _uncertainty_realizations(
    case: Case, points: Mapping[str, tuple[Scenario, ControlState]], observation_names: Sequence[str]
) -> tuple[_Realization, ...]:
    config = case.data["identification"]
    if "uncertainty" not in config:
        return ()
    path = "identification.uncertainty"
    declared = config["uncertainty"]
    if not isinstance(declared, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if not declared:
        raise ValueError(f"{path} requires at least one named realization")
    parameters = resolve_parameter_set(case)
    calibration_specs = parameters.specs(ParameterRole.CALIBRATION)
    observation_specs = {
        name: spec for name, spec in parameters.specs(ParameterRole.OPERATING).items() if name in observation_names
    }
    defaults = default_params(case)
    terminal_fit = (case.data.get("target") or {}).get("objective") == "terminal_vi_fit"
    realizations: list[_Realization] = []
    for name, entry in declared.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip() or name == "nominal":
            raise ValueError(f"{path} requires unique nonempty names other than 'nominal'")
        field = f"{path}.{name}"
        spec = mapping(entry, field)
        reject_unknown(spec, {"calibration", "observations"}, field)
        calibration = _numeric_changes(spec.get("calibration"), calibration_specs, defaults, f"{field}.calibration")
        rows = mapping(spec.get("observations"), f"{field}.observations")
        reject_unknown(rows, set(points), f"{field}.observations")
        observations = {}
        for scenario_id, overrides in rows.items():
            scenario, control = points[scenario_id]
            nominal = {**defaults, **scenario.values, **control.values}
            row = f"{field}.observations.{scenario_id}"
            changes = _numeric_changes(overrides, observation_specs, nominal, row)
            if changes and terminal_fit:
                try:
                    terminal_observation_impedance({**nominal, **changes})
                except ValueError as exc:
                    raise ValueError(f"{row}: {exc}") from exc
            if changes:
                observations[scenario_id] = changes
        if not calibration and not observations:
            raise ValueError(f"{field} does not change the nominal inputs")
        if any(item.calibration == calibration and item.observations == observations for item in realizations):
            raise ValueError(f"{field} duplicates an earlier realization")
        realizations.append(_Realization(name, calibration, observations))
    return tuple(realizations)


def _realized_case(case: Case, inputs: _IdentificationInputs, realization: _Realization) -> Case:
    points = [
        (replace(scenario, values={**scenario.values, **realization.observations.get(name, {})}), control)
        for name, (scenario, control) in inputs.points.items()
    ]
    data = _phase_case(case, case.case_id, points).data
    data["identification"] = {
        name: value
        for name, value in case.data["identification"].items()
        if name not in {"uncertainty", "observation_table_columns"}
    }
    data.pop("output", None)
    for block in _variable_blocks(data):
        for name in block.keys() & realization.calibration.keys():
            spec = mapping(block[name], name)
            block[name] = {**spec, "default": realization.calibration[name]}
    return Case(path=case.path, data=data)


def identification_inputs(case: Case) -> _IdentificationInputs:
    """Validate the same fit/holdout problem for preflight and execution."""

    settings = IdentificationSettings.from_case(case)
    _verify_observation_table_metadata(case, settings)
    latent_specs = _bounded_latent_specs(case)
    if settings.expected and set(settings.expected) != set(latent_specs):
        raise ValueError("identification.expected must name every and only the latent parameters")
    study = study_spec_from_case(case)
    declared = {scenario.scenario_id for scenario in study.scenarios}
    selected = set(settings.fit_scenarios) | set(settings.holdout_scenarios)
    if selected != declared:
        raise ValueError(
            f"identification partitions must cover every scenario; missing={sorted(declared - selected)}, extra={sorted(selected - declared)}"
        )
    if len(study.objectives) != 1 or study.objectives[0].direction != "minimize":
        raise ValueError("identification requires exactly one minimized fit objective")
    measurement = (case.data.get("target") or {}).get("objective", "waveform_l2")
    if measurement == "waveform_l2":
        raise ValueError(
            "waveform_l2 uses a shared target file; identification requires observations for each scenario"
        )
    if measurement in {"impedance_match", "rf_load"}:
        raise ValueError(f"identification requires an observed-data residual objective for {measurement!r}")
    if measurement == "terminal_vi_fit":
        if study.objectives[0].metric not in {"loss", "normalized_impedance_error"}:
            raise ValueError("identification requires an observed-data residual objective for 'terminal_vi_fit'")
        if set(settings.observed_metrics) - {"resistance_ohm", "reactance_ohm"}:
            raise ValueError(
                "terminal_vi_fit sensitivity requires observed impedance components (resistance or reactance)"
            )
    elif not settings.observation_parameters:
        raise ValueError("identification with a custom metric requires identification.observation_parameters")
    policy = CaseControlPolicy(case, study.scenarios)
    candidate = Candidate("identification")
    points = {}
    for scenario in study.scenarios:
        controls = policy.controls(study, candidate, scenario)
        if len(controls) != 1:
            raise ValueError("identification requires observed controls to be fixed in each scenario")
        points[scenario.scenario_id] = (scenario, controls[0])
    frequency_parameter = terminal_observation_frequency_parameter(case) if measurement == "terminal_vi_fit" else None
    split_evidence, condition_values = _split_evidence(case, settings, points, frequency_parameter)
    realizations = _uncertainty_realizations(case, points, split_evidence["observation_parameters"])
    return _IdentificationInputs(
        settings, latent_specs, points, split_evidence, condition_values, frequency_parameter, realizations
    )


def _holdout_breakdown(
    inputs: _IdentificationInputs,
    objective_data: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    available_observations: frozenset[str],
) -> dict[str, Any]:
    """Show coverage and error without changing the declared row-weighted objective."""

    objective = Objective.from_dict(objective_data)
    names = inputs.settings.holdout_scenarios
    acquisition_groups = inputs.settings.acquisition_groups
    frequency_parameter = inputs.frequency_parameter
    weights = {name: inputs.points[name][0].weight for name in names}
    weight_scale = max(weights.values())
    weight_total = math.fsum(weight / weight_scale for weight in weights.values())
    errors = {
        row["scenario_id"]: row["objectives"].get(objective.metric)
        for row in (result or {}).get("best", {}).get("conditions", [])
        if row["scenario_id"] in available_observations
    }

    def summarize(members: list[str]) -> dict[str, Any]:
        available = [(float(errors[name]), weights[name]) for name in members if errors.get(name) is not None]
        return {
            "scenario_ids": members,
            "n_observations": len(members),
            "n_available": len(available),
            "weight_fraction": math.fsum(weights[name] / weight_scale for name in members) / weight_total,
            "value": aggregate_objective(available, objective) if len(available) == len(members) else None,
        }

    groups: dict[str, list[str]] = {}
    conditions: list[tuple[dict[str, Any], list[str]]] = []
    for name in names:
        if acquisition_groups:
            groups.setdefault(acquisition_groups[name], []).append(name)
        values = inputs.condition_values[name]
        members = next(
            (members for condition, members in conditions if _same_conditions(values, condition, frequency_parameter)),
            None,
        )
        if members is None:
            conditions.append((values, [name]))
        else:
            members.append(name)
    return {
        "weighting": "declared_row_weights",
        "by_acquisition_group": [
            {"acquisition_group": group, **summarize(members)} for group, members in groups.items()
        ],
        "by_condition": [{"values": to_plain(values), **summarize(members)} for values, members in conditions],
    }


def _evaluate_identification(
    case: Case,
    base: Path,
    input_files: Mapping[str, str],
    inputs: _IdentificationInputs,
    *,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate one complete fit and its independent holdout evidence."""

    settings, latent_specs = inputs.settings, inputs.latent_specs
    fit_points = tuple(inputs.points[name] for name in settings.fit_scenarios)
    holdout_points = tuple(inputs.points[name] for name in settings.holdout_scenarios)
    fit_case = _phase_case(case, "fit", fit_points)
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
    observations, available = _observation_table(base / "fit", settings.observed_metrics, settings.acquisition_groups)
    observations.to_csv(base / "fit_observations.csv", index=False)
    fit = _phase_evidence(
        fit_result, settings.fit_scenarios, objective, settings.fit_loss_max, available_observations=available
    )
    holdout_result = None
    holdout_observations = frozenset()
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
        "fit_result": f"fit/{fit_result['artifacts']['generation']}/{GENERATION_RESULT_FILE}",
        "fit_observations": "fit_observations.csv",
        "case": "input/" + input_files["case"],
        "input_manifest": "input/input_manifest.json",
        "input_case": "input/" + input_files.get("input_case", "case.yaml"),
    }
    estimated = {}
    recovery = None
    if fit["value"] is not None:
        estimated = dict(fit_result["best"]["candidate"]["values"])
        if set(estimated) != set(latent_specs):
            raise ValueError("identified candidate does not contain exactly the declared latent parameters")
        holdout_result = _run_parameter_study(
            _phase_case(case, "holdout", holdout_points, estimated),
            base,
            n_trials=1,
            optimizer_name="grid",
            solver_override=solver_override,
            seed=0,
            proposed_role=ParameterRole.LATENT,
        )
        observations, holdout_observations = _observation_table(
            base / "holdout", settings.observed_metrics, settings.acquisition_groups
        )
        observations.to_csv(base / "holdout_observations.csv", index=False)
        holdout = _phase_evidence(
            holdout_result,
            settings.holdout_scenarios,
            objective,
            settings.holdout_loss_max,
            available_observations=holdout_observations,
        )
        sensitivity = _local_sensitivity(
            _phase_case(case, "sensitivity", fit_points), base, estimated, settings, solver_override
        )
        write_json(base / "sensitivity.json", sensitivity)
        artifacts.update(
            {
                "holdout_result": f"holdout/{holdout_result['artifacts']['generation']}/{GENERATION_RESULT_FILE}",
                "holdout_observations": "holdout_observations.csv",
                "sensitivity": "sensitivity.json",
            }
        )
        recovery = _recovery(estimated, settings)
    holdout.update(_holdout_breakdown(inputs, objective, holdout_result, holdout_observations))
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
    names = ("case", "fit_result", "fit_observations", "holdout_result", "holdout_observations")
    artifact_sha256 = capture_evidence(base, {name: artifacts[name] for name in names if name in artifacts})
    return {
        "schema": "identification_result.v1",
        "case_id": case.case_id,
        "status": "identified" if passed else "not_identified",
        **({"reason": f"{unavailable}_evidence_unavailable"} if unavailable else {}),
        "physical_scope": "effective_electrical_terminal_model_only",
        "estimated_latent": estimated,
        "fit": fit,
        "holdout": holdout,
        "validation": inputs.split_evidence,
        "identifiability": sensitivity,
        **({"recovery": recovery} if recovery is not None else {}),
        "run_root": str(base),
        "artifacts": artifacts,
        "artifact_sha256": artifact_sha256,
    }


def _ranges(records: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, dict[str, float | int | None]]:
    ranges = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        ranges[key] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n_realizations": len(values),
        }
    return ranges


def _summarize_realizations(
    nominal: dict[str, Any],
    variants: Sequence[tuple[_Realization, dict[str, Any], str]],
    inputs: _IdentificationInputs,
) -> dict[str, Any]:
    settings = inputs.settings
    fits = []
    predictions: dict[str, list[dict[str, float]]] = {name: [] for name in settings.holdout_scenarios}
    rows = []
    for realization, result, artifact in [(None, nominal, None), *variants]:
        rows.append(
            {
                "name": realization.name if realization else "nominal",
                "status": result["status"],
                "reason": result.get("reason"),
                "fit_value": result["fit"]["value"],
                "fit_passed": result["fit"]["passed"],
                "holdout_value": result["holdout"]["value"],
                "holdout_passed": result["holdout"]["passed"],
                "local_rank": result["identifiability"]["rank"],
                "local_condition_number": result["identifiability"]["condition_number"],
                "locally_identifiable": result["identifiability"]["identifiable"],
                "estimated_latent": result["estimated_latent"],
                **(
                    {
                        "calibration": realization.calibration,
                        "observations": realization.observations,
                        "artifact": artifact,
                    }
                    if realization
                    else {}
                ),
            }
        )
        if not (result["fit"]["passed"] and result["identifiability"]["identifiable"]):
            continue
        fits.append(result["estimated_latent"])
        if result["holdout"]["status"] != "available":
            continue
        path = Path(result["run_root"]) / result["artifacts"]["holdout_observations"]
        columns = {metric: f"predicted.{metric}" for metric in settings.observed_metrics}
        _, observations = read_csv_rows(path, ("scenario_id", *columns.values()))
        for observation in observations:
            values = {}
            for metric, column in columns.items():
                if observation[column]:
                    values[metric] = _finite_number(observation[column], column)
            predictions[observation["scenario_id"]].append(values)
    n_identified = sum(row["status"] == "identified" for row in rows)
    n_unavailable = sum(row["reason"] is not None for row in rows)
    status = "evidence_unavailable" if n_unavailable else "not_all_identified"
    if n_identified == len(rows):
        status = "all_identified"
    return {
        "scope": "declared_deterministic_realizations_only",
        "nominal_status": nominal["status"],
        "status": status,
        "n_realizations": len(rows),
        "n_identified": n_identified,
        "n_unavailable": n_unavailable,
        "realizations": rows,
        "parameter_ranges": _ranges(fits, tuple(inputs.latent_specs)),
        "holdout_prediction_ranges": {
            name: _ranges(values, settings.observed_metrics) for name, values in predictions.items()
        },
    }


def run_identification(
    case: Case,
    run_root: str | Path = "runs",
    *,
    n_trials: int | None = None,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit and verify latent parameters under nominal and declared uncertainty inputs."""

    identification_inputs(case)
    base = create_run_directory(run_root, f"{case.case_id}_identification")
    case, input_files = archive_case_bundle(case, base / "input")
    inputs = identification_inputs(case)
    options: dict[str, Any] = {
        "n_trials": n_trials,
        "optimizer_name": optimizer_name,
        "solver_override": solver_override,
        "seed": seed,
    }
    payload = _evaluate_identification(case, base, input_files, inputs, **options)
    if inputs.realizations:
        variants = []
        variant_paths = {}
        for index, realization in enumerate(inputs.realizations, 1):
            destination = base / "uncertainty" / f"{index:03d}"
            realized, files = archive_case_bundle(_realized_case(case, inputs, realization), destination / "input")
            result = _evaluate_identification(realized, destination, files, identification_inputs(realized), **options)
            path = destination / "identification_result.json"
            write_json(path, result)
            artifact = f"uncertainty_{index:03d}"
            variants.append((realization, result, artifact))
            variant_paths[artifact] = path.relative_to(base).as_posix()
        payload["artifacts"].update(variant_paths)
        payload["artifact_sha256"].update(capture_evidence(base, variant_paths, within=base))
        uncertainty = _summarize_realizations(payload, variants, inputs)
        payload["uncertainty"] = uncertainty
        if uncertainty["status"] != "all_identified":
            if payload["status"] == "identified":
                payload["reason"] = (
                    "uncertainty_evidence_unavailable"
                    if uncertainty["n_unavailable"]
                    else "uncertainty_acceptance_not_met"
                )
            payload["status"] = "not_identified"
    if "output" in case.data:
        from .reporting import render_requested_output
        from .reporting.study import render_identification_outputs

        payload["output"] = render_requested_output(lambda: render_identification_outputs(payload, case))
        if summary := payload["output"]["summary"]:
            payload["artifacts"]["output_summary"] = summary
    write_json(base / "identification_result.json", payload)
    return payload
