"""Scenario aggregation and feasibility-first candidate ordering."""

from __future__ import annotations

import math

from .models import Candidate, CandidateResult, ControlState, EvaluationResult, Objective, ScenarioResult, StudySpec


def _finite_float(value: object) -> float | None:
    """Normalize a metric value once for aggregation and ordering."""

    if value is None:
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    total = sum(weight for _value, weight in items)
    return sum(value * weight for value, weight in items) / total


def control_margin(selected: ControlState, available: tuple[ControlState, ...]) -> float | None:
    """Return selected numeric-grid headroom on a 0 (edge) to 1 (center) scale.

    Every axis must be numeric and finite in every available state to
    participate. A single-valued numeric axis is neutral at 1.0, while
    categorical axes are ignored. The minimum axis margin is returned because
    one saturated actuator is sufficient to remove tuning headroom.
    """

    margins: list[float] = []
    for name in selected.values:
        values: list[float] = []
        for control in available:
            if name not in control.values:
                values = []
                break
            value = control.values[name]
            if isinstance(value, bool):
                values = []
                break
            number = _finite_float(value)
            if number is None:
                values = []
                break
            values.append(number)
        if not values:
            continue

        selected_value = _finite_float(selected.values[name])
        if selected_value is None:
            continue
        low, high = min(values), max(values)
        if low == high:
            margins.append(1.0)
            continue
        distance = min(selected_value - low, high - selected_value)
        margins.append(max(0.0, min(1.0, 2.0 * distance / (high - low))))
    return min(margins) if margins else None


def _weighted_cvar(items: list[tuple[float, float]], objective: Objective) -> float:
    """Average the worst ``alpha`` probability mass, including a split boundary."""

    reverse = objective.direction == "minimize"
    ordered = sorted(items, key=lambda item: item[0], reverse=reverse)
    total_weight = sum(weight for _value, weight in ordered)
    tail_weight = objective.cvar_alpha * total_weight
    remaining = tail_weight
    weighted = 0.0
    for value, weight in ordered:
        take = min(weight, remaining)
        weighted += value * take
        remaining -= take
        if remaining <= 1e-15:
            break
    return weighted / tail_weight


def aggregate_objective(items: list[tuple[float, float]], objective: Objective) -> float | None:
    if not items:
        return None
    if objective.aggregation == "mean":
        return _weighted_mean(items)
    if objective.aggregation == "cvar":
        return _weighted_cvar(items, objective)
    values = [value for value, _weight in items]
    return max(values) if objective.direction == "minimize" else min(values)


def aggregate_candidate(
    study: StudySpec,
    candidate: Candidate,
    scenarios: tuple[ScenarioResult, ...],
) -> CandidateResult:
    evaluations = tuple(item.selected for item in scenarios)
    weights = {item.scenario_id: item.weight for item in study.scenarios}
    total_weight = sum(weights.values())
    successful_weight = sum(weights[item.request.scenario.scenario_id] for item in evaluations if item.raw.ok)
    feasible_weight = sum(weights[item.request.scenario.scenario_id] for item in evaluations if item.feasible)
    violation = (
        sum(weights[item.request.scenario.scenario_id] * item.total_violation for item in evaluations) / total_weight
    )

    aggregates: dict[str, float | None] = {}
    for objective in study.objectives:
        values: list[tuple[float, float]] = []
        for item in evaluations:
            value = item.metrics.values.get(objective.metric)
            if not item.raw.ok:
                continue
            if value is None:
                continue
            number = _finite_float(value)
            if number is not None:
                values.append((number, weights[item.request.scenario.scenario_id]))
        aggregates[objective.metric] = aggregate_objective(values, objective)

    return CandidateResult(
        candidate=candidate,
        scenarios=scenarios,
        aggregates=aggregates,
        feasible_fraction=feasible_weight / total_weight,
        success_fraction=successful_weight / total_weight,
        total_violation=violation,
    )


def evaluation_rank_key(study: StudySpec, result: EvaluationResult) -> tuple[float, ...]:
    """Select a scenario's operating point with feasibility before objectives."""

    key: list[float] = [0.0 if result.feasible else 1.0, result.total_violation]
    for objective in study.objectives:
        number = _finite_float(result.metrics.values.get(objective.metric))
        fallback = float("inf")
        key.append(fallback if number is None else (number if objective.direction == "minimize" else -number))
    return tuple(key)


def candidate_rank_key(study: StudySpec, result: CandidateResult) -> tuple[float, ...]:
    """Sort by evidence coverage, violation, objectives, then control margin."""

    complete_and_feasible = result.success_fraction == 1.0 and result.feasible_fraction == 1.0
    key: list[float] = [
        0.0 if complete_and_feasible else 1.0,
        1.0 - result.success_fraction,
        1.0 - result.feasible_fraction,
        result.total_violation,
    ]
    for objective in study.objectives:
        number = _finite_float(result.aggregates.get(objective.metric))
        fallback = float("inf")
        key.append(fallback if number is None else (number if objective.direction == "minimize" else -number))
    margin = result.control_margin
    key.append(float("inf") if margin is None else -margin)
    return tuple(key)
