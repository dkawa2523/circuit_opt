"""Scenario aggregation and feasibility-first candidate ordering."""

from __future__ import annotations

import math
from collections.abc import Iterable

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
    weights = _normalized_weights([weight for _value, weight in items])
    return math.fsum(value * weight for (value, _original), weight in zip(items, weights, strict=True))


def _normalized_weights(weights: list[float]) -> list[float]:
    """Normalize positive weights without overflowing their original sum."""

    scale = max(weights)
    scaled = [weight / scale for weight in weights]
    total = math.fsum(scaled)
    return [weight / total for weight in scaled]


def control_margin(selected: ControlState, available: tuple[ControlState, ...]) -> float | None:
    """Return selected numeric-grid headroom on a 0 (edge) to 1 (center) scale.

    Every axis must be numeric and finite in every available state to
    participate. A single-valued numeric axis is neutral at 1.0, while
    categorical axes are ignored. The minimum axis margin is returned because
    one saturated actuator is sufficient to remove tuning headroom.
    """

    margins: list[float] = []
    for name in selected.values:
        if name in selected.numeric_ranges:
            low, high = selected.numeric_ranges[name]
            number = float(selected.values[name])
            margins.append(
                1.0 if low == high else max(0.0, min(1.0, 2.0 * min(number - low, high - number) / (high - low)))
            )
            continue
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
    weights = _normalized_weights([weight for _value, weight in items])
    ordered = sorted(zip((value for value, _weight in items), weights, strict=True), reverse=reverse)
    remaining = objective.cvar_alpha
    contributions: list[float] = []
    for value, weight in ordered:
        take = min(weight, remaining)
        contributions.append(value * (take / objective.cvar_alpha))
        remaining -= take
        if remaining <= 0.0:
            break
    return math.fsum(contributions)


def aggregate_objective(items: list[tuple[float, float]], objective: Objective) -> float | None:
    if not items:
        return None
    if objective.aggregation == "mean":
        return _weighted_mean(items)
    if objective.aggregation == "rms":
        weights = _normalized_weights([weight for _value, weight in items])
        return math.hypot(*(value * math.sqrt(weight) for (value, _raw), weight in zip(items, weights, strict=True)))
    if objective.aggregation == "cvar":
        return _weighted_cvar(items, objective)
    values = [value for value, _weight in items]
    return max(values) if objective.direction == "minimize" else min(values)


def aggregate_candidate(
    study: StudySpec,
    candidate: Candidate,
    scenarios: tuple[ScenarioResult, ...],
) -> CandidateResult:
    if tuple(item.scenario.scenario_id for item in scenarios) != tuple(item.scenario_id for item in study.scenarios):
        raise ValueError("candidate results must cover every declared scenario in study order")
    evaluations = tuple(item.selected for item in scenarios)
    weights = dict(
        zip(
            (item.scenario_id for item in study.scenarios),
            _normalized_weights([item.weight for item in study.scenarios]),
            strict=True,
        )
    )
    total_weight = math.fsum(weights.values())
    successful_weight = math.fsum(weights[item.request.scenario.scenario_id] for item in evaluations if item.raw.ok)
    feasible_weight = math.fsum(weights[item.request.scenario.scenario_id] for item in evaluations if item.feasible)
    violation = (
        math.fsum(weights[item.request.scenario.scenario_id] * item.total_violation for item in evaluations)
        / total_weight
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
    """Select a scenario's operating point without preferring missing evidence."""

    # A solver/measurement failure is unknown, not a small engineering
    # violation.  It must never beat an evaluated control merely because the
    # evaluated point exceeds a normalized limit by more than one.
    key: list[float] = [
        0.0 if result.raw.ok else 1.0,
        0.0 if result.feasible else 1.0,
        result.total_violation,
    ]
    for objective in study.objectives:
        number = _finite_float(result.metrics.values.get(objective.metric))
        fallback = float("inf")
        key.append(fallback if number is None else (number if objective.direction == "minimize" else -number))
    return tuple(key)


def candidate_rank_key(study: StudySpec, result: CandidateResult) -> tuple[float, ...]:
    """Sort by evidence coverage, violation, objectives, then control margin."""

    complete_and_feasible = result.feasible and result.success_fraction == 1.0 and result.feasible_fraction == 1.0
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


def _pareto_vector(study: StudySpec, result: CandidateResult) -> tuple[float, ...] | None:
    values: list[float] = []
    for objective in study.objectives:
        number = _finite_float(result.aggregates.get(objective.metric))
        if number is None:
            return None
        values.append(number if objective.direction == "minimize" else -number)
    return tuple(values)


def candidate_is_pareto_eligible(study: StudySpec, result: CandidateResult) -> bool:
    """Require complete electrical evidence and full declared feasibility.

    ``success_fraction`` covers each scenario's selected control. Pareto
    evidence is stricter: every explored control must have solved so a failed
    alternative cannot hide a potentially better operating point.
    """

    return (
        result.feasible
        and result.success_fraction == 1.0
        and result.feasible_fraction == 1.0
        and result.total_violation == 0.0
        and all(evaluation.raw.ok for evaluation in result.control_evaluations)
        and _pareto_vector(study, result) is not None
    )


def pareto_front(study: StudySpec, results: Iterable[CandidateResult]) -> tuple[CandidateResult, ...]:
    """Return the nondominated set among fully evidenced feasible candidates."""

    eligible = [
        (result, vector)
        for result in results
        if candidate_is_pareto_eligible(study, result)
        for vector in (_pareto_vector(study, result),)
        if vector is not None
    ]

    def dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        return all(a <= b for a, b in zip(left, right, strict=True)) and any(
            a < b for a, b in zip(left, right, strict=True)
        )

    return tuple(
        result
        for index, (result, vector) in enumerate(eligible)
        if not any(
            other_index != index and dominates(other_vector, vector)
            for other_index, (_other, other_vector) in enumerate(eligible)
        )
    )
