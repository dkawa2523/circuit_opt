"""Scenario aggregation and feasibility-first candidate ordering."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

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


def _control_number(value: object) -> float | None:
    # Numeric-looking categorical labels must retain their declared type.
    return _finite_float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


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

    An axis needs a nonzero numeric range. Fixed and categorical values do
    not provide headroom. Use the minimum margin of the remaining axes.
    """

    margins: list[float] = []
    for name, value in selected.values.items():
        number = _control_number(value)
        if number is None:
            continue
        if name in selected.numeric_ranges:
            low, high = selected.numeric_ranges[name]
        else:
            values = [_control_number(control.values.get(name)) for control in available]
            numeric = [value for value in values if value is not None]
            if not numeric or len(numeric) != len(available):
                continue
            low, high = min(numeric), max(numeric)
        if low == high:
            continue
        if not low < number < high:
            margins.append(0.0)
            continue
        if not math.isfinite(high - low):
            low, high, number = low / 2.0, high / 2.0, number / 2.0
        margins.append(2.0 * (min(number - low, high - number) / (high - low)))
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
    declared_weights = {item.scenario_id: item.weight for item in study.scenarios}
    weights = dict(
        zip(
            declared_weights,
            _normalized_weights(list(declared_weights.values())),
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
                values.append((number, declared_weights[item.request.scenario.scenario_id]))
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
    return (
        0.0 if result.evidence_available else 1.0,
        0.0 if result.feasible else 1.0,
        result.total_violation,
        *objective_rank(study, result.metrics.values),
    )


def objective_rank(study: StudySpec, values: Mapping[str, object]) -> tuple[float, ...]:
    """Normalize objective direction for both control and candidate ordering."""

    key = []
    for objective in study.objectives:
        number = _finite_float(values.get(objective.metric))
        key.append(float("inf") if number is None else (number if objective.direction == "minimize" else -number))
    return tuple(key)


def candidate_rank_key(study: StudySpec, result: CandidateResult) -> tuple[float, ...]:
    """Rank accepted, measured rejected, then incomplete candidates."""

    unavailable = {item.request.scenario.scenario_id for item in result.evaluations if not item.evidence_available}
    evidence_shortfall = 0.0
    if unavailable:
        weights = _normalized_weights([scenario.weight for scenario in study.scenarios])
        evidence_shortfall = math.fsum(
            weight
            for scenario, weight in zip(study.scenarios, weights, strict=True)
            if scenario.scenario_id in unavailable
        )
    # Even a negligible weight cannot move missing evidence into the measured tier.
    tier = 2.0 if unavailable else (0.0 if result.feasible else 1.0)
    margin = result.control_margin
    return (
        tier,
        evidence_shortfall,
        1.0 - result.feasible_fraction,
        result.total_violation,
        *objective_rank(study, result.aggregates),
        float("inf") if margin is None else -margin,
    )


def _pareto_vector(study: StudySpec, result: CandidateResult) -> tuple[float, ...] | None:
    if (
        not result.feasible
        or result.total_violation
        or not all(evaluation.evidence_available for evaluation in result.control_evaluations)
    ):
        return None
    values = objective_rank(study, result.aggregates)
    return values if all(map(math.isfinite, values)) else None


def candidate_is_pareto_eligible(study: StudySpec, result: CandidateResult) -> bool:
    """Require complete electrical evidence and full declared feasibility.

    Every explored control needs usable acceptance evidence so an unknown
    alternative cannot hide a better operating point.
    """

    return _pareto_vector(study, result) is not None


def pareto_front(study: StudySpec, results: Iterable[CandidateResult]) -> tuple[CandidateResult, ...]:
    """Return the nondominated set among fully evidenced feasible candidates."""

    eligible = [
        (result, vector) for result in results for vector in (_pareto_vector(study, result),) if vector is not None
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
