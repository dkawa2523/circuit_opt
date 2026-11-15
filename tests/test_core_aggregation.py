from __future__ import annotations

from dataclasses import replace

import pytest

from pcd.core import (
    Candidate,
    CandidateResult,
    ConstraintResult,
    ControlState,
    EvaluationRequest,
    EvaluationResult,
    MetricSet,
    Objective,
    RawResult,
    Scenario,
    ScenarioResult,
    StudySpec,
)
from pcd.core.aggregation import (
    aggregate_candidate,
    aggregate_objective,
    candidate_is_pareto_eligible,
    candidate_rank_key,
    control_margin,
    evaluation_rank_key,
    pareto_front,
)


def _evaluation(
    scenario: Scenario,
    value: object,
    *,
    status: str = "ok",
    violation: float = 0.0,
) -> EvaluationResult:
    request = EvaluationRequest(Candidate("design"), scenario, ControlState())
    constraints = (ConstraintResult("limit", violation == 0, violation),) if status == "ok" else ()
    return EvaluationResult(request, RawResult(status), MetricSet({"score": value}), constraints)


def test_objective_aggregation_supports_weighted_mean_worst_and_cvar():
    values = [(1.0, 1.0), (10.0, 3.0)]
    assert aggregate_objective([], Objective("score")) is None
    assert aggregate_objective(values, Objective("score", aggregation="mean")) == pytest.approx(7.75)
    assert aggregate_objective(values, Objective("score", aggregation="worst")) == 10.0
    assert aggregate_objective(values, Objective("score", direction="maximize", aggregation="worst")) == 1.0
    assert aggregate_objective(values, Objective("score", aggregation="cvar", cvar_alpha=0.25)) == 10.0
    assert (
        aggregate_objective(
            values,
            Objective("score", direction="maximize", aggregation="cvar", cvar_alpha=0.25),
        )
        == 1.0
    )


@pytest.mark.parametrize("scale", [1.0, 1e-16, 1e300])
def test_weighted_decisions_are_invariant_to_a_common_weight_scale(scale):
    values = [(value, scale) for value in (10.0, 6.0, 2.0, 0.0)]
    assert aggregate_objective(values, Objective("score", aggregation="mean")) == pytest.approx(4.5)
    assert aggregate_objective(values, Objective("score", aggregation="cvar", cvar_alpha=0.5)) == pytest.approx(8.0)
    scenarios = (Scenario("accepted", weight=scale), Scenario("failed", weight=scale))
    trials = (_evaluation(scenarios[0], 1.0), _evaluation(scenarios[1], 2.0, status="failed"))
    result = aggregate_candidate(
        StudySpec("weights", scenarios=scenarios, objectives=(Objective("score"),)),
        Candidate("design"),
        tuple(ScenarioResult(scenario, trial, (trial,)) for scenario, trial in zip(scenarios, trials, strict=True)),
    )
    assert result.success_fraction == result.feasible_fraction == 0.5


def test_candidate_aggregation_ignores_failed_missing_and_invalid_metrics():
    scenarios = (
        Scenario("valid", weight=2.0),
        Scenario("failed", weight=1.0),
        Scenario("missing", weight=1.0),
        Scenario("invalid", weight=1.0),
    )
    evaluations = (
        _evaluation(scenarios[0], "3.5"),
        _evaluation(scenarios[1], 1.0, status="failed"),
        _evaluation(scenarios[2], None),
        _evaluation(scenarios[3], "not-a-number"),
    )
    results = tuple(
        ScenarioResult(scenario, evaluation, (evaluation,))
        for scenario, evaluation in zip(scenarios, evaluations, strict=True)
    )
    study = StudySpec(
        "weighted",
        scenarios=scenarios,
        objectives=(Objective("score", aggregation="mean"),),
    )

    result = aggregate_candidate(study, Candidate("design"), results)
    assert result.aggregates["score"] == pytest.approx(3.5)
    assert result.success_fraction == pytest.approx(0.8)
    assert result.feasible_fraction == pytest.approx(0.8)
    assert result.total_violation == pytest.approx(0.2)


def test_partial_objective_uses_original_weight_of_surviving_conditions():
    scenarios = (Scenario("valid", weight=1e-300), Scenario("failed", weight=1e300))
    trials = (_evaluation(scenarios[0], 2.0), _evaluation(scenarios[1], 3.0, status="failed"))
    result = aggregate_candidate(
        StudySpec("weights", scenarios=scenarios, objectives=(Objective("score", aggregation="mean"),)),
        Candidate("design"),
        tuple(ScenarioResult(scenario, trial, (trial,)) for scenario, trial in zip(scenarios, trials, strict=True)),
    )

    assert result.aggregates["score"] == 2.0
    assert not result.successful


def test_evaluation_order_is_feasibility_first_then_direction_aware():
    scenario = Scenario("nominal")
    study = StudySpec("maximize", objectives=(Objective("score", direction="maximize"),))
    feasible = _evaluation(scenario, 4.0)
    infeasible = _evaluation(scenario, 100.0, violation=0.1)
    invalid = _evaluation(scenario, "not-a-number")

    assert evaluation_rank_key(study, feasible) < evaluation_rank_key(study, infeasible)
    assert evaluation_rank_key(study, feasible)[-1] == -4.0
    assert evaluation_rank_key(study, invalid)[-1] == float("inf")


def test_control_margin_uses_numeric_grid_edges_and_ignores_categories():
    controls = (
        ControlState({"tune": 0.0, "fixed": 3.0, "mode": "low", "enabled": False}),
        ControlState({"tune": 2.0, "fixed": 3.0, "mode": "auto", "enabled": True}),
        ControlState({"tune": 10.0, "fixed": 3.0, "mode": "high", "enabled": False}),
    )

    # Fixed, categorical, and bool axes contribute no headroom.
    assert control_margin(controls[0], controls) == 0.0
    assert control_margin(controls[1], controls) == pytest.approx(0.4)
    assert control_margin(ControlState({"fixed": 3.0, "mode": "auto"}), controls) is None
    assert control_margin(ControlState({"mode": "auto"}), controls) is None
    recipe_codes = tuple(ControlState({"recipe": code}) for code in ("001", "002", "003"))
    assert control_margin(recipe_codes[1], recipe_codes) is None
    assert control_margin(ControlState({"tune": 1.0}, {"tune": (0.0, 4.0)}), ()) == 0.5
    assert control_margin(ControlState({"tune": 0.0}, {"tune": (-1e308, 1e308)}), ()) == 1.0


def test_candidate_order_requires_complete_feasibility_before_objectives():
    nominal, light = Scenario("nominal"), Scenario("light", weight=1e-30)
    study = StudySpec("rank", scenarios=(nominal, light), objectives=(Objective("score"),))

    def evaluate(name, score, violation):
        trials = (_evaluation(nominal, score), _evaluation(light, score, violation=violation))
        results = tuple(ScenarioResult(item.request.scenario, item, (item,)) for item in trials)
        return aggregate_candidate(study, Candidate(name), results)

    accepted = evaluate("accepted", 2.0, 0.0)
    rejected = evaluate("rejected", 1.0, 1.0)
    assert rejected.feasible_fraction == accepted.feasible_fraction == 1.0
    assert candidate_rank_key(study, accepted) < candidate_rank_key(study, rejected)


def test_incomplete_candidate_order_prioritizes_evidence_then_feasible_coverage():
    light, heavy = Scenario("light", weight=1.0), Scenario("heavy", weight=3.0)
    study = StudySpec("coverage-rank", scenarios=(light, heavy), objectives=(Objective("score"),))

    def summarize(*trials):
        scenarios = tuple(ScenarioResult(item.request.scenario, item, (item,)) for item in trials)
        return aggregate_candidate(study, Candidate("design"), scenarios)

    def unavailable(scenario):
        return replace(
            _evaluation(scenario, 0.0),
            constraints=(ConstraintResult("limit", False, 1.0, evidence_available=False),),
        )

    higher_coverage = summarize(_evaluation(light, 100.0, violation=400.0), _evaluation(heavy, 10.0))
    lower_coverage = summarize(_evaluation(light, 10.0), _evaluation(heavy, 100.0, violation=2.0))
    incomplete_light = summarize(unavailable(light), _evaluation(heavy, 0.0))
    incomplete_heavy = summarize(_evaluation(light, 0.0), unavailable(heavy))
    failed_heavy = summarize(_evaluation(light, 0.0), _evaluation(heavy, 0.0, status="failed"))

    assert higher_coverage.success_fraction == incomplete_light.success_fraction == 1.0
    assert candidate_rank_key(study, higher_coverage) < candidate_rank_key(study, lower_coverage)
    assert candidate_rank_key(study, lower_coverage) < candidate_rank_key(study, incomplete_light)
    assert candidate_rank_key(study, incomplete_light) < candidate_rank_key(study, incomplete_heavy)
    assert candidate_rank_key(study, incomplete_light) < candidate_rank_key(study, failed_heavy)


def test_candidate_order_uses_worst_control_margin_only_after_objectives():
    scenario = Scenario("nominal")
    study = StudySpec("margin-rank", scenarios=(scenario,), objectives=(Objective("score"),))
    endpoint_evaluation = _evaluation(scenario, 2.0)
    centered_evaluation = EvaluationResult(
        EvaluationRequest(Candidate("design"), scenario, ControlState({"tune": 5.0})),
        RawResult("ok"),
        MetricSet({"score": 2.0}),
        (ConstraintResult("limit", True),),
    )
    endpoint = aggregate_candidate(
        study,
        Candidate("endpoint"),
        (ScenarioResult(scenario, endpoint_evaluation, (endpoint_evaluation,), 0.0),),
    )
    centered = aggregate_candidate(
        study,
        Candidate("centered"),
        (ScenarioResult(scenario, centered_evaluation, (centered_evaluation,), 1.0),),
    )
    better_objective_at_edge = type(endpoint)(
        candidate=Candidate("better-objective"),
        scenarios=endpoint.scenarios,
        aggregates={"score": 1.0},
        feasible_fraction=1.0,
        success_fraction=1.0,
        total_violation=0.0,
    )

    assert candidate_rank_key(study, centered) < candidate_rank_key(study, endpoint)
    assert candidate_rank_key(study, better_objective_at_edge) < candidate_rank_key(study, centered)


def test_pareto_front_uses_declared_directions_and_only_complete_feasible_evidence():
    scenario = Scenario("nominal")
    study = StudySpec(
        "tradeoff",
        scenarios=(scenario,),
        objectives=(Objective("error"), Objective("efficiency", direction="maximize")),
    )

    def candidate_result(
        candidate_id: str,
        error: float,
        efficiency: float,
        *,
        feasible: bool = True,
        failed_alternative: bool = False,
    ) -> CandidateResult:
        candidate = Candidate(candidate_id)
        selected = EvaluationResult(
            EvaluationRequest(candidate, scenario, ControlState({"tune": 0})),
            RawResult("ok"),
            MetricSet({"error": error, "efficiency": efficiency}),
            (ConstraintResult("limit", feasible, 0.0 if feasible else 1.0),),
        )
        trials = [selected]
        if failed_alternative:
            trials.append(
                EvaluationResult(
                    EvaluationRequest(candidate, scenario, ControlState({"tune": 1})),
                    RawResult("failed", error="solver failed"),
                )
            )
        return CandidateResult(
            candidate,
            (ScenarioResult(scenario, selected, tuple(trials)),),
            {"error": error, "efficiency": efficiency},
            feasible_fraction=1.0 if feasible else 0.0,
            success_fraction=1.0,
            total_violation=0.0 if feasible else 1.0,
        )

    low_error = candidate_result("low-error", 1.0, 0.5)
    high_efficiency = candidate_result("high-efficiency", 2.0, 0.8)
    same_tradeoff = candidate_result("same-tradeoff", 2.0, 0.8)
    dominated = candidate_result("dominated", 3.0, 0.7)
    incomplete = candidate_result("incomplete", 0.5, 0.9, failed_alternative=True)
    infeasible = candidate_result("infeasible", 0.2, 1.0, feasible=False)

    assert candidate_is_pareto_eligible(study, low_error)
    assert not candidate_is_pareto_eligible(study, incomplete)
    assert not candidate_is_pareto_eligible(study, infeasible)
    assert [
        item.candidate.candidate_id
        for item in pareto_front(
            study,
            (low_error, high_efficiency, same_tradeoff, dominated, incomplete, infeasible),
        )
    ] == ["low-error", "high-efficiency", "same-tradeoff"]
