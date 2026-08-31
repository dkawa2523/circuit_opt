from __future__ import annotations

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
    candidate_rank_key,
    control_margin,
    evaluation_rank_key,
)


def _evaluation(
    scenario: Scenario,
    value: object,
    *,
    status: str = "ok",
    violation: float = 0.0,
) -> EvaluationResult:
    request = EvaluationRequest(Candidate("design"), scenario, ControlState())
    constraints = (ConstraintResult("limit", violation == 0, violation),)
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


def test_candidate_aggregation_ignores_failed_missing_and_invalid_metrics():
    scenarios = (
        Scenario("valid", weight=2.0),
        Scenario("failed", weight=1.0),
        Scenario("missing", weight=1.0),
        Scenario("invalid", weight=1.0),
    )
    evaluations = (
        _evaluation(scenarios[0], "3.5"),
        _evaluation(scenarios[1], 1.0, status="failed", violation=1.0),
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

    # Physical distance is normalized against the declared numeric range. A
    # single numeric point is neutral, and categorical/bool axes do not enter.
    assert control_margin(controls[0], controls) == 0.0
    assert control_margin(controls[1], controls) == pytest.approx(0.4)
    assert control_margin(ControlState({"fixed": 3.0, "mode": "auto"}), controls) == 1.0
    assert control_margin(ControlState({"mode": "auto"}), controls) is None


def test_candidate_order_requires_complete_feasibility_before_objectives():
    scenario = Scenario("nominal")
    study = StudySpec("rank", scenarios=(scenario,), objectives=(Objective("score"),))
    evaluation = _evaluation(scenario, 2.0)
    scenario_result = ScenarioResult(scenario, evaluation, (evaluation,))
    complete = aggregate_candidate(study, Candidate("complete"), (scenario_result,))
    incomplete = type(complete)(
        candidate=Candidate("incomplete"),
        scenarios=(scenario_result,),
        aggregates={"score": 0.0},
        feasible_fraction=0.5,
        success_fraction=1.0,
        total_violation=0.0,
    )
    missing_metric = type(complete)(
        candidate=Candidate("missing"),
        scenarios=(scenario_result,),
        aggregates={"score": None},
        feasible_fraction=1.0,
        success_fraction=1.0,
        total_violation=0.0,
    )

    assert candidate_rank_key(study, complete) < candidate_rank_key(study, incomplete)
    assert candidate_rank_key(study, missing_metric)[-2] == float("inf")


def test_incomplete_candidate_order_prioritizes_evidence_then_feasible_coverage():
    scenario = Scenario("nominal")
    study = StudySpec("coverage-rank", scenarios=(scenario,), objectives=(Objective("score"),))
    evaluation = _evaluation(scenario, 2.0)
    scenario_result = ScenarioResult(scenario, evaluation, (evaluation,))

    successful = CandidateResult(
        Candidate("successful"),
        (scenario_result,),
        {"score": 100.0},
        feasible_fraction=0.5,
        success_fraction=1.0,
        total_violation=100.0,
    )
    failed_evidence = CandidateResult(
        Candidate("failed-evidence"),
        (scenario_result,),
        {"score": 0.0},
        feasible_fraction=0.75,
        success_fraction=0.75,
        total_violation=0.0,
    )
    higher_coverage = CandidateResult(
        Candidate("higher-coverage"),
        (scenario_result,),
        {"score": 100.0},
        feasible_fraction=0.5,
        success_fraction=1.0,
        total_violation=100.0,
    )
    lower_coverage = CandidateResult(
        Candidate("lower-coverage"),
        (scenario_result,),
        {"score": 0.0},
        feasible_fraction=0.25,
        success_fraction=1.0,
        total_violation=0.0,
    )

    assert candidate_rank_key(study, successful) < candidate_rank_key(study, failed_evidence)
    assert candidate_rank_key(study, higher_coverage) < candidate_rank_key(study, lower_coverage)


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
