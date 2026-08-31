from __future__ import annotations

from dataclasses import dataclass

import pytest

from pcd.core import (
    Candidate,
    ConstraintResult,
    ControlState,
    EvaluationRequest,
    MetricSet,
    Objective,
    RawResult,
    Scenario,
    StudyRunner,
    StudySpec,
)
from pcd.results import FileResultStore


@dataclass
class CountingEvaluator:
    calls: int = 0

    def evaluate(self, request):
        self.calls += 1
        values = request.merged_inputs()
        return RawResult("ok", {"score": float(values["x"]) + float(values.get("offset", 0.0))})


class ScoreMetric:
    def compute(self, request, raw):
        del request
        return MetricSet({"loss": raw.observations["score"]})


class LimitConstraint:
    def __init__(self, limit):
        self.limit = limit

    def evaluate(self, request, raw, metrics):
        del request, raw
        value = float(metrics.values["loss"])
        return ConstraintResult("loss_limit", value <= self.limit, max(0.0, value - self.limit), value, self.limit)


def _study():
    return StudySpec(
        "generic",
        scenarios=(Scenario("nominal", {"offset": 0.0}, 1.0), Scenario("corner", {"offset": 2.0}, 1.0)),
        objectives=(Objective("loss", aggregation="worst"),),
    )


def test_role_collisions_are_rejected_before_evaluation():
    request = EvaluationRequest(
        Candidate("c", {"shared": 1}),
        Scenario("s", {"shared": 2}),
        ControlState(),
    )
    with pytest.raises(ValueError, match="belongs to both candidate and scenario"):
        request.merged_inputs()


def test_worst_scenario_and_feasibility_are_aggregated(tmp_path):
    evaluator = CountingEvaluator()
    runner = StudyRunner(
        _study(),
        evaluator,
        metrics=(ScoreMetric(),),
        constraints=(LimitConstraint(3.0),),
        store=FileResultStore(tmp_path, "generic"),
    )
    result = runner.evaluate_candidate(Candidate("candidate", {"x": 1.5}))
    assert result.aggregates["loss"] == pytest.approx(3.5)
    assert result.success_fraction == 1.0
    assert result.feasible_fraction == 0.5
    assert result.total_violation == pytest.approx(0.25)


def test_identical_evaluations_are_resumed_from_content_cache(tmp_path):
    evaluator = CountingEvaluator()
    store = FileResultStore(tmp_path, "generic", {"backend": "test-v1"})
    runner = StudyRunner(_study(), evaluator, metrics=(ScoreMetric(),), store=store)
    candidate = Candidate("same", {"x": 1.0})

    first = runner.evaluate_candidate(candidate)
    second = runner.evaluate_candidate(candidate)

    assert evaluator.calls == 2  # two scenarios, only on the first call
    assert not any(item.from_cache for item in first.evaluations)
    assert all(item.from_cache for item in second.evaluations)
    assert len(list((store.root / "evaluations").glob("*/result.json"))) == 2
    assert len(list((store.root / "raw").glob("*/raw_result.json"))) == 2
    assert all(item.raw_cache_key for item in second.evaluations)


def test_physical_cache_ignores_attribution_ids_and_scenario_weights(tmp_path):
    study = StudySpec(
        "attribution",
        scenarios=(
            Scenario("nominal-a", {"offset": 0.0}, 1.0),
            Scenario("nominal-b", {"offset": 0.0}, 3.0),
        ),
        objectives=(Objective("loss"),),
    )
    evaluator = CountingEvaluator()
    store = FileResultStore(tmp_path, study.study_id)
    runner = StudyRunner(study, evaluator, metrics=(ScoreMetric(),), store=store)

    first = runner.evaluate_candidate(Candidate("first-label", {"x": 1.0}))
    second = runner.evaluate_candidate(Candidate("second-label", {"x": 1.0}))

    assert evaluator.calls == 1
    assert [item.from_cache for item in first.evaluations] == [False, True]
    assert all(item.from_cache for item in second.evaluations)
    assert len({item.raw_cache_key for item in (*first.evaluations, *second.evaluations)}) == 1
    assert len({item.cache_key for item in (*first.evaluations, *second.evaluations)}) == 4
    assert len(list((store.root / "raw").glob("*/raw_result.json"))) == 1
    assert len(list((store.root / "evaluations").glob("*/result.json"))) == 4
    assert {item.request.candidate.candidate_id for item in second.evaluations} == {"second-label"}


def test_current_metrics_and_constraints_are_recomputed_from_cached_raw_output(tmp_path):
    class ScaledScoreMetric:
        def __init__(self, factor):
            self.factor = factor

        def compute(self, request, raw):
            del request
            return MetricSet({"loss": self.factor * float(raw.observations["score"])})

    evaluator = CountingEvaluator()
    raw_fingerprint = {"backend": "unchanged"}
    candidate = Candidate("same", {"x": 1.0})
    first_store = FileResultStore(
        tmp_path,
        "generic",
        {"evaluation": "v1"},
        raw_runtime_fingerprint=raw_fingerprint,
    )
    first = StudyRunner(
        _study(),
        evaluator,
        metrics=(ScaledScoreMetric(1.0),),
        constraints=(LimitConstraint(4.0),),
        store=first_store,
    ).evaluate_candidate(candidate)

    second_store = FileResultStore(
        tmp_path,
        "generic",
        {"evaluation": "v2"},
        raw_runtime_fingerprint=raw_fingerprint,
    )
    second = StudyRunner(
        _study(),
        evaluator,
        metrics=(ScaledScoreMetric(10.0),),
        constraints=(LimitConstraint(4.0),),
        store=second_store,
    ).evaluate_candidate(candidate)

    assert evaluator.calls == 2
    assert [item.metrics.values["loss"] for item in first.evaluations] == [1.0, 3.0]
    assert [item.metrics.values["loss"] for item in second.evaluations] == [10.0, 30.0]
    assert all(item.feasible for item in first.evaluations)
    assert not any(item.feasible for item in second.evaluations)
    assert all(item.from_cache for item in second.evaluations)
    assert len(list((first_store.root / "raw").glob("*/raw_result.json"))) == 2
    assert len(list((first_store.root / "evaluations").glob("*/result.json"))) == 4


def test_measurement_failure_does_not_poison_reusable_raw_output(tmp_path):
    class BrokenMetric:
        def compute(self, request, raw):
            del request, raw
            raise RuntimeError("metric definition is broken")

    evaluator = CountingEvaluator()
    raw_fingerprint = {"backend": "same"}
    candidate = Candidate("candidate", {"x": 1.0})
    broken = StudyRunner(
        _study(),
        evaluator,
        metrics=(BrokenMetric(),),
        store=FileResultStore(
            tmp_path,
            "generic",
            {"measurement": "broken"},
            raw_runtime_fingerprint=raw_fingerprint,
        ),
    ).evaluate_candidate(candidate)
    recovered = StudyRunner(
        _study(),
        evaluator,
        metrics=(ScoreMetric(),),
        store=FileResultStore(
            tmp_path,
            "generic",
            {"measurement": "fixed"},
            raw_runtime_fingerprint=raw_fingerprint,
        ),
    ).evaluate_candidate(candidate)

    assert evaluator.calls == 2
    assert all(item.raw.diagnostics["stage"] == "measure" for item in broken.evaluations)
    assert all(item.raw.ok and item.from_cache for item in recovered.evaluations)


def test_inner_control_search_selects_a_different_operating_point_per_scenario(tmp_path):
    class Controls:
        def controls(self, study, candidate, scenario):
            del study, candidate, scenario
            return (ControlState({"tune": -2.0}), ControlState({"tune": 0.0}), ControlState({"tune": 2.0}))

    class TrackingEvaluator:
        def evaluate(self, request):
            values = request.merged_inputs()
            return RawResult("ok", {"score": abs(values["offset"] + values["tune"])})

    runner = StudyRunner(
        _study(),
        TrackingEvaluator(),
        metrics=(ScoreMetric(),),
        control_policy=Controls(),
        store=FileResultStore(tmp_path, "tuned"),
    )
    result = runner.evaluate_candidate(Candidate("candidate", {"x": 0.0}))
    assert len(result.control_evaluations) == 6
    assert result.scenarios[0].selected.request.control.values["tune"] == 0.0
    assert result.scenarios[1].selected.request.control.values["tune"] == -2.0
    assert result.aggregates["loss"] == 0.0
    assert result.scenarios[0].control_margin == 1.0
    assert result.scenarios[1].control_margin == 0.0
    assert result.control_margin == 0.0
    assert result.to_dict()["control_margin"] == 0.0
    assert result.to_dict()["scenarios"][0]["control_margin"] == 1.0


def test_inner_control_search_uses_margin_only_to_break_an_objective_tie():
    class Controls:
        def controls(self, study, candidate, scenario):
            del study, candidate, scenario
            return (
                ControlState({"tune": -1.0}),
                ControlState({"tune": 0.0}),
                ControlState({"tune": 1.0}),
            )

    class FlatEvaluator:
        def evaluate(self, request):
            del request
            return RawResult("ok", {"score": 1.0})

    scenario = Scenario("nominal")
    study = StudySpec("control-tie", scenarios=(scenario,), objectives=(Objective("loss"),))
    result = StudyRunner(
        study,
        FlatEvaluator(),
        metrics=(ScoreMetric(),),
        control_policy=Controls(),
    ).evaluate_candidate(Candidate("candidate", {"x": 0.0}))

    assert result.evaluations[0].request.control.values["tune"] == 0.0
    assert result.control_margin == 1.0

    class ShapedEvaluator:
        def evaluate(self, request):
            tune = float(request.control.values["tune"])
            return RawResult("ok", {"score": abs(tune + 1.0)})

    better_edge = StudyRunner(
        study,
        ShapedEvaluator(),
        metrics=(ScoreMetric(),),
        control_policy=Controls(),
    ).evaluate_candidate(Candidate("candidate", {"x": 0.0}))
    assert better_edge.evaluations[0].request.control.values["tune"] == -1.0
    assert better_edge.control_margin == 0.0


def test_control_margin_acceptance_selects_an_interior_operating_point():
    class Controls:
        def controls(self, study, candidate, scenario):
            del study, candidate, scenario
            return (ControlState({"tune": -1.0}), ControlState({"tune": 0.0}), ControlState({"tune": 1.0}))

    class EdgeIsSlightlyBetter:
        def evaluate(self, request):
            tune = float(request.control.values["tune"])
            return RawResult("ok", {"score": abs(tune + 1.0) * 0.1})

    study = StudySpec(
        "control-reserve",
        scenarios=(Scenario("nominal"),),
        objectives=(Objective("loss"),),
        control_margin_min=0.5,
    )
    result = StudyRunner(
        study,
        EdgeIsSlightlyBetter(),
        metrics=(ScoreMetric(),),
        control_policy=Controls(),
    ).evaluate_candidate(Candidate("candidate"))

    selected = result.scenarios[0].selected
    assert selected.request.control.values["tune"] == 0.0
    assert selected.feasible
    assert result.control_margin == 1.0
    assert not result.edge_limited
    margin_constraint = next(item for item in selected.constraints if item.name == "min_control_margin")
    assert margin_constraint.satisfied
    assert margin_constraint.value == 1.0


def test_electrically_reachable_edge_is_retained_when_no_jointly_feasible_control_exists(tmp_path):
    class Controls:
        def controls(self, study, candidate, scenario):
            del study, candidate, scenario
            return (ControlState({"tune": -1.0}), ControlState({"tune": 0.0}), ControlState({"tune": 1.0}))

    class OnlyEdgeMeetsElectricalLimit:
        def evaluate(self, request):
            return RawResult("ok", {"score": abs(float(request.control.values["tune"]) + 1.0)})

    study = StudySpec(
        "edge-limited",
        scenarios=(Scenario("nominal"),),
        objectives=(Objective("loss"),),
        control_margin_min=0.5,
    )
    store = FileResultStore(tmp_path, study.study_id)
    result = StudyRunner(
        study,
        OnlyEdgeMeetsElectricalLimit(),
        metrics=(ScoreMetric(),),
        constraints=(LimitConstraint(0.0),),
        control_policy=Controls(),
        store=store,
    ).evaluate_candidate(Candidate("candidate"))

    selected = result.scenarios[0].selected
    assert selected.request.control.values["tune"] == -1.0
    assert not selected.feasible
    assert result.feasible_fraction == 0.0
    assert result.edge_limited
    assert result.to_dict()["scenarios"][0]["edge_limited"] is True
    persisted = next((store.root / "evaluations").glob("*/result.json")).read_text(encoding="utf-8")
    assert "min_control_margin" in persisted


def test_failed_evaluation_is_recorded_and_does_not_abort_the_study(tmp_path):
    class Broken:
        def evaluate(self, request):
            del request
            raise RuntimeError("backend stopped")

    runner = StudyRunner(_study(), Broken(), metrics=(ScoreMetric(),), store=FileResultStore(tmp_path, "broken"))
    result = runner.evaluate_candidate(Candidate("candidate", {"x": 1.0}))
    assert result.success_fraction == 0.0
    assert result.feasible_fraction == 0.0
    assert all(item.raw.status == "failed" for item in result.evaluations)
    assert all("backend stopped" in str(item.raw.error) for item in result.evaluations)
    assert list((tmp_path / "broken" / "raw").glob("*/raw_result.json")) == []


def test_failed_raw_evaluation_is_retried_instead_of_cached(tmp_path):
    class Recovering:
        def __init__(self):
            self.calls = 0

        def evaluate(self, request):
            del request
            self.calls += 1
            return RawResult("failed") if self.calls <= 2 else RawResult("ok", {"score": 1.0})

    evaluator = Recovering()
    runner = StudyRunner(_study(), evaluator, metrics=(ScoreMetric(),), store=FileResultStore(tmp_path, "retry"))
    candidate = Candidate("candidate", {"x": 1.0})

    first = runner.evaluate_candidate(candidate)
    second = runner.evaluate_candidate(candidate)

    assert first.success_fraction == 0.0
    assert second.success_fraction == 1.0
    assert evaluator.calls == 4
    assert not any(item.from_cache for item in second.evaluations)


def test_feasible_candidates_rank_before_lower_but_infeasible_loss(tmp_path):
    runner = StudyRunner(
        _study(),
        CountingEvaluator(),
        metrics=(ScoreMetric(),),
        constraints=(LimitConstraint(3.0),),
        store=FileResultStore(tmp_path, "rank"),
    )

    # x=1 is feasible at both scenarios (worst=3); x=-10 has a lower loss and
    # is also feasible, so add a constraint that makes only the negative one bad.
    class NonNegative:
        def evaluate(self, request, raw, metrics):
            del raw, metrics
            value = float(request.candidate.values["x"])
            return ConstraintResult("nonnegative", value >= 0, max(0.0, -value), value, 0.0)

    runner.constraints = (LimitConstraint(3.0), NonNegative())
    ordered = runner.run([Candidate("infeasible", {"x": -10.0}), Candidate("feasible", {"x": 1.0})])
    assert ordered[0].candidate.candidate_id == "feasible"


def test_an_empty_control_policy_is_rejected_before_backend_work():
    class NoControls:
        def controls(self, study, candidate, scenario):
            del study, candidate, scenario
            return ()

    runner = StudyRunner(_study(), CountingEvaluator(), metrics=(ScoreMetric(),), control_policy=NoControls())
    with pytest.raises(ValueError, match="no operating point"):
        runner.evaluate_candidate(Candidate("candidate", {"x": 1.0}))


def test_duplicate_metric_names_become_a_recorded_measurement_failure():
    runner = StudyRunner(
        _study(),
        CountingEvaluator(),
        metrics=(ScoreMetric(), ScoreMetric()),
    )
    result = runner.evaluate_candidate(Candidate("candidate", {"x": 1.0}))
    assert all(item.raw.status == "failed" for item in result.evaluations)
    assert all(item.raw.diagnostics["stage"] == "measure" for item in result.evaluations)
    assert all("duplicate values" in str(item.raw.error) for item in result.evaluations)
    assert all(item.cache_key == "" for item in result.evaluations)


def test_missing_objective_metric_becomes_a_recorded_measurement_failure():
    class IncompleteMetric:
        def compute(self, request, raw):
            del request, raw
            return MetricSet({"diagnostic_only": 1.0})

    result = StudyRunner(_study(), CountingEvaluator(), metrics=(IncompleteMetric(),)).evaluate_candidate(
        Candidate("candidate", {"x": 1.0})
    )

    assert all(item.raw.status == "failed" for item in result.evaluations)
    assert all("objective metric 'loss'" in str(item.raw.error) for item in result.evaluations)


def test_a_backend_reported_failure_needs_no_exception_or_result_store():
    class Failed:
        def evaluate(self, request):
            del request
            return RawResult("failed")

    result = StudyRunner(_study(), Failed(), metrics=(ScoreMetric(),)).evaluate_candidate(
        Candidate("candidate", {"x": 1.0})
    )
    assert result.success_fraction == 0.0
    assert all(item.constraints[-1].detail == "failed" for item in result.evaluations)
