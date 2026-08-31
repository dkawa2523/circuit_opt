from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from pcd.core import (
    Candidate,
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
from pcd.core.models import to_plain
from pcd.core.spaces import grid_values, parameter_grid
from pcd.results import raw_evaluation_key


def _request(scenario: Scenario | None = None) -> EvaluationRequest:
    return EvaluationRequest(
        Candidate("candidate", {"x": 1.0}),
        scenario or Scenario("nominal"),
        ControlState({"tune": 2.0}),
    )


def _evaluation(scenario: Scenario | None = None, *, cache_key: str = "key") -> EvaluationResult:
    return EvaluationResult(
        request=_request(scenario),
        raw=RawResult("ok", {"waveform": [1.0, 2.0]}, {"file": "wave.csv"}),
        metrics=MetricSet({"loss": 0.25}),
        constraints=(ConstraintResult("limit", True, value=0.25, limit=1.0),),
        cache_key=cache_key,
        duration_s=0.1,
    )


def test_nested_inputs_are_copied_frozen_and_json_ready():
    original = {"nested": [{"value": 2}], "tags": {"rf", "plasma"}, "scalar": np.float64(1.5)}
    candidate = Candidate("design", original)
    original["nested"][0]["value"] = 99

    assert candidate.values["nested"][0]["value"] == 2
    mutable_view = cast(dict[str, object], candidate.values)
    with pytest.raises(TypeError):
        mutable_view["new"] = 1
    plain = candidate.to_dict()["values"]
    assert plain["nested"] == [{"value": 2}]
    assert set(plain["tags"]) == {"rf", "plasma"}
    assert plain["scalar"] == 1.5


def test_to_plain_keeps_an_object_when_its_scalar_conversion_is_not_valid():
    class NotAScalar:
        def item(self):
            raise ValueError("not scalar")

    value = NotAScalar()
    assert to_plain(value) is value


def test_metric_values_must_be_persistable_finite_numbers():
    with pytest.raises(ValueError, match=r"metric\.loss must be finite or null"):
        MetricSet({"loss": np.float64(np.inf)})


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Candidate(" "), "candidate id"),
        (lambda: Scenario("s", weight=0), "weight"),
        (lambda: Objective("loss", direction="sideways"), "direction"),
        (lambda: Objective("loss", aggregation="median"), "aggregation"),
        (lambda: Objective("loss", cvar_alpha=0), "cvar_alpha"),
        (lambda: RawResult("maybe"), "unsupported"),
        (lambda: ConstraintResult("limit", False, violation=-1), "non-negative"),
    ],
)
def test_invalid_domain_values_are_rejected_at_the_boundary(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scenarios": ()}, "at least one scenario"),
        ({"objectives": ()}, "at least one objective"),
        ({"scenarios": (Scenario("same"), Scenario("same"))}, "scenario ids must be unique"),
    ],
)
def test_a_study_rejects_empty_or_ambiguous_axes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StudySpec("study", **kwargs)


@pytest.mark.parametrize("margin", [-0.01, 1.01, np.nan, np.inf])
def test_a_study_rejects_an_invalid_control_margin_limit(margin):
    with pytest.raises(ValueError, match="control_margin_min must be between 0 and 1"):
        StudySpec("study", control_margin_min=margin)


@pytest.mark.parametrize("margin", [0.0, 0.25, 1.0])
def test_a_study_accepts_a_normalized_control_margin_limit(margin):
    assert StudySpec("study", control_margin_min=margin).to_dict()["control_margin_min"] == margin


def test_evaluation_result_round_trip_preserves_all_boundary_fields():
    original = _evaluation()
    restored = EvaluationResult.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored.feasible
    assert restored.total_violation == 0.0
    assert restored.request.merged_inputs() == {"x": 1.0, "tune": 2.0}


def test_raw_key_describes_physics_not_study_attribution():
    first = EvaluationRequest(
        Candidate("candidate-a", {"x": 1.0}),
        Scenario("scenario-a", {"offset": 2.0}, weight=1.0),
        ControlState({"tune": 3.0}),
    )
    relabeled = EvaluationRequest(
        Candidate("candidate-b", {"x": 1.0}),
        Scenario("scenario-b", {"offset": 2.0}, weight=9.0),
        ControlState({"tune": 3.0}),
    )
    changed_input = EvaluationRequest(
        relabeled.candidate,
        relabeled.scenario,
        ControlState({"tune": 4.0}),
    )

    assert raw_evaluation_key(first, {"circuit": "v1"}) == raw_evaluation_key(relabeled, {"circuit": "v1"})
    assert raw_evaluation_key(first, {"circuit": "v1"}) != raw_evaluation_key(changed_input, {"circuit": "v1"})
    assert raw_evaluation_key(first, {"circuit": "v1"}) != raw_evaluation_key(first, {"circuit": "v2"})


def test_scenario_result_requires_a_real_selected_trial():
    scenario = Scenario("nominal")
    selected = _evaluation(scenario, cache_key="selected")
    other = _evaluation(scenario, cache_key="other")
    with pytest.raises(ValueError, match="at least one"):
        ScenarioResult(scenario, selected, ())
    with pytest.raises(ValueError, match="one of"):
        ScenarioResult(scenario, selected, (other,))


def test_grid_values_cover_discrete_integer_linear_and_log_spaces():
    assert grid_values({"choices": ["a", "b"]}) == ["a", "b"]
    assert grid_values({"values": [1, 3]}) == [1, 3]
    assert grid_values({"default": 7}) == [7]
    assert grid_values({"bounds": [1, 5], "type": "int", "grid": 3}) == [1, 3, 5]
    assert grid_values({"bounds": [0.0, 1.0], "grid": 3}) == [0.0, 0.5, 1.0]
    assert grid_values({"bounds": [1e-3, 1e3], "scale": "log", "grid": 3}) == pytest.approx([1e-3, 1.0, 1e3])


def test_invalid_or_empty_parameter_spaces_fail_directly():
    with pytest.raises(ValueError, match="log-scale"):
        grid_values({"bounds": [0.0, 1.0], "scale": "log"})
    with pytest.raises(ValueError, match="no values"):
        grid_values({"values": []})
    with pytest.raises(ValueError, match="budget"):
        parameter_grid({"x": {"values": [1]}}, budget=0)


def test_parameter_grid_is_complete_and_rejects_an_incomplete_budget():
    specs = {
        "a": {"values": [1, 2], "default": 2},
        "b": {"values": ["x", "y", "z"], "default": "y"},
    }
    complete = parameter_grid(specs, budget=6)
    assert len(complete) == 6
    assert parameter_grid({}, budget=1) == [{}]
    with pytest.raises(ValueError, match="complete control grid has 6 states"):
        parameter_grid(specs, budget=3)
