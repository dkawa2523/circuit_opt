"""Circuit metrics and candidate-search behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcd import metrics as metrics_module
from pcd.metrics import MetricLimitConstraint, interpolate_to_target, measure_response
from pcd.search import create_optimizer, validate_proposal
from pcd.simulation import SimulationResult

RTOL_LOSS = 1e-12


def test_optimizer_proposals_are_normalized_at_the_design_space_boundary(make_case):
    case = make_case(
        {
            "case_id": "proposal",
            "variables": {
                "turns": {"type": "int", "bounds": [1, 5], "default": 3},
                "topology": {"choices": ["l", "pi"], "default": "pi"},
            },
        }
    )
    assert validate_proposal(case, {"turns": np.int64(4)}) == {"turns": 4, "topology": "pi"}
    with pytest.raises(ValueError, match="undeclared"):
        validate_proposal(case, {"turns": 4, "unknown": 1})
    with pytest.raises(ValueError, match="outside"):
        validate_proposal(case, {"turns": 9})
    with pytest.raises(ValueError, match="must be int"):
        validate_proposal(case, {"turns": 2.5})


def test_optimizer_proposals_reject_missing_and_invalid_typed_values(make_case):
    case = make_case(
        {
            "case_id": "typed_proposal",
            "variables": {
                "enabled": {"type": "bool"},
                "gain": {"type": "float", "bounds": [0.1, 10.0]},
                "label": {"type": "string"},
                "mode": {"choices": ["a", "b"]},
            },
        }
    )
    with pytest.raises(TypeError, match="mapping"):
        validate_proposal(case, [])
    with pytest.raises(ValueError, match="missing candidate variables"):
        validate_proposal(case, {})

    baseline = {"enabled": True, "gain": 1.0, "label": "ok", "mode": "a"}
    for name, value, message in (
        ("enabled", 1, "must be bool"),
        ("gain", "high", "must be numeric"),
        ("gain", float("nan"), "must be finite"),
        ("label", 3, "must be a string"),
        ("mode", "c", "must be one of"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_proposal(case, {**baseline, name: value})

    numeric_case = make_case(
        {"case_id": "numeric_proposal", "variables": {"x": {"bounds": [1, 2]}, "positive": {"scale": "log"}}}
    )
    with pytest.raises(ValueError, match="within its bounds"):
        validate_proposal(numeric_case, {"x": [], "positive": 1})
    with pytest.raises(ValueError, match="positive on a log scale"):
        validate_proposal(numeric_case, {"x": 1.5, "positive": 0})


def _waveform_response(frame: pd.DataFrame) -> SimulationResult:
    return SimulationResult(time_s=frame["time_s"].to_numpy(float), voltage_V=frame["voltage_V"].to_numpy(float))


def _candidate_frame(case, n: int, seed: int) -> pd.DataFrame:
    optimizer = create_optimizer(case, optimizer_name="random", seed=seed)
    return pd.DataFrame([optimizer.ask() for _ in range(n)])


# --- objectives ------------------------------------------------------------


def test_scoring_a_target_against_itself_is_zero_error(rc_case):
    """Boundary condition: a perfect match must give zero normalized RMSE."""

    target = pd.read_csv(rc_case.base_dir / "target_rc.csv")
    metrics = measure_response(rc_case, _waveform_response(target))
    assert metrics["normalized_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rmse_V"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("sample", "expected_peak"), [(20.0, 20.0), (np.nan, None)])
def test_waveform_peak_limit_uses_complete_solver_observations(rc_case, monkeypatch, sample, expected_peak):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario

    def metric(_case, _params, response):
        response.voltage_V.fill(0.0)
        return {"loss": 0.0, "peak_abs_voltage_V": 0.0}

    monkeypatch.setattr(metrics_module, "get_metric", lambda *_args, **_kwargs: metric)
    response = SimulationResult(np.array([0.0, 1.0, 2.0]), np.array([0.0, sample, 0.0]))
    values = measure_response(rc_case, response)
    assert values["peak_abs_voltage_V"] == expected_peak

    request = EvaluationRequest(Candidate("point"), Scenario("condition"), ControlState())
    limit = MetricLimitConstraint("peak_abs_voltage_V", "max", 10.0)
    result = limit.evaluate(request, RawResult("ok"), MetricSet(values))
    assert not result.satisfied
    if expected_peak is None:
        assert "full waveform" in str(result.detail)


def test_terminal_vi_fit_uses_delivered_current_to_recover_complex_impedance(make_case):
    case = make_case(
        {
            "case_id": "terminal_fit",
            "source": {"type": "sine_voltage", "frequency_Hz": 1.0e6},
            "solver": {"ac": {"frequency_Hz": 1.0e6}},
            "target": {"objective": "terminal_vi_fit", "impedance_scale_ohm": 50.0},
        }
    )
    impedance = 20.0 - 30.0j
    delivered_current = 1.0 / impedance
    response = SimulationResult(
        time_s=np.asarray([], dtype=float),
        voltage_V=np.asarray([], dtype=float),
        status="ok",
        frequency_response=pd.DataFrame(
            {
                "frequency_Hz": [1.0e6],
                "voltage_re": [1.0],
                "voltage_im": [0.0],
                "current_re": [-delivered_current.real],
                "current_im": [-delivered_current.imag],
            }
        ),
    )

    metrics = measure_response(
        case,
        response,
        {
            "target_voltage_re_V": 1.0,
            "target_voltage_im_V": 0.0,
            "target_current_re_A": delivered_current.real,
            "target_current_im_A": delivered_current.imag,
        },
    )

    assert metrics["normalized_impedance_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["resistance_ohm"] == pytest.approx(20.0)
    assert metrics["reactance_ohm"] == pytest.approx(-30.0)


def test_terminal_vi_fit_selects_the_solved_observation_frequency(make_case):
    from pcd.core.models import MeasurementUnavailableError

    case = make_case(
        {
            "source": {"type": "sine_voltage", "frequency_Hz": 1e6},
            "solver": {"ac": {"sweep": "lin", "points": 3, "start_Hz": 1e6, "stop_Hz": 3e6}},
            "target": {
                "objective": "terminal_vi_fit",
                "impedance_scale_ohm": 1.0,
                "observation_frequency_parameter": "f_observed",
            },
        }
    )
    ac = pd.DataFrame(
        {
            "frequency_Hz": [1e6, 2e6, 3e6],
            "voltage_re": [1.0, 1.0, 1.0],
            "voltage_im": [0.0, 0.0, 0.0],
            "current_re": [-0.1, -0.05, -1 / 30],
            "current_im": [0.0, 0.0, 0.0],
        }
    )
    params = {
        "f_observed": 2e6,
        "target_voltage_re_V": 1.0,
        "target_voltage_im_V": 0.0,
        "target_current_re_A": 0.05,
        "target_current_im_A": 0.0,
    }
    empty = np.asarray([])
    measured = measure_response(case, SimulationResult(empty, empty, frequency_response=ac), params)
    assert measured["observed_frequency_Hz"] == measured["match_frequency_Hz"] == 2e6
    assert measured["loss"] == pytest.approx(0.0)

    missing = SimulationResult(empty, empty, frequency_response=ac.iloc[[0, 2]])
    with pytest.raises(MeasurementUnavailableError, match="requires a solved sample"):
        measure_response(case, missing, params)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "requires scenario value"),
        (
            {
                "target_voltage_re_V": "bad",
                "target_voltage_im_V": 0.0,
                "target_current_re_A": 1.0,
                "target_current_im_A": 0.0,
            },
            "must be numeric",
        ),
        (
            {
                "target_voltage_re_V": float("inf"),
                "target_voltage_im_V": 0.0,
                "target_current_re_A": 1.0,
                "target_current_im_A": 0.0,
            },
            "must be finite",
        ),
        (
            {
                "target_voltage_re_V": 1.0,
                "target_voltage_im_V": 0.0,
                "target_current_re_A": 0.0,
                "target_current_im_A": 0.0,
            },
            "must be nonzero",
        ),
        (
            {
                "target_voltage_re_V": 1.0e300,
                "target_voltage_im_V": 0.0,
                "target_current_re_A": 1.0e-20,
                "target_current_im_A": 0.0,
            },
            "target impedance must be finite",
        ),
    ],
)
def test_terminal_vi_fit_rejects_invalid_target_observations(values, message):
    from pcd.metrics import terminal_observation_impedance

    with pytest.raises(ValueError, match=message):
        terminal_observation_impedance(values)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ({}, "requires target.impedance_scale_ohm"),
        ({"impedance_scale_ohm": "bad"}, "must be numeric"),
        ({"impedance_scale_ohm": 0.0}, "must be positive and finite"),
        ({"impedance_scale_ohm": float("inf")}, "must be positive and finite"),
    ],
)
def test_terminal_vi_fit_requires_a_positive_finite_scale(make_case, target, message):
    from pcd.metrics import _target_impedance_scale

    case = make_case({"case_id": "invalid_terminal_scale", "target": target})
    with pytest.raises(ValueError, match=message):
        _target_impedance_scale(case)


def test_the_objective_ignores_waveform_row_order(rc_case):
    """Interpolation sorts by time, so a shuffled waveform must score the same."""

    frame = pd.read_csv(rc_case.base_dir / "target_rc.csv")[["time_s", "voltage_V"]]
    a = measure_response(rc_case, _waveform_response(frame))
    b = measure_response(rc_case, _waveform_response(frame.sample(frac=1.0, random_state=0)))
    assert a["loss"] == pytest.approx(b["loss"], rel=RTOL_LOSS)


def test_a_failed_response_is_not_measured_as_a_fake_large_loss(topology_case):
    response = SimulationResult(time_s=np.array([]), voltage_V=np.array([]), status="failed")
    with pytest.raises(ValueError, match="status 'failed'"):
        measure_response(topology_case, response)


# --- interpolation and non-finite data -------------------------------------


def test_an_empty_waveform_interpolates_to_nan_rather_than_crashing():
    target = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [0.0, 1.0]})
    _t, _vt, v = interpolate_to_target(target, pd.DataFrame(columns=["time_s", "voltage_V"]))
    assert v.shape == (2,)
    assert np.isnan(v).all()


def test_interpolation_marks_times_outside_the_observed_range_as_unavailable():
    target = pd.DataFrame({"time_s": [-1.0, 0.0, 1.0, 2.0], "voltage_V": [0.0, 0.0, 1.0, 1.0]})
    waveform = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [0.0, 1.0]})

    _time, _target, aligned = interpolate_to_target(target, waveform)

    assert np.isnan(aligned[[0, -1]]).all()
    assert aligned[1:3].tolist() == [0.0, 1.0]


def test_waveform_objective_rejects_partial_time_coverage(rc_case):
    target = pd.read_csv(rc_case.base_dir / "target_rc.csv")
    partial = target.iloc[1:-1][["time_s", "voltage_V"]]

    with pytest.raises(ValueError, match="does not cover"):
        measure_response(rc_case, _waveform_response(partial))


def test_non_finite_samples_are_dropped_before_interpolation():
    target = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [0.0, 1.0]})
    noisy = pd.DataFrame(
        {
            "time_s": [0.0, float("nan"), 0.5, float("inf"), 1.0],
            "voltage_V": [0.0, 5.0, 0.5, 5.0, 1.0],
        }
    )
    _t, _vt, v = interpolate_to_target(target, noisy)
    assert np.all(np.isfinite(v))
    assert v.max() <= 1.0 + 1e-12


def test_duplicate_timestamps_average_only_the_exact_target_tie():
    target = pd.DataFrame({"time_s": [0.0, 0.5, 1.0, 1.5, 2.0], "voltage_V": np.zeros(5)})
    waveform = pd.DataFrame({"time_s": [0.0, 1.0, 1.0, 2.0], "voltage_V": [0.0, 2.0, 4.0, 6.0]})
    _t, _vt, v = interpolate_to_target(target, waveform)
    assert v.tolist() == pytest.approx([0.0, 1.0, 3.0, 5.0, 6.0])


@pytest.mark.parametrize(("n_target", "n_source"), [(2, 2), (8, 3), (3, 8), (64, 17)])
def test_interpolation_preserves_shape_and_never_extrapolates(n_target, n_source):
    target = pd.DataFrame({"time_s": np.linspace(0.0, 1e-6, n_target), "voltage_V": np.zeros(n_target)})
    source_v = np.linspace(-5.0, 5.0, n_source)
    waveform = pd.DataFrame({"time_s": np.linspace(0.0, 1e-6, n_source), "voltage_V": source_v})

    t, vt, v = interpolate_to_target(target, waveform)
    assert t.shape == vt.shape == v.shape == (n_target,)
    assert v.dtype == np.float64
    assert source_v.min() - 1e-9 <= v.min()
    assert v.max() <= source_v.max() + 1e-9


# --- optimizers ------------------------------------------------------------


def test_proposing_candidates_never_touches_the_simulation_layer(tmp_path, topology_case):
    frame = _candidate_frame(topology_case, n=5, seed=1)
    assert len(frame) == 5
    assert {"topology_choice", "load_model"} <= set(frame.columns)
    assert list(tmp_path.rglob("debug/manifest.json")) == []


def test_the_same_seed_reproduces_the_same_candidates(topology_case):
    first = _candidate_frame(topology_case, n=8, seed=42)
    second = _candidate_frame(topology_case, n=8, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_explore_different_points(topology_case):
    first = _candidate_frame(topology_case, n=8, seed=1)
    second = _candidate_frame(topology_case, n=8, seed=2)
    assert not first["C1"].equals(second["C1"])


def test_sampled_candidates_respect_every_declared_bound(topology_case):
    from pcd.case import variable_specs

    candidates = _candidate_frame(topology_case, n=64, seed=7)
    for name, spec in variable_specs(topology_case).items():
        if "bounds" not in spec or name not in candidates.columns:
            continue
        lo, hi = float(spec["bounds"][0]), float(spec["bounds"][1])
        values = candidates[name].to_numpy(float)
        assert values.min() >= lo, f"{name} sampled below its lower bound"
        assert values.max() <= hi, f"{name} sampled above its upper bound"


# --- contracts the loop relies on ------------------------------------------


def test_the_base_optimizer_requires_an_ask_implementation(topology_case):
    from pcd.search import BaseOptimizer

    with pytest.raises(NotImplementedError):
        BaseOptimizer(case=topology_case).ask()


def test_optimizer_state_tracks_the_best_told_result(topology_case):
    from pcd.search import BaseOptimizer

    opt = BaseOptimizer(case=topology_case)
    opt.tell({"x": 1}, {"loss": 5.0})
    opt.tell({"x": 2}, {})  # no loss: must be skipped, not crash
    opt.tell({"x": 3}, {"loss": 2.0})

    state = opt.state()
    assert state["n_observations"] == 3
    assert state["best"]["params"] == {"x": 3}


def test_an_optimizer_with_no_results_has_no_best(topology_case):
    from pcd.search import BaseOptimizer

    assert BaseOptimizer(case=topology_case).state()["best"] is None


def test_a_metric_may_return_a_named_objective_instead_of_loss(tmp_path, rc_case):
    from pcd.api import execute_case
    from pcd.metric_registry import register

    @register("missing_loss")
    def missing_loss(case, params, response):
        del case, params, response
        return {"something_else": 1.0}

    run = execute_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    case = rc_case
    case.data["target"]["objective"] = "missing_loss"
    assert measure_response(case, run.response, run.record.params)["something_else"] == 1.0


def test_rf_load_is_a_named_study_metric_with_exact_port_current(make_case):
    frequency_hz = 1.0e6
    time_s = np.linspace(0.0, 8.0 / frequency_hz, 801)
    phase = 2.0 * np.pi * frequency_hz * time_s
    response = SimulationResult(
        time_s=time_s,
        voltage_V=10.0 * np.sin(phase),
        # Source current follows ngspice's into-source sign convention.
        current_A=-2.0 * np.sin(phase),
        probes={"source_voltage_V": 10.0 * np.sin(phase), "load_current_A": 2.0 * np.sin(phase)},
    )
    case = make_case(
        {
            "case_id": "rf_load",
            "source": {"frequency_Hz": frequency_hz},
            "load": {"name": "resistor", "R_ohm": 5},
            "measurement": {"load_current": "load_current_A"},
            "target": {"objective": "rf_load"},
        }
    )

    metrics = measure_response(case, response)

    assert "loss" not in metrics
    assert metrics["load_real_power_W"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["load_i_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["periodic_settled"] is True
    assert metrics["voltage_harmonic_amplitude_V"]["h1"] == pytest.approx(10.0, rel=2e-2)


def test_a_non_finite_loss_is_a_failed_measurement_not_a_store_error(tmp_path, rc_case):
    from pcd.api import execute_case
    from pcd.metric_registry import register

    @register("nonfinite_loss")
    def nonfinite_loss(case, params, response):
        del case, params, response
        return {"loss": float("inf")}

    run = execute_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    rc_case.data["target"]["objective"] = "nonfinite_loss"
    with pytest.raises(ValueError, match="non-finite loss"):
        measure_response(rc_case, run.response, run.record.params)


@pytest.mark.parametrize(("source_voltage", "expected_loss"), [(1.0, 1.0), (0.0, None)])
def test_open_port_distinguishes_total_reflection_from_zero_excitation(make_case, source_voltage, expected_loss):
    case = make_case(
        {
            "case_id": "open_match",
            "source": {"frequency_Hz": 13.56e6},
            "measurement": {"reference_impedance_ohm": 50.0},
            "target": {"objective": "impedance_match"},
        }
    )
    response = SimulationResult(
        time_s=np.asarray([]),
        voltage_V=np.asarray([]),
        frequency_response=pd.DataFrame(
            {
                "frequency_Hz": [13.56e6],
                "voltage_re": [source_voltage],
                "voltage_im": [0.0],
                "current_re": [0.0],
                "current_im": [0.0],
            }
        ),
    )
    metrics = measure_response(case, response)

    assert metrics["loss"] == expected_loss
    if expected_loss is None:
        assert metrics["loss_unavailable_reason"] == metrics["reflection_magnitude_unavailable_reason"]
    assert metrics["vswr"] is None
    assert metrics["resistance_ohm"] is None

    from pcd.core.models import MeasurementUnavailableError
    from pcd.metrics import terminal_vi_fit

    case.data["target"] = {"objective": "terminal_vi_fit", "impedance_scale_ohm": 1}
    targets = {"target_voltage_re_V": 1, "target_voltage_im_V": 0, "target_current_re_A": 0.1, "target_current_im_A": 0}
    with pytest.raises(MeasurementUnavailableError, match="open or undefined"):
        terminal_vi_fit(case, targets, response)


def test_a_waveform_objective_needs_a_target_file(make_case):
    from pcd.metrics import load_target_waveform

    with pytest.raises(ValueError, match=r"target.waveform_file is required"):
        load_target_waveform(make_case({"case_id": "x", "target": {"objective": "waveform_l2"}}))


def test_a_target_waveform_must_have_the_canonical_columns(tmp_path, make_case):
    from pcd.metrics import load_target_waveform

    (tmp_path / "bad_target.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    case = make_case({"case_id": "x", "target": {"waveform_file": "bad_target.csv"}})
    with pytest.raises(ValueError, match="missing columns"):
        load_target_waveform(case)


def test_the_voltage_constraint_marks_only_an_overshoot_infeasible(make_case):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario
    from pcd.metrics import constraints_from_case

    case = make_case(
        {
            "case_id": "c",
            "target": {"constraints": {"metric_bounds": {"peak_abs_voltage_V": {"max": 100.0}}}},
        }
    )
    constraint = constraints_from_case(case)[0]
    request = EvaluationRequest(Candidate("c"), Scenario("s"), ControlState())
    raw = RawResult("ok")
    within = constraint.evaluate(request, raw, MetricSet({"peak_abs_voltage_V": 90.0}))
    assert within.satisfied
    assert within.violation == 0.0
    assert within.margin == pytest.approx(10.0)

    over = constraint.evaluate(request, raw, MetricSet({"peak_abs_voltage_V": 200.0}))
    assert not over.satisfied
    assert over.violation == pytest.approx(1.0)
    assert over.margin == pytest.approx(-100.0)


def test_no_declared_constraint_means_no_constraint(make_case):
    from pcd.metrics import constraints_from_case

    assert constraints_from_case(make_case({"case_id": "c"})) == ()


def test_metric_bounds_define_engineering_feasibility_without_changing_the_objective(make_case):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario
    from pcd.metrics import constraints_from_case

    case = make_case(
        {
            "case_id": "bounded",
            "target": {
                "constraints": {
                    "metric_bounds": {
                        "reflection_magnitude": {"max": 0.316227766},
                        "delivered_power_W": {"min": 100.0},
                    }
                }
            },
        }
    )
    request = EvaluationRequest(Candidate("c"), Scenario("s"), ControlState())
    constraints = constraints_from_case(case)

    passing = MetricSet({"reflection_magnitude": 0.1, "delivered_power_W": 120.0})
    assert all(item.evaluate(request, RawResult("ok"), passing).satisfied for item in constraints)

    failing = MetricSet({"reflection_magnitude": 0.5, "delivered_power_W": 80.0})
    results = {item.name: item.evaluate(request, RawResult("ok"), failing) for item in constraints}
    assert results["max_reflection_magnitude"].violation > 0.0
    assert results["max_reflection_magnitude"].margin == pytest.approx(0.316227766 - 0.5)
    assert results["min_delivered_power_W"].violation == pytest.approx(0.2)
    assert results["min_delivered_power_W"].margin == pytest.approx(-20.0)


def test_a_missing_bounded_metric_is_infeasible(make_case):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario
    from pcd.metrics import constraints_from_case

    case = make_case(
        {
            "case_id": "missing_metric",
            "target": {"constraints": {"metric_bounds": {"reflection_magnitude": {"max": 0.316}}}},
        }
    )
    request = EvaluationRequest(Candidate("c"), Scenario("s"), ControlState())
    result = constraints_from_case(case)[0].evaluate(request, RawResult("ok"), MetricSet())
    assert not result.satisfied
    assert "unavailable" in str(result.detail)


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ([], "target.constraints must be a mapping"),
        ({"metric_bounds": []}, "metric_bounds must be a mapping"),
        ({"metric_bounds": {"x": 1}}, "bound for 'x' must be a mapping"),
        ({"metric_bounds": {"x": {}}}, "must declare min or max"),
    ],
)
def test_malformed_metric_bounds_are_rejected_at_study_construction(make_case, constraints, message):
    from pcd.metrics import constraints_from_case

    case = make_case({"case_id": "bad_bound", "target": {"constraints": constraints}})
    with pytest.raises(ValueError, match=message):
        constraints_from_case(case)


def test_metric_limit_definition_and_values_must_be_numeric(make_case):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario
    from pcd.metrics import MetricLimitConstraint

    with pytest.raises(ValueError, match="bound"):
        MetricLimitConstraint("x", "equal", 1.0)
    with pytest.raises(ValueError, match="metric name"):
        MetricLimitConstraint(" ", "max", float("inf"))

    request = EvaluationRequest(Candidate("c"), Scenario("s"), ControlState())
    result = MetricLimitConstraint("x", "max", 1.0).evaluate(
        request,
        RawResult("ok"),
        MetricSet({"x": "not-a-number"}),
    )
    assert not result.satisfied


# --- exact finite search ----------------------------------------------------


def test_grid_optimizer_enumerates_each_discrete_candidate_once(make_case):
    case = make_case(
        {
            "case_id": "finite_parts",
            "variables": {
                "fixed": {"choices": [10.0], "default": 10.0},
                "C1": {"choices": [1.0, 2.0], "default": 1.0},
                "L1": {"choices": [3.0, 4.0], "default": 3.0},
            },
        }
    )
    optimizer = create_optimizer(case, optimizer_name="grid")

    candidates = [optimizer.ask() for _ in range(4)]

    assert candidates == [
        {"C1": 1.0, "L1": 3.0},
        {"C1": 1.0, "L1": 4.0},
        {"C1": 2.0, "L1": 3.0},
        {"C1": 2.0, "L1": 4.0},
    ]
    with pytest.raises(RuntimeError, match="exhausted its 4"):
        optimizer.ask()


def test_grid_optimizer_rejects_a_continuous_axis(make_case):
    case = make_case({"case_id": "continuous", "variables": {"C1": {"bounds": [1.0, 2.0]}}})
    with pytest.raises(ValueError, match="requires finite choices"):
        create_optimizer(case, optimizer_name="grid")


def test_grid_optimizer_reports_completion_and_best_result(make_case):
    case = make_case({"case_id": "finite", "variables": {"C1": {"choices": [1.0, 2.0]}}})
    optimizer = create_optimizer(case, optimizer_name="grid")
    losses = iter((2.0, 1.0))

    def evaluate(params):
        feedback = {"loss": next(losses)}
        optimizer.tell(params, feedback)
        return feedback

    optimizer.optimize(evaluate, 2)

    assert optimizer.outcome == {"termination": "declared_grid_completed"}
    state = optimizer.state()
    assert state["complete"] is True
    assert state["n_points"] == 2
    assert state["best"]["params"] == {"C1": 2.0}


# --- bounded continuous search --------------------------------------------


def _evolution_trace(case, seed: int, trials: int) -> list[tuple[dict, dict]]:
    optimizer = create_optimizer(case, optimizer_name="differential_evolution", seed=seed)
    trace = []
    for _index in range(trials):
        params = optimizer.ask()
        loss = (float(params["x"]) - 1.25) ** 2 + (np.log10(float(params["gain"])) - 1.0) ** 2
        optimizer.tell(params, {"rank": (0.0, 0.0, 0.0, 0.0, loss), "loss": loss})
        trace.append((params, optimizer.proposal_metadata()))
    return trace


def test_differential_evolution_is_seeded_bounded_and_traceable(make_case):
    case = make_case(
        {
            "case_id": "continuous",
            "run": {"trials": 16},
            "variables": {
                "x": {"bounds": [-5.0, 5.0], "default": 2.0},
                "gain": {"bounds": [1.0, 100.0], "scale": "log", "default": 10.0},
                "fixed": {"choices": ["pi"], "default": "pi"},
            },
        }
    )

    first = _evolution_trace(case, seed=17, trials=16)
    second = _evolution_trace(case, seed=17, trials=16)

    assert first == second
    assert first[0][0] == {"x": 2.0, "gain": 10.0}
    assert all(-5.0 <= row[0]["x"] <= 5.0 and 1.0 <= row[0]["gain"] <= 100.0 for row in first)
    assert first[0][1] == {"phase": "initialization", "generation": 0, "population_index": 0, "accepted": True}
    assert first[8][1]["phase"] == "evolution"
    assert first[8][1]["strategy"] == "DE/rand/1/bin"
    assert len(first[8][1]["parents"]) == 3


@pytest.mark.parametrize(("bounds", "default"), [([5e-10, 1e-7], 5e-10), ([1e-10, 2e-10], 2e-10)])
def test_differential_evolution_preserves_log_axis_boundary_defaults(make_case, bounds, default):
    case = make_case(
        {
            "case_id": "log_boundary",
            "run": {"trials": 8},
            "variables": {"C": {"bounds": bounds, "scale": "log", "default": default}},
        }
    )
    optimizer = create_optimizer(case, optimizer_name="differential_evolution")

    assert validate_proposal(optimizer.case, optimizer.ask()) == {"C": default}


@pytest.mark.parametrize("optimum", [1.0, 8.0])
def test_least_squares_can_move_interior_from_a_declared_bound(make_case, optimum):
    case = make_case(
        {
            "case_id": "terminal_boundary",
            "variables": {"R": {"role": "latent", "bounds": [1.0, 10.0], "default": 1.0}},
            "target": {"objective": "terminal_vi_fit"},
            "study": {
                "candidate_role": "latent",
                "objectives": [{"metric": "normalized_impedance_error", "aggregation": "rms"}],
            },
            "optimizer": {"name": "least_squares"},
        }
    )
    optimizer = create_optimizer(case)
    proposed = []

    def evaluate(params):
        proposed.append(params["R"])
        return {"residuals": [(params["R"] - optimum) / 100.0]}

    optimizer.optimize(evaluate, 32)

    assert optimizer.outcome["termination"] == "converged"
    assert proposed[0] == 1.0
    assert min(proposed, key=lambda value: abs(value - optimum)) == pytest.approx(optimum, abs=1e-6)


def test_differential_evolution_does_not_spend_trials_on_duplicate_points(make_case):
    case = make_case(
        {
            "case_id": "unique_continuous",
            "run": {"trials": 24},
            "variables": {"c": {"bounds": [1e-10, 1e-7], "scale": "log", "default": 1e-9}},
        }
    )
    optimizer = create_optimizer(case, optimizer_name="differential_evolution", seed=3)
    proposals = []
    metadata = []
    for _index in range(24):
        params = optimizer.ask()
        value = float(params["c"])
        proposals.append(value)
        feasible = value >= 5.1e-10
        rank = (
            0.0 if feasible else 1.0,
            0.0,
            0.0,
            max(0.0, 5.1e-10 - value) / 5.1e-10,
            abs(np.log(value / 3e-10)),
        )
        optimizer.tell(params, {"rank": rank})
        metadata.append(optimizer.proposal_metadata())

    assert len(set(proposals)) == len(proposals)
    assert any(row.get("duplicate_fallback") == "uniform_resample" for row in metadata)


def test_differential_evolution_uses_feasibility_first_replacement(make_case):
    case = make_case(
        {
            "case_id": "constrained",
            "run": {"trials": 8},
            "variables": {"x": {"bounds": [-2.0, 2.0], "default": 0.0}},
        }
    )
    optimizer = create_optimizer(case, optimizer_name="differential_evolution", seed=3)
    for index in range(4):
        params = optimizer.ask()
        rank = (1.0, 0.0, 1.0, 1.0, 0.0) if index == 0 else (0.0, 0.0, 0.0, 0.0, 10.0)
        optimizer.tell(params, {"rank": rank})

    child = optimizer.ask()
    optimizer.tell(child, {"rank": (0.0, 0.0, 0.0, 0.0, 1000.0)})

    assert optimizer.proposal_metadata()["population_index"] == 0
    assert optimizer.proposal_metadata()["accepted"] is True


def test_differential_evolution_rejects_incomplete_or_discrete_spaces(make_case):
    too_small = make_case({"case_id": "small", "run": {"trials": 7}, "variables": {"x": {"bounds": [0.0, 1.0]}}})
    with pytest.raises(ValueError, match="need at least 8"):
        create_optimizer(too_small, optimizer_name="differential_evolution")

    discrete = make_case(
        {
            "case_id": "discrete",
            "run": {"trials": 8},
            "variables": {"x": {"bounds": [0.0, 1.0]}, "topology": {"choices": ["l", "pi"]}},
        }
    )
    with pytest.raises(ValueError, match="declares 2 choices"):
        create_optimizer(discrete, optimizer_name="differential_evolution")
