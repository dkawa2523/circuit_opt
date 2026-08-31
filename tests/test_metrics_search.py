"""Circuit metrics and candidate-search behaviour.

This layer only ever reads saved waveforms, so it must be robust to non-finite
and out-of-order data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.metrics import interpolate_to_target, measure_record
from pcd.search import create_optimizer

RTOL_LOSS = 1e-12


def _saved_waveform_record(tmp_path: Path, frame: pd.DataFrame, name: str = "run") -> dict:
    """Persist the same artifact shape produced by simulation, without a fake importer API."""

    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    frame.to_csv(run_dir / "waveform.csv", index=False)
    record = {
        "schema": "simulation_record.v2",
        "run_dir": str(run_dir),
        "status": "ok",
        "params": {},
        "artifacts": {"waveform": "waveform.csv"},
    }
    (run_dir / "sim_manifest.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _candidate_frame(case, n: int, seed: int) -> pd.DataFrame:
    optimizer = create_optimizer(case, optimizer_name="random", seed=seed)
    return pd.DataFrame([optimizer.ask() for _ in range(n)])


# --- objectives ------------------------------------------------------------


def test_scoring_a_target_against_itself_is_zero_error(tmp_path, rc_case):
    """Boundary condition: a perfect match must give zero normalized RMSE."""

    target = pd.read_csv(rc_case.base_dir / "target_rc.csv")
    record = _saved_waveform_record(tmp_path, target[["time_s", "voltage_V"]])
    metrics = measure_record(rc_case, record)
    assert metrics["normalized_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rmse_V"] == pytest.approx(0.0, abs=1e-12)


def test_the_objective_ignores_waveform_row_order(tmp_path, rc_case):
    """Interpolation sorts by time, so a shuffled waveform must score the same."""

    frame = pd.read_csv(rc_case.base_dir / "target_rc.csv")[["time_s", "voltage_V"]]
    a = measure_record(rc_case, _saved_waveform_record(tmp_path, frame, "ordered"))
    b = measure_record(
        rc_case,
        _saved_waveform_record(tmp_path, frame.sample(frac=1.0, random_state=0), "shuffled"),
    )
    assert a["loss"] == pytest.approx(b["loss"], rel=RTOL_LOSS)


def test_a_failed_record_is_not_measured_as_a_fake_large_loss(tmp_path, topology_case):
    (tmp_path / "sim_manifest.json").write_text(
        json.dumps({"run_dir": str(tmp_path), "status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="status 'failed'"):
        measure_record(topology_case, tmp_path)


# --- interpolation and non-finite data -------------------------------------


def test_an_empty_waveform_interpolates_to_nan_rather_than_crashing():
    target = pd.DataFrame({"time_s": [0.0, 1.0], "voltage_V": [0.0, 1.0]})
    _t, _vt, v = interpolate_to_target(target, pd.DataFrame(columns=["time_s", "voltage_V"]))
    assert v.shape == (2,)
    assert np.isnan(v).all()


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


def test_duplicate_timestamps_are_averaged():
    target = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "voltage_V": [0.0, 1.0, 2.0]})
    waveform = pd.DataFrame({"time_s": [0.0, 1.0, 1.0, 2.0], "voltage_V": [0.0, 2.0, 4.0, 6.0]})
    _t, _vt, v = interpolate_to_target(target, waveform)
    assert v[1] == pytest.approx(3.0)


@pytest.mark.property
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
    assert list(tmp_path.rglob("sim_manifest.json")) == []


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
    from pcd.metric_registry import register
    from pcd.sim_core import simulate_case

    @register("missing_loss")
    def missing_loss(case, record, waveform):
        return {"something_else": 1.0}

    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    case = rc_case
    case.data["target"]["objective"] = "missing_loss"
    assert measure_record(case, rec.manifest())["something_else"] == 1.0


def test_rf_load_is_a_named_study_metric_with_exact_port_current(tmp_path, make_case):
    frequency_hz = 1.0e6
    time_s = np.linspace(0.0, 8.0 / frequency_hz, 801)
    phase = 2.0 * np.pi * frequency_hz * time_s
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "voltage_V": 10.0 * np.sin(phase),
            # Source current follows ngspice's into-source sign convention.
            "current_A": -2.0 * np.sin(phase),
            "source_voltage_V": 10.0 * np.sin(phase),
            "load_current_A": 2.0 * np.sin(phase),
        }
    )
    case = make_case(
        {
            "case_id": "rf_load",
            "source": {"frequency_Hz": frequency_hz},
            "measurement": {"load_current": "load_current_A"},
            "target": {"objective": "rf_load"},
        }
    )

    record = _saved_waveform_record(tmp_path, frame)
    metrics = measure_record(case, record)

    assert "loss" not in metrics
    assert metrics["load_real_power_W"] == pytest.approx(10.0, rel=1e-3)
    assert metrics["load_i_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["periodic_settled"] is True
    assert metrics["voltage_harmonic_amplitude_V"]["h1"] == pytest.approx(10.0, rel=2e-2)


def test_a_non_finite_loss_is_a_failed_measurement_not_a_store_error(tmp_path, rc_case):
    from pcd.metric_registry import register
    from pcd.sim_core import simulate_case

    @register("nonfinite_loss")
    def nonfinite_loss(case, record, waveform):
        return {"loss": float("inf")}

    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    rc_case.data["target"]["objective"] = "nonfinite_loss"
    with pytest.raises(ValueError, match="non-finite loss"):
        measure_record(rc_case, rec.manifest())


def test_total_reflection_keeps_a_finite_loss_and_null_unbounded_values(tmp_path, make_case):
    from pcd.metrics import impedance_match

    np.savetxt(tmp_path / "ac.csv", [[13.56e6, 1.0, 0.0, 13.56e6, 0.0, 0.0]])
    case = make_case(
        {
            "case_id": "open_match",
            "source": {"frequency_Hz": 13.56e6},
            "measurement": {"reference_impedance_ohm": 50.0},
            "target": {"objective": "impedance_match"},
        }
    )
    record = {"run_dir": str(tmp_path), "artifacts": {"frequency_response": "ac.csv"}, "params": {}}
    metrics = impedance_match(case, record, pd.DataFrame())

    assert metrics["loss"] == pytest.approx(1.0)
    assert metrics["vswr"] is None
    assert metrics["resistance_ohm"] is None


def test_a_waveform_objective_needs_a_target_file(make_case):
    from pcd.metrics import load_target_waveform

    with pytest.raises(ValueError, match=r"target.waveform_file is required"):
        load_target_waveform(make_case({"case_id": "x", "target": {"objective": "waveform_l2"}}))


def test_a_target_waveform_must_have_the_canonical_columns(tmp_path, make_case):
    from pcd.metrics import load_target_waveform

    (tmp_path / "bad_target.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    case = make_case({"case_id": "x", "target": {"waveform_file": "bad_target.csv"}})
    with pytest.raises(ValueError, match="must contain time_s and voltage_V"):
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

    over = constraint.evaluate(request, raw, MetricSet({"peak_abs_voltage_V": 200.0}))
    assert not over.satisfied
    assert over.violation == pytest.approx(1.0)


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
    assert results["min_delivered_power_W"].violation == pytest.approx(0.2)


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
        {"fixed": 10.0, "C1": 1.0, "L1": 3.0},
        {"fixed": 10.0, "C1": 1.0, "L1": 4.0},
        {"fixed": 10.0, "C1": 2.0, "L1": 3.0},
        {"fixed": 10.0, "C1": 2.0, "L1": 4.0},
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
    for loss in (2.0, 1.0):
        params = optimizer.ask()
        optimizer.tell(params, {"loss": loss})

    state = optimizer.state()
    assert state["complete"] is True
    assert state["n_points"] == 2
    assert state["best"]["params"] == {"C1": 2.0}
