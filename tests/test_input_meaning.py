"""Specified conditions, limits, units, and measurements keep their meaning."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pcd.case import Case, load_case
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
from pcd.core.aggregation import aggregate_candidate, candidate_rank_key, control_margin
from pcd.results import best_decision_summary
from pcd.simulation import AcSweep, AnalysisRequest, SimulationResult, TransientAnalysis
from pcd.study_config import CaseControlPolicy, study_spec_from_case
from pcd.validation import validate_case


@pytest.mark.parametrize(
    ("suffix", "text"),
    [
        ("yaml", "schema: case_yaml.v1\nsolver: {tran: {stop_s: 1, stop_s: 2}}\n"),
        ("json", '{"schema":"case_yaml.v1","solver":{"tran":{"stop_s":1,"stop_s":2}}}'),
    ],
)
def test_duplicate_authored_fields_cannot_replace_a_requested_value(tmp_path, suffix, text):
    path = tmp_path / f"case.{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate input field 'stop_s'"):
        load_case(path)


def test_top_level_typo_cannot_replace_an_explicit_ac_request_with_default_transient(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("schema: case_yaml.v1\nslver: {ac: {frequency_Hz: 1000000}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="slver"):
        load_case(path)


def test_yaml_anchor_defaults_keep_their_explicit_override(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        "schema: case_yaml.v1\nextensions: {base: &base {stop_s: 1}}\nsolver: {tran: {<<: *base, stop_s: 2}}\n",
        encoding="utf-8",
    )
    assert load_case(path).data["solver"]["tran"]["stop_s"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"solver": {"ac": {"frequency": 1e6}}},
        {"solver": {"tran": {"stop": 1e-6}}},
        {"measurement": {"reference_impedance": 100}},
        {"target": {"constraints": {"max_voltage_V": 10}}},
        {"target": {"constraints": {"metric_bounds": {"peak_abs_voltage_V": {"max": 10, "minimum": 1}}}}},
        {"study": {"objectives": [{"metric": "loss", "reduce": "mean"}]}},
        {"study": {"controls": {"by_scenario": {"misspelled": {"C": 1}}}}},
        {"circuit": {"builder": "$missing"}},
        {"run": {"trials": 1.5}},
        {"optimizer": {"name": "random", "seed": 1.5}},
        {"optimizer": {"name": "random", "population_size": 20}},
        {"variables": {"R": {"bounds": [1, 10], "scale": "Log"}}},
        {"variables": {"R": {"default": 10, "bound": [1, 100]}}},
        {"study": {"design_variables": {"R": {"default": 10, "bound": [1, 100]}}}},
        {"study": {"controls": {"variables": {"R": {"default": 10, "bound": [1, 100]}}}}},
        {"variables": {"R": {"choices": [1, 2], "values": [3, 4]}}},
        {"variables": {"R": {"bounds": [1, 10], "grid": 5}}},
        {"study": {"design_variables": {"R": {"role": "latent", "default": 10}}}},
        {"study": {"controls": {"variables": {"R": {"role": "design", "choices": [1, 2]}}}}},
        {"measurement": {"source_voltage": "i(Vsrc)"}},
        {"measurement": {"source_voltage": "v(different_port)"}},
    ],
)
def test_unused_owned_settings_are_invalid_before_solving(make_case, change):
    case = make_case({"source": {"type": "dc_voltage", "voltage_V": 1}, **change})
    assert not validate_case(case).ok


@pytest.mark.parametrize(
    "binding", ["source_voltage_V", "voltage_V", "current_A", "missing", "voltage_named_current_A"]
)
def test_load_current_binding_requires_an_actual_declared_current_vector(make_case, binding):
    from pcd.simulation_input import resolve_simulation_case

    case = make_case(
        {
            "source": {"type": "dc_voltage", "voltage_V": 1},
            "measurement": {"load_current": binding, "probes": {"voltage_named_current_A": "v(src)"}},
        }
    )
    assert not validate_case(case).ok
    with pytest.raises(ValueError, match="declared current probe"):
        resolve_simulation_case(case)


def test_named_load_probe_keeps_its_declared_vector_and_polarity(make_case):
    from pcd.simulation_input import resolve_simulation_case

    case = make_case(
        {
            "source": {"type": "dc_voltage", "voltage_V": 1},
            "measurement": {"load_current": "into_load", "probes": {"into_load": "i(Vmeter)"}},
        }
    )
    probes = resolve_simulation_case(case).probes
    assert probes.load_current_column == "into_load"
    assert probes.ac[0].vector == "i(Vmeter)"


def test_condition_controls_pin_search_axes_without_fabricating_headroom(make_case):
    case = make_case(
        {
            "study": {
                "scenarios": [{"id": "A", "controls": {"C": 1}}],
                "controls": {
                    "defaults": {"bias": 5},
                    "variables": {"C": {"choices": [1, 2, 3]}, "L": {"values": [4, 6]}},
                },
            }
        }
    )
    study = study_spec_from_case(case)
    controls = CaseControlPolicy(case).controls(study, Candidate("design"), study.scenarios[0])
    assert [dict(item.values) for item in controls] == [{"bias": 5, "L": 4, "C": 1}, {"bias": 5, "L": 6, "C": 1}]
    assert control_margin(controls[0], controls) == 0


def test_scenario_table_defaults_and_requested_weight_column_are_effective(make_case, tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("scenario_id,R,confidence\nA,10,3\n", encoding="utf-8")
    case = make_case(
        {
            "study": {
                "scenario_table": {
                    "table_file": path.name,
                    "values": {"R": "R"},
                    "defaults": {"T": 300},
                    "weight_column": "confidence",
                }
            }
        }
    )
    scenario = study_spec_from_case(case).scenarios[0]
    assert dict(scenario.values) == {"R": 10, "T": 300}
    assert scenario.weight == 3
    data = deepcopy(case.data)
    data["study"]["scenario_table"]["weight_column"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        study_spec_from_case(Case(case.path, data))


@pytest.mark.parametrize("status", ["failed", "ok"])
def test_tiny_condition_weights_cannot_turn_missing_or_failed_limits_into_acceptance(status):
    scenarios = (Scenario("normal"), Scenario("rare", weight=1e-30))
    study = StudySpec("limits", scenarios, (Objective("score"),))
    rows = []
    for index, scenario in enumerate(scenarios):
        evaluation = EvaluationResult(
            EvaluationRequest(Candidate("bad"), scenario, ControlState()),
            RawResult(status if index else "ok"),
            MetricSet({"score": 0.0}),
            (ConstraintResult("limit", not index, float(index)),),
        )
        rows.append(ScenarioResult(scenario, evaluation, (evaluation,)))
    bad = aggregate_candidate(study, Candidate("bad"), tuple(rows))
    assert bad.feasible_fraction == 1.0  # Statistical rounding is not a logical decision.
    assert best_decision_summary(study, bad)["status"] != "meets_declared_acceptance"
    good_rows = []
    for scenario in scenarios:
        value = EvaluationResult(
            EvaluationRequest(Candidate("good"), scenario, ControlState()), RawResult("ok"), MetricSet({"score": 100.0})
        )
        good_rows.append(ScenarioResult(scenario, value, (value,)))
    good = aggregate_candidate(study, Candidate("good"), tuple(good_rows))
    assert candidate_rank_key(study, good) < candidate_rank_key(study, bad)


def test_successful_solver_results_must_cover_the_requested_numerical_domain():
    transient = SimulationResult(np.array([0.0, 0.5]), np.ones(2))
    with pytest.raises(ValueError, match="requested stop_s"):
        transient.validate_for(AnalysisRequest(TransientAnalysis(1.0, 1.0), None))
    import pandas as pd

    ac = SimulationResult(
        np.array([]),
        np.array([]),
        frequency_response=pd.DataFrame(
            {"frequency_Hz": [1.0], "voltage_re": [1.0], "voltage_im": [0.0], "current_re": [-1.0], "current_im": [0.0]}
        ),
    )
    with pytest.raises(ValueError, match="requested sweep"):
        ac.validate_for(AnalysisRequest(None, AcSweep("lin", 3, 1.0, 3.0)))


def test_missing_solver_is_a_saved_unavailable_fit_without_fabricated_estimates(tmp_path):
    from pcd.identification import run_identification

    example = Path(__file__).resolve().parents[1] / "examples/terminal_identification.yaml"
    original = load_case(example)
    data = deepcopy(original.data)
    data["solver"]["executable"] = str(tmp_path / "missing_ngspice")
    result = run_identification(Case(example, data), tmp_path / "runs")
    assert result["status"] == "not_identified"
    assert result["estimated_latent"] == {}
    assert result["fit"]["value"] is None
    assert result["holdout"]["status"] == "not_run"
    fit = json.loads((Path(result["run_root"]) / result["artifacts"]["fit_result"]).read_text(encoding="utf-8"))
    assert fit["execution"]["optimizer_outcome"]["termination"] == "residual_unavailable"
    assert fit["n_candidates"] == 1


NGSPICE = shutil.which("ngspice_con.exe") or shutil.which("ngspice")


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_display_node_and_ac_marker_do_not_change_transient_load_measurements(make_case, tmp_path):
    case = make_case(
        {
            "case_id": "separate_observations",
            "source": {"type": "sine_voltage", "amplitude_V": 2, "frequency_Hz": 1e6},
            "circuit": {"output_node": "out", "components": [{"ref": "R1", "n1": "src", "n2": "out", "value": 50}]},
            "load": {"name": "resistor", "R_ohm": 50, "ports": {"p": "out", "n": "0"}},
            "measurement": {"voltage_node": "src", "load_current": "auto"},
            "solver": {
                "tran": {"step_s": 1e-8, "stop_s": 8e-6},
                "ac": {"sweep": "lin", "points": 2, "start_Hz": 1e6, "stop_Hz": 2e6},
            },
        }
    )
    from pcd.reporting.run import analyze_run
    from pcd.sim_core import simulate_case

    record = simulate_case(case, run_root=tmp_path / "runs")
    assert record.status == "ok"
    summary = analyze_run(record.run_dir, frequency_hz=2e6)
    assert summary["measurement"]["ac_frequency_Hz"] == 2e6
    assert summary["measurement"]["periodic_fundamental_Hz"] == 1e6
    transient = summary["analyses"]["transient"]
    assert transient["waveform_voltage_peak_V"] == pytest.approx(2, rel=1e-3)
    assert transient["periodic"]["load_v_peak_V"] == pytest.approx(1, rel=1e-3)
    assert transient["periodic"]["load_real_power_W"] == pytest.approx(0.01, rel=1e-3)
    assert summary["measurement"]["impedance_reference_plane"] == "source_port"
    assert summary["analyses"]["ac"]["resistance_ohm"] == pytest.approx(100)


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_least_squares_minimizes_the_declared_weighted_rms_error(make_case, tmp_path):
    observations = []
    for name, resistance, weight in [("a", 10, 1), ("b", 30, 3), ("holdout", 25, 1)]:
        observations.append(
            {
                "id": name,
                "weight": weight,
                "values": {
                    "target_voltage_re_V": 1,
                    "target_voltage_im_V": 0,
                    "target_current_re_A": 1 / resistance,
                    "target_current_im_A": 0,
                },
            }
        )
    case = make_case(
        {
            "case_id": "weighted_identification",
            "variables": {"R": {"role": "latent", "default": 20, "bounds": [1, 50]}},
            "source": {"type": "sine_voltage", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"output_node": "src", "components": [{"ref": "R1", "n1": "src", "n2": "0", "value": "R"}]},
            "solver": {"ac": {"frequency_Hz": 1e6}},
            "target": {"objective": "terminal_vi_fit", "impedance_scale_ohm": 1},
            "run": {"trials": 32},
            "optimizer": {"name": "least_squares"},
            "study": {
                "scenarios": observations,
                "objectives": [{"metric": "normalized_impedance_error", "aggregation": "rms"}],
            },
            "identification": {
                "holdout_mode": "replicates",
                "fit_scenarios": ["a", "b"],
                "holdout_scenarios": ["holdout"],
                "observed_metrics": ["resistance_ohm"],
                "metric_scales": {"resistance_ohm": 1},
                "fit_loss_max": 10,
                "holdout_loss_max": 1e-5,
            },
        }
    )
    command = [sys.executable, "-m", "pcd", "identify", str(case.path), "--output", str(tmp_path / "runs"), "--json"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["estimated_latent"]["R"] == pytest.approx(25, abs=1e-5)
    assert result["fit"]["value"] == pytest.approx(np.sqrt(75), rel=1e-6)


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_current_source_keeps_its_units_and_uses_ac_current_in_ac_measurements(make_case, tmp_path):
    from pcd.analysis.measurements import ac_operating_metrics
    from pcd.sim_core import execute_case

    case = make_case(
        {
            "source": {"type": "current_dc", "current_A": 0.001, "ac_magnitude_A": 1},
            "circuit": {"output_node": "src"},
            "load": {"name": "resistor", "R_ohm": 100, "ports": {"p": "src", "n": "0"}},
            "measurement": {"load_current": "auto"},
            "solver": {"tran": {"step_s": 1e-7, "stop_s": 1e-6}, "ac": {"frequency_Hz": 1e6}},
        }
    )
    run = execute_case(case, run_root=tmp_path / "runs")
    assert run.response.status == "ok"
    assert np.mean(run.response.voltage_V) == pytest.approx(-0.1)
    assert run.response.frequency_response is not None
    metrics = ac_operating_metrics(run.response.frequency_response, 1e6, load_current_column="load_current_A")
    assert metrics["resistance_ohm"] == pytest.approx(100)
    assert metrics["source_real_power_W"] == pytest.approx(50)
    assert metrics["load_real_power_W"] == pytest.approx(50)
