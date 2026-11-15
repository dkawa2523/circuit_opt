"""Latent-parameter identification stays on the ordinary study pathway."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pcd.case import load_case
from pcd.identification import (
    IdentificationSettings,
    _bounded_latent_specs,
    _sensitivity_column,
    _sensitivity_diagnosis,
    identification_inputs,
    run_identification,
)
from pcd.metric_registry import register as register_metric
from pcd.records import read_sim_record
from pcd.results import read_evaluation_table
from pcd.sim_registry import register_solver
from pcd.simulation import SimulationResult


@register_solver("test_identification_ac")
def _identification_solver(request):
    ac = request.simulation.analysis.ac
    assert ac is not None
    params = read_sim_record(request.run_dir)["params"]
    gain = params.get("fixture_gain", 1.0)
    resistance = 3 + gain * (params["latent_R"] - 4) + ac.start_hz
    response = pd.DataFrame(
        {
            "frequency_Hz": [ac.start_hz],
            "voltage_re": [1.0],
            "voltage_im": [0.0],
            "current_re": [-1.0 / resistance],
            "current_im": [0.0],
        }
    )
    return SimulationResult(np.asarray([]), np.asarray([]), frequency_response=response)


def _scenario(name, condition, resistance, weight=1.0):
    return {
        "id": name,
        "weight": weight,
        "values": {
            "condition": condition,
            "target_voltage_re_V": 1.0,
            "target_voltage_im_V": 0.0,
            "target_current_re_A": 1.0 / resistance,
            "target_current_im_A": 0.0,
        },
    }


def _identification_case(make_case):
    return make_case(
        {
            "case_id": "identify_fake",
            "run": {"trials": 8},
            "variables": {
                "latent_R": {"role": "latent", "bounds": [1.0, 10.0], "default": 3.0},
                "condition": {"role": "operating", "default": 1.0},
            },
            "source": {
                "type": "sine_voltage",
                "name": "Vsrc",
                "p": "src",
                "n": "0",
                "amplitude_V": 1.0,
                "frequency_Hz": "condition",
            },
            "circuit": {
                "builder": "from_yaml",
                "output_node": "src",
                "components": [{"ref": "R1", "n1": "src", "n2": "0", "value": "latent_R"}],
            },
            "solver": {"name": "test_identification_ac", "ac": {"frequency_Hz": "condition"}},
            "target": {"objective": "terminal_vi_fit", "impedance_scale_ohm": 1.0},
            "study": {
                "scenarios": [
                    _scenario("fit", 1.0, 4.0),
                    _scenario("holdout", 2.0, 5.0),
                ],
                "objectives": [{"metric": "normalized_impedance_error", "aggregation": "worst"}],
            },
            "optimizer": {"name": "differential_evolution", "seed": 4},
            "identification": {
                "fit_scenarios": ["fit"],
                "holdout_scenarios": ["holdout"],
                "observed_metrics": ["resistance_ohm"],
                "metric_scales": {"resistance_ohm": 1.0},
                "fit_loss_max": 10.0,
                "holdout_loss_max": 10.0,
                "condition_number_max": 1.0e9,
            },
        },
        name="identify.yaml",
    )


def test_identification_uses_latent_columns_and_independent_holdout(make_case, tmp_path):
    case = _identification_case(make_case)

    result = run_identification(case, tmp_path / "runs")

    root = Path(result["run_root"])
    assert result["status"] == "identified"
    assert result["fit"]["scenario_ids"] == ["fit"]
    assert result["holdout"]["scenario_ids"] == ["holdout"]
    assert result["holdout"]["by_acquisition_group"] == []
    assert result["validation"]["acquisition_group_separation"] == "not_declared"
    assert result["identifiability"]["rank"] == 1
    assert result["identifiability"]["required_rank"] == 1
    assert (root / "identification_result.json").is_file()
    fit = pd.read_csv(root / "fit_observations.csv").iloc[0]
    holdout = pd.read_csv(root / "holdout_observations.csv").iloc[0]
    assert fit["target.resistance_ohm"] == pytest.approx(4.0)
    assert holdout["target.resistance_ohm"] == pytest.approx(5.0)
    assert holdout["predicted.resistance_ohm"] - fit["predicted.resistance_ohm"] == pytest.approx(1.0)
    assert holdout["residual.resistance_ohm"] == pytest.approx(fit["residual.resistance_ohm"])
    fit_table = read_evaluation_table(root / "fit")
    assert "latent.latent_R" in fit_table
    assert "design.latent_R" not in fit_table
    assert set(fit_table["scenario_id"]) == {"fit"}
    original = (root / "fit" / "study_result.json").read_bytes()
    repeated = run_identification(case, tmp_path / "runs")
    assert Path(repeated["run_root"]) != root
    assert (root / "fit" / "study_result.json").read_bytes() == original


def test_waveform_identification_rejects_a_shared_target(make_case, tmp_path):
    case = _identification_case(make_case)
    target = tmp_path / "target.csv"
    target.write_text("time_s,voltage_V\n0,0\n1,1\n", encoding="utf-8")
    case.data["target"] = {"objective": "waveform_l2", "waveform_file": target.name}
    case.data["study"]["objectives"][0]["metric"] = "normalized_rmse"

    with pytest.raises(ValueError, match="waveform_l2 uses a shared target file"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_identification_rejects_observations_inherited_from_defaults(make_case, tmp_path):
    case = _identification_case(make_case)
    field = "target_voltage_re_V"
    case.data["variables"][field] = {"role": "operating", "default": 1.0}
    del case.data["study"]["scenarios"][1]["values"][field]

    with pytest.raises(ValueError, match=rf"holdout.*requires explicit observation values.*{field}"):
        run_identification(case, tmp_path / "runs")


def test_public_identification_rejects_observation_at_a_different_planned_frequency(make_case, tmp_path):
    from pcd.validation import validate_case

    example = Path(__file__).resolve().parents[1] / "examples" / "terminal_identification.yaml"
    raw = deepcopy(load_case(example).authored_data)
    raw["identification"]["observations"] = (
        pd.read_csv(example.with_name("terminal_observations.csv")).iloc[[0, 2]].to_dict(orient="records")
    )
    raw["analysis"]["ac"]["frequency_Hz"] = "10 MHz"
    case = make_case(raw)

    assert case.data["target"]["observation_frequency_parameter"] == "frequency_Hz"
    message = r"holdout_20MHz.*observed frequency.*differs from planned AC point"
    assert "observed frequency" in validate_case(case).format_text()
    with pytest.raises(ValueError, match=message):
        run_identification(case, tmp_path / "runs")
    legacy_data = deepcopy(case.data)
    del legacy_data["target"]["observation_frequency_parameter"]
    with pytest.raises(ValueError, match=message):
        run_identification(type(case)(case.path, legacy_data), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()

    del raw["identification"]["observations"][0]["frequency_Hz"]
    with pytest.raises(ValueError, match="row 1 is missing 'frequency_Hz'"):
        make_case(raw, "missing_frequency.yaml")


def test_advanced_observation_frequency_must_be_explicit_in_every_row(make_case, tmp_path):
    case = _identification_case(make_case)
    case.data["target"]["observation_frequency_parameter"] = "condition"
    case.data["variables"]["bias"] = {"role": "operating", "default": 0.5}
    case.data["identification"]["condition_parameters"] = ["bias"]
    assert identification_inputs(case).split_evidence["condition_parameters"] == ["bias", "condition"]
    del case.data["study"]["scenarios"][1]["values"]["condition"]

    with pytest.raises(ValueError, match=r"holdout.*requires explicit observation frequency.*condition"):
        run_identification(case, tmp_path / "runs")


@pytest.mark.parametrize("ac", [{"frequency_Hz": 1.0}, {"sweep": "lin", "points": 3, "start_Hz": 0.5, "stop_Hz": 1.5}])
def test_identification_treats_roundoff_around_one_ac_point_as_one_condition(make_case, tmp_path, ac):
    case = _identification_case(make_case)
    case.data["target"]["observation_frequency_parameter"] = "condition"
    case.data["solver"]["ac"] = ac
    fit, holdout = case.data["study"]["scenarios"]
    fit["values"]["condition"] = 1.0 - 0.75e-12
    holdout["values"]["condition"] = 1.0 + 0.75e-12

    with pytest.raises(ValueError, match="holdout repeats fit operating conditions"):
        run_identification(case, tmp_path / "runs")

    case.data["identification"]["holdout_mode"] = "replicates"
    evidence = identification_inputs(case).split_evidence
    assert evidence["overlapping_conditions"] == [{"fit": "fit", "holdout": "holdout"}]


@pytest.mark.parametrize("missing_prediction", [False, True], ids=["optional-report-fields", "missing-prediction"])
def test_custom_identification_requires_predictions_but_allows_missing_report_fields(
    make_case, tmp_path, missing_prediction
):
    case = _identification_case(make_case)
    metric_name = f"test_custom_observation_{missing_prediction}"

    @register_metric(metric_name)
    def observation_residual(_case, params, response):
        assert response.frequency_response is not None
        prediction = -1.0 / float(response.frequency_response.iloc[0]["current_re"])
        target = params["observed_R"]
        residual = prediction - target
        values: dict[str, float | None] = {"loss": abs(residual), "resistance_ohm": prediction}
        if params["condition"] != 2.0:
            values.update(target_resistance_ohm=target, residual_resistance_ohm=residual)
        elif missing_prediction:
            values["resistance_ohm"] = None
        return values

    case.data["target"] = {"objective": metric_name}
    case.data["study"]["objectives"] = [{"metric": "loss"}]
    case.data["variables"]["observed_R"] = {"role": "operating"}
    for point, target in zip(case.data["study"]["scenarios"], [4.0, 5.0], strict=True):
        point["values"] = {"condition": point["values"]["condition"], "observed_R": target}

    with pytest.raises(ValueError, match=r"custom metric requires identification\.observation_parameters"):
        run_identification(case, tmp_path / "runs")
    case.data["identification"]["observation_parameters"] = ["observed_R"]

    result = run_identification(case, tmp_path / "runs")
    assert result["fit"]["passed"]
    observations = pd.read_csv(Path(result["run_root"]) / result["artifacts"]["holdout_observations"])
    assert observations[["target.resistance_ohm", "residual.resistance_ohm"]].isna().all(axis=None)
    if missing_prediction:
        assert result["status"] == "not_identified"
        assert result["reason"] == "holdout_evidence_unavailable"
        assert result["holdout"]["value"] is None
        assert result["holdout"]["by_condition"][0]["n_available"] == 0
    else:
        assert result["status"] == "identified"


def test_identification_rejects_overlapping_fit_and_holdout(make_case):
    case = _identification_case(make_case)
    data = case.data
    data["identification"]["holdout_scenarios"] = ["fit"]
    overlapping = type(case)(case.path, data)

    with pytest.raises(ValueError, match="must be disjoint"):
        IdentificationSettings.from_case(overlapping)


def test_identification_uncertainty_cannot_change_an_operating_condition(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "terminal_identification.yaml"
    case = load_case(example)
    case.data["identification"]["uncertainty"] = {
        "frequency_shift": {"observations": {"holdout_40_68MHz": {"frequency_Hz": 40e6}}}
    }

    with pytest.raises(ValueError, match=r"observations\.holdout_40_68MHz.*frequency_Hz"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_identification_uncertainty_ranges_exclude_an_unidentifiable_fit(make_case, tmp_path):
    case = _identification_case(make_case)
    case.data["variables"]["fixture_gain"] = {"role": "calibration", "default": 1.0}
    case.data["identification"]["uncertainty"] = {"zero_gain": {"calibration": {"fixture_gain": 0.0}}}

    result = run_identification(case, tmp_path / "r")

    uncertainty = result["uncertainty"]
    nominal, zero_gain = uncertainty["realizations"]
    assert nominal["status"] == "identified"
    assert zero_gain["fit_passed"]
    assert zero_gain["holdout_passed"]
    assert zero_gain["local_rank"] == 0
    assert zero_gain["locally_identifiable"] is False
    assert uncertainty["status"] == "not_all_identified"
    assert uncertainty["n_unavailable"] == 0
    assert uncertainty["parameter_ranges"]["latent_R"]["n_realizations"] == 1
    assert uncertainty["holdout_prediction_ranges"]["holdout"]["resistance_ohm"]["n_realizations"] == 1


def test_terminal_identification_rejects_unrelated_fit_and_sensitivity_metrics(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "terminal_identification.yaml"
    case = load_case(example)
    objective = case.data["study"]["objectives"][0]
    original = objective["metric"]
    objective["metric"] = "source_real_power_W"
    with pytest.raises(ValueError, match="observed-data residual objective"):
        run_identification(case, tmp_path / "runs", optimizer_name="random")

    objective["metric"] = original
    case.data["identification"].update(
        observed_metrics=["source_real_power_W"], metric_scales={"source_real_power_W": 1.0}
    )
    with pytest.raises(ValueError, match="sensitivity requires observed impedance components"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_identification_publishes_estimate_when_requested_output_raises(make_case, tmp_path, monkeypatch):
    case = _identification_case(make_case)
    case.data["output"] = {}

    def fail(*_args):
        raise RuntimeError("output unavailable")

    monkeypatch.setattr("pcd.reporting.study.render_identification_outputs", fail)
    result = run_identification(case, tmp_path / "runs")
    assert result["status"] == "identified"
    assert result["estimated_latent"]
    assert result["output"] == {"status": "partial", "summary": None, "error": "RuntimeError: output unavailable"}
    assert "output_summary" not in result["artifacts"]
    archived = Path(result["run_root"]) / "identification_result.json"
    assert json.loads(archived.read_text(encoding="utf-8")) == result


def test_design_entry_point_does_not_accept_latent_candidate_role(make_case, tmp_path):
    from pcd.study import run_case_study

    case = _identification_case(make_case)
    data = case.data
    data["study"]["candidate_role"] = "latent"
    latent_case = type(case)(case.path, data)

    with pytest.raises(ValueError, match="use run_identification"):
        run_case_study(latent_case, tmp_path / "runs")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fit_scenarios", [], "non-empty list"),
        ("fit_scenarios", ["fit", "fit"], "unique non-empty names"),
        ("sensitivity_step", 0.5, "less than 0.5"),
        ("fit_loss_max", "not-a-number", "must be numeric"),
        ("fit_loss_max", float("inf"), "must be finite"),
        ("fit_loss_max", 0.0, "must be positive"),
    ],
)
def test_identification_settings_reject_malformed_limits(make_case, field, value, message):
    case = _identification_case(make_case)
    data = deepcopy(case.data)
    data["identification"][field] = value

    with pytest.raises(ValueError, match=message):
        IdentificationSettings.from_case(type(case)(case.path, data))


def test_identification_settings_require_every_observed_metric_scale(make_case):
    case = _identification_case(make_case)
    data = deepcopy(case.data)
    data["identification"]["metric_scales"] = {}

    with pytest.raises(ValueError, match="metric_scales is missing"):
        IdentificationSettings.from_case(type(case)(case.path, data))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"role": "fixed", "default": 3.0}, "at least one parameter"),
        ({"role": "latent", "default": 3.0}, "requires two numeric bounds"),
        ({"role": "latent", "bounds": [3.0, 3.0], "default": 3.0}, "increasing bounds"),
    ],
)
def test_identification_requires_valid_bounded_latent_parameters(
    make_case,
    replacement,
    message,
):
    case = _identification_case(make_case)
    data = deepcopy(case.data)
    data["variables"]["latent_R"] = replacement

    with pytest.raises(ValueError, match=message):
        _bounded_latent_specs(type(case)(case.path, data))


@pytest.mark.parametrize("matrix", [np.zeros((3, 2)), np.eye(2) * 1e-20])
def test_numerically_empty_sensitivity_is_not_identifiable(matrix):
    singular, rank, condition, identifiable = _sensitivity_diagnosis(
        matrix,
        2,
        IdentificationSettings(
            fit_scenarios=("fit",),
            holdout_scenarios=("holdout",),
            observed_metrics=("loss",),
            metric_scales={"loss": 1.0},
            fit_loss_max=1.0,
            holdout_loss_max=1.0,
            sensitivity_step=0.02,
            condition_number_max=100.0,
            rank_tolerance=1.0e-6,
            expected={},
            recovery_relative_tolerance=0.1,
        ),
    )
    assert np.max(singular) <= 1e-20
    assert rank == 0
    assert condition is None
    assert identifiable is False


def test_unrepresentable_sensitivity_step_retains_unavailable_evidence(make_case):
    case = _identification_case(make_case)
    settings = replace(IdentificationSettings.from_case(case), sensitivity_step=1e-30)
    runner = Mock()
    derivative, evaluations, evidence = _sensitivity_column(
        "latent_R", {"bounds": [1.0, 10.0]}, {"latent_R": 5.0}, runner, settings
    )
    assert derivative is None
    assert evaluations == 0
    assert evidence[0]["parameter"] == "latent_R"
    runner.evaluate_candidate.assert_not_called()


def test_identification_partitions_must_cover_every_declared_scenario(make_case, tmp_path):
    case = _identification_case(make_case)
    data = deepcopy(case.data)
    data["study"]["scenarios"].append({"id": "unused", "values": {"condition": 3.0}})

    with pytest.raises(ValueError, match="must cover every scenario"):
        run_identification(type(case)(case.path, data), tmp_path / "runs")


@pytest.mark.parametrize(
    ("failed_phase", "partial"),
    [("fit", False), ("fit", True), ("holdout", False), ("holdout", True), ("sensitivity", False)],
)
def test_solver_failure_in_each_identification_phase_leaves_a_complete_result(
    make_case, tmp_path, failed_phase, partial
):
    case = _identification_case(make_case)
    if partial:
        case.data["study"]["scenarios"].append(_scenario("secondary", 3.0, 6.0))
        case.data["identification"][f"{failed_phase}_scenarios"].append("secondary")
        if failed_phase == "holdout":
            case.data["identification"]["acquisition_groups"] = {"fit": "A", "holdout": "B", "secondary": "B"}
    solver_name = f"test_identification_failure_{failed_phase}_{partial}"

    @register_solver(solver_name)
    def failing_solver(request):
        response = _identification_solver(request)
        fail = failed_phase in request.run_dir.parts and (
            not partial or read_sim_record(request.run_dir)["params"]["condition"] == 3.0
        )
        return replace(response, status="failed", log=f"injected {failed_phase} failure") if fail else response

    result = run_identification(
        case,
        tmp_path / "runs",
        solver_override=solver_name,
    )
    assert result["status"] == "not_identified"
    assert result["reason"] == f"{failed_phase}_evidence_unavailable"
    assert (
        json.loads((Path(result["run_root"]) / "identification_result.json").read_text())["status"] == "not_identified"
    )
    if failed_phase == "fit":
        assert result["fit"]["status"] == "unavailable"
        assert result["fit"]["value"] is None
        assert result["estimated_latent"] == {}
        assert result["holdout"]["status"] == "not_run"
        assert result["identifiability"]["status"] == "not_run"
    elif failed_phase == "sensitivity":
        for evidence in result["identifiability"]["unavailable_evidence"]:
            assert (Path(result["run_root"]) / evidence["artifacts"]["manifest"]).is_file()
    else:
        assert result["holdout"]["value"] is None
        assert result["holdout"]["passed"] is False
        assert (Path(result["run_root"]) / result["artifacts"]["holdout_result"]).is_file()
        if partial:
            group = result["holdout"]["by_acquisition_group"][0]
            assert group["n_observations"] == 2
            assert group["n_available"] == 1
            assert group["value"] is None


def test_zero_holdout_observed_current_is_rejected_before_fitting(tmp_path):
    from pcd.case import load_case
    from pcd.validation import validate_case

    example = Path(__file__).resolve().parents[1] / "examples" / "terminal_identification.yaml"
    case = load_case(example)
    observations = pd.read_csv(example.with_name("terminal_observations.csv"))
    observations.loc[observations["scenario_id"] == "holdout_20MHz", ["current_re_A", "current_im_A"]] = 0.0
    path = tmp_path / "observations.csv"
    observations.to_csv(path, index=False)
    case.data["study"]["scenario_table"]["table_file"] = str(path)

    report = validate_case(case)
    assert not report.ok
    assert "holdout_20MHz" in report.format_text()
    with pytest.raises(ValueError, match=r"holdout_20MHz.*current must be nonzero"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_identification_preserves_inline_and_by_scenario_observed_controls(make_case, tmp_path):
    case = _identification_case(make_case)
    case.data["study"]["scenarios"][0]["controls"] = {"bias": 1}
    case.data["study"]["controls"] = {
        "variables": {"bias": {"choices": [1, 2]}},
        "by_scenario": {"holdout": {"bias": 2}},
    }
    result = run_identification(case, tmp_path / "runs")
    root = Path(result["run_root"])
    assert set(read_evaluation_table(root / "fit")["control.bias"]) == {1}
    assert set(read_evaluation_table(root / "holdout")["control.bias"]) == {2}


def test_new_ids_alone_cannot_claim_an_unseen_condition_holdout(make_case, tmp_path):
    case = _identification_case(make_case)
    case.data["study"]["scenarios"][1]["values"]["condition"] = "1.0"
    case.data["identification"]["acquisition_groups"] = {"fit": "fit_batch", "holdout": "new_batch"}
    with pytest.raises(ValueError, match="holdout repeats fit operating conditions"):
        run_identification(case, tmp_path / "runs")
    case.data["identification"]["holdout_mode"] = "replicates"
    result = run_identification(case, tmp_path / "runs")
    assert result["status"] == "identified"
    assert result["validation"]["scope"] == "same_condition_repeatability_only"
    assert result["validation"]["overlapping_conditions"] == [{"fit": "fit", "holdout": "holdout"}]
    assert result["validation"]["independent_acquisition_verified"] is False


@pytest.mark.parametrize(("column", "replacement"), [("partition", "fit"), ("acquisition_group", "fit_group")])
def test_observation_csv_metadata_drift_is_rejected(make_case, tmp_path, column, replacement):
    from pcd.case import load_case
    from pcd.validation import validate_case

    example = Path(__file__).resolve().parents[1] / "examples" / "terminal_identification.yaml"
    raw = deepcopy(load_case(example).authored_data)
    observations = pd.read_csv(example.with_name("terminal_observations.csv"))
    observations["acquisition_group"] = observations["partition"].map({"fit": "fit_group", "holdout": "holdout_group"})
    table = tmp_path / "terminal_observations.csv"
    observations.to_csv(table, index=False)
    case = make_case(raw)
    assert validate_case(case).ok

    observations.loc[observations["partition"] == "holdout", column] = replacement
    observations.to_csv(table, index=False)

    with pytest.raises(ValueError, match="observation CSV partition or acquisition groups changed after loading"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("mode", ["new_conditions", "replicates"])
def test_fit_and_holdout_must_not_share_acquisition_groups(make_case, tmp_path, mode):
    case = _identification_case(make_case)
    case.data["identification"].update(
        {"holdout_mode": mode, "acquisition_groups": {"fit": "shared", "holdout": " shared "}}
    )
    with pytest.raises(ValueError, match="fit and holdout share acquisition groups"):
        run_identification(case, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_grouped_holdout_reports_condition_coverage_and_explicit_weighting(make_case, tmp_path):
    import csv

    case = _identification_case(make_case)
    case.data["study"]["scenarios"].extend(
        [
            _scenario("holdout_repeat", 2, 5.1, weight=3.0),
            _scenario("holdout_other", 3, 6.0, weight=2.0),
        ]
    )
    case.data["study"]["objectives"][0]["aggregation"] = "rms"
    case.data["identification"]["holdout_scenarios"].extend(["holdout_repeat", "holdout_other"])
    case.data["identification"]["acquisition_groups"] = {
        "fit": "fitting",
        "holdout": "001",
        "holdout_repeat": "001",
        "holdout_other": "NA",
    }

    result = run_identification(case, tmp_path / "runs")
    assert result["status"] == "identified"
    assert result["validation"]["acquisition_group_separation"] == "declared_disjoint"
    assert result["validation"]["acquisition_groups"] == {"fit": ["fitting"], "holdout": ["001", "NA"]}
    assert result["validation"]["independent_acquisition_verified"] is False
    holdout = result["holdout"]
    assert holdout["weighting"] == "declared_row_weights"
    first, second = holdout["by_acquisition_group"]
    assert (first["acquisition_group"], second["acquisition_group"]) == ("001", "NA")
    assert first["n_observations"] == first["n_available"] == 2
    assert first["weight_fraction"] == pytest.approx(2 / 3)
    assert second["weight_fraction"] == pytest.approx(1 / 3)
    assert holdout["value"] == pytest.approx(
        np.hypot(np.sqrt(2 / 3) * first["value"], np.sqrt(1 / 3) * second["value"])
    )
    assert [row["values"] for row in holdout["by_condition"]] == [{"condition": 2}, {"condition": 3}]
    assert holdout["by_condition"][0]["scenario_ids"] == ["holdout", "holdout_repeat"]
    with (Path(result["run_root"]) / result["artifacts"]["holdout_observations"]).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "input.acquisition_group" not in reader.fieldnames
        assert [row["acquisition_group"] for row in reader] == ["001", "001", "NA"]
