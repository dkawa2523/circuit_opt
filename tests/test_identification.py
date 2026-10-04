"""Latent-parameter identification stays on the ordinary study pathway."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.identification import (
    IdentificationSettings,
    _axis_unit_value,
    _axis_value,
    _bounded_latent_specs,
    _fix_latent_specs,
    _objective_value,
    _recovery,
    _selected_scenarios,
    _sensitivity_diagnosis,
    run_identification,
)
from pcd.results import read_evaluation_table
from pcd.study_config import study_spec_from_case


def _identification_case(make_case, tmp_path: Path):
    target_path = tmp_path / "target.csv"
    time_s = np.linspace(0.0, 1.0e-6, 101)
    pd.DataFrame({"time_s": time_s, "voltage_V": np.full_like(time_s, 0.8)}).to_csv(
        target_path,
        index=False,
    )
    return make_case(
        {
            "case_id": "identify_fake",
            "run": {"trials": 8},
            "variables": {
                "latent_R": {"role": "latent", "bounds": [1.0, 10.0], "default": 3.0},
                "condition": {"role": "operating", "default": 1.0},
            },
            "source": {
                "type": "voltage_pulse",
                "name": "Vsrc",
                "p": "src",
                "n": "0",
                "v1_V": 0.0,
                "v2_V": 1.0,
                "width_s": 2.0e-6,
                "period_s": 3.0e-6,
            },
            "circuit": {
                "builder": "from_yaml",
                "output_node": "out",
                "components": [{"ref": "R1", "n1": "src", "n2": "out", "value": "latent_R"}],
            },
            "load": {"name": "none"},
            "measurement": {"voltage_node": "out", "current_source": "Vsrc"},
            "solver": {"name": "test_fake", "tran": {"step_s": 1.0e-8, "stop_s": 1.0e-6}},
            "target": {"objective": "waveform_l2", "waveform_file": target_path.name},
            "study": {
                "scenarios": [
                    {"id": "fit", "values": {"condition": 1.0}},
                    {"id": "holdout", "values": {"condition": 2.0}},
                ],
                "objectives": [{"metric": "normalized_rmse", "direction": "minimize", "aggregation": "worst"}],
            },
            "optimizer": {"name": "differential_evolution", "seed": 4},
            "identification": {
                "fit_scenarios": ["fit"],
                "holdout_scenarios": ["holdout"],
                "observed_metrics": ["peak_abs_voltage_V"],
                "metric_scales": {"peak_abs_voltage_V": 1.0},
                "fit_loss_max": 10.0,
                "holdout_loss_max": 10.0,
                "condition_number_max": 1.0e9,
            },
        },
        name="identify.yaml",
    )


def test_identification_uses_latent_columns_and_independent_holdout(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)

    result = run_identification(case, tmp_path / "runs")

    root = Path(result["run_root"])
    assert result["status"] == "identified"
    assert result["fit"]["scenario_ids"] == ["fit"]
    assert result["holdout"]["scenario_ids"] == ["holdout"]
    assert result["identifiability"]["rank"] == 1
    assert result["identifiability"]["required_rank"] == 1
    assert (root / "identification_result.json").is_file()
    assert (root / "fit_observations.csv").is_file()
    assert (root / "holdout_observations.csv").is_file()
    fit_table = read_evaluation_table(root / "fit")
    assert "latent.latent_R" in fit_table
    assert "design.latent_R" not in fit_table
    assert set(fit_table["scenario_id"]) == {"fit"}


def test_identification_rejects_overlapping_fit_and_holdout(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
    data = case.data
    data["identification"]["holdout_scenarios"] = ["fit"]
    overlapping = type(case)(case.path, data)

    with pytest.raises(ValueError, match="must be disjoint"):
        IdentificationSettings.from_case(overlapping)


def test_design_entry_point_does_not_accept_latent_candidate_role(make_case, tmp_path):
    from pcd.study import run_case_study

    case = _identification_case(make_case, tmp_path)
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
def test_identification_settings_reject_malformed_limits(make_case, tmp_path, field, value, message):
    case = _identification_case(make_case, tmp_path)
    data = deepcopy(case.data)
    data["identification"][field] = value

    with pytest.raises(ValueError, match=message):
        IdentificationSettings.from_case(type(case)(case.path, data))


def test_identification_settings_require_every_observed_metric_scale(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
    data = deepcopy(case.data)
    data["identification"]["metric_scales"] = {}

    with pytest.raises(ValueError, match="metric_scales is missing"):
        IdentificationSettings.from_case(type(case)(case.path, data))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"role": "fixed", "default": 3.0}, "at least one parameter"),
        ({"role": "latent", "default": 3.0}, "requires two numeric bounds"),
        ({"role": "latent", "bounds": [4.0, 2.0], "default": 3.0}, "invalid bounds"),
        ({"role": "latent", "bounds": [0.0, 2.0], "default": 1.0, "scale": "log"}, "invalid bounds"),
    ],
)
def test_identification_requires_valid_bounded_latent_parameters(
    make_case,
    tmp_path,
    replacement,
    message,
):
    case = _identification_case(make_case, tmp_path)
    data = deepcopy(case.data)
    data["variables"]["latent_R"] = replacement

    with pytest.raises(ValueError, match=message):
        _bounded_latent_specs(type(case)(case.path, data))


def test_identification_helpers_reject_unknown_scenarios_and_objectives(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
    catalog = {scenario.scenario_id: scenario for scenario in study_spec_from_case(case).scenarios}

    with pytest.raises(ValueError, match="unknown scenarios"):
        _selected_scenarios(catalog, ["missing"], "identification.fit_scenarios")
    with pytest.raises(ValueError, match="has no objective"):
        _objective_value({})
    with pytest.raises(ValueError, match="must be minimized"):
        _objective_value({"study": {"objectives": [{"metric": "loss", "direction": "maximize"}]}})


def test_log_axis_mapping_and_rank_deficiency_are_explicit():
    spec = {"bounds": [1.0e-3, 1.0e1], "scale": "log"}
    assert _axis_unit_value(1.0e-1, spec) == pytest.approx(0.5)
    assert _axis_value(0.5, spec) == pytest.approx(1.0e-1)

    singular, rank, condition, identifiable = _sensitivity_diagnosis(
        np.zeros((3, 2)),
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
    assert singular.tolist() == [0.0, 0.0]
    assert rank == 0
    assert condition is None
    assert identifiable is False


def test_recovery_reports_each_parameter_and_rejects_name_mismatch(make_case, tmp_path):
    settings = IdentificationSettings.from_case(_identification_case(make_case, tmp_path))
    settings = replace(
        settings,
        expected={"latent_R": 4.0},
        recovery_relative_tolerance=0.1,
    )

    passed = _recovery({"latent_R": 4.2}, settings)
    failed = _recovery({"latent_R": 5.0}, settings)
    assert passed is not None
    assert passed["passed"] is True
    assert failed is not None
    assert failed["passed"] is False
    assert failed["parameters"]["latent_R"]["relative_error"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="every and only"):
        _recovery({"other": 4.0}, settings)


def test_fix_latent_values_handles_all_variable_blocks_and_missing_names():
    data = {
        "variables": {"root": {"role": "latent", "bounds": [1.0, 2.0], "scale": "log"}},
        "source": {"variables": {"source_value": 1.0}},
        "circuit": {"variables": {"circuit_value": {"bounds": [1.0, 2.0]}}},
        "load": {"variables": {"load_value": {"bounds": [1.0, 2.0]}}},
        "sources": [{"variables": {"second_source": {"bounds": [1.0, 2.0]}}}, None],
    }
    values = {
        "root": 1.5,
        "source_value": 2.0,
        "circuit_value": 3.0,
        "load_value": 4.0,
        "second_source": 5.0,
    }

    _fix_latent_specs(data, values)

    assert data["variables"]["root"] == {"role": "latent", "default": 1.5, "choices": [1.5]}
    source = data["source"]
    assert isinstance(source, dict)
    source_value = source["variables"]["source_value"]
    assert isinstance(source_value, dict)
    assert source_value["default"] == 2.0
    second_source = data["sources"][0]
    assert isinstance(second_source, dict)
    assert second_source["variables"]["second_source"]["choices"] == [5.0]
    with pytest.raises(ValueError, match="not declared"):
        _fix_latent_specs(data, {"missing": 1.0})


def test_identification_partitions_must_cover_every_declared_scenario(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
    data = deepcopy(case.data)
    data["study"]["scenarios"].append({"id": "unused", "values": {"condition": 3.0}})

    with pytest.raises(ValueError, match="must cover every scenario"):
        run_identification(type(case)(case.path, data), tmp_path / "runs")
