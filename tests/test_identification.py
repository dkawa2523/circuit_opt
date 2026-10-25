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
    _bounded_latent_specs,
    _sensitivity_diagnosis,
    run_identification,
)
from pcd.results import read_evaluation_table


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
    original = (root / "fit" / "study_result.json").read_bytes()
    repeated = run_identification(case, tmp_path / "runs")
    assert Path(repeated["run_root"]) != root
    assert (root / "fit" / "study_result.json").read_bytes() == original


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


def test_identification_partitions_must_cover_every_declared_scenario(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
    data = deepcopy(case.data)
    data["study"]["scenarios"].append({"id": "unused", "values": {"condition": 3.0}})

    with pytest.raises(ValueError, match="must cover every scenario"):
        run_identification(type(case)(case.path, data), tmp_path / "runs")


@pytest.mark.parametrize("failed_phase", ["fit", "holdout", "sensitivity"])
def test_solver_failure_in_each_identification_phase_leaves_a_complete_result(make_case, tmp_path, failed_phase):
    import json

    from pcd.sim_registry import register_solver
    from tests.fakes import fake_solver

    solver_name = f"test_identification_failure_{failed_phase}"

    @register_solver(solver_name)
    def failing_solver(request):
        response = fake_solver(request)
        return (
            replace(response, status="failed", log=f"injected {failed_phase} failure")
            if failed_phase in request.run_dir.parts
            else response
        )

    result = run_identification(
        _identification_case(make_case, tmp_path),
        tmp_path / "runs",
        solver_override=solver_name,
    )
    assert result["status"] == "not_identified"
    assert result["reason"] == f"{failed_phase}_evidence_unavailable"
    assert (
        json.loads((Path(result["run_root"]) / "identification_result.json").read_text())["status"] == "not_identified"
    )
    if failed_phase == "fit":
        assert result["estimated_latent"] == {}
        assert result["holdout"]["status"] == "not_run"
    elif failed_phase == "sensitivity":
        for evidence in result["identifiability"]["unavailable_evidence"]:
            assert (Path(result["run_root"]) / evidence["artifacts"]["manifest"]).is_file()
    else:
        assert result["holdout"]["value"] is None
        assert result["holdout"]["passed"] is False
        assert (Path(result["run_root"]) / result["artifacts"]["holdout_result"]).is_file()


def test_identification_preserves_inline_and_by_scenario_observed_controls(make_case, tmp_path):
    case = _identification_case(make_case, tmp_path)
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
    case = _identification_case(make_case, tmp_path)
    case.data["study"]["scenarios"][1]["values"]["condition"] = 1.0
    with pytest.raises(ValueError, match="holdout repeats fit operating conditions"):
        run_identification(case, tmp_path / "runs")
    case.data["identification"]["holdout_mode"] = "replicates"
    result = run_identification(case, tmp_path / "runs")
    assert result["status"] == "identified"
    assert result["validation"]["scope"] == "same_condition_repeatability_only"
    assert result["validation"]["overlapping_conditions"] == [{"fit": "fit", "holdout": "holdout"}]
    assert result["validation"]["independent_acquisition_verified"] is False
