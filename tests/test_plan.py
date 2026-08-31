"""The public RF input must resolve once into an explicit, replayable plan."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from pcd.case import load_case
from pcd.plan import compile_rf_case
from pcd.sim_core import prepare_case
from pcd.study_config import study_spec_from_case
from pcd.validation import validate_case

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "cases"


def _base() -> dict:
    return {
        "schema": "pcd.rf.v1",
        "case_id": "small_rf",
        "frequency_Hz": 13.56e6,
        "network": {"type": "pi_match", "fixed": {"C1": 250e-12, "L1": 1.2e-6, "C2": 8e-12}},
        "load": {
            "type": "impedance_point",
            "resistance_ohm": 25,
            "reactance_ohm": -80,
            "reference_plane": "electrode_terminal",
            "evidence": {"origin": "test"},
        },
        "acceptance": {"reflected_power_fraction_max": 0.1},
    }


def test_fixed_table_input_resolves_roles_defaults_and_acceptance():
    case = load_case(BENCH / "match_full_tuner.yaml")

    assert case.authored_data["network"]["fixed"] == {"L1": 6.43146488676e-7}
    assert case.data["source"]["frequency_Hz"] == 13.56e6
    assert case.data["load"]["resistance_ohm"] == "load_resistance_ohm"
    assert case.data["study"]["design_variables"] == ["L1"]
    assert case.data["study"]["controls"]["budget"] == 49
    assert case.data["study"]["control_margin_min"] == 0.2
    assert case.data["target"]["constraints"]["metric_bounds"]["reflection_magnitude"]["max"] == pytest.approx(
        math.sqrt(0.1)
    )
    assert case.data["variables"]["load_resistance_ohm"]["default"] == 50.0


def test_frequency_table_drives_source_load_and_solver_with_one_value():
    case = load_case(BENCH / "match_independent_frequency_points.yaml")

    assert case.data["source"]["frequency_Hz"] == "rf_frequency_Hz"
    assert case.data["load"]["model_frequency_Hz"] == "rf_frequency_Hz"
    assert case.data["solver"]["ac"]["frequency_Hz"] == "rf_frequency_Hz"
    assert case.data["variables"]["rf_frequency_Hz"]["default"] == 10e6


def test_absolute_drive_exposes_every_named_matching_component():
    case = load_case(BENCH / "match_high_drive_stress.yaml")
    components = {item["ref"]: item for item in case.data["circuit"]["components"]}

    assert case.data["circuit"]["builder"] == "from_yaml"
    assert {ref for ref, item in components.items() if item.get("observe")} == {"C1", "L1", "C2"}
    assert components["L1"]["series_resistance_ohm"] == 0.5
    assert case.data["source"]["amplitude_V"] == "drive_amplitude_V"
    assert case.data["measurement"]["load_current"] == "auto"


def test_explicit_drive_without_limits_still_reports_named_component_stress():
    data = _base()
    data["drive_peak_V"] = 80
    plan = compile_rf_case(data, Path.cwd())

    components = {item["ref"]: item for item in plan.case["circuit"]["components"]}
    assert plan.case["circuit"]["builder"] == "from_yaml"
    assert all(item.get("observe") is True for item in components.values())
    assert "load_current" in plan.case["measurement"]


def test_source_limits_and_control_margin_compile_into_their_existing_execution_layers():
    data = _base()
    data["drive_peak_V"] = 80
    data["network"]["fixed"].pop("C1")
    data["network"]["tuning"] = {"C1": [200e-12, 250e-12, 300e-12]}
    data["acceptance"].update(
        {
            "source_limits": {"current_rms_A_max": 2.0, "apparent_power_VA_max": 100.0},
            "control_margin_min": 0.2,
        }
    )
    plan = compile_rf_case(data, Path.cwd())

    bounds = plan.case["target"]["constraints"]["metric_bounds"]
    assert bounds["source_current_rms_A"] == {"max": 2.0}
    assert bounds["source_apparent_power_VA"] == {"max": 100.0}
    assert "control_margin" not in bounds
    assert plan.case["study"]["control_margin_min"] == 0.2


def test_run_archives_authored_input_and_resolved_plan(tmp_path):
    case = load_case(BENCH / "match_fixed_nominal.yaml")
    record = prepare_case(case, run_root=tmp_path)

    assert yaml.safe_load((record.run_dir / "input_case.yaml").read_text(encoding="utf-8"))["schema"] == "pcd.rf.v1"
    resolved = yaml.safe_load((record.run_dir / "resolved_plan.yaml").read_text(encoding="utf-8"))
    assert resolved["source_schema"] == "pcd.rf.v1"
    assert resolved["case"] == yaml.safe_load((record.run_dir / "case.yaml").read_text(encoding="utf-8"))
    assert record.manifest()["artifacts"]["resolved_plan"] == "resolved_plan.yaml"
    assert record.manifest()["artifacts"]["input_manifest"] == "input_manifest.json"
    assert record.provenance["input_schema"] == "pcd.rf.v1"

    manifest = json.loads((record.run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "input_manifest.v1"
    assert len(manifest["inputs"]) == 1
    archived_input = record.run_dir / manifest["inputs"][0]["artifact"]
    assert archived_input.read_bytes() == (BENCH.parent / "load_scenarios.csv").read_bytes()

    archived_case = load_case(record.run_dir / "case.yaml")
    archived_table = archived_case.data["study"]["scenario_table"]["table_file"]
    assert not Path(archived_table).is_absolute()
    assert (archived_case.base_dir / archived_table).resolve() == archived_input.resolve()


def test_archived_case_replays_its_scenarios_after_the_source_table_is_removed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    table = source / "points.csv"
    table.write_text(
        "scenario_id,resistance_ohm,reactance_ohm\nnominal,25,-80\nhigh,50,-40\n",
        encoding="utf-8",
    )
    data = _base()
    data["load"] = {"type": "impedance_table", "file": table.name, "reference_plane": "electrode"}
    case_path = source / "case.yaml"
    case_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    record = prepare_case(load_case(case_path), run_root=tmp_path / "runs")
    table.unlink()
    case_path.unlink()

    replay = load_case(record.run_dir / "case.yaml")
    assert [item.scenario_id for item in study_spec_from_case(replay).scenarios] == ["nominal", "high"]


def test_search_defaults_are_resolved_and_fixed_runs_are_single_candidate_grids():
    data = _base()
    data["network"]["fixed"].pop("C1")
    data["network"]["search"] = {"C1": {"values": [1e-11, 1e-10, 1e-9]}}
    plan = compile_rf_case(data, Path.cwd())
    assert plan.trials == 3
    assert plan.case["variables"]["C1"]["default"] == pytest.approx(1e-11)

    fixed = _base()
    fixed_plan = compile_rf_case(fixed, Path.cwd())
    assert fixed_plan.optimizer == "grid"
    assert fixed_plan.trials == 1


def test_discrete_search_is_a_complete_grid_without_trial_count_input():
    data = _base()
    data["network"]["fixed"].pop("C1")
    data["network"]["search"] = {"C1": {"values": [100e-12, 200e-12, 300e-12]}}

    plan = compile_rf_case(data, Path.cwd())

    assert plan.optimizer == "grid"
    assert plan.trials == 3
    assert plan.case["run"] == {"trials": 3}
    assert any("all 3 discrete hardware candidates" in item for item in plan.inferences)


def test_public_search_rejects_continuous_partial_or_oversized_spaces():
    continuous = _base()
    continuous["network"]["fixed"].pop("C1")
    continuous["network"]["search"] = {"C1": {"range": [100e-12, 300e-12]}}
    with pytest.raises(ValueError, match=r"unsupported fields.*range"):
        compile_rf_case(continuous, Path.cwd())

    partial = _base()
    partial["network"]["fixed"].pop("C1")
    partial["network"]["search"] = {"C1": {"values": [100e-12, 200e-12, 300e-12]}}
    partial["execution"] = {"trials": 2}
    with pytest.raises(ValueError, match=r"unsupported fields.*trials"):
        compile_rf_case(partial, Path.cwd())

    oversized = _base()
    oversized["network"]["fixed"].pop("C1")
    oversized["network"]["search"] = {"C1": {"values": [float(index + 1) for index in range(251)]}}
    with pytest.raises(ValueError, match="251 exact candidates"):
        compile_rf_case(oversized, Path.cwd())


def test_public_component_values_are_positive_unique_and_replayable():
    data = _base()
    data["network"]["fixed"].pop("C1")

    for spec, message in [
        ({"values": [100e-12, 100e-12]}, "must not contain duplicates"),
        ({"values": [0.0, 100e-12]}, "must be positive"),
        ({"values": [100e-12, 200e-12], "default": 300e-12}, "must be one of"),
        ({"range": [100e-12, 200e-12], "default": 300e-12}, "unsupported fields.*range"),
    ]:
        invalid = deepcopy(data)
        invalid["network"]["search"] = {"C1": spec}
        with pytest.raises(ValueError, match=message):
            compile_rf_case(invalid, Path.cwd())


def test_direct_reflection_limit_drive_and_execution_remain_explicit():
    data = _base()
    data["drive_peak_V"] = 80
    data["acceptance"] = {"reflection_magnitude_max": 0.2}
    data["execution"] = {"solver": "ngspice_cli"}
    plan = compile_rf_case(data, Path.cwd())

    assert plan.case["source"]["amplitude_V"] == 80
    assert plan.case["target"]["constraints"]["metric_bounds"]["reflection_magnitude"] == {"max": 0.2}
    assert plan.to_dict()["execution"] == {
        "solver": "ngspice_cli",
        "optimizer": "grid",
        "trials": 1,
        "seed": 0,
    }


def test_ccp_and_icp_are_explicit_load_choices_not_plasma_state_solvers():
    ccp = _base()
    ccp["load"] = {
        "type": "ccp_lumped",
        "reference_plane": "electrode",
        "parameters": {"R_eff_ohm": 10, "L_eff_H": 1e-6, "C_sheath_eq_F": 100e-12},
    }
    assert compile_rf_case(ccp, Path.cwd()).case["load"]["name"] == "ccp_lumped"

    icp = _base()
    icp["load"] = {
        "type": "icp_transformer",
        "reference_plane": "coil_terminal",
        "parameters": {
            "R_coil_ohm": 0.2,
            "L_coil_H": 2e-6,
            "reflected_inductance_H": 0.18e-6,
            "secondary_damping_rate_rad_s": 4e6,
        },
    }
    resolved = compile_rf_case(icp, Path.cwd()).case["load"]
    assert resolved["name"] == "icp_transformer"
    assert resolved["reflected_inductance_H"] == pytest.approx(0.18e-6)


def test_characterization_is_optional_but_visible_as_a_strict_warning(tmp_path):
    data = _base()
    data["load"].pop("evidence")
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    case = load_case(path)

    assert validate_case(case).ok
    assert any(issue.code == "load.missing_characterization" for issue in validate_case(case, strict=True).issues)


def _bad_topology(data):
    data["network"]["type"] = "mystery"


def _missing_component(data):
    data["network"]["fixed"].pop("C2")


def _extra_component(data):
    data["network"]["fixed"]["R9"] = 1


def _role_overlap(data):
    data["network"]["tuning"] = {"C1": [1e-12]}


def _empty_tuning(data):
    data["network"]["fixed"].pop("C1")
    data["network"]["tuning"] = {"C1": []}


def _control_limit(data):
    data["network"]["fixed"].pop("C1")
    data["network"]["tuning"] = {"C1": list(range(1, 252))}


def _two_reflection_limits(data):
    data["acceptance"]["reflection_magnitude_max"] = 0.2


def _bad_reflection_limit(data):
    data["acceptance"]["reflected_power_fraction_max"] = 2


def _missing_reference_plane(data):
    data["load"].pop("reference_plane")


def _missing_drive(data):
    data["acceptance"]["component_limits"] = {"L1": {"current_rms_A_max": 1}}


def _loss_limit_without_loss(data):
    data["drive_peak_V"] = 100
    data["acceptance"]["component_limits"] = {"L1": {"loss_W_max": 1}}


def _unknown_loss_component(data):
    data["network"]["loss_ohm"] = {"R9": 1}


def _unknown_top_level(data):
    data["frequncy_Hz"] = data["frequency_Hz"]


def _negative_loss(data):
    data["network"]["loss_ohm"] = {"L1": -1}


def _loss_balance_without_loss(data):
    data["acceptance"]["loss_balance_fraction_max"] = 1e-5


def _missing_frequency(data):
    data.pop("frequency_Hz")


def _bad_evidence(data):
    data["load"]["evidence"] = "measured-ish"


def _missing_load_value(data):
    data["load"].pop("reactance_ohm")


def _unknown_limit_ref(data):
    data["acceptance"]["component_limits"] = {"R9": {"current_rms_A_max": 1}}


def _empty_limits(data):
    data["acceptance"]["component_limits"] = {"L1": {}}


def _unknown_limit(data):
    data["acceptance"]["component_limits"] = {"L1": {"temperature_C_max": 100}}


def _negative_limit(data):
    data["acceptance"]["component_limits"] = {"L1": {"current_rms_A_max": -1}}


def _source_limit_without_drive(data):
    data["acceptance"]["source_limits"] = {"current_rms_A_max": 1}


def _empty_source_limits(data):
    data["acceptance"]["source_limits"] = {}


def _unknown_source_limit(data):
    data["drive_peak_V"] = 100
    data["acceptance"]["source_limits"] = {"power_factor_min": 0.9}


def _negative_source_limit(data):
    data["drive_peak_V"] = 100
    data["acceptance"]["source_limits"] = {"apparent_power_VA_max": -1}


def _margin_without_tuning(data):
    data["acceptance"]["control_margin_min"] = 0.2


def _invalid_margin(data):
    data["network"]["fixed"].pop("C1")
    data["network"]["tuning"] = {"C1": [200e-12, 250e-12, 300e-12]}
    data["acceptance"]["control_margin_min"] = 1.1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_bad_topology, "network.type"),
        (_missing_component, "missing component"),
        (_extra_component, "not used"),
        (_role_overlap, "one role"),
        (_empty_tuning, "non-empty list"),
        (_control_limit, "exceeding the safety limit"),
        (_two_reflection_limits, "exactly one"),
        (_bad_reflection_limit, "between 0 and 1"),
        (_missing_reference_plane, "reference_plane"),
        (_missing_drive, "drive_peak_V"),
        (_loss_limit_without_loss, "loss_ohm"),
        (_unknown_loss_component, "not used"),
        (_unknown_top_level, "unsupported fields"),
        (_negative_loss, "non-negative"),
        (_loss_balance_without_loss, "loss_ohm"),
        (_missing_frequency, "frequency_Hz is required"),
        (_bad_evidence, "evidence must be a mapping"),
        (_missing_load_value, "reactance_ohm is required"),
        (_unknown_limit_ref, "not used by the network"),
        (_empty_limits, "at least one limit"),
        (_unknown_limit, "unknown component limit"),
        (_negative_limit, "must be non-negative"),
        (_source_limit_without_drive, "drive_peak_V"),
        (_empty_source_limits, "at least one limit"),
        (_unknown_source_limit, "unknown source limit"),
        (_negative_source_limit, "must be non-negative"),
        (_margin_without_tuning, "requires network.tuning"),
        (_invalid_margin, "between 0 and 1"),
    ],
)
def test_invalid_engineering_inputs_fail_at_the_compile_boundary(change, message):
    data = deepcopy(_base())
    change(data)
    with pytest.raises(ValueError, match=message):
        compile_rf_case(data, Path.cwd())


def test_impedance_table_has_one_canonical_shape(tmp_path):
    data = _base()
    data["load"] = {
        "type": "impedance_table",
        "file": "bad.csv",
        "reference_plane": "electrode",
    }
    (tmp_path / "bad.csv").write_text("scenario_id,R,X\nnominal,25,-80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical columns"):
        compile_rf_case(data, tmp_path)


def test_search_axes_reject_ambiguous_or_invalid_spaces():
    base = _base()
    base["network"]["fixed"].pop("C1")

    choices = deepcopy(base)
    choices["network"]["search"] = {"C1": {"values": [100e-12, 200e-12]}}
    assert compile_rf_case(choices, Path.cwd()).case["variables"]["C1"]["default"] == 100e-12

    invalid = [
        ({"values": [1], "range": [1, 2]}, "unsupported fields.*range"),
        ({}, "non-empty list"),
        ({"values": []}, "non-empty list"),
        ({"range": [1]}, "unsupported fields.*range"),
        ({"range": [2, 1]}, "unsupported fields.*range"),
        ({"range": [-1, 2], "scale": "log"}, "unsupported fields"),
    ]
    for spec, message in invalid:
        data = deepcopy(base)
        data["network"]["search"] = {"C1": spec}
        with pytest.raises(ValueError, match=message):
            compile_rf_case(data, Path.cwd())


def test_table_rejects_missing_empty_and_frequency_ambiguous_data(tmp_path):
    data = _base()
    data["load"] = {"type": "impedance_table", "file": "points.csv", "reference_plane": "electrode"}

    with pytest.raises(ValueError, match="not found"):
        compile_rf_case(data, tmp_path)

    table = tmp_path / "points.csv"
    table.write_text("scenario_id,resistance_ohm,reactance_ohm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        compile_rf_case(data, tmp_path)

    table.write_text("scenario_id,frequency_Hz,resistance_ohm,reactance_ohm\na,1e6,25,-80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already supplied"):
        compile_rf_case(data, tmp_path)

    data.pop("frequency_Hz")
    table.write_text("scenario_id,resistance_ohm,reactance_ohm\na,25,-80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required when the impedance table"):
        compile_rf_case(data, tmp_path)

    data["frequency_Hz"] = 13.56e6
    table.write_text("scenario_id,resistance_ohm,reactance_ohm\na,,-80\n", encoding="utf-8")
    with pytest.raises(ValueError, match="value is empty"):
        compile_rf_case(data, tmp_path)


def test_table_and_inline_conditions_are_not_implicitly_crossed():
    case = load_case(BENCH / "match_fixed_nominal.yaml")
    data = deepcopy(case.authored_data)
    data["conditions"] = [{"id": "hot", "drive_peak_V": 10}]
    with pytest.raises(ValueError, match="cannot be combined"):
        compile_rf_case(data, case.base_dir)


def test_table_rows_can_be_complete_load_frequency_and_drive_points(tmp_path):
    table = tmp_path / "operating_points.csv"
    table.write_text(
        "scenario_id,frequency_Hz,drive_peak_V,resistance_ohm,reactance_ohm\nproduction,13560000,100,25,-80\n",
        encoding="utf-8",
    )
    data = _base()
    data.pop("frequency_Hz")
    data["load"] = {
        "type": "impedance_table",
        "file": table.name,
        "reference_plane": "electrode",
    }
    data["acceptance"]["component_limits"] = {"L1": {"current_rms_A_max": 1}}
    plan = compile_rf_case(data, tmp_path)

    assert plan.case["source"]["amplitude_V"] == "drive_amplitude_V"
    values = plan.case["study"]["scenario_table"]["values"]
    assert values["drive_amplitude_V"] == "drive_peak_V"
    assert plan.case["variables"]["drive_amplitude_V"]["default"] == 100


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.pop("case_id"), "case_id is required"),
        (lambda data: data.__setitem__("network", None), "network must be a mapping"),
        (lambda data: data.__setitem__("load", []), "load must be a mapping"),
        (lambda data: data.__setitem__("conditions", []), "conditions must be a non-empty list"),
        (lambda data: data.__setitem__("conditions", [{"drive_peak_V": 10}]), "id is required"),
    ],
)
def test_required_public_sections_fail_with_a_local_error(change, message):
    data = _base()
    change(data)
    with pytest.raises(ValueError, match=message):
        compile_rf_case(data, Path.cwd())


def test_conditions_must_supply_frequency_and_drive_consistently():
    data = _base()
    data.pop("frequency_Hz")
    data["conditions"] = [{"id": "a", "frequency_Hz": 10e6, "drive_peak_V": 10}, {"id": "b", "drive_peak_V": 20}]
    with pytest.raises(ValueError, match="frequency_Hz"):
        compile_rf_case(data, Path.cwd())

    data["conditions"] = [
        {"id": "a", "frequency_Hz": 10e6, "drive_peak_V": 10},
        {"id": "b", "frequency_Hz": 20e6},
    ]
    with pytest.raises(ValueError, match="drive_peak_V"):
        compile_rf_case(data, Path.cwd())

    data["conditions"] = [
        {"id": "a", "frequency_Hz": 10e6, "drive_peak_V": 10},
        {"id": "b", "frequency_Hz": 20e6, "drive_peak_V": 20},
    ]
    plan = compile_rf_case(data, Path.cwd())
    assert plan.case["source"] == {
        "type": "sine_voltage",
        "name": "Vsrc",
        "p": "src",
        "n": "0",
        "amplitude_V": "drive_amplitude_V",
        "frequency_Hz": "rf_frequency_Hz",
    }
