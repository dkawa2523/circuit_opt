"""User-written circuits keep physical roles, units, data, and execution scope."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from pcd.api import describe_case, simulate_case
from pcd.case import load_case
from pcd.cli import main
from pcd.core import Candidate
from pcd.study import run_case_study
from pcd.study_config import CaseControlPolicy, select_simulation_inputs, study_spec_from_case
from pcd.validation import validate_case

ROOT = Path(__file__).resolve().parents[1]
NGSPICE = shutil.which("ngspice_con.exe") or shutil.which("ngspice")


def _circuit() -> dict[str, Any]:
    return {
        "schema": "pcd.rf.v1",
        "case_id": "simple",
        "parameters": {"fixed": {"R": "50 ohm"}},
        "source": {"type": "dc_voltage", "voltage_V": 1},
        "circuit": {"output_node": "src", "components": [{"ref": "R1", "n1": "src", "n2": 0, "value": "R"}]},
        "analysis": {"ac": {"frequency_Hz": "1 MHz"}},
    }


@pytest.mark.parametrize(
    "example",
    ["circuit_external.yaml", "circuit_transient.yaml", "circuit_design.yaml", "terminal_identification.yaml"],
)
def test_public_examples_validate_without_internal_configuration_or_warnings(example):
    case = load_case(ROOT / "examples" / example)
    report = validate_case(case, strict=True)
    assert report.ok, report.format_text()
    assert not report.issues


def test_fixed_roles_and_dimensioned_values_are_resolved_once(make_case):
    authored = _circuit()
    authored["parameters"]["fixed"]["C"] = {"default": "100 pF", "unit": "F"}
    case = make_case(authored)
    assert case.data["variables"]["C"] == {"default": 1e-10, "unit": "F", "role": "fixed"}
    assert case.data["solver"]["ac"]["frequency_Hz"] == 1e6
    assert describe_case(case)["workflow"] == "simulation"
    with pytest.raises(ValueError, match="forward simulation"):
        run_case_study(case, "unused")


@pytest.mark.parametrize(
    "spec", [{"default": 10, "bounds": [9, 11]}, {"default": 100, "unit": "pF"}, {"default": "100 V", "unit": "F"}]
)
def test_fixed_tolerances_and_ambiguous_units_cannot_change_a_model(make_case, spec):
    authored = _circuit()
    authored["parameters"]["fixed"]["R"] = spec
    with pytest.raises(ValueError, match=r"fixed per condition|SI metadata|expected F"):
        make_case(authored)


def test_imported_source_is_observed_without_a_false_missing_source_warning():
    case = load_case(ROOT / "examples/circuit_external.yaml")
    from pcd.simulation_input import resolve_simulation_case

    probes = resolve_simulation_case(case).probes
    assert probes.source_name == "V1"
    assert probes.source_voltage_vector == "v(in)"
    assert validate_case(case, strict=True).ok


def test_multiple_sources_require_an_explicit_observed_port(make_case, tmp_path):
    (tmp_path / "multi.cir").write_text("two sources\nV1 a 0 AC 1\nV2 b 0 AC 1\nR1 a 0 50\nR2 b 0 50\n.end\n")
    case = make_case(
        {
            "schema": "pcd.rf.v1",
            "case_id": "multi",
            "circuit": {"file": "multi.cir", "output_node": "a"},
            "analysis": {"ac": {"frequency_Hz": 1e6}},
        }
    )
    assert not validate_case(case).ok


@pytest.mark.parametrize("conditional", [False, True])
def test_preserved_or_conditional_sources_are_not_silently_replaced_or_guessed(make_case, tmp_path, conditional):
    deck = "title\nV1 src 0 AC 1\nR1 src 0 50\n.end\n"
    if conditional:
        deck = "title\n.param state=1\n.if state\nV1 a 0 AC 1\n.else\nV1 b 0 AC 1\n.endif\nR1 a 0 50\nR2 b 0 50\n.end\n"
    (tmp_path / "sources.cir").write_text(deck)
    data: dict[str, Any] = {
        "schema": "pcd.rf.v1",
        "case_id": "sources",
        "circuit": {"file": "sources.cir", "output_node": "src"},
        "analysis": {"ac": {"frequency_Hz": 1e6}},
    }
    if not conditional:
        data["source"] = {"type": "dc_voltage", "name": "V1", "voltage_V": 1}
    report = validate_case(make_case(data))
    assert not report.ok
    assert any(
        "source_voltage" in issue.message if conditional else "preserved netlist" in issue.message
        for issue in report.issues
    )


def test_an_imported_branch_meter_can_have_an_explicit_source_plane(make_case, tmp_path):
    (tmp_path / "meter.cir").write_text("title\nV1 in 0 AC 1\nVmeter mid in DC 0\nR1 mid 0 50\n.end\n")
    case = make_case(
        {
            "schema": "pcd.rf.v1",
            "case_id": "meter",
            "circuit": {"file": "meter.cir", "output_node": "mid"},
            "measurement": {"current_source": "Vmeter", "source_voltage": "v(in)"},
            "analysis": {"ac": {"frequency_Hz": 1e6}},
        }
    )
    assert validate_case(case, strict=True).ok


def test_a_study_is_not_silently_reduced_to_one_default_point(tmp_path, capsys):
    path = ROOT / "examples/rf_ccp_lumped.yaml"
    with pytest.raises(ValueError, match="multiple conditions"):
        simulate_case(load_case(path), run_root=tmp_path / "unused")
    with pytest.raises(SystemExit) as error:
        main(["sim-run", str(path), "--run-root", str(tmp_path / "unused"), "--json"])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
    assert not (tmp_path / "unused").exists()
    point = select_simulation_inputs(load_case(path), {"L1": 1e-6, "C1": 2e-10}, scenario_id="nominal_frequency")
    assert point["rf_frequency_Hz"] == 13560000


def test_csv_and_inline_control_conditions_resolve_to_the_same_settings(make_case, tmp_path):
    authored = _circuit()
    authored["parameters"] = {"control": {"R": {"choices": [25, 50]}}}
    authored["conditions"] = [{"R": 25}, {"R": 50}]
    inline = make_case(authored, "inline.yaml")
    (tmp_path / "controls.csv").write_text("R\n25\n50\n")
    authored["conditions"] = {"file": "controls.csv"}
    table = make_case(authored, "table.yaml")
    resolved = []
    for case in (inline, table):
        report = validate_case(case)
        assert report.ok, report.format_text()
        spec = study_spec_from_case(case)
        policy = CaseControlPolicy(case)
        resolved.append(
            [
                (row.scenario_id, dict(policy.controls(spec, Candidate("fixed"), row)[0].values))
                for row in spec.scenarios
            ]
        )
    assert resolved == [[("point_1", {"R": 25}), ("point_2", {"R": 50})]] * 2


def test_inline_and_csv_impedance_points_have_identical_scenarios_and_auto_ids(make_case, tmp_path):
    data = {
        "schema": "pcd.rf.v1",
        "case_id": "points",
        "frequency_Hz": "1 MHz",
        "reference_impedance_ohm": 75,
        "network": {"type": "l_match", "fixed": {"L1": "100 nH", "C1": "100 pF"}},
        "load": {
            "type": "impedance_table",
            "reference_plane": "terminal",
            "points": [{"resistance_ohm": 25, "reactance_ohm": -80}, {"resistance_ohm": 50, "reactance_ohm": -40}],
        },
        "acceptance": {"reflected_power_fraction_max": 0.1},
    }
    inline = make_case(data, "inline.yaml")
    (tmp_path / "points.csv").write_text("resistance_ohm,reactance_ohm\n25,-80\n50,-40\n")
    data["load"].pop("points")
    data["load"]["file"] = "points.csv"
    table = make_case(data, "table.yaml")
    assert [row.to_dict() for row in study_spec_from_case(inline).scenarios] == [
        row.to_dict() for row in study_spec_from_case(table).scenarios
    ]
    assert inline.data["measurement"]["reference_impedance_ohm"] == 75


def test_identification_observations_inline_or_csv_need_no_target_variable_mapping(make_case):
    original = load_case(ROOT / "examples/terminal_identification.yaml")
    raw = deepcopy(original.authored_data)
    import csv

    with (ROOT / "examples/terminal_observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    raw["identification"]["observations"] = rows
    inline = make_case(raw)
    assert inline.data["identification"] == original.data["identification"]
    assert [row.to_dict() for row in study_spec_from_case(inline).scenarios] == [
        row.to_dict() for row in study_spec_from_case(original).scenarios
    ]
    assert "expected" not in original.data["identification"]


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_imported_public_circuit_and_explicit_port_measurements_match_algebra(tmp_path):
    from pcd.reporting import analyze_run

    case = load_case(ROOT / "examples/circuit_external.yaml")
    run = simulate_case(case, run_root=tmp_path)
    analysis = analyze_run(run.run_dir)
    assert run.status == "ok"
    assert run.summary()["scope"] == "single_resolved_parameter_point"
    assert analysis["analyses"]["ac"]["resistance_ohm"] == pytest.approx(100)
    assert analysis["analyses"]["ac"]["source_real_power_W"] == pytest.approx(0.005)
