"""Prescribed component profiles stay inside the advanced circuit boundary."""

from __future__ import annotations

import json

import pytest

from pcd.netlist import build_circuit, build_load_subckt
from pcd.sim_core import archive_case_bundle
from pcd.study import _simulation_fingerprint


def _write_profile(tmp_path, text: str = "time_s,resistance_ohm\n0,25\n1e-6,100\n2e-6,25\n") -> None:
    (tmp_path / "resistance.csv").write_text(text, encoding="utf-8")


def _profile_value(**overrides):
    return {
        "profile": "resistance.csv",
        "time_column": "time_s",
        "value_column": "resistance_ohm",
        "interpolation": "linear",
        **overrides,
    }


def _profile_case(make_case, *, value=None, reference: str = "Rload", solver=None):
    return make_case(
        {
            "case_id": "profiled_resistor",
            "source": {"type": "dc_voltage", "name": "Vsrc", "voltage_V": 10},
            "circuit": {
                "builder": "from_yaml",
                "output_node": "out",
                "components": [{"ref": reference, "n1": "src", "n2": "out", "value": value or _profile_value()}],
            },
            "load": {"name": "none"},
            "solver": solver or {"tran": {"step_s": 1e-8, "stop_s": 3e-6}},
        }
    )


def test_linear_profile_becomes_a_bounded_behavioral_resistor(tmp_path, make_case):
    _write_profile(tmp_path)
    _name, circuit = build_circuit(_profile_case(make_case), {})

    line = circuit.components[0].to_spice()
    assert line.startswith("Rload src out R = '")
    assert "pwl(time, 0.0, 25.0, 1e-06, 100.0, 2e-06, 25.0)" in line
    assert line.endswith(": 25.0'")


def test_hold_and_repeat_have_explicit_piecewise_and_cycle_semantics(tmp_path, make_case):
    _write_profile(tmp_path)
    value = _profile_value(interpolation="hold", repeat_period_s=2e-6)
    _name, circuit = build_circuit(_profile_case(make_case, value=value), {})

    line = circuit.components[0].to_spice()
    assert "floor(time/2e-06)" in line
    assert "< 1e-06 ? 25.0" in line
    assert "< 2e-06 ? 100.0 : 25.0" in line


def test_profile_is_available_to_load_from_yaml_too(tmp_path, make_case):
    _write_profile(tmp_path)
    case = make_case(
        {
            "source": {"type": "dc_voltage", "name": "Vsrc", "voltage_V": 10},
            "load": {
                "name": "from_yaml",
                "components": [{"ref": "Rplasma", "n1": "p", "n2": "n", "value": _profile_value()}],
            },
            "solver": {"tran": {"step_s": 1e-8, "stop_s": 3e-6}},
        }
    )

    _name, subcircuit = build_load_subckt(case, {})
    assert "Rplasma p n R = '" in subcircuit


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("time_s,resistance_ohm\n1e-9,25\n1e-6,50\n", "start at time 0"),
        ("time_s,resistance_ohm\n0,25\n0,50\n", "strictly increasing"),
        ("time_s,resistance_ohm\n0,0\n1e-6,50\n", "must be positive"),
        ("time_s,value\n0,25\n1e-6,50\n", "missing columns"),
    ],
)
def test_profile_data_errors_are_rejected_by_the_component_owner(tmp_path, make_case, csv_text, message):
    _write_profile(tmp_path, csv_text)
    with pytest.raises(ValueError, match=message):
        build_circuit(_profile_case(make_case), {})


def test_only_resistance_profiles_are_accepted(tmp_path, make_case):
    _write_profile(tmp_path)
    with pytest.raises(ValueError, match=r"only resistance R\(t\)"):
        build_circuit(_profile_case(make_case, reference="Cload"), {})


def test_profile_requires_transient_resolution_no_coarser_than_its_samples(tmp_path, make_case):
    _write_profile(tmp_path)
    coarse = {"tran": {"step_s": 1.1e-6, "stop_s": 3e-6}}
    with pytest.raises(ValueError, match="must not exceed"):
        build_circuit(_profile_case(make_case, solver=coarse), {})

    ac_only = {"ac": {"frequency_Hz": 1e6}}
    with pytest.raises(ValueError, match=r"requires solver\.tran"):
        build_circuit(_profile_case(make_case, solver=ac_only), {})


def test_repeat_requires_an_explicit_continuous_cycle_boundary(tmp_path, make_case):
    _write_profile(tmp_path)
    wrong_period = _profile_value(repeat_period_s=3e-6)
    with pytest.raises(ValueError, match="final time must equal"):
        build_circuit(_profile_case(make_case, value=wrong_period), {})

    _write_profile(tmp_path, "time_s,resistance_ohm\n0,25\n1e-6,100\n2e-6,30\n")
    discontinuous = _profile_value(repeat_period_s=2e-6)
    with pytest.raises(ValueError, match="final resistance must equal"):
        build_circuit(_profile_case(make_case, value=discontinuous), {})


def test_profile_is_archived_for_replay_and_changes_simulation_identity(tmp_path, make_case):
    _write_profile(tmp_path)
    case = _profile_case(make_case)
    before = _simulation_fingerprint(case, "test_fake")
    snapshot, _files = archive_case_bundle(case, tmp_path / "bundle")
    manifest = json.loads((snapshot.base_dir / "input_manifest.json").read_text(encoding="utf-8"))

    profile_entry = next(item for item in manifest["inputs"] if item["field"].endswith(".value.profile"))
    archived = snapshot.data["circuit"]["components"][0]["value"]["profile"]
    assert archived == profile_entry["artifact"]
    (tmp_path / "resistance.csv").unlink()
    assert "Rload src out R = '" in build_circuit(snapshot, {})[1].components[0].to_spice()

    _write_profile(tmp_path, "time_s,resistance_ohm\n0,25\n1e-6,80\n2e-6,25\n")
    assert _simulation_fingerprint(case, "test_fake") != before
