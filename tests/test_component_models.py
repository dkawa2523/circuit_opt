from __future__ import annotations

import pytest

from pcd.analysis import ac_probe_plan, probe_plan
from pcd.case import default_params
from pcd.component_models import observed_components, series_resistance_ohm
from pcd.netlist import build_circuit, build_load_subckt, render_ngspice_netlist


def _observed_case(make_case):
    return make_case(
        {
            "case_id": "observed_lossy_component",
            "variables": {"L1": {"default": 1e-6}, "L1_DCR": {"default": 0.5}},
            "source": {
                "type": "sine_voltage",
                "name": "Vsrc",
                "p": "src",
                "n": "0",
                "amplitude_V": 100,
                "frequency_Hz": 1e6,
            },
            "circuit": {
                "builder": "from_yaml",
                "output_node": "load",
                "components": [
                    {
                        "ref": "L1",
                        "n1": "src",
                        "n2": "load",
                        "value": "L1",
                        "series_resistance_ohm": "L1_DCR",
                        "observe": True,
                    }
                ],
            },
            "load": {"name": "resistor", "ports": {"p": "load", "n": "0"}, "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": 1e6}},
            "target": {"objective": "impedance_match"},
        }
    )


def test_structured_component_expands_to_meter_loss_and_core(make_case):
    case = _observed_case(make_case)
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    netlist = render_ngspice_netlist(case, circuit, load, params)

    assert "Vobserve_L1 src observe_L1_meter DC 0" in netlist
    assert "Rloss_L1 observe_L1_meter observe_L1_core 0.5" in netlist
    assert "L1 observe_L1_core load {L1}" in netlist
    assert "Vsrc src 0 SIN(0 100 1000000 0 0 0) AC 100" in netlist
    assert "wrdata ac.csv v(src) i(Vsrc) v(load) v(src,load) i(Vobserve_L1) i(Vload_meter)" in netlist


def test_component_observation_names_are_shared_by_ac_and_transient(make_case):
    case = _observed_case(make_case)
    transient_vectors, transient_names = probe_plan(case)
    ac_vectors, ac_names = ac_probe_plan(case)

    expected = ["component_L1_voltage_V", "component_L1_current_A", "load_current_A"]
    assert ac_names == expected
    assert transient_names[:3] == expected
    assert ac_vectors == ["v(src,load)", "i(Vobserve_L1)", "i(Vload_meter)"]
    assert transient_vectors[-1] == "v(src)"


def test_named_user_probes_keep_friendly_artifact_columns(make_case):
    case = make_case(
        {
            "case_id": "named_probes",
            "source": {"type": "sine_voltage", "p": "src"},
            "measurement": {"probes": {"mid_voltage_V": "v(mid)", "branch_current_A": "i(Vsense)"}},
        }
    )
    vectors, names = probe_plan(case)
    assert vectors[:2] == ["v(mid)", "i(Vsense)"]
    assert names[:2] == ["mid_voltage_V", "branch_current_A"]


def test_negative_or_unresolved_series_resistance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        series_resistance_ohm({"ref": "C1", "series_resistance_ohm": -0.1})
    with pytest.raises(ValueError, match="resolve to a number"):
        series_resistance_ohm({"ref": "C1", "series_resistance_ohm": "missing"})


def test_raw_lines_cannot_claim_automatic_component_observation(make_case):
    case = make_case(
        {
            "case_id": "raw_observe",
            "circuit": {"builder": "from_yaml", "components": [{"raw": "L1 src out 1u", "observe": True}]},
        }
    )
    with pytest.raises(ValueError, match="raw circuit components cannot use observe"):
        observed_components(case)
    with pytest.raises(ValueError, match="raw circuit components cannot use observe"):
        build_circuit(case, {})
