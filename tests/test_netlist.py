"""pcd.netlist — turning a case into ngspice netlist text.

Nothing here runs a solver; these tests read the generated text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcd.case import default_params, load_case
from pcd.netlist import (
    SOURCE_RENDERERS,
    Circuit,
    Component,
    build_circuit,
    build_load_subckt,
    render_ngspice_netlist,
    render_source,
    select_circuit_name,
    select_load_name,
)

EX = Path(__file__).resolve().parents[1] / "examples"


# --- the Circuit model -----------------------------------------------------


def test_a_circuit_reports_its_nodes_and_problems():
    circuit = Circuit(output_node="out")
    circuit.add("R1", "src", "out", 50)
    circuit.add("R1", "out", "0", 50)  # duplicate reference
    assert circuit.nodes() == {"0", "src", "out"}
    assert "duplicate component reference names detected" in circuit.warnings()


def test_an_output_node_with_no_component_is_flagged():
    circuit = Circuit(output_node="electrode")
    circuit.add("R1", "src", "out", 50)
    assert any("electrode" in w for w in circuit.warnings())


def test_a_raw_line_is_emitted_verbatim():
    circuit = Circuit()
    circuit.raw("Bsrc out 0 V=1")
    assert circuit.components[0].to_spice() == "Bsrc out 0 V=1"


def test_an_incomplete_component_is_rejected():
    with pytest.raises(ValueError, match="invalid component"):
        Component(ref="R1", n1="src").to_spice()


# --- choosing methods ------------------------------------------------------


def test_topology_and_load_can_be_chosen_by_a_design_variable(topology_case):
    """`builder: $topology_choice` is what makes topology a categorical variable."""

    params = {"topology_choice": "l_match", "load_model": "impedance_point"}
    assert select_circuit_name(topology_case, params) == "l_match"
    assert select_load_name(topology_case, params) == "impedance_point"


def test_a_fixed_builder_name_is_used_as_is(rc_case):
    assert select_circuit_name(rc_case, {}) == "from_yaml"
    assert select_load_name(rc_case, {}) == "none"


def test_a_case_without_a_load_section_selects_none(make_case):
    assert select_load_name(make_case({"case_id": "x"}), {}) == "none"


def test_a_builder_must_return_a_circuit(make_case):
    from pcd.sim_registry import register

    @register("circuit", "not_a_circuit")
    def broken(case, params):
        return "nope"

    case = make_case({"case_id": "bad", "circuit": {"builder": "not_a_circuit"}})
    with pytest.raises(TypeError, match="must return Circuit"):
        build_circuit(case, {})


# --- sources ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"type": "sine_voltage", "amplitude_V": 600, "frequency_Hz": 13.56e6}, "SIN(0 600 13560000 0 0 0)"),
        ({"type": "dc_voltage", "voltage_V": 5}, "DC 5"),
        ({"type": "current_dc", "current_A": 2}, "DC 2"),
        ({"type": "pulse", "v1_V": 0, "v2_V": 5}, "PULSE(0 5 0 1e-09 1e-09 1e-06 2e-06)"),
    ],
)
def test_each_source_type_renders_its_spice_form(make_case, source, expected):
    case = make_case({"case_id": "src", "source": {"name": "Vsrc", "p": "src", "n": "0", **source}})
    assert render_source(case, {})[0] == f"Vsrc src 0 {expected}"


def test_a_source_field_may_reference_a_design_variable(make_case):
    case = make_case(
        {
            "case_id": "ref",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": "Vamp", "frequency_Hz": 1e6},
        }
    )
    assert "SIN(0 600 1000000 0 0 0)" in render_source(case, {"Vamp": 600})[0]


def test_a_raw_source_line_bypasses_rendering(make_case):
    case = make_case({"case_id": "raw", "source": {"raw": "Vsrc src 0 EXP(0 1 1n 2n)"}})
    assert render_source(case, {}) == ["Vsrc src 0 EXP(0 1 1n 2n)"]


def test_an_unknown_source_type_lists_the_available_ones(make_case):
    case = make_case({"case_id": "bad", "source": {"type": "tesla_coil"}})
    with pytest.raises(ValueError, match="unknown source type: tesla_coil"):
        render_source(case, {})


def test_multiple_sources_are_all_rendered(make_case):
    case = make_case(
        {
            "case_id": "multi",
            "sources": [
                {"type": "dc_voltage", "name": "V1", "voltage_V": 1},
                {"type": "dc_voltage", "name": "V2", "voltage_V": 2},
            ],
        }
    )
    assert len(render_source(case, {})) == 2


def test_every_registered_source_type_is_callable():
    assert {"sine_voltage", "dc_voltage", "pulse", "current_dc"} <= set(SOURCE_RENDERERS)
    assert all(callable(fn) for fn in SOURCE_RENDERERS.values())


# --- full netlist ----------------------------------------------------------


def test_a_case_without_a_load_emits_no_subcircuit(rc_case):
    params = default_params(rc_case)
    _, circuit = build_circuit(rc_case, params)
    load_name, subckt = build_load_subckt(rc_case, params)
    text = render_ngspice_netlist(rc_case, circuit, subckt, params)

    assert load_name == "none"
    assert "Xload" not in text
    assert "R1 src out {R1}" in text
    assert "C1 out 0 {C1}" in text


def test_the_netlist_carries_params_and_the_transient_block(rc_case):
    params = default_params(rc_case)
    _, circuit = build_circuit(rc_case, params)
    text = render_ngspice_netlist(rc_case, circuit, "", params)

    assert ".param R1=1000" in text
    assert ".control" in text
    assert "tran 2e-09 2e-06" in text
    assert "wrdata waveform.csv time v(out) i(Vsrc)" in text
    assert text.rstrip().endswith(".end")


def _render(case):
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, subckt = build_load_subckt(case, params)
    return render_ngspice_netlist(case, circuit, subckt, params)


def test_a_load_subcircuit_is_instantiated_at_its_ports(topology_case):
    topology_case.data["measurement"].pop("load_current", None)
    assert "Xload electrode 0 load_model" in _render(topology_case)


def test_auto_metering_puts_an_ammeter_in_series_with_the_load(topology_case):
    """A zero-volt source changes no voltage and makes the load current readable."""

    topology_case.data["measurement"]["load_current"] = "auto"
    text = _render(topology_case)

    assert "Vload_meter electrode electrode_metered DC 0" in text
    assert "Xload electrode_metered 0 load_model" in text
    # The netlist writer and the result parser must agree it was recorded.
    assert "wrdata waveform.csv time v(electrode) i(Vsrc) i(Vload_meter) v(src)" in text


def test_without_auto_metering_no_ammeter_appears(topology_case):
    topology_case.data["measurement"].pop("load_current", None)
    text = _render(topology_case)
    assert "Vload_meter" not in text


def test_an_impedance_point_emits_the_single_frequency_equivalent():
    case = load_case(EX / "rf_impedance_point_study.yaml")
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, subckt = build_load_subckt(case, params)
    text = render_ngspice_netlist(case, circuit, subckt, params)

    assert "load model: impedance_point" in text
    assert "exact at 13560000 Hz" in text
    assert "Rpoint p nx" in text
    assert "Cpoint nx n" in text
    assert "tran " not in text
    assert "wrdata waveform.csv" not in text
    assert "wrdata ac.csv" in text


def test_one_frequency_drives_source_load_and_ac_from_the_same_scenario_value(make_case):
    case = make_case(
        {
            "case_id": "measured_frequency_point",
            "variables": {
                "rf_frequency_Hz": {"default": 13.56e6},
                "load_R": {"default": 20.0},
                "load_X": {"default": -80.0},
            },
            "source": {
                "type": "sine_voltage",
                "name": "Vsrc",
                "p": "src",
                "n": "0",
                "amplitude_V": 1,
                "frequency_Hz": "rf_frequency_Hz",
            },
            "circuit": {
                "builder": "from_yaml",
                "output_node": "port",
                "components": [{"raw": "Rfixture src port 1e-9"}],
            },
            "load": {
                "name": "impedance_point",
                "ports": {"p": "port", "n": "0"},
                "reference_plane": "port",
                "characterization": {"origin": "measured"},
                "resistance_ohm": "load_R",
                "reactance_ohm": "load_X",
                "model_frequency_Hz": "rf_frequency_Hz",
            },
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": "rf_frequency_Hz"}},
        }
    )
    params = {"rf_frequency_Hz": 27.12e6, "load_R": 32.0, "load_X": 45.0}
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    text = render_ngspice_netlist(case, circuit, load, params)

    assert "SIN(0 1 27120000" in text
    assert "exact at 27120000 Hz" in text
    assert "Rpoint p nx 32" in text
    assert "Lpoint nx n" in text
    assert "ac lin 1 2.712e+07 2.712e+07" in text


def test_the_default_load_voltage_is_differential_across_declared_ports(make_case):
    case = make_case(
        {
            "case_id": "differential",
            "source": {"type": "sine_voltage", "name": "Vsrc", "p": "src", "n": "0", "frequency_Hz": 1e6},
            "circuit": {
                "builder": "from_yaml",
                "output_node": "pos",
                "components": [{"raw": "Rwire src pos 1"}, {"raw": "Rreturn neg 0 1"}],
            },
            "load": {"name": "resistor", "R_ohm": 50, "ports": {"p": "pos", "n": "neg"}},
        }
    )
    assert "wrdata waveform.csv time v(pos,neg) i(Vsrc)" in _render(case)
