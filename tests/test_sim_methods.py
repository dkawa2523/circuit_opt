"""pcd.sim_methods — the built-in circuits, load models, and solvers.

Every load model emits a ``load_model`` subcircuit that the netlist viewer can
expand. RF-specific models expose only effective electrical parameters.
"""

from __future__ import annotations

import pytest

from pcd.netlist import build_load_subckt
from pcd.sim_registry import available

# name -> (config, expected element lines in order)
LOAD_MODELS = {
    "resistor": (
        {"R_ohm": 50},
        ["Rload p n 50"],
    ),
    "impedance_point": (
        {"resistance_ohm": 20, "reactance_ohm": -80, "model_frequency_Hz": 13.56e6},
        ["Rpoint p nx 20", "Cpoint nx n"],
    ),
    "ccp_lumped": (
        {"R_eff_ohm": 20, "L_eff_H": 2e-7, "C_sheath_eq_F": 5e-11},
        ["Reffective p nb 20", "Leffective nb ns 2e-07", "Csheath_eq ns n 5e-11"],
    ),
    "icp_transformer": (
        {
            "R_coil_ohm": 0.4,
            "L_coil_H": 2e-6,
            "reflected_inductance_H": 2.45e-7,
            "secondary_damping_rate_rad_s": 2 / 3e-7,
            "C_parallel_F": 2e-11,
        },
        [
            "Rcoil p np 0.4",
            "Lcoil np n 2e-06",
            "Lsecondary n nr 2e-06",
            "Rsecondary nr n 13.3333333333",
            "Kload Lcoil Lsecondary 0.35",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(LOAD_MODELS))
def test_each_load_model_emits_its_subcircuit(make_case, name):
    config, expected = LOAD_MODELS[name]
    case = make_case({"case_id": name, "source": {"frequency_Hz": 13.56e6}, "load": {"name": name, **config}})
    selected, subckt = build_load_subckt(case, {})

    assert selected == name
    assert f"* load model: {name}" in subckt
    assert ".subckt load_model p n" in subckt.splitlines()
    assert subckt.splitlines()[-1] == ".ends load_model"
    for line in expected:
        assert line in subckt, f"{name} should emit {line!r}"


def test_impedance_point_rejects_a_drive_frequency_different_from_its_anchor(make_case):
    case = make_case(
        {
            "case_id": "mismatched_point",
            "source": {"frequency_Hz": 27.12e6},
            "load": {
                "name": "impedance_point",
                "resistance_ohm": 20,
                "reactance_ohm": -80,
                "model_frequency_Hz": 13.56e6,
            },
        }
    )

    with pytest.raises(ValueError, match="must equal"):
        build_load_subckt(case, {})


def test_zero_icp_coil_resistance_is_an_ideal_wire_not_a_zero_ohm_spice_resistor(make_case):
    case = make_case(
        {
            "case_id": "ideal_coil",
            "source": {"frequency_Hz": 13.56e6},
            "load": {
                "name": "icp_transformer",
                "R_coil_ohm": 0.0,
                "L_coil_H": 2e-6,
                "reflected_inductance_H": 2.5e-7,
                "secondary_damping_rate_rad_s": 8.52e5,
            },
        }
    )

    _selected, subckt = build_load_subckt(case, {})
    assert "Rcoil" not in subckt
    assert "Lcoil p n 2e-06" in subckt


def test_the_none_load_emits_nothing(make_case):
    case = make_case({"case_id": "bare", "load": {"name": "none"}})
    assert build_load_subckt(case, {}) == ("none", "")


def test_load_values_may_reference_design_variables(make_case):
    """`R_ohm: Rload` means "use the design variable named Rload"."""

    case = make_case({"case_id": "ref", "load": {"name": "resistor", "R_ohm": "Rload"}})
    _name, subckt = build_load_subckt(case, {"Rload": 75.0})
    assert "Rload p n 75" in subckt


def test_an_unset_load_value_becomes_a_netlist_parameter(make_case):
    """With no config and no param, the model falls back to a `.param` name."""

    case = make_case({"case_id": "unset", "load": {"name": "resistor"}})
    _name, subckt = build_load_subckt(case, {})
    assert "Rload p n {Rload}" in subckt


def test_a_load_built_from_yaml_components(make_case):
    case = make_case(
        {
            "case_id": "yaml_load",
            "load": {
                "name": "from_yaml",
                "components": [
                    {"ref": "Rx", "n1": "p", "n2": "mid", "value": 10},
                    {"raw": "Bx mid n I=V(mid,n)*1e-3"},
                ],
            },
        }
    )
    _name, subckt = build_load_subckt(case, {})
    assert "Rx p mid 10" in subckt
    assert "Bx mid n I=V(mid,n)*1e-3" in subckt


# --- circuit builders ------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "expected_refs"),
    [
        ("l_match", {"L1", "C1"}),
        ("pi_match", {"C1", "L1", "C2"}),
        ("pi_match_harmonic", {"C1", "L1", "C2", "Lh", "Ch"}),
    ],
)
def test_each_matching_topology_places_its_elements(make_case, builder, expected_refs):
    from pcd.netlist import build_circuit

    case = make_case({"case_id": builder, "circuit": {"builder": builder, "output_node": "electrode"}})
    name, circuit = build_circuit(case, {})

    assert name == builder
    assert {c.ref for c in circuit.components} == expected_refs
    assert circuit.output_node == "electrode"


def test_the_harmonic_topology_adds_a_shunt_trap(make_case):
    from pcd.netlist import build_circuit

    case = make_case({"case_id": "h", "circuit": {"builder": "pi_match_harmonic"}})
    _name, circuit = build_circuit(case, {})
    assert any("harmonic" in note for note in circuit.notes)


def test_every_registered_method_is_reachable_by_name():
    methods = available()
    assert set(methods["load"]) == {
        "none",
        "resistor",
        "from_yaml",
        "impedance_point",
        "ccp_lumped",
        "icp_transformer",
    }
    assert set(methods["circuit"]) >= {"from_yaml", "l_match", "pi_match", "pi_match_harmonic"}
    assert "ladder" not in methods["circuit"]


# --- an externally authored netlist ----------------------------------------


def test_an_external_netlist_is_used_verbatim(make_case):
    from pcd.netlist import build_circuit

    case = make_case({"case_id": "ext", "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"}})
    (case.base_dir / "c.cir").write_text(
        "* hand written\nR1 src out 50\nC1 out 0 1e-9\n.subckt blk p n\nRi p n 7\n.ends blk\nXb out 0 blk\n.end\n",
        encoding="utf-8",
    )
    _name, circuit = build_circuit(case, {})
    emitted = circuit.preamble
    assert "R1 src out 50" in emitted
    assert ".subckt blk p n" in emitted
    assert "Xb out 0 blk" in emitted


def test_a_case_source_replaces_the_one_in_the_netlist(make_case):
    """The default policy replaces only a source with the generated name."""

    from pcd.netlist import build_circuit

    case = make_case(
        {
            "case_id": "ext",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"},
        }
    )
    (case.base_dir / "c.cir").write_text("Vsrc src 0 DC 1\nR1 src out 50\n.end\n", encoding="utf-8")
    _name, circuit = build_circuit(case, {})
    assert "Vsrc src 0 DC 1" not in circuit.preamble
    assert any("replaced same-named source line" in note for note in circuit.notes)


def test_a_netlist_without_a_case_source_keeps_its_own(make_case):
    from pcd.netlist import build_circuit

    case = make_case({"case_id": "ext", "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"}})
    (case.base_dir / "c.cir").write_text("Vold src 0 DC 1\nR1 src out 50\n.end\n", encoding="utf-8")
    _name, circuit = build_circuit(case, {})
    assert "Vold src 0 DC 1" in circuit.preamble


def test_coupled_inductors_can_be_declared():
    from pcd.netlist import Circuit

    circuit = Circuit()
    circuit.add("Lp", "src", "0", 1e-6)
    circuit.add("Ls", "out", "0", 9e-6)
    circuit.couple("Kx", "Lp", "Ls", 0.95)
    assert circuit.components[-1].to_spice() == "Kx Lp Ls 0.95"


def test_only_a_conflicting_source_line_is_dropped(make_case):
    """A 0 V ammeter is a source line too; dropping it would delete a measurement."""

    from pcd.netlist import build_circuit

    case = make_case(
        {
            "case_id": "ammeter",
            "source": {"type": "sine_voltage", "name": "Vsrc", "p": "src", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"},
        }
    )
    (case.base_dir / "c.cir").write_text("R1 src mid 50\nVload mid out DC 0\nR2 out 0 50\n.end\n", encoding="utf-8")
    _name, circuit = build_circuit(case, {})
    spice = circuit.preamble
    assert any("Vload" in line for line in spice), "the ammeter must survive"
    assert not any("ignored conflicting source line" in note for note in circuit.notes)


def test_a_differently_named_source_is_not_inferred_to_be_a_conflict(make_case):
    from pcd.netlist import build_circuit

    case = make_case(
        {
            "case_id": "clash",
            "source": {"type": "sine_voltage", "name": "Vsrc", "p": "src", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"},
        }
    )
    (case.base_dir / "c.cir").write_text("Vother src 0 DC 5\nR1 src out 50\n.end\n", encoding="utf-8")
    _name, circuit = build_circuit(case, {})
    assert any("Vother" in line for line in circuit.preamble)
    assert not circuit.notes


def test_preserve_source_policy_keeps_even_a_same_named_source(make_case):
    from pcd.netlist import build_circuit

    case = make_case(
        {
            "case_id": "preserve",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {
                "builder": "from_netlist",
                "netlist_file": "c.cir",
                "source_policy": "preserve",
            },
        }
    )
    (case.base_dir / "c.cir").write_text("Vsrc isolated 0 DC 5\nR1 src out 50\n", encoding="utf-8")

    _name, circuit = build_circuit(case, {})

    assert "Vsrc isolated 0 DC 5" in circuit.preamble


def test_external_netlist_keeps_semantic_directives_but_not_execution_control(make_case):
    from pcd.netlist import build_circuit

    case = make_case({"case_id": "modelled", "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"}})
    include = case.base_dir / "models.lib"
    include.write_text(".model local_diode D(Is=1e-12)\n", encoding="utf-8")
    (case.base_dir / "c.cir").write_text(
        ".param gain=2\n"
        ".model inline_diode D(Is=2e-12)\n"
        '.include "models.lib"\n'
        "D1 src out inline_diode\n"
        ".control\nrun\n.endc\n"
        ".tran 1n 1u\n.end\n",
        encoding="utf-8",
    )

    _name, circuit = build_circuit(case, {})
    assert ".param gain=2" in circuit.preamble
    assert ".model inline_diode D(Is=2e-12)" in circuit.preamble
    assert ".model local_diode D(Is=1e-12)" in circuit.preamble
    assert not any(line.lower().startswith(".include") for line in circuit.preamble)
    assert "D1 src out inline_diode" in circuit.preamble
    assert not any(line.lower().startswith((".control", ".tran", ".end")) for line in circuit.preamble)


def test_execution_import_handles_common_include_and_library_forms(tmp_path):
    from pcd.netlist_import import executable_netlist_lines, flatten_netlist_file

    library = tmp_path / "models.lib"
    library.write_text(".lib typical\n.model d D\n.endl typical\n", encoding="utf-8")
    absolute = library.resolve().as_posix()

    assert executable_netlist_lines("\n.model d D\n")[0].text == ".model d D"
    assert executable_netlist_lines(".include relative.lib")[0].text == ".include relative.lib"
    assert executable_netlist_lines(".include", tmp_path)[0].text == ".include"
    assert executable_netlist_lines('.include ""', tmp_path)[0].text == '.include ""'
    assert executable_netlist_lines(".lib typical", tmp_path)[0].text == ".lib typical"
    assert executable_netlist_lines(f'.include "{absolute}"', tmp_path)[0].text == f'.include "{absolute}"'
    assert executable_netlist_lines(".lib models.lib typical", tmp_path)[0].text == f'.lib "{absolute}" typical'
    unterminated = executable_netlist_lines(".include 'models.lib", tmp_path)[0].text
    assert unterminated == f'.include "{absolute}"'

    deck = tmp_path / "deck.cir"
    deck.write_text(".lib models.lib typical\nR1 src out 1\n.end\n", encoding="utf-8")
    flattened, dependencies = flatten_netlist_file(deck)
    assert ".model d D" in flattened
    assert ".lib models.lib typical" not in flattened
    assert dependencies == (library.resolve(),)


def test_execution_import_distinguishes_complete_decks_from_fragments():
    from pcd.netlist_import import executable_netlist_lines

    text = "Exported circuit title\nR1 src out 50\n.end\n"
    assert [item.text for item in executable_netlist_lines(text, mode="deck")] == ["R1 src out 50"]
    assert executable_netlist_lines("R1 src out 50\n", mode="fragment")[0].text == "R1 src out 50"

    with pytest.raises(ValueError, match="netlist_mode"):
        executable_netlist_lines(text, mode="guess")


def test_archived_external_netlist_is_self_contained_with_included_models(tmp_path, make_case):
    import json

    from pcd.netlist import build_circuit
    from pcd.sim_core import archive_case_bundle

    case = make_case({"case_id": "portable", "circuit": {"builder": "from_netlist", "netlist_file": "c.cir"}})
    include = case.base_dir / "models.inc"
    include.write_text(".model bundled D(Is=3e-12)\n", encoding="utf-8")
    (case.base_dir / "c.cir").write_text('.include "models.inc"\nD1 src out bundled\n.end\n', encoding="utf-8")

    snapshot, files = archive_case_bundle(case, tmp_path / "bundle")
    include.unlink()
    _name, circuit = build_circuit(snapshot, {})

    assert ".model bundled D(Is=3e-12)" in circuit.preamble
    assert files["imported_netlist"] == "imported_netlist.cir"
    manifest = json.loads((snapshot.base_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["inputs"]) == 2
    assert all(item["status"] == "archived" for item in manifest["inputs"])
