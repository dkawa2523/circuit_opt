"""pcd.netlist_parse — reading a SPICE netlist back into a graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcd.netlist_parse import netlist_summary, node_sort_key, parse_netlist


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "netlist.cir"
    path.write_text(body, encoding="utf-8")
    return path


def test_control_blocks_and_directives_are_ignored(tmp_path):
    netlist = _write(
        tmp_path,
        "* comment\n.param R1=50\n.options gmin=1e-12\n"
        "Vsrc src 0 SIN(0 1 1e6 0 0 0)\nR1 src out {R1}\nC1 out 0 1e-9\n"
        ".save v(out)\n.control\ntran 1n 1u\n.endc\n.end\n",
    )
    parsed = parse_netlist(netlist)
    assert {e.ref for e in parsed.edges} == {"Vsrc", "R1", "C1"}
    assert {e.kind for e in parsed.edges} == {"voltage", "resistor", "capacitor"}


def test_a_netlist_can_be_passed_as_text_instead_of_a_path():
    parsed = parse_netlist("Vsrc src 0 DC 1\nR1 src out 50\n.end\n")
    assert {e.ref for e in parsed.edges} == {"Vsrc", "R1"}


def test_unknown_component_prefixes_are_skipped(tmp_path):
    netlist = _write(tmp_path, "Vsrc src 0 DC 1\nQ1 c b e npn\nR1 src out 50\n.end\n")
    assert {e.ref for e in parse_netlist(netlist).edges} == {"Vsrc", "R1"}


def test_truncated_lines_are_skipped(tmp_path):
    netlist = _write(tmp_path, "R1 src\nVsrc src 0 DC 1\n.end\n")
    assert {e.ref for e in parse_netlist(netlist).edges} == {"Vsrc"}


def test_subcircuits_are_expanded_so_load_internals_stay_visible(tmp_path):
    netlist = _write(
        tmp_path,
        "Vsrc src 0 SIN(0 1 1e6 0 0 0)\nR1 src out 50\n"
        ".subckt load_model p n\nRbulk p nb 20\nLbulk nb ns 2e-7\nCsh ns n 5e-11\n.ends load_model\n"
        "Xload out 0 load_model\n.end\n",
    )
    summary = netlist_summary(netlist)
    assert summary["subckts"] == ["load_model"]
    refs = {edge["ref"] for edge in summary["edges"]}
    assert {"Xload:Rbulk", "Xload:Lbulk", "Xload:Csh"} <= refs
    # Internal nodes are namespaced so two instances cannot collide.
    assert any(edge["n2"].startswith("Xload.") for edge in summary["edges"])


def test_subcircuit_ports_map_onto_the_callers_nets(tmp_path):
    netlist = _write(
        tmp_path,
        "Vsrc src 0 DC 1\n.subckt load_model p n\nRbulk p n 20\n.ends load_model\nXload out 0 load_model\n.end\n",
    )
    edge = next(e for e in parse_netlist(netlist).edges if e.ref == "Xload:Rbulk")
    assert (edge.n1, edge.n2) == ("out", "0")


def test_an_unresolved_subcircuit_becomes_a_single_edge(tmp_path):
    netlist = _write(tmp_path, "Vsrc src 0 DC 1\nXmystery src 0 not_defined\n.end\n")
    assert "subckt" in {e.kind for e in parse_netlist(netlist).edges}


def test_the_summary_counts_components_by_kind(tmp_path):
    netlist = _write(tmp_path, "Vsrc src 0 DC 1\nR1 src a 50\nR2 a out 50\nC1 out 0 1e-9\n.end\n")
    summary = netlist_summary(netlist)
    assert summary["component_counts"] == {"voltage": 1, "resistor": 2, "capacitor": 1}
    assert summary["n_edges"] == 4
    assert summary["n_nodes"] == 4


def test_an_empty_netlist_parses_to_nothing(tmp_path):
    parsed = parse_netlist(_write(tmp_path, "* nothing here\n.end\n"))
    assert parsed.edges == []
    assert parsed.nodes == []


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [(["0", "out", "src"], ["src", "out", "0"]), (["b", "a", "src"], ["src", "a", "b"])],
)
def test_nodes_are_ordered_source_first_ground_last(nodes, expected):
    assert sorted(nodes, key=node_sort_key) == expected


def test_coupled_inductors_are_parsed_as_couplings_not_branches(tmp_path):
    """A `K` line is a relationship between two inductors, not an element."""

    netlist = _write(tmp_path, "Lp src 0 1e-6\nLs out 0 9e-6\nKx Lp Ls 0.95\n.end\n")
    parsed = parse_netlist(netlist)
    assert {e.ref for e in parsed.edges} == {"Lp", "Ls"}
    assert len(parsed.couplings) == 1
    coupling = parsed.couplings[0]
    assert (coupling.ref, coupling.inductors, coupling.coefficient) == ("Kx", ("Lp", "Ls"), 0.95)


def test_a_transmission_line_is_drawn_along_its_signal_path(tmp_path):
    netlist = _write(tmp_path, "Vsrc src 0 DC 1\nT1 src 0 out 0 Z0=50 TD=1n\n.end\n")
    edge = next(e for e in parse_netlist(netlist).edges if e.ref == "T1")
    assert (edge.n1, edge.n2, edge.kind) == ("src", "out", "tline")


def test_couplings_appear_in_the_summary(tmp_path):
    netlist = _write(tmp_path, "Lp src 0 1e-6\nLs out 0 9e-6\nKx Lp Ls 0.9\n.end\n")
    assert netlist_summary(netlist)["couplings"][0]["ref"] == "Kx"
