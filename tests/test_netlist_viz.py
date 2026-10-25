"""A successful circuit drawing must retain its electrical connectivity."""

from pathlib import Path

import matplotlib
import pytest

from pcd.netlist_parse import parse_netlist
from pcd.netlist_viz import render_netlist_schematic


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "circuit.cir"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("extension", ["png", "svg", "pdf"])
def test_all_parallel_disconnected_sources_and_device_pins_are_represented(tmp_path, extension):
    netlist = _write(
        tmp_path,
        "Resonant circuit example\nV1 src 0 DC 1\nV2 aux 0 DC 2\nR1 src out 50\nRparallel src out 100\nRout out 0 50\nRaux aux 0 33\nQ1 out base 0 MOD\n.model MOD NPN\n.end\n",
    )
    output = tmp_path / f"circuit.{extension}"
    summary = render_netlist_schematic(netlist, output)
    assert summary["status"] == "complete"
    assert set(summary["represented_components"]) == {"V1", "V2", "R1", "Rparallel", "Rout", "Raux", "Q1"}
    assert "Resonant" not in summary["represented_components"]
    assert next(edge for edge in summary["edges"] if edge["ref"] == "Q1")["terminals"] == ["out", "base", "0"]
    assert output.stat().st_size > 1000
    assert matplotlib.get_backend().lower() == "agg"


def test_measurement_sources_are_explicitly_collapsed_or_shown_as_current_meters(tmp_path):
    netlist = _write(
        tmp_path,
        "* case\n.param R=1000\nVsrc src 0 SIN(0 1 1e6)\nVobserve_R1 src sense DC 0\nR1 sense out {R}\nC1 out 0 1n\n.end\n",
    )
    physical = render_netlist_schematic(netlist, tmp_path / "physical.svg")
    instrumented = render_netlist_schematic(netlist, tmp_path / "instrumented.svg", view="instrumented")
    assert physical["collapsed_measurement_sources"] == ["Vobserve_R1"]
    assert physical["node_aliases"]["src"] == physical["node_aliases"]["sense"]
    assert set(instrumented["represented_components"]) == {"Vsrc", "Vobserve_R1", "R1", "C1"}
    assert "current probe (0 V)" in (tmp_path / "instrumented.svg").read_text()
    assert "1 kohm" in (tmp_path / "physical.svg").read_text()


def test_unknown_topology_and_recursive_subcircuits_are_partial_not_successful(tmp_path):
    netlist = _write(
        tmp_path, "* unresolved\nVsrc src 0 DC 1\nYunknown src out custom\nXunknown out 0 not_defined\n.end\n"
    )
    summary = render_netlist_schematic(netlist, tmp_path / "partial.svg")
    assert summary["status"] == "partial"
    assert any("Yunknown" in issue for issue in summary["issues"])
    assert any("Xunknown" in issue for issue in summary["issues"])
    assert "PARTIAL DRAWING" in (tmp_path / "partial.svg").read_text()


def test_nested_subcircuits_and_magnetic_couplings_keep_their_ports(tmp_path):
    netlist = _write(
        tmp_path,
        "* nested\n.subckt coil p n\nL1 p mid 1u\nL2 mid n 2u\nK1 L1 L2 .9\n.ends\n.subckt unit a b\nXinner a b coil\n.ends\nV1 src 0 AC 1\nXouter src out unit\nR1 out 0 50\n.end\n",
    )
    parsed = parse_netlist(netlist)
    summary = render_netlist_schematic(netlist, tmp_path / "coupled.svg")
    assert summary["status"] == "complete"
    assert {"Xouter:Xinner:L1", "Xouter:Xinner:L2"} <= set(summary["represented_components"])
    assert parsed.couplings[0].inductors == ("Xouter:Xinner:L1", "Xouter:Xinner:L2")
    assert "Xouter:Xinner:K1" in (tmp_path / "coupled.svg").read_text()
