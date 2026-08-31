"""pcd.netlist_viz — drawing a parsed netlist as a schematic PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from pcd.case import default_params, load_case
from pcd.netlist import build_circuit, build_load_subckt, render_ngspice_netlist
from pcd.netlist_parse import netlist_summary, parse_netlist
from pcd.netlist_viz import _main_signal_path, _series_shunt_pairs, render_netlist_schematic

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "netlist.cir"
    path.write_text(body, encoding="utf-8")
    return path


# --- layout ----------------------------------------------------------------


def test_the_main_path_follows_the_signal_from_the_source(tmp_path):
    parsed = parse_netlist(_write(tmp_path, "Vsrc src 0 DC 1\nR1 src mid 50\nL1 mid out 1e-6\nC1 out 0 1e-9\n.end\n"))
    assert _main_signal_path([n for n in parsed.nodes if n != "0"], parsed.edges) == ["src", "mid", "out"]


# A longer main chain plus a two-element branch hanging off node `a`.  The
# branch only stays off the main path because src-a-b-out is longer than
# src-a-h; that is exactly when a series+shunt pair needs collapsing.
_BRANCHED = "Vsrc src 0 DC 1\nR1 src a 50\nR2 a b 50\nR3 b out 50\nLh a h 1e-7\nCh h 0 1e-10\n.end\n"


def test_a_series_element_feeding_one_grounded_element_is_collapsed(tmp_path):
    """The pair must draw as one branch, not two unrelated stubs."""

    parsed = parse_netlist(_write(tmp_path, _BRANCHED))
    main = _main_signal_path([n for n in parsed.nodes if n != "0"], parsed.edges)
    assert main == ["src", "a", "b", "out"]

    pairs = _series_shunt_pairs(parsed.edges, set(main))
    assert [(edge.ref, ground.ref, base, branch) for edge, ground, base, branch in pairs] == [("Lh", "Ch", "a", "h")]


def test_a_branch_with_two_grounded_partners_is_not_collapsed(tmp_path):
    """Ambiguous branches are left to the ordinary shunt drawing."""

    parsed = parse_netlist(_write(tmp_path, _BRANCHED.replace(".end", "Rh h 0 1e3\n.end")))
    main = _main_signal_path([n for n in parsed.nodes if n != "0"], parsed.edges)
    assert _series_shunt_pairs(parsed.edges, set(main)) == []


def test_a_collapsed_branch_renders(tmp_path):
    out = tmp_path / "branched.png"
    render_netlist_schematic(_write(tmp_path, _BRANCHED), out)
    assert out.stat().st_size > 0


# --- rendering -------------------------------------------------------------


def test_rendering_produces_a_schematic_and_a_summary(tmp_path):
    case = load_case(ROOT / "examples" / "rf_ccp_lumped.yaml")
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    netlist = _write(tmp_path, render_ngspice_netlist(case, circuit, load, params))

    out = tmp_path / "level1.png"
    render_netlist_schematic(netlist, out, title="Level 1")
    summary = netlist_summary(netlist)

    assert out.stat().st_size > 0
    assert summary["n_nodes"] >= 5
    assert summary["component_counts"]["inductor"] >= 2
    assert any(edge["ref"] == "Xload:Reffective" for edge in summary["edges"])


def test_a_diode_renders_through_the_generic_element(tmp_path):
    """schemdraw has no `Box`; unknown two-terminal parts use RBox."""

    netlist = _write(tmp_path, "Vsrc src 0 SIN(0 1 1e6 0 0 0)\nR1 src out 50\nD1 out mid DMOD\nC1 mid 0 1e-9\n.end\n")
    out = tmp_path / "diode.png"
    render_netlist_schematic(netlist, out)
    assert out.stat().st_size > 0


def test_a_netlist_with_no_drawable_path_still_produces_a_file(tmp_path):
    out = tmp_path / "empty.png"
    render_netlist_schematic(_write(tmp_path, "* nothing drawable\n.end\n"), out)
    assert out.stat().st_size > 0


def test_a_netlist_with_no_source_still_renders(tmp_path):
    out = tmp_path / "no_source.png"
    render_netlist_schematic(_write(tmp_path, "R1 a b 50\nC1 b 0 1e-9\n.end\n"), out)
    assert out.stat().st_size > 0


def test_rendering_never_requires_a_display(tmp_path):
    """Rendering writes a file; an interactive backend fails on a CI runner."""

    netlist = _write(tmp_path, "Vsrc src 0 SIN(0 1 1e6 0 0 0)\nR1 src out 50\nC1 out 0 1e-9\n.end\n")
    render_netlist_schematic(netlist, tmp_path / "s.png")
    assert matplotlib.get_backend().lower() == "agg"


# --- magnetic coupling ------------------------------------------------------


COUPLED_NETLIST = """\
Vsrc src 0 SIN(0 600 13.56e6)
L1 src mid 8e-7
C1 mid 0 2e-10
L2 mid electrode 6e-7
K1 L1 L2 0.72
Rload electrode 0 50
.end
"""


def test_a_coupling_between_drawn_coils_is_drawn_not_just_noted(tmp_path):
    """A `K` line is a real part of the circuit, so it belongs in the picture."""

    from pcd.netlist_parse import parse_netlist
    from pcd.netlist_viz import _draw_couplings

    netlist = tmp_path / "coupled.cir"
    netlist.write_text(COUPLED_NETLIST, encoding="utf-8")
    parsed = parse_netlist(netlist)
    assert len(parsed.couplings) == 1

    image = tmp_path / "coupled.png"
    render_netlist_schematic(netlist, image, title="coupled")
    assert image.stat().st_size > 1000

    # Both coils are on the main chain, so nothing should be left for the footer.
    coils = {"l1": _FakeCoil((0.0, 0.0)), "l2": _FakeCoil((3.0, 0.0))}
    assert _draw_couplings(_FakeDrawing(), parsed.couplings, coils) == []


def test_a_coupling_to_a_coil_that_was_not_drawn_is_reported_instead(tmp_path):
    """A winding inside an unexpanded subckt has no position to tie to."""

    from pcd.netlist_parse import parse_netlist
    from pcd.netlist_viz import _draw_couplings

    netlist = tmp_path / "coupled.cir"
    netlist.write_text(COUPLED_NETLIST, encoding="utf-8")
    parsed = parse_netlist(netlist)

    undrawn = _draw_couplings(_FakeDrawing(), parsed.couplings, {"l1": _FakeCoil((0.0, 0.0))})
    assert [c.ref for c in undrawn] == ["K1"]


class _FakeCoil:
    """Stands in for a schemdraw element: only its centre is used."""

    def __init__(self, center):
        self.center = center


class _FakeDrawing:
    def __iadd__(self, _element):
        return self

    def push(self):
        pass

    def pop(self):
        pass
