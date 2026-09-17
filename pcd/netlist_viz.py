"""Drawing a parsed netlist as a conventional schematic.

Layout model, in one sentence: the longest non-ground path becomes a horizontal
series chain, anything tied to ground hangs below it, and a series element
feeding a single grounded element collapses into one vertical branch.

Parsing lives in :mod:`pcd.netlist_parse`; this module only draws.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .netlist_parse import Coupling, NetlistEdge, ParsedNetlist, node_sort_key, parse_netlist

#: Grid spacing between series elements, in schemdraw units.
UNIT = 3.1

GROUND = "0"


def _is_grounded(edge: NetlistEdge) -> bool:
    """True for an element with one terminal on ground: it is drawn as a shunt."""

    return GROUND in {edge.n1, edge.n2}


def _live_node(edge: NetlistEdge) -> str:
    """The non-ground terminal of a grounded element."""

    return edge.n2 if edge.n1 == GROUND else edge.n1


def render_netlist_schematic(netlist_path: str | Path, out: str | Path, title: str | None = None) -> None:
    """Draw a netlist as a schematic PNG."""

    import schemdraw

    _use_headless_backend()
    parsed = parse_netlist(netlist_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    main = _main_signal_path([n for n in parsed.nodes if n != "0"], parsed.edges)
    with schemdraw.Drawing(show=False, unit=UNIT) as drawing:
        drawing.config(fontsize=8)
        if title:
            _label_at(drawing, title, (UNIT * max(len(main), 3) / 2, UNIT * 1.8), fontsize=12)
        undrawn = _draw_circuit(drawing, parsed, main) if main else list(parsed.couplings)
        if not main:
            _label_at(drawing, "no drawable two-terminal components", (0, 0), fontsize=10)
        _draw_footer(drawing, parsed, main, undrawn)
        drawing.save(str(out), dpi=170)


def _use_headless_backend() -> None:
    """Select matplotlib's file-only backend before any figure is created.

    Rendering always writes a PNG and never shows a window.  Left to its
    default, matplotlib picks an interactive backend when a display exists,
    which allocates GUI resources during batch rendering and fails outright on
    a headless CI runner.
    """

    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------


def _main_signal_path(nodes: list[str], edges: list[NetlistEdge]) -> list[str]:
    """The longest non-ground path, preferring one that ends at a shunt."""

    graph: dict[str, set[str]] = {n: set() for n in nodes}
    for edge in edges:
        if _is_grounded(edge):
            continue
        graph.setdefault(edge.n1, set()).add(edge.n2)
        graph.setdefault(edge.n2, set()).add(edge.n1)
    if not graph:
        return []

    grounded = {_live_node(e) for e in edges if _is_grounded(e)}

    def score(path: list[str]) -> tuple[int, int, int]:
        return (len(path), int(path[-1] in grounded), -node_sort_key(path[-1])[0])

    start = "src" if "src" in graph else sorted(graph, key=node_sort_key)[0]
    best = [start]

    def walk(node: str, path: list[str]) -> None:
        nonlocal best
        if score(path) > score(best):
            best = list(path)
        for nxt in sorted(graph.get(node, set()), key=node_sort_key):
            if nxt not in path:
                walk(nxt, [*path, nxt])

    walk(start, [start])
    return best


def _main_edges(main: list[str], edges: list[NetlistEdge]) -> list[NetlistEdge]:
    """The edges along the main path, oriented left to right."""

    out: list[NetlistEdge] = []
    for n1, n2 in itertools.pairwise(main):
        edge = next((e for e in edges if {e.n1, e.n2} == {n1, n2}), None)
        if edge is None:
            continue
        oriented = edge if (edge.n1, edge.n2) == (n1, n2) else NetlistEdge(edge.ref, n1, n2, edge.value, edge.kind)
        out.append(oriented)
    return out


def _series_shunt_pairs(
    edges: list[NetlistEdge], main_nodes: set[str]
) -> list[tuple[NetlistEdge, NetlistEdge, str, str]]:
    """Series element + its single grounded partner, drawn as one branch.

    Without this, an L-match's series inductor and its shunt capacitor would be
    drawn as two unrelated stubs instead of the branch they physically form.
    """

    grounded_by_node: dict[str, list[NetlistEdge]] = {}
    for edge in edges:
        if _is_grounded(edge):
            grounded_by_node.setdefault(_live_node(edge), []).append(edge)

    pairs = []
    for edge in edges:
        if _is_grounded(edge):
            continue
        on_main = (edge.n1 in main_nodes, edge.n2 in main_nodes)
        if on_main[0] == on_main[1]:
            continue
        base, branch = (edge.n1, edge.n2) if on_main[0] else (edge.n2, edge.n1)
        partners = grounded_by_node.get(branch, [])
        if len(partners) == 1:
            pairs.append((edge, partners[0], base, branch))
    return pairs


def _branch_offsets(n: int) -> list[float]:
    """Fan several branches out horizontally so they do not overlap."""

    return [0.0] if n <= 1 else [(-0.55 * (n - 1) / 2.0) + 0.55 * i for i in range(n)]


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------


def _draw_circuit(drawing: Any, parsed: ParsedNetlist, main: list[str]) -> list[Coupling]:
    """Draw the circuit; return the couplings that could not be drawn."""

    source, shunts, collapsed = _partition_edges(parsed, main)
    # Where each element ended up, so magnetic couplings can be tied to coils
    # the three layout passes place independently.
    coils: dict[str, Any] = {}
    node_points = _draw_main_chain(drawing, parsed, main, source, coils)
    _draw_shunts(drawing, shunts, node_points, coils)
    _draw_collapsed_pairs(drawing, collapsed, node_points, coils)
    return _draw_couplings(drawing, parsed.couplings, coils)


def _partition_edges(
    parsed: ParsedNetlist, main: list[str]
) -> tuple[NetlistEdge | None, list[NetlistEdge], list[tuple[NetlistEdge, NetlistEdge, str, str]]]:
    """Decide how each edge is drawn: the source, a shunt, or a collapsed pair.

    Every edge lands in exactly one group, which is what keeps an element from
    being drawn twice.
    """

    sources = [e for e in parsed.edges if e.kind in {"voltage", "current"} and _is_grounded(e)]
    collapsed = _series_shunt_pairs(parsed.edges, set(main))
    collapsed_ids = {id(edge) for pair in collapsed for edge in pair[:2]}
    drawn_elsewhere = {id(e) for e in sources} | collapsed_ids
    shunts = [e for e in parsed.edges if _is_grounded(e) and id(e) not in drawn_elsewhere]
    return (sources[0] if sources else None), shunts, collapsed


def _draw_main_chain(
    drawing: Any, parsed: ParsedNetlist, main: list[str], source: NetlistEdge | None, coils: dict[str, Any]
) -> dict[str, tuple[float, float]]:
    """Draw the source and the horizontal series chain; return node positions."""

    import schemdraw.elements as elm

    if source:
        drawing += _schemdraw_source(source).up().at((0.0, 0)).label(_clean_ref(source.ref), loc="left", fontsize=8)
    else:
        drawing += elm.Dot().at((0.0, UNIT))
    drawing += elm.Line().right().length(UNIT * 0.45)

    node_points = {main[0]: drawing.here}
    drawing += elm.Dot()
    _draw_node_label(drawing, main[0], drawing.here)
    for edge in _main_edges(main, parsed.edges):
        element = _schemdraw_element(edge).right().label(_clean_ref(edge.ref), loc="top", fontsize=8)
        drawing += element
        _remember_coil(coils, edge, element)
        drawing += elm.Dot()
        node_points[edge.n2] = drawing.here
        _draw_node_label(drawing, edge.n2, drawing.here)

    if source:
        drawing.push()
        drawing += elm.Ground().at((0, 0))
        drawing.pop()
    return node_points


def _draw_shunts(
    drawing: Any, shunts: list[NetlistEdge], node_points: dict[str, tuple[float, float]], coils: dict[str, Any]
) -> None:
    """Hang each grounded element below the node it connects to."""

    import schemdraw.elements as elm

    by_node: dict[str, list[NetlistEdge]] = {}
    for edge in shunts:
        node = _live_node(edge)
        if node in node_points:
            by_node.setdefault(node, []).append(edge)

    for node, edges in by_node.items():
        base = node_points[node]
        for edge, offset in zip(edges, _branch_offsets(len(edges)), strict=True):
            top = (base[0] + offset, base[1])
            drawing.push()
            if abs(offset) > 1e-9:
                line = elm.Line().left() if offset < 0 else elm.Line().right()
                drawing += line.length(abs(offset)).at(base)
                drawing += elm.Dot()
            element = _schemdraw_element(edge).down().at(top)
            drawing += element
            _remember_coil(coils, edge, element)
            label_x = top[0] + (-0.72 if offset < 0 else 0.72)
            _label_at(drawing, _clean_ref(edge.ref), (label_x, top[1] - UNIT * 0.48), 7)
            drawing += elm.Ground()
            drawing.pop()


def _draw_collapsed_pairs(
    drawing: Any,
    pairs: list[tuple[NetlistEdge, NetlistEdge, str, str]],
    node_points: dict[str, tuple[float, float]],
    coils: dict[str, Any],
) -> None:
    """Draw a series element and its grounded partner stacked in one branch."""

    import schemdraw.elements as elm

    seen: dict[str, int] = {}
    for edge, ground_edge, base, branch in pairs:
        if base not in node_points:
            continue
        index = seen.get(base, 0)
        seen[base] = index + 1
        origin = node_points[base]
        offset = -0.85 - index * 0.75
        top = (origin[0] + offset, origin[1])

        drawing.push()
        drawing += elm.Line().left().at(origin).length(abs(offset))
        drawing += elm.Dot()
        _draw_node_label(drawing, branch, top, force=True)
        element = _schemdraw_element(edge).down().at(top)
        drawing += element
        _remember_coil(coils, edge, element)
        _label_at(drawing, _clean_ref(edge.ref), (top[0] + 0.72, top[1] - UNIT * 0.25), 7)
        second = drawing.here
        ground_element = _schemdraw_element(ground_edge).down()
        drawing += ground_element
        _remember_coil(coils, ground_edge, ground_element)
        _label_at(drawing, _clean_ref(ground_edge.ref), (second[0] + 0.72, second[1] - UNIT * 0.45), 7)
        drawing += elm.Ground()
        drawing.pop()


def _remember_coil(coils: dict[str, Any], edge: NetlistEdge, element: Any) -> None:
    """Record where an inductor was drawn, keyed by ref, for coupling lines."""

    if edge.kind == "inductor":
        coils[_clean_ref(edge.ref).lower()] = element


#: How far above the coils the coupling tie is routed, in schemdraw units.
COUPLING_LIFT = 1.15


def _draw_couplings(drawing: Any, couplings: list[Coupling], coils: dict[str, Any]) -> list[Coupling]:
    """Tie magnetically coupled inductors together.

    A `K` line is not a branch, so it has no place in the series/shunt layout.
    The two coils are drawn wherever the circuit puts them and joined by a
    dashed tie carrying the coupling coefficient -- the standard annotation
    when a schematic does not place the windings side by side.  A transformer
    symbol would require exactly that adjacency, which this layout cannot
    promise.

    The tie is routed *above* the coils rather than straight between them:
    coupled inductors usually both sit in the horizontal chain, and a direct
    line would then run along the wire and read as part of the circuit.
    """

    import schemdraw.elements as elm

    undrawn: list[Coupling] = []
    for index, coupling in enumerate(couplings):
        first, second = (coils.get(ref.lower()) for ref in coupling.inductors)
        if first is None or second is None:
            undrawn.append(coupling)  # a coil inside an unexpanded subckt
            continue

        (x1, y1), (x2, y2) = tuple(first.center), tuple(second.center)
        top = max(y1, y2) + COUPLING_LIFT + index * 0.5
        for start, end in itertools.pairwise([(x1, y1), (x1, top), (x2, top), (x2, y2)]):
            drawing += elm.Line().at(start).to(end).color("gray").linestyle("--")
        _label_at(drawing, f"{_clean_ref(coupling.ref)}  k={coupling.coefficient:g}", ((x1 + x2) / 2, top + 0.3), 7)
    return undrawn


def _draw_footer(drawing: Any, parsed: ParsedNetlist, main: list[str], undrawn: list[Coupling]) -> None:
    """Only what the drawing itself could not say."""

    notes = []
    if parsed.subckt_names:
        notes.append("expanded subckts: " + ", ".join(parsed.subckt_names))
    if undrawn:
        notes.append(
            "coupled (coils not drawn): "
            + ", ".join(f"{c.ref}({c.inductors[0]},{c.inductors[1]}) k={c.coefficient:g}" for c in undrawn)
        )
    if notes:
        _label_at(drawing, "   ".join(notes), (0, -1.45), fontsize=7)
    _label_at(drawing, "rendered with Schemdraw", (UNIT * max(len(main), 3), -1.45), fontsize=7)


# -----------------------------------------------------------------------------
# schemdraw primitives
# -----------------------------------------------------------------------------


def _schemdraw_element(edge: NetlistEdge) -> Any:
    import schemdraw.elements as elm

    by_kind = {
        "resistor": elm.Resistor,
        "capacitor": elm.Capacitor,
        "inductor": elm.Inductor,
        "voltage": elm.SourceSin,
        "current": elm.SourceI,
    }
    # RBox is the generic two-terminal box: diodes and unresolved subckts land
    # here.  schemdraw has no element called `Box`.
    return by_kind.get(edge.kind, elm.RBox)()


def _schemdraw_source(edge: NetlistEdge) -> Any:
    import schemdraw.elements as elm

    return elm.SourceI() if edge.kind == "current" else elm.SourceSin()


def _label_at(drawing: Any, text: str, point: tuple[float, float], fontsize: int = 8) -> None:
    import schemdraw.elements as elm

    drawing.push()
    drawing += elm.Label().at(point).label(text, fontsize=fontsize)
    drawing.pop()


def _draw_node_label(drawing: Any, node: str, point: Any, force: bool = False) -> None:
    if force or not node.startswith("Xload."):
        _label_at(drawing, _short_node(node), (float(point[0]), float(point[1]) + 0.38), fontsize=7)


def _clean_ref(ref: str) -> str:
    return ref.replace("Xload:", "load.")


def _short_node(node: str) -> str:
    return node.replace("Xload.", "load.")
