"""Complete connectivity drawings using Schemdraw symbols and shared geometry.

A spanning forest places nodes. Every remaining branch gets its own route;
parallel edges, disconnected sections and extra sources are never discarded.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from .figures.circuit import CircuitDiagram, CircuitStyle, CircuitViewport
from .netlist_parse import NetlistEdge, ParsedNetlist, node_sort_key, parse_netlist


def _instrument(edge: NetlistEdge) -> bool:
    return (
        edge.kind == "voltage"
        and (edge.ref.lower().startswith("vobserve_") or edge.ref.lower() == "vload_meter")
        and bool(re.fullmatch(r"(?:dc\s+)?[+-]?0(?:\.0*)?", edge.value.strip(), re.IGNORECASE))
    )


def _physical_edges(parsed: ParsedNetlist, view: str) -> tuple[list[NetlistEdge], dict[str, str], list[str]]:
    parents: dict[str, str] = {}

    def find(node: str) -> str:
        if node in parents:
            parents[node] = find(parents[node])
            return parents[node]
        return node

    hidden = []
    if view == "physical":
        for edge in parsed.edges:
            if _instrument(edge):
                first, second = find(edge.n1), find(edge.n2)
                if first != second:
                    preferred = min(
                        (first, second), key=lambda name: ("observe_" in name or "meter" in name, node_sort_key(name))
                    )
                    parents[second if preferred == first else first] = preferred
                hidden.append(edge.ref)
    mapping = {node: find(node) for node in parsed.nodes}
    edges = [
        NetlistEdge(e.ref, mapping[e.n1], mapping[e.n2], e.value, e.kind, tuple(mapping[n] for n in e.nodes))
        for e in parsed.edges
        if e.ref not in hidden
    ]
    return edges, mapping, hidden


def _positions(edges: list[NetlistEdge]) -> dict[str, tuple[float, float]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        nodes = [node for node in edge.nodes if node != "0"]
        for node in nodes:
            graph[node].update(other for other in nodes if other != node)
    remaining = set(graph)
    points: dict[str, tuple[float, float]] = {}
    base = 0.0
    while remaining:
        start = min(remaining, key=node_sort_key)
        queue = deque([(start, 0)])
        levels: dict[int, list[str]] = defaultdict(list)
        seen = {start}
        while queue:
            node, level = queue.popleft()
            levels[level].append(node)
            for other in sorted(graph[node], key=node_sort_key):
                if other not in seen:
                    seen.add(other)
                    queue.append((other, level + 1))
        for level, nodes in levels.items():
            for row, node in enumerate(nodes):
                points[node] = (4.5 * level, base - 4.5 * row)
        remaining -= seen
        base -= 4.5 * (max(map(len, levels.values())) - 1) + 6.3
    return points


def _value(edge: NetlistEdge, parameters: dict[str, str]) -> str:
    value = re.sub(r"\{([\w.]+)\}", lambda m: parameters.get(m[1].lower(), m[0]), edge.value)
    if edge.kind in {"voltage", "current"}:
        unit = "V" if edge.kind == "voltage" else "A"
        sine = re.search(r"(?i)sin\(\s*([^\s,]+)[\s,]+([^\s,]+)[\s,]+([^\s,)]+)", value)
        pulse = re.search(r"(?i)pulse\(\s*([^\s,]+)[\s,]+([^\s,]+)", value)
        if sine:
            return f"{sine[2]} {unit} peak, f={sine[3]} Hz"
        if pulse:
            return f"pulse {pulse[1]} to {pulse[2]} {unit}"
    unit = {"resistor": "ohm", "capacitor": "F", "inductor": "H"}.get(edge.kind)
    try:
        number = float(value)
    except ValueError:
        return value if len(value) <= 65 else value[:62] + "..."
    if not unit or number == 0:
        return value + (" " + unit if unit else "")
    power = max(-12, min(9, 3 * math.floor(math.log10(abs(number)) / 3)))
    prefix = {-12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}[power]
    return f"{number / 10**power:.4g} {prefix}{unit}"


def _factory(edge: NetlistEdge) -> Any:
    import schemdraw.elements as elm

    if _instrument(edge):
        return elm.MeterI
    if edge.kind == "voltage":
        low = edge.value.lower()
        if "sin(" in low:
            return elm.SourceSin
        if "pulse(" in low:
            return elm.SourcePulse
        return lambda: elm.SourceV().reverse()
    return {
        "resistor": elm.Resistor,
        "capacitor": elm.Capacitor,
        "inductor": elm.Inductor,
        "diode": elm.Diode,
        "current": elm.SourceI,
        "behavioral": elm.SourceControlled,
    }.get(edge.kind, elm.RBox)


def render_netlist_schematic(
    netlist_path: str | Path,
    out: str | Path,
    title: str | None = None,
    *,
    view: str = "physical",
    netlist_mode: str = "auto",
    signals: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import schemdraw.elements as elm

    if view not in {"physical", "instrumented"}:
        raise ValueError("schematic view must be physical or instrumented")
    parsed = parse_netlist(netlist_path, netlist_mode=netlist_mode)
    edges, mapping, hidden = _physical_edges(parsed, view)
    points = _positions(edges)
    parameters = {**parsed.parameters, **{name.lower(): str(value) for name, value in (params or {}).items()}}
    issues = list(parsed.issues)
    fig, axis = plt.subplots(figsize=(10, 6))
    diagram = CircuitDiagram(axis, CircuitStyle(component_length=1.6, font_size=8.5))
    centers: dict[str, tuple[float, float]] = {}
    bounds = list(points.values()) or [(0.0, 0.0)]
    used_pairs: Counter = Counter()
    grounded: Counter = Counter()
    extras = 0

    def wire(a: tuple[float, float], b: tuple[float, float]) -> None:
        diagram.wire(a, b)
        bounds.extend((a, b))

    def symbol(edge: NetlistEdge, a: tuple[float, float], b: tuple[float, float]) -> None:
        label = edge.ref + "\n" + ("current probe (0 V)" if _instrument(edge) else _value(edge, parameters))
        diagram.component(
            _factory(edge), a, b, label=label, label_location="top" if a[1] == b[1] else "right", name=edge.ref
        )
        centers[edge.ref.lower()] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        bounds.extend((a, b))
        if edge.kind == "voltage" and not _instrument(edge):
            axis.text(a[0] + 0.2, a[1] - 0.2, "+", fontsize=10, color="#35495c")

    for edge in edges:
        if len(edge.nodes) != 2:
            continue
        if "0" in edge.nodes and edge.n1 != edge.n2:
            live = edge.n2 if edge.n1 == "0" else edge.n1
            p = points[live]
            index = grounded[live]
            grounded[live] += 1
            top = (p[0] + 2.2 * index, p[1])
            bottom = (top[0], top[1] - 3.1)
            wire(p, top)
            symbol(edge, bottom if edge.n1 == "0" else top, top if edge.n1 == "0" else bottom)
            diagram.ground(bottom)
            bounds.append((bottom[0], bottom[1] - 0.6))
            continue
        p, q = points.get(edge.n1, (0.0, 0.0)), points.get(edge.n2, (0.0, 0.0))
        pair = tuple(sorted((edge.n1, edge.n2)))
        index = used_pairs[pair]
        used_pairs[pair] += 1
        if p[1] == q[1] and p[0] != q[0] and index == 0:
            symbol(edge, p, q)
        elif p[0] != q[0]:
            extras += 1
            lane = max(p[1], q[1], 0) + 2.4 * extras
            a, b = (p[0], lane), (q[0], lane)
            wire(p, a)
            symbol(edge, a, b)
            wire(b, q)
        else:
            # Same-level or self-loop branches route outside the node column.
            extras += 1
            x = max((point[0] for point in points.values()), default=0) + 3 + 2 * extras
            a, b = (x, p[1]), (x, q[1] if p != q else q[1] + 3)
            wire(p, a)
            symbol(edge, a, b)
            wire(b, (q[0], b[1]))
            wire((q[0], b[1]), q)
    for node, p in points.items():
        diagram.node(p)
        axis.text(p[0] - 0.12, p[1] + 0.35, node, fontsize=8, ha="right", color="#35495c")
    # Multi-terminal symbols retain every pin through explicit named-net stubs.
    block_x = max((point[0] for point in bounds), default=0) + 5
    block_y = 0.0
    roles = {
        "bjt": ["C", "B", "E", "S"],
        "mosfet": ["D", "G", "S", "B"],
        "jfet": ["D", "G", "S"],
        "tline": ["1+", "1-", "2+", "2-"],
    }
    for edge in edges:
        if len(edge.nodes) == 2:
            continue
        height = max(3.0, len(edge.nodes) * 0.9)
        names = roles.get(edge.kind, [str(i + 1) for i in range(len(edge.nodes))])
        pins = [
            elm.IcPin(
                name=names[i],
                pin=node,
                side="L",
                slot=f"{len(edge.nodes) - i}/{len(edge.nodes)}",
                lblsize=8.5,
                pinlblsize=8.5,
            )
            for i, node in enumerate(edge.nodes)
        ]
        diagram.drawing.add(
            elm.Ic(pins=pins, size=(4.2, height))
            .theta(0)
            .at((block_x, block_y))
            .label(edge.ref + "\n" + edge.kind + "\n" + _value(edge, parameters), loc="center")
        )
        centers[edge.ref.lower()] = (block_x + 2.1, block_y + height / 2)
        bounds.extend(((block_x - 4, block_y - 1), (block_x + 4.5, block_y + height + 1)))
        block_y -= height + 2
    for coupling in parsed.couplings:
        first, second = (centers.get(ref.lower()) for ref in coupling.inductors)
        if first is None or second is None:
            issues.append(f"coupling has an unavailable winding: {coupling.ref}")
            continue
        top = max(p[1] for p in bounds) + 1.3
        axis.plot(
            [first[0], first[0], second[0], second[0]], [first[1], top, top, second[1]], "--", color="#6b7280", lw=1
        )
        axis.text(
            (first[0] + second[0]) / 2, top + 0.15, f"{coupling.ref}: k={coupling.coefficient}", ha="center", fontsize=8
        )
        bounds.append((second[0], top + 1))
    probe_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    current_arrows: set[str] = set()
    nodes_by_lower = {node.lower(): mapped for node, mapped in mapping.items()}
    for name, item in (signals or {}).get("signals", {}).items():
        vector = item.get("vector", "")
        voltage = re.fullmatch(r"v\(([^(),]+)(?:,([^()]+))?\)", vector, re.IGNORECASE)
        current = re.fullmatch(r"[+-]?i\(([^()]+)\)", vector, re.IGNORECASE)
        if voltage and voltage[1].lower() in nodes_by_lower:
            p = points.get(nodes_by_lower[voltage[1].lower()])
            if p:
                probe_groups[(vector, item.get("unit", ""))].append(item.get("label", name))
                axis.plot([p[0]], [p[1]], "o", ms=4, mfc="white", mec="#2a6f97", zorder=10)
        elif current:
            ref = current[1].lower()
            p = centers.get(ref) or centers.get(ref.removeprefix("vobserve_"))
            if p:
                probe_groups[(vector, item.get("unit", ""))].append(item.get("label", name))
                if ref not in current_arrows:
                    current_arrows.add(ref)
                    branch_ref = ref.removeprefix("vobserve_")
                    branch = next((edge for edge in edges if edge.ref.lower() in {ref, branch_ref}), None)
                    vertical = branch is not None and (
                        "0" in branch.nodes or points.get(branch.n1, (0, 0))[0] == points.get(branch.n2, (0, 0))[0]
                    )
                    if vertical:
                        sign = 1 if branch is not None and branch.n1 == "0" else -1
                        if branch is not None and "0" not in branch.nodes:
                            sign = 1 if points[branch.n2][1] > points[branch.n1][1] else -1
                        dx, dy = 0.0, 0.6 * sign
                    else:
                        sign = 1 if branch is None or points[branch.n2][0] > points[branch.n1][0] else -1
                        dx, dy = 0.6 * sign, 0.0
                    if vector.strip().startswith("-"):
                        dx, dy = -dx, -dy
                    axis.annotate(
                        "",
                        xy=(p[0] + dx / 2, p[1] + 0.7 + dy / 2),
                        xytext=(p[0] - dx / 2, p[1] + 0.7 - dy / 2),
                        arrowprops={"arrowstyle": "->", "color": "#2a6f97", "lw": 1},
                    )
    if not edges:
        issues.append("no drawable components")
    left, right = min(p[0] for p in bounds) - 2.2, max(p[0] for p in bounds) + 2.5
    bottom, top = min(p[1] for p in bounds) - 2, max(p[1] for p in bounds) + 2
    notes = []
    if hidden:
        notes.append("Physical circuit; blue arrows indicate positive current-probe direction.")
    if any(len(edge.nodes) > 2 for edge in edges):
        notes.append("Multi-terminal blocks use named nets; equal net names are electrically connected.")
    notes.extend(
        f"{', '.join(dict.fromkeys(names))}: {vector} [{unit}]" for (vector, unit), names in probe_groups.items()
    )
    if issues:
        notes.insert(0, "PARTIAL DRAWING: " + "; ".join(dict.fromkeys(issues)))
    if title:
        axis.set_title(title, fontsize=11, pad=15)
    note_text = "\n".join(notes)
    if note_text:
        axis.text(left, bottom - 0.3, note_text, fontsize=7.5, va="top", color="#a33322" if issues else "#35495c")
        bottom -= max(1, len(notes)) * 0.48
    width, height = right - left, top - bottom
    fig.set_size_inches(min(20, max(7, width * 0.48)), min(26, max(4, height * 0.48)))
    diagram.finish(CircuitViewport((left, right), (bottom - 0.5, top)))
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        **parsed.summary(),
        "status": "partial" if issues else "complete",
        "issues": list(dict.fromkeys(issues)),
        "view": view,
        "represented_components": [edge.ref for edge in edges],
        "collapsed_measurement_sources": hidden,
        "node_aliases": mapping,
    }
