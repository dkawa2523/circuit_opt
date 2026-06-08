from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NetlistEdge:
    ref: str
    n1: str
    n2: str
    value: str = ""
    kind: str = "component"


@dataclass
class ParsedNetlist:
    edges: list[NetlistEdge]
    subckt_names: list[str]

    @property
    def nodes(self) -> list[str]:
        out = sorted({e.n1 for e in self.edges} | {e.n2 for e in self.edges}, key=_node_sort_key)
        return out


def parse_netlist(path_or_text: str | Path) -> ParsedNetlist:
    raw = str(path_or_text)
    if isinstance(path_or_text, Path) or ("\n" not in raw and Path(raw).exists()):
        text = Path(path_or_text).read_text(encoding="utf-8")
    else:
        text = raw
    lines = [_strip_comment(line) for line in text.splitlines()]
    subckts: dict[str, tuple[list[str], list[str]]] = {}
    top_lines: list[str] = []
    in_control = False
    current_subckt: str | None = None
    current_ports: list[str] = []
    current_body: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low == ".control":
            in_control = True
            continue
        if low == ".endc":
            in_control = False
            continue
        if in_control or low.startswith(".param") or low.startswith(".options") or low.startswith(".save"):
            continue
        if low.startswith(".subckt"):
            parts = line.split()
            current_subckt = parts[1]
            current_ports = parts[2:]
            current_body = []
            continue
        if low.startswith(".ends"):
            if current_subckt:
                subckts[current_subckt] = (current_ports, list(current_body))
            current_subckt = None
            current_ports = []
            current_body = []
            continue
        if low.startswith("."):
            continue
        if current_subckt:
            current_body.append(line)
        else:
            top_lines.append(line)

    edges: list[NetlistEdge] = []
    for line in top_lines:
        edges.extend(_parse_component_line(line, subckts))
    return ParsedNetlist(edges=edges, subckt_names=sorted(subckts))


def render_netlist_schematic(netlist_path: str | Path, out: str | Path, title: str | None = None) -> None:
    try:
        _render_netlist_schematic_schemdraw(netlist_path, out, title=title)
    except ImportError:
        _render_netlist_schematic_matplotlib(netlist_path, out, title=title)


def _render_netlist_schematic_schemdraw(netlist_path: str | Path, out: str | Path, title: str | None = None) -> None:
    import schemdraw
    import schemdraw.elements as elm

    parsed = parse_netlist(netlist_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    main = _main_signal_path([n for n in parsed.nodes if n != "0"], parsed.edges)
    if not main:
        _render_netlist_schematic_matplotlib(netlist_path, out, title=title)
        return
    main_edges = _main_edges(main, parsed.edges)
    main_edge_ids = {id(edge) for edge in main_edges}
    source_edges = [e for e in parsed.edges if e.kind in {"voltage", "current"} and "0" in {e.n1, e.n2}]
    shunt_edges = [e for e in parsed.edges if "0" in {e.n1, e.n2} and e not in source_edges]
    collapsed_pairs = _series_shunt_pairs(parsed.edges, set(main))
    collapsed_edge_ids = {id(edge) for pair in collapsed_pairs for edge in pair[:2]}

    unit = 3.1
    with schemdraw.Drawing(show=False, unit=unit) as d:
        d.config(fontsize=8)
        if title:
            d += elm.Label().label(title, fontsize=12).at((unit * max(len(main), 3) / 2, unit * 1.8))

        start_x = 0.0
        source = source_edges[0] if source_edges else None
        if source:
            d += _schemdraw_source(source).up().at((start_x, 0)).label(_clean_ref(source.ref), loc="left", fontsize=8)
            d += elm.Line().right().length(unit * 0.45)
        else:
            d += elm.Dot().at((start_x, unit))
            d += elm.Line().right().length(unit * 0.45)

        node_points: dict[str, tuple[float, float]] = {main[0]: d.here}
        d += elm.Dot()
        _draw_schemdraw_node_label(d, main[0], d.here)
        for edge in main_edges:
            d += _schemdraw_element(edge).right().label(_clean_ref(edge.ref), loc="top", fontsize=8)
            d += elm.Dot()
            node_points[edge.n2] = d.here
            _draw_schemdraw_node_label(d, edge.n2, d.here)

        _draw_ground_bus(d, source, node_points, main, unit)
        _draw_schemdraw_shunts(d, shunt_edges, collapsed_edge_ids, node_points, unit)
        _draw_schemdraw_collapsed_pairs(d, collapsed_pairs, node_points, unit)

        if parsed.subckt_names:
            d += elm.Label().at((0, -1.45)).label("expanded subckts: " + ", ".join(parsed.subckt_names), fontsize=7)
        d += elm.Label().at((unit * max(len(main), 3), -1.45)).label("rendered with Schemdraw", fontsize=7)
        d.save(str(out), dpi=170)


def _render_netlist_schematic_matplotlib(netlist_path: str | Path, out: str | Path, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parsed = parse_netlist(netlist_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    layout = _schematic_layout(parsed)
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_title(title or f"Netlist topology: {Path(netlist_path).name}", fontsize=13)
    ax.axis("off")

    if not parsed.edges:
        ax.text(0.5, 0.5, "No drawable components found", transform=ax.transAxes, ha="center", va="center")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        return

    y_signal = layout["y_signal"]
    y_ground = layout["y_ground"]
    node_pos: dict[str, tuple[float, float]] = layout["node_pos"]
    main_nodes = set(layout["main_nodes"])
    ground_xs = [layout["x_min"], layout["x_max"]]

    ax.plot(ground_xs, [y_ground, y_ground], color="#202020", lw=1.3)
    _draw_ground_symbol(ax, (sum(ground_xs) / 2.0, y_ground))

    shunt_offsets = _shunt_offsets(parsed.edges)
    used_shunts: dict[str, int] = {}
    collapsed_pairs = _series_shunt_pairs(parsed.edges, main_nodes)
    skip_edges = {edge for edge, _ground_edge, _base, _branch in collapsed_pairs}
    skip_edges.update({ground_edge for _edge, ground_edge, _base, _branch in collapsed_pairs})
    collapsed_nodes = {branch for _edge, _ground_edge, _base, branch in collapsed_pairs}

    for edge, ground_edge, base, branch in collapsed_pairs:
        base_x, base_y = node_pos[base]
        mid_y = (base_y + y_ground) / 2.0
        mid = (base_x, mid_y)
        _draw_component(ax, edge.kind, (base_x, base_y), mid, edge.ref)
        _draw_connection_dot(ax, mid)
        ax.text(base_x + 0.72, mid_y, _short_node(branch), ha="left", va="center", fontsize=8.5, color="#202020")
        _draw_component(ax, ground_edge.kind, mid, (base_x, y_ground), ground_edge.ref)
        _draw_connection_dot(ax, (base_x, base_y))

    for edge in parsed.edges:
        if edge in skip_edges:
            continue
        if edge.n1 == "0" or edge.n2 == "0":
            node = edge.n2 if edge.n1 == "0" else edge.n1
            if node not in node_pos:
                continue
            idx = used_shunts.get(node, 0)
            used_shunts[node] = idx + 1
            offset = shunt_offsets.get(node, [0.0])[idx]
            node_x, node_y = node_pos[node]
            branch_x = node_x + offset
            _draw_wire(ax, (node_x, node_y), (branch_x, node_y))
            _draw_component(ax, edge.kind, (branch_x, node_y), (branch_x, y_ground), edge.ref)
            _draw_connection_dot(ax, (node_x, node_y))
        else:
            if edge.n1 not in node_pos or edge.n2 not in node_pos:
                continue
            p1 = node_pos[edge.n1]
            p2 = node_pos[edge.n2]
            if abs(p1[1] - p2[1]) > 1e-9:
                via1 = (p1[0], p2[1])
                _draw_wire(ax, p1, via1)
                _draw_component(ax, edge.kind, via1, p2, edge.ref)
            else:
                _draw_component(ax, edge.kind, p1, p2, edge.ref)
            _draw_connection_dot(ax, p1)
            _draw_connection_dot(ax, p2)

    for node, (x, y) in node_pos.items():
        if node in collapsed_nodes:
            continue
        _draw_connection_dot(ax, (x, y))
        label_y = y + (0.38 if y >= y_signal else -0.38)
        ax.text(x, label_y, _short_node(node), ha="center", va="center", fontsize=9, color="#202020")

    if parsed.subckt_names:
        ax.text(
            0.01,
            0.02,
            "expanded subckts: " + ", ".join(parsed.subckt_names),
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    ax.text(
        0.99,
        0.02,
        "schematic-style layout inferred from netlist",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#555555",
    )
    ax.set_xlim(layout["x_min"] - 0.6, layout["x_max"] + 0.6)
    ax.set_ylim(y_ground - 0.9, layout["y_max"] + 0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.08)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def netlist_summary(path: str | Path) -> dict[str, Any]:
    parsed = parse_netlist(path)
    counts: dict[str, int] = {}
    for edge in parsed.edges:
        counts[edge.kind] = counts.get(edge.kind, 0) + 1
    return {
        "nodes": parsed.nodes,
        "n_nodes": len(parsed.nodes),
        "n_edges": len(parsed.edges),
        "component_counts": counts,
        "subckts": parsed.subckt_names,
        "edges": [edge.__dict__ for edge in parsed.edges],
    }


def _parse_component_line(line: str, subckts: dict[str, tuple[list[str], list[str]]]) -> list[NetlistEdge]:
    parts = line.split()
    if len(parts) < 3:
        return []
    ref = parts[0]
    kind = ref[0].upper()
    if kind == "X" and len(parts) >= 4:
        model = parts[-1]
        external_nodes = parts[1:-1]
        if model in subckts:
            ports, body = subckts[model]
            mapping = {port: external_nodes[i] for i, port in enumerate(ports) if i < len(external_nodes)}
            expanded: list[NetlistEdge] = []
            for body_line in body:
                expanded.extend(_parse_subckt_component_line(body_line, ref, mapping))
            return expanded
        return [NetlistEdge(ref=ref, n1=external_nodes[0], n2=external_nodes[1], value=model, kind="subckt")]
    if kind in {"R", "C", "L", "V", "I", "D"}:
        return [NetlistEdge(ref=ref, n1=parts[1], n2=parts[2], value=" ".join(parts[3:]), kind=_kind_name(kind))]
    return []


def _parse_subckt_component_line(line: str, instance: str, mapping: dict[str, str]) -> list[NetlistEdge]:
    parts = line.split()
    if len(parts) < 3:
        return []
    ref = f"{instance}:{parts[0]}"
    kind = parts[0][0].upper()
    if kind not in {"R", "C", "L", "V", "I", "D"}:
        return []
    n1 = _map_subckt_node(parts[1], instance, mapping)
    n2 = _map_subckt_node(parts[2], instance, mapping)
    return [NetlistEdge(ref=ref, n1=n1, n2=n2, value=" ".join(parts[3:]), kind=_kind_name(kind))]


def _map_subckt_node(node: str, instance: str, mapping: dict[str, str]) -> str:
    if node in mapping:
        return mapping[node]
    if node == "0":
        return "0"
    return f"{instance}.{node}"


def _schematic_layout(parsed: ParsedNetlist) -> dict[str, Any]:
    nodes = [n for n in parsed.nodes if n != "0"]
    main = _main_signal_path(nodes, parsed.edges)
    if not main:
        main = nodes
    x_step = 3.0
    y_signal = 3.5
    y_ground = 0.0
    node_pos: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(main):
        node_pos[node] = (1.5 + i * x_step, y_signal)

    branch_edges = [e for e in parsed.edges if e.n1 != "0" and e.n2 != "0" and (e.n1 not in main or e.n2 not in main)]
    branch_count: dict[str, int] = {}
    main_index = {node: i for i, node in enumerate(main)}
    for edge in branch_edges:
        base = edge.n1 if edge.n1 in main else edge.n2 if edge.n2 in main else None
        other = edge.n2 if base == edge.n1 else edge.n1 if base == edge.n2 else None
        if base is None or other is None:
            continue
        idx = branch_count.get(base, 0)
        branch_count[base] = idx + 1
        bx, by = node_pos[base]
        side = -1 if main_index.get(base, 0) > 0 else 1
        node_pos[other] = (bx + side * (1.35 + idx * 1.35), by + 1.45)

    for node in nodes:
        if node not in node_pos:
            node_pos[node] = (1.5 + len(node_pos) * x_step, y_signal)

    xs = [p[0] for p in node_pos.values()]
    y_max = max([p[1] for p in node_pos.values()] + [y_signal])
    return {
        "node_pos": node_pos,
        "main_nodes": main,
        "y_signal": y_signal,
        "y_ground": y_ground,
        "y_max": y_max,
        "x_min": min(xs + [0.5]),
        "x_max": max(xs + [8.5]),
    }


def _main_signal_path(nodes: list[str], edges: list[NetlistEdge]) -> list[str]:
    graph: dict[str, set[str]] = {n: set() for n in nodes}
    for edge in edges:
        if edge.n1 == "0" or edge.n2 == "0":
            continue
        graph.setdefault(edge.n1, set()).add(edge.n2)
        graph.setdefault(edge.n2, set()).add(edge.n1)
    if not graph:
        return []
    start = "src" if "src" in graph else sorted(graph, key=_node_sort_key)[0]

    best: list[str] = [start]

    def score(path: list[str]) -> tuple[int, int, int]:
        terminal = path[-1]
        terminal_has_shunt = int(any((e.n1 == terminal and e.n2 == "0") or (e.n2 == terminal and e.n1 == "0") for e in edges))
        return (len(path), terminal_has_shunt, -_node_sort_key(terminal)[0])

    def dfs(node: str, path: list[str]) -> None:
        nonlocal best
        if score(path) > score(best):
            best = list(path)
        for nxt in sorted(graph.get(node, set()), key=_node_sort_key):
            if nxt in path:
                continue
            dfs(nxt, path + [nxt])

    dfs(start, [start])
    return best


def _shunt_offsets(edges: list[NetlistEdge]) -> dict[str, list[float]]:
    by_node: dict[str, list[NetlistEdge]] = {}
    for edge in edges:
        if edge.n1 == "0" or edge.n2 == "0":
            node = edge.n2 if edge.n1 == "0" else edge.n1
            by_node.setdefault(node, []).append(edge)
    out: dict[str, list[float]] = {}
    for node, node_edges in by_node.items():
        n = len(node_edges)
        if n == 1:
            out[node] = [0.0]
        elif n == 2:
            out[node] = [-0.45, 0.45]
        else:
            out[node] = [(-0.55 * (n - 1) / 2.0) + 0.55 * i for i in range(n)]
    return out


def _series_shunt_pairs(
    edges: list[NetlistEdge],
    main_nodes: set[str],
) -> list[tuple[NetlistEdge, NetlistEdge, str, str]]:
    pairs: list[tuple[NetlistEdge, NetlistEdge, str, str]] = []
    ground_edges_by_node: dict[str, list[NetlistEdge]] = {}
    for edge in edges:
        if edge.n1 == "0" or edge.n2 == "0":
            node = edge.n2 if edge.n1 == "0" else edge.n1
            ground_edges_by_node.setdefault(node, []).append(edge)
    for edge in edges:
        if edge.n1 == "0" or edge.n2 == "0":
            continue
        n1_main = edge.n1 in main_nodes
        n2_main = edge.n2 in main_nodes
        if n1_main == n2_main:
            continue
        base = edge.n1 if n1_main else edge.n2
        branch = edge.n2 if n1_main else edge.n1
        ground_edges = ground_edges_by_node.get(branch, [])
        if len(ground_edges) == 1:
            pairs.append((edge, ground_edges[0], base, branch))
    return pairs


def _main_edges(main: list[str], edges: list[NetlistEdge]) -> list[NetlistEdge]:
    out: list[NetlistEdge] = []
    for n1, n2 in zip(main, main[1:]):
        edge = next((e for e in edges if {e.n1, e.n2} == {n1, n2}), None)
        if edge is None:
            continue
        if edge.n1 == n1 and edge.n2 == n2:
            out.append(edge)
        else:
            out.append(NetlistEdge(ref=edge.ref, n1=n1, n2=n2, value=edge.value, kind=edge.kind))
    return out


def _schemdraw_element(edge: NetlistEdge) -> Any:
    import schemdraw.elements as elm

    if edge.kind == "resistor":
        return elm.Resistor()
    if edge.kind == "capacitor":
        return elm.Capacitor()
    if edge.kind == "inductor":
        return elm.Inductor()
    if edge.kind == "voltage":
        return elm.SourceSin()
    if edge.kind == "current":
        return elm.SourceI()
    return elm.Box(w=1.2, h=0.6)


def _schemdraw_source(edge: NetlistEdge) -> Any:
    import schemdraw.elements as elm

    return elm.SourceI() if edge.kind == "current" else elm.SourceSin()


def _draw_ground_bus(d: Any, source: NetlistEdge | None, node_points: dict[str, tuple[float, float]], main: list[str], unit: float) -> None:
    import schemdraw.elements as elm

    if source:
        d.push()
        d += elm.Ground().at((0, 0))
        d.pop()


def _draw_schemdraw_shunts(
    d: Any,
    shunt_edges: list[NetlistEdge],
    skip_edge_ids: set[int],
    node_points: dict[str, tuple[float, float]],
    unit: float,
) -> None:
    import schemdraw.elements as elm

    by_node: dict[str, list[NetlistEdge]] = {}
    for edge in shunt_edges:
        if id(edge) in skip_edge_ids:
            continue
        node = edge.n2 if edge.n1 == "0" else edge.n1
        if node in node_points:
            by_node.setdefault(node, []).append(edge)

    for node, edges in by_node.items():
        base = node_points[node]
        offsets = _schemdraw_branch_offsets(len(edges))
        for edge, offset in zip(edges, offsets):
            branch_top = (base[0] + offset, base[1])
            d.push()
            if abs(offset) > 1e-9:
                direction = elm.Line().left().length(abs(offset)) if offset < 0 else elm.Line().right().length(abs(offset))
                d += direction.at(base)
                d += elm.Dot()
            d += _schemdraw_element(edge).down().at(branch_top)
            label_dx = -0.72 if offset < 0 else 0.72
            _draw_schemdraw_text(d, _clean_ref(edge.ref), (branch_top[0] + label_dx, branch_top[1] - unit * 0.48), fontsize=7)
            d += elm.Ground()
            d.pop()


def _draw_schemdraw_collapsed_pairs(
    d: Any,
    pairs: list[tuple[NetlistEdge, NetlistEdge, str, str]],
    node_points: dict[str, tuple[float, float]],
    unit: float,
) -> None:
    import schemdraw.elements as elm

    count_by_base: dict[str, int] = {}
    for edge, ground_edge, base, branch in pairs:
        if base not in node_points:
            continue
        idx = count_by_base.get(base, 0)
        count_by_base[base] = idx + 1
        base_point = node_points[base]
        offset = -0.85 - idx * 0.75
        top = (base_point[0] + offset, base_point[1])
        d.push()
        d += elm.Line().left().at(base_point).length(abs(offset))
        d += elm.Dot()
        _draw_schemdraw_node_label(d, branch, top, force=True)
        d += _schemdraw_element(edge).down().at(top)
        _draw_schemdraw_text(d, _clean_ref(edge.ref), (top[0] + 0.72, top[1] - unit * 0.25), fontsize=7)
        second_start = d.here
        d += _schemdraw_element(ground_edge).down()
        _draw_schemdraw_text(d, _clean_ref(ground_edge.ref), (second_start[0] + 0.72, second_start[1] - unit * 0.45), fontsize=7)
        d += elm.Ground()
        d.pop()


def _schemdraw_branch_offsets(n: int) -> list[float]:
    if n <= 1:
        return [0.0]
    if n == 2:
        return [-0.55, 0.55]
    return [(-0.55 * (n - 1) / 2.0) + 0.55 * i for i in range(n)]


def _draw_schemdraw_node_label(d: Any, node: str, point: Any, force: bool = False) -> None:
    import schemdraw.elements as elm

    if not force and not _should_label_node(node):
        return
    x, y = float(point[0]), float(point[1])
    d.push()
    d += elm.Label().at((x, y + 0.38)).label(_short_node(node), fontsize=7)
    d.pop()


def _draw_schemdraw_text(d: Any, text: str, point: tuple[float, float], fontsize: int = 8) -> None:
    import schemdraw.elements as elm

    d.push()
    d += elm.Label().at(point).label(text, fontsize=fontsize)
    d.pop()


def _should_label_node(node: str) -> bool:
    return node in {"src", "out", "electrode", "harmonic_mid"} or not node.startswith("Xload.")


def _clean_ref(ref: str) -> str:
    return ref.replace("Xload:", "load.")


def _draw_component(ax: Any, kind: str, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    if kind == "resistor":
        _draw_resistor(ax, p1, p2, label)
    elif kind == "capacitor":
        _draw_capacitor(ax, p1, p2, label)
    elif kind == "inductor":
        _draw_inductor(ax, p1, p2, label)
    elif kind == "voltage":
        _draw_voltage_source(ax, p1, p2, label)
    elif kind == "current":
        _draw_current_source(ax, p1, p2, label)
    else:
        _draw_box_component(ax, p1, p2, label)


def _draw_resistor(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    u, v, length = _basis(p1, p2)
    lead = min(0.45, length * 0.2)
    start = _add(p1, _mul(u, lead))
    end = _add(p2, _mul(u, -lead))
    _draw_wire(ax, p1, start)
    _draw_wire(ax, end, p2)
    points = [start]
    amp = 0.16
    segments = 6
    for i in range(1, segments):
        frac = i / segments
        sign = 1 if i % 2 else -1
        base = _add(start, _mul(u, frac * _dist(start, end)))
        points.append(_add(base, _mul(v, amp * sign)))
    points.append(end)
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#202020", lw=1.7)
    _label_component(ax, p1, p2, label)


def _draw_capacitor(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    u, v, length = _basis(p1, p2)
    center = _mid(p1, p2)
    gap = min(0.18, length * 0.08)
    plate_len = 0.55
    a = _add(center, _mul(u, -gap))
    b = _add(center, _mul(u, gap))
    _draw_wire(ax, p1, a)
    _draw_wire(ax, b, p2)
    for point in [a, b]:
        p_left = _add(point, _mul(v, -plate_len / 2))
        p_right = _add(point, _mul(v, plate_len / 2))
        ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color="#202020", lw=2.0)
    _label_component(ax, p1, p2, label)


def _draw_inductor(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    u, v, length = _basis(p1, p2)
    lead = min(0.45, length * 0.18)
    start = _add(p1, _mul(u, lead))
    end = _add(p2, _mul(u, -lead))
    _draw_wire(ax, p1, start)
    _draw_wire(ax, end, p2)
    coil_len = _dist(start, end)
    loops = 4
    xs: list[float] = []
    ys: list[float] = []
    for i in range(80):
        t = i / 79
        base = _add(start, _mul(u, coil_len * t))
        offset = math.sin(t * loops * math.pi) * 0.22
        point = _add(base, _mul(v, offset))
        xs.append(point[0])
        ys.append(point[1])
    ax.plot(xs, ys, color="#202020", lw=1.7)
    _label_component(ax, p1, p2, label)


def _draw_voltage_source(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    from matplotlib.patches import Circle

    u, v, length = _basis(p1, p2)
    center = _mid(p1, p2)
    r = min(0.36, length * 0.2)
    a = _add(center, _mul(u, -r))
    b = _add(center, _mul(u, r))
    _draw_wire(ax, p1, a)
    _draw_wire(ax, b, p2)
    circle = ax.add_patch(Circle(center, r, fill=False, ec="#202020", lw=1.7))
    circle.set_zorder(2)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(36):
        t = (i / 35 - 0.5) * 1.2
        base = _add(center, _mul(v, t * r))
        wave = math.sin((i / 35) * 2 * math.pi) * r * 0.28
        point = _add(base, _mul(u, wave))
        xs.append(point[0])
        ys.append(point[1])
    ax.plot(xs, ys, color="#202020", lw=1.2)
    _label_component(ax, p1, p2, label)


def _draw_current_source(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    u, _v, length = _basis(p1, p2)
    center = _mid(p1, p2)
    r = min(0.36, length * 0.2)
    _draw_voltage_source(ax, p1, p2, label)
    start = _add(center, _mul(u, -r * 0.45))
    end = _add(center, _mul(u, r * 0.45))
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#202020"})


def _draw_box_component(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    u, v, length = _basis(p1, p2)
    center = _mid(p1, p2)
    width = min(0.95, length * 0.45)
    height = 0.45
    start = _add(center, _mul(u, -width / 2))
    end = _add(center, _mul(u, width / 2))
    _draw_wire(ax, p1, start)
    _draw_wire(ax, end, p2)
    corners = [
        _add(_add(center, _mul(u, -width / 2)), _mul(v, -height / 2)),
        _add(_add(center, _mul(u, width / 2)), _mul(v, -height / 2)),
        _add(_add(center, _mul(u, width / 2)), _mul(v, height / 2)),
        _add(_add(center, _mul(u, -width / 2)), _mul(v, height / 2)),
        _add(_add(center, _mul(u, -width / 2)), _mul(v, -height / 2)),
    ]
    ax.plot([p[0] for p in corners], [p[1] for p in corners], color="#202020", lw=1.5)
    _label_component(ax, p1, p2, label)


def _draw_wire(ax: Any, p1: tuple[float, float], p2: tuple[float, float]) -> None:
    if _dist(p1, p2) < 1e-9:
        return
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#202020", lw=1.3, solid_capstyle="round", zorder=1)


def _draw_connection_dot(ax: Any, p: tuple[float, float]) -> None:
    ax.scatter([p[0]], [p[1]], s=26, c="#202020", zorder=4)


def _draw_ground_symbol(ax: Any, p: tuple[float, float]) -> None:
    x, y = p
    ax.plot([x, x], [y, y - 0.18], color="#202020", lw=1.3)
    widths = [0.48, 0.32, 0.16]
    for i, w in enumerate(widths):
        yy = y - 0.18 - i * 0.12
        ax.plot([x - w / 2, x + w / 2], [yy, yy], color="#202020", lw=1.3)


def _label_component(ax: Any, p1: tuple[float, float], p2: tuple[float, float], label: str) -> None:
    _u, v, _length = _basis(p1, p2)
    center = _mid(p1, p2)
    offset = _mul(v, 0.38)
    if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
        offset = (0.72, 0.0)
    pos = _add(center, offset)
    ax.text(pos[0], pos[1], label.replace("Xload:", "load."), ha="center", va="center", fontsize=8.5, color="#202020")


def _basis(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float], float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(math.hypot(dx, dy), 1e-9)
    u = (dx / length, dy / length)
    v = (-u[1], u[0])
    return u, v, length


def _mid(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _add(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    return (p1[0] + p2[0], p1[1] + p2[1])


def _mul(p: tuple[float, float], scalar: float) -> tuple[float, float]:
    return (p[0] * scalar, p[1] * scalar)


def _layout_nodes(nodes: list[str], edges: list[NetlistEdge]) -> dict[str, tuple[float, float]]:
    if not nodes:
        return {}
    graph: dict[str, set[str]] = {n: set() for n in nodes}
    for edge in edges:
        graph[edge.n1].add(edge.n2)
        graph[edge.n2].add(edge.n1)

    start = "src" if "src" in graph else next((n for n in nodes if n != "0"), nodes[0])
    level: dict[str, int] = {start: 0}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for nxt in sorted(graph[node], key=_node_sort_key):
            if nxt == "0" or nxt in level:
                continue
            level[nxt] = level[node] + 1
            queue.append(nxt)
    for node in nodes:
        if node != "0" and node not in level:
            level[node] = max(level.values(), default=0) + 1

    buckets: dict[int, list[str]] = {}
    for node, lvl in level.items():
        buckets.setdefault(lvl, []).append(node)
    positions: dict[str, tuple[float, float]] = {}
    for lvl, bucket in sorted(buckets.items()):
        bucket = sorted(bucket, key=_node_sort_key)
        for i, node in enumerate(bucket):
            spread = len(bucket) - 1
            y = 1.1 - (i - spread / 2.0) * 1.15
            positions[node] = (float(lvl) * 2.1, y)
    if "0" in nodes:
        xs = [p[0] for p in positions.values()] or [0.0]
        positions["0"] = ((min(xs) + max(xs)) / 2.0, -1.85)
    return positions


def _draw_edge(ax: Any, p1: tuple[float, float], p2: tuple[float, float], edge: NetlistEdge, rad: float) -> None:
    color = {
        "resistor": "#20639B",
        "capacitor": "#3CAEA3",
        "inductor": "#ED553B",
        "voltage": "#8A5CF6",
        "current": "#8A5CF6",
    }.get(edge.kind, "#555555")
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "lw": 2.0,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=2,
    )
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max(math.hypot(dx, dy), 1e-9)
    ox, oy = -dy / length * (0.18 + abs(rad) * 0.35), dx / length * (0.18 + abs(rad) * 0.35)
    label = edge.ref
    ax.text(
        mx + ox,
        my + oy,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "lw": 0.6, "alpha": 0.88},
        zorder=3,
    )


def _strip_comment(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("*"):
        return ""
    return line


def _kind_name(kind: str) -> str:
    return {
        "R": "resistor",
        "C": "capacitor",
        "L": "inductor",
        "V": "voltage",
        "I": "current",
        "D": "diode",
    }.get(kind.upper(), "component")


def _short_node(node: str) -> str:
    return node.replace("Xload.", "load.")


def _node_sort_key(node: str) -> tuple[int, str]:
    if node == "src":
        return (0, node)
    if node in {"out", "electrode"}:
        return (1, node)
    if node == "0":
        return (99, node)
    return (10, node)
