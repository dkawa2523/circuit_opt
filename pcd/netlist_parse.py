"""Reading a SPICE netlist back into a graph.

The inverse of :mod:`pcd.netlist`: text in, two-terminal edges out.  Subcircuits
are expanded inline so a load model's internals stay visible rather than
collapsing into one opaque block.

This module is pure text processing — it imports nothing else from ``pcd`` and
knows nothing about cases, solvers, or drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: SPICE reference-designator prefixes this reader understands.
TWO_TERMINAL_PREFIXES = {"R", "C", "L", "V", "I", "D"}

_KIND_BY_PREFIX = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "V": "voltage",
    "I": "current",
    "D": "diode",
}

#: Four-terminal transmission lines.  Drawn as a two-terminal block between the
#: input and output conductors; the two return terminals are usually ground.
TRANSMISSION_PREFIXES = {"T", "O"}

#: Subcircuit definition: (ports, body lines), keyed by subckt name.
Subckts = dict[str, tuple[list[str], list[str]]]


@dataclass(frozen=True)
class NetlistEdge:
    """One two-terminal element between two nodes."""

    ref: str
    n1: str
    n2: str
    value: str = ""
    kind: str = "component"


@dataclass(frozen=True)
class Coupling:
    """A `K` line: magnetic coupling between two inductors, not a branch."""

    ref: str
    inductors: tuple[str, str]
    coefficient: float


@dataclass
class ParsedNetlist:
    edges: list[NetlistEdge]
    subckt_names: list[str]
    couplings: list[Coupling] = field(default_factory=list)

    @property
    def nodes(self) -> list[str]:
        return sorted({e.n1 for e in self.edges} | {e.n2 for e in self.edges}, key=node_sort_key)


def parse_netlist(path_or_text: str | Path) -> ParsedNetlist:
    """Parse a netlist into drawable edges, expanding subcircuits inline."""

    top_lines, subckts = split_netlist(_read_netlist(path_or_text))
    edges: list[NetlistEdge] = []
    couplings: list[Coupling] = []
    for line in top_lines:
        coupling = _parse_coupling_line(line)
        if coupling:
            couplings.append(coupling)
            continue
        edges.extend(_parse_component_line(line, subckts))
    return ParsedNetlist(edges=edges, subckt_names=sorted(subckts), couplings=couplings)


def _parse_coupling_line(line: str) -> Coupling | None:
    """`Kname La Lb coefficient` couples two inductors into a transformer."""

    parts = line.split()
    if len(parts) < 4 or parts[0][0].upper() != "K":
        return None
    try:
        coefficient = float(parts[3])
    except ValueError:
        return None
    return Coupling(ref=parts[0], inductors=(parts[1], parts[2]), coefficient=coefficient)


def netlist_summary(path: str | Path) -> dict[str, Any]:
    """Machine-readable topology summary, written by `visualize-netlist`."""

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
        "couplings": [c.__dict__ for c in parsed.couplings],
        "edges": [edge.__dict__ for edge in parsed.edges],
    }


def node_sort_key(node: str) -> tuple[int, str]:
    """Order nodes source -> output -> everything else -> ground."""

    if node == "src":
        return (0, node)
    if node in {"out", "electrode"}:
        return (1, node)
    if node == "0":
        return (99, node)
    return (10, node)


# -----------------------------------------------------------------------------
# Splitting the file
# -----------------------------------------------------------------------------


def _read_netlist(path_or_text: str | Path) -> str:
    raw = str(path_or_text)
    if isinstance(path_or_text, Path) or ("\n" not in raw and Path(raw).exists()):
        return Path(path_or_text).read_text(encoding="utf-8")
    return raw


def _strip_comment(line: str) -> str:
    return "" if line.strip().startswith("*") else line


def _is_ignorable(low: str) -> bool:
    """Directives that carry no topology."""

    return low.startswith((".param", ".options", ".save")) or (low.startswith(".") and not low.startswith(".subckt"))


def split_netlist(text: str) -> tuple[list[str], Subckts]:
    """Separate top-level component lines from subcircuit definitions.

    ``.control`` blocks hold simulator commands rather than topology, so
    everything between ``.control`` and ``.endc`` is skipped.
    """

    subckts: Subckts = {}
    top_lines: list[str] = []
    in_control = False
    name: str | None = None
    ports: list[str] = []
    body: list[str] = []

    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        low = line.lower()
        if low in {".control", ".endc"}:
            in_control = low == ".control"
        elif in_control:
            continue
        elif low.startswith(".subckt"):
            parts = line.split()
            name, ports, body = parts[1], parts[2:], []
        elif low.startswith(".ends"):
            if name:
                subckts[name] = (ports, body)
            name, ports, body = None, [], []
        elif _is_ignorable(low):
            continue
        elif name:
            body.append(line)
        else:
            top_lines.append(line)
    return top_lines, subckts


# -----------------------------------------------------------------------------
# Component lines
# -----------------------------------------------------------------------------


def _parse_component_line(line: str, subckts: Subckts) -> list[NetlistEdge]:
    parts = line.split()
    if len(parts) < 3:
        return []
    ref = parts[0]
    prefix = ref[0].upper()
    if prefix == "X" and len(parts) >= 4:
        return _expand_subckt(ref, parts[1:-1], parts[-1], subckts)
    if prefix in TRANSMISSION_PREFIXES and len(parts) >= 5:
        # T n1 nref1 n2 nref2 ...: draw the signal path n1 -> n2.
        return [NetlistEdge(ref=ref, n1=parts[1], n2=parts[3], value=" ".join(parts[5:]), kind="tline")]
    if prefix in TWO_TERMINAL_PREFIXES:
        kind = _KIND_BY_PREFIX[prefix]
        return [NetlistEdge(ref=ref, n1=parts[1], n2=parts[2], value=" ".join(parts[3:]), kind=kind)]
    return []


def _expand_subckt(ref: str, external_nodes: list[str], model: str, subckts: Subckts) -> list[NetlistEdge]:
    """Inline a subcircuit's body, or keep it as one edge if it is undefined."""

    if model not in subckts:
        return [NetlistEdge(ref=ref, n1=external_nodes[0], n2=external_nodes[1], value=model, kind="subckt")]
    ports, body = subckts[model]
    mapping = dict(zip(ports, external_nodes, strict=False))
    edges: list[NetlistEdge] = []
    for line in body:
        edges.extend(_parse_subckt_component_line(line, ref, mapping))
    return edges


def _parse_subckt_component_line(line: str, instance: str, mapping: dict[str, str]) -> list[NetlistEdge]:
    parts = line.split()
    if len(parts) < 3:
        return []
    prefix = parts[0][0].upper()
    if prefix not in TWO_TERMINAL_PREFIXES:
        return []
    return [
        NetlistEdge(
            ref=f"{instance}:{parts[0]}",
            n1=_map_subckt_node(parts[1], instance, mapping),
            n2=_map_subckt_node(parts[2], instance, mapping),
            value=" ".join(parts[3:]),
            kind=_KIND_BY_PREFIX[prefix],
        )
    ]


def _map_subckt_node(node: str, instance: str, mapping: dict[str, str]) -> str:
    """Port nodes map to the caller's nets; internal nodes get namespaced."""

    if node in mapping:
        return mapping[node]
    return "0" if node == "0" else f"{instance}.{node}"
