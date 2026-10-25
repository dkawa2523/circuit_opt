"""Read drawable connectivity without silently discarding circuit statements.

Simulation still owns SPICE interpretation. Unknown or conditional topology is
reported as partial rather than presented as a complete equivalent circuit.
"""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .netlist_import import flatten_netlist_file

_KINDS = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "V": "voltage",
    "I": "current",
    "D": "diode",
    "Q": "bjt",
    "M": "mosfet",
    "J": "jfet",
    "E": "vcvs",
    "G": "vccs",
    "F": "cccs",
    "H": "ccvs",
    "B": "behavioral",
    "T": "tline",
    "O": "tline",
    "S": "switch",
    "W": "switch",
}
_PORTS = {"Q": 3, "M": 4, "J": 3, "E": 4, "G": 4, "T": 4, "O": 4, "S": 4}
Subckts = dict[str, tuple[list[str], list[str]]]


@dataclass(frozen=True)
class NetlistEdge:
    ref: str
    n1: str
    n2: str
    value: str = ""
    kind: str = "component"
    terminals: tuple[str, ...] = ()

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.terminals or (self.n1, self.n2)


@dataclass(frozen=True)
class Coupling:
    ref: str
    inductors: tuple[str, str]
    coefficient: str | float


@dataclass
class ParsedNetlist:
    edges: list[NetlistEdge]
    subckt_names: list[str]
    couplings: list[Coupling] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)

    @property
    def nodes(self) -> list[str]:
        return sorted({node for edge in self.edges for node in edge.nodes}, key=node_sort_key)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.kind] = counts.get(edge.kind, 0) + 1
        return {
            "status": "partial" if self.issues else "complete",
            "issues": self.issues,
            "nodes": self.nodes,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "component_counts": counts,
            "subckts": self.subckt_names,
            "couplings": [c.__dict__ for c in self.couplings],
            "edges": [{**edge.__dict__, "terminals": list(edge.nodes)} for edge in self.edges],
            "parameters": self.parameters,
        }


def node_sort_key(node: str) -> tuple[int, str]:
    return (
        0 if node.lower() == "src" else 1 if node.lower() in {"out", "electrode"} else 99 if node == "0" else 10,
        node,
    )


def _read_netlist(path_or_text: str | Path) -> str:
    raw = str(path_or_text)
    if isinstance(path_or_text, Path) or ("\n" not in raw and Path(raw).is_file()):
        return flatten_netlist_file(Path(path_or_text))[0]
    return raw


def _is_device_line(line: str) -> bool:
    parts = line.split()
    if not parts:
        return False
    prefix = parts[0][0].upper()
    if any(char.isdigit() for char in parts[0]):
        return parts[0][0].isalpha()
    if prefix == "X":
        return len(parts) >= 4
    if prefix not in _KINDS or len(parts) < 4:
        return False
    value = parts[3].lower()
    return (
        prefix in _PORTS
        or prefix == "D"
        or bool(re.match(r"[+\-\d.{]", value))
        or value.startswith(("dc", "ac", "sin(", "pulse(", "pwl("))
    )


def _logical_lines(text: str, mode: str) -> list[str]:
    if mode not in {"auto", "deck", "fragment"}:
        raise ValueError("netlist format must be auto, deck, or fragment")
    physical = text.splitlines()
    if physical and (
        mode == "deck"
        or (
            mode == "auto"
            and physical[0].strip()
            and not physical[0].lstrip().startswith(("*", "."))
            and not _is_device_line(physical[0])
        )
    ):
        physical = physical[1:]
    lines: list[str] = []
    for raw in physical:
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and lines:
            lines[-1] += " " + line[1:].strip()
        else:
            lines.append(re.split(r"\s[;$]", line, maxsplit=1)[0])
    return lines


def split_netlist(text: str, mode: str = "auto") -> tuple[list[str], Subckts]:
    subckts: Subckts = {}
    top: list[str] = []
    block: list[str] | None = None
    in_control = False
    for line in _logical_lines(text, mode):
        parts = line.split()
        low = parts[0].lower()
        if low == ".control":
            in_control = True
        elif low == ".endc":
            in_control = False
        elif in_control:
            continue
        elif low == ".subckt" and len(parts) >= 2:
            ports = []
            for token in parts[2:]:
                if "=" in token or token.lower() == "params:":
                    break
                ports.append(token)
            block = []
            subckts[parts[1].lower()] = (ports, block)
        elif low == ".ends":
            block = None
        elif block is not None:
            block.append(line)
        else:
            top.append(line)
    return top, subckts


def parse_netlist(path_or_text: str | Path, *, netlist_mode: str = "auto") -> ParsedNetlist:
    top, subckts = split_netlist(_read_netlist(path_or_text), netlist_mode)
    parsed = ParsedNetlist([], sorted(subckts))
    node_names: dict[str, str] = {}
    models = {line.split()[1].lower() for line in top if line.lower().startswith(".model ") and len(line.split()) >= 3}

    def visit(
        line: str, namespace: str = "", mapping: dict[str, str] | None = None, stack: tuple[str, ...] = ()
    ) -> None:
        parts = line.split()
        if not parts:
            return
        low = parts[0].lower()
        if low == ".param":
            for key, value in re.findall(r"([A-Za-z_][\w.]*)\s*=\s*(\{[^}]*\}|[^\s]+)", " ".join(parts[1:])):
                if not namespace:
                    parsed.parameters[key.lower()] = value
            return
        if low.startswith("."):
            if low in {".if", ".elseif", ".else", ".endif", ".include", ".inc", ".lib"}:
                parsed.issues.append(f"topology not resolved by the drawing reader: {namespace}{line}")
            return
        ref = f"{namespace}{parts[0]}"
        prefix = parts[0][0].upper()
        ports = mapping or {}

        def node(value: str) -> str:
            resolved = ports.get(value.lower(), "0" if value == "0" else namespace.replace(":", ".") + value)
            return node_names.setdefault(resolved.lower(), resolved)

        if prefix == "K" and len(parts) >= 4:
            coefficient: str | float = parts[3]
            with suppress(ValueError):
                coefficient = float(coefficient)
            parsed.couplings.append(Coupling(ref, (namespace + parts[1], namespace + parts[2]), coefficient))
            return
        if prefix == "X":
            positional = [token for token in parts[1:] if "=" not in token and token.lower() != "params:"]
            if len(positional) < 3:
                parsed.issues.append(f"invalid subcircuit instance: {ref}")
                return
            model = positional[-1].lower()
            external = [node(item) for item in positional[:-1]]
            if model in subckts and model not in stack:
                names, body = subckts[model]
                if len(names) != len(external):
                    parsed.issues.append(f"subcircuit port count mismatch: {ref}")
                else:
                    for child in body:
                        visit(
                            child,
                            ref + ":",
                            dict(zip((name.lower() for name in names), external, strict=True)),
                            (*stack, model),
                        )
                    return
            else:
                parsed.issues.append(f"unexpanded subcircuit: {ref} ({model})")
            parsed.edges.append(NetlistEdge(ref, external[0], external[1], model, "subckt", tuple(external)))
            return
        count = _PORTS.get(prefix, 2)
        if prefix == "Q" and len(parts) >= 6 and parts[4].lower() not in models:
            if parts[5].lower() in models:
                count = 4
            elif not re.match(r"[+\-\d.]", parts[5]) and "=" not in parts[5]:
                parsed.issues.append(f"ambiguous BJT substrate/model terminals: {ref}")
        if prefix not in _KINDS or len(parts) < count + 2:
            parsed.issues.append(f"unsupported or truncated component: {ref}: {line}")
            return
        terminals = tuple(node(item) for item in parts[1 : count + 1])
        # Keep the conventional signal-path endpoints for old two-terminal readers.
        n2 = terminals[2] if prefix in {"T", "O"} else terminals[1]
        parsed.edges.append(NetlistEdge(ref, terminals[0], n2, " ".join(parts[count + 1 :]), _KINDS[prefix], terminals))

    for line in top:
        visit(line)
    refs = [edge.ref.lower() for edge in parsed.edges]
    if len(set(refs)) != len(refs):
        parsed.issues.append("duplicate component names (possibly conditional alternatives)")
    return parsed


def netlist_summary(path: str | Path, *, netlist_mode: str = "auto") -> dict[str, Any]:
    return parse_netlist(path, netlist_mode=netlist_mode).summary()
