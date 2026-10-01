"""Solver-neutral graph representation for structured circuit templates.

The graph is deliberately smaller than a SPICE parser.  It represents the
structured components produced by a circuit builder as component, terminal,
and net nodes.  Netlist rendering remains in :mod:`pcd.netlist`; ML-specific
tensors and model dependencies remain in :mod:`pcd.ml`.

Arbitrary raw SPICE statements are not guessed into this representation.  An
adapter must either supply their terminal semantics explicitly or report that
the circuit is not graph-ready.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

GraphNodeKind = Literal["component", "terminal", "net"]
GraphValue = float | str


def _required_text(value: str, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _graph_value(value: object, reference: str) -> GraphValue:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"component {reference!r} value must be a number or string")
    if isinstance(value, str):
        return _required_text(value, f"component {reference!r} value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"component {reference!r} numeric value must be finite")
    return number


@dataclass(frozen=True)
class GraphTerminal:
    """One named terminal of a component and the electrical net it joins."""

    name: str
    net: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "terminal name"))
        object.__setattr__(self, "net", _required_text(self.net, "terminal net"))


@dataclass(frozen=True)
class GraphComponent:
    """A logical component with terminals, value, and optional series loss.

    ``series_resistance_ohm`` belongs to the physical component description.
    Solver-only devices used to measure current are deliberately absent from
    this record.
    """

    reference: str
    kind: str
    terminals: tuple[GraphTerminal, ...]
    value: GraphValue
    series_resistance_ohm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference", _required_text(self.reference, "component reference"))
        object.__setattr__(self, "kind", _required_text(self.kind, "component kind"))
        if len(self.terminals) < 2:
            raise ValueError(f"component {self.reference!r} must have at least two terminals")
        names = [terminal.name for terminal in self.terminals]
        if len(names) != len(set(names)):
            raise ValueError(f"component {self.reference!r} has duplicate terminal names")
        object.__setattr__(self, "value", _graph_value(self.value, self.reference))
        resistance = self.series_resistance_ohm
        if resistance is not None:
            if isinstance(resistance, bool) or not isinstance(resistance, (int, float)):
                raise TypeError(f"component {self.reference!r} series resistance must be numeric")
            number = float(resistance)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"component {self.reference!r} series resistance must be finite and non-negative")
            object.__setattr__(self, "series_resistance_ohm", number)


@dataclass(frozen=True)
class GraphPort:
    """A semantic circuit boundary attached to one net."""

    role: str
    net: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required_text(self.role, "port role"))
        object.__setattr__(self, "net", _required_text(self.net, "port net"))


@dataclass(frozen=True)
class GraphNode:
    """One node in the expanded component-terminal-net projection."""

    node_id: str
    kind: GraphNodeKind
    label: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    """One typed relation in the expanded heterogeneous graph."""

    source: str
    target: str
    relation: Literal["has_terminal", "connected_to_net"]


@dataclass(frozen=True)
class CircuitGraph:
    """Canonical structured-circuit record shared by data producers and ML."""

    SCHEMA: ClassVar[str] = "circuit_graph.v1"
    REQUIRED_PORT_ROLES: ClassVar[frozenset[str]] = frozenset({"source", "load", "ground"})

    topology_family: str
    components: tuple[GraphComponent, ...]
    ports: tuple[GraphPort, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_family", _required_text(self.topology_family, "topology family"))
        if not self.components:
            raise ValueError("circuit graph must contain at least one component")
        references = [component.reference for component in self.components]
        if len(references) != len(set(references)):
            raise ValueError("circuit graph component references must be unique")
        roles = [port.role for port in self.ports]
        if len(roles) != len(set(roles)):
            raise ValueError("circuit graph port roles must be unique")
        missing_roles = sorted(self.REQUIRED_PORT_ROLES - set(roles))
        if missing_roles:
            raise ValueError(f"circuit graph is missing required ports: {missing_roles}")
        nets = self.net_names
        unknown = sorted({port.net for port in self.ports} - set(nets))
        if unknown:
            raise ValueError(f"circuit graph ports reference unknown nets: {unknown}")

    @property
    def net_names(self) -> tuple[str, ...]:
        return tuple(sorted({terminal.net for component in self.components for terminal in component.terminals}))

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        port_roles = {
            net: tuple(sorted(port.role for port in self.ports if port.net == net))
            for net in {port.net for port in self.ports}
        }
        nodes: list[GraphNode] = []
        for component in self._ordered_components():
            nodes.append(GraphNode(f"component:{component.reference}", "component", label=component.kind))
            nodes.extend(
                GraphNode(f"terminal:{component.reference}:{terminal.name}", "terminal", label=terminal.name)
                for terminal in component.terminals
            )
        nodes.extend(GraphNode(f"net:{net}", "net", roles=port_roles.get(net, ())) for net in self.net_names)
        return tuple(nodes)

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        edges: list[GraphEdge] = []
        for component in self._ordered_components():
            component_id = f"component:{component.reference}"
            for terminal in component.terminals:
                terminal_id = f"terminal:{component.reference}:{terminal.name}"
                edges.append(GraphEdge(component_id, terminal_id, "has_terminal"))
                edges.append(GraphEdge(terminal_id, f"net:{terminal.net}", "connected_to_net"))
        return tuple(edges)

    @property
    def topology_fingerprint(self) -> str:
        """Stable authored-structure identity, excluding component values.

        The explicit ``topology_family`` is the cross-design grouping key.
        This fingerprint additionally detects wiring changes inside that
        family.  It is deterministic under declaration reordering, but is not
        presented as a general graph-isomorphism proof.
        """

        return _sha256(self._record(include_values=False))

    @property
    def instance_fingerprint(self) -> str:
        """Stable identity of topology, wiring, and resolved component values."""

        return _sha256(self._record(include_values=True))

    def to_dict(self) -> dict[str, Any]:
        """Return the portable graph contract without framework-specific tensors."""

        record = self._record(include_values=True)
        return {
            "schema": self.SCHEMA,
            **record,
            "topology_fingerprint": self.topology_fingerprint,
            "instance_fingerprint": self.instance_fingerprint,
        }

    def _ordered_components(self) -> tuple[GraphComponent, ...]:
        return tuple(sorted(self.components, key=lambda component: component.reference))

    def _record(self, *, include_values: bool) -> dict[str, Any]:
        components: list[dict[str, Any]] = []
        for component in self._ordered_components():
            item: dict[str, Any] = {
                "reference": component.reference,
                "kind": component.kind,
                "terminals": [
                    {"name": terminal.name, "net": terminal.net}
                    for terminal in sorted(component.terminals, key=lambda terminal: terminal.name)
                ],
            }
            if include_values:
                item["value"] = component.value
                item["series_resistance_ohm"] = component.series_resistance_ohm
            components.append(item)
        return {
            "topology_family": self.topology_family,
            "ports": [{"role": port.role, "net": port.net} for port in sorted(self.ports, key=lambda port: port.role)],
            "components": components,
        }


def _sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
