"""Deterministic numeric views of ``circuit_graph.v1`` records.

The flat view is intentionally simple and gives non-graph baselines a fair
description of component values.  The relational view keeps component/net
nodes and positive/negative terminal directions for the GNN comparison.
Neither view reads files or knows how a study was executed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_KINDS = ("resistor", "inductor", "capacitor", "other")
_PORT_ROLES = ("source", "load", "ground")
_RELATIONS = {("p", False): 0, ("n", False): 1, ("p", True): 2, ("n", True): 3}


@dataclass(frozen=True)
class GraphBatch:
    """Padded component/net graphs aligned with a sample table."""

    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray
    component_mask: np.ndarray
    source_mask: np.ndarray
    load_mask: np.ndarray
    ground_mask: np.ndarray

    def take(self, rows: np.ndarray) -> GraphBatch:
        return GraphBatch(
            self.node_features[rows],
            self.adjacency[rows],
            self.node_mask[rows],
            self.component_mask[rows],
            self.source_mask[rows],
            self.load_mask[rows],
            self.ground_mask[rows],
        )


@dataclass(frozen=True)
class FlatGraphFeatures:
    """Fixed-width graph descriptors for ridge and MLP baselines."""

    matrix: np.ndarray
    names: tuple[str, ...]


def encode_graph_batch(
    graphs: Mapping[str, Mapping[str, Any]],
    graph_ids: Sequence[str],
) -> GraphBatch:
    """Encode sample-aligned graphs as padded relation tensors."""

    parsed = [_graph_parts(_named_graph(graphs, graph_id)) for graph_id in graph_ids]
    max_nodes = max(len(components) + len(nets) for components, nets, _ports in parsed)
    rows = len(parsed)
    features = np.zeros((rows, max_nodes, 12), dtype=float)
    adjacency = np.zeros((rows, 4, max_nodes, max_nodes), dtype=float)
    node_mask = np.zeros((rows, max_nodes), dtype=float)
    component_mask = np.zeros_like(node_mask)
    port_masks = {role: np.zeros_like(node_mask) for role in _PORT_ROLES}

    for row, (components, nets, ports) in enumerate(parsed):
        _encode_one_graph(
            components,
            nets,
            ports,
            features[row],
            adjacency[row],
            node_mask[row],
            component_mask[row],
            port_masks,
            row,
        )
    _normalize_adjacency(adjacency)
    return GraphBatch(
        features,
        adjacency,
        node_mask,
        component_mask,
        port_masks["source"],
        port_masks["load"],
        port_masks["ground"],
    )


def flat_graph_features(
    graphs: Mapping[str, Mapping[str, Any]],
    graph_ids: Sequence[str],
) -> FlatGraphFeatures:
    """Return compact value/count descriptors without encoding graph identity."""

    names = (
        "graph.component_count",
        "graph.net_count",
        *(f"graph.{kind}_count" for kind in _KINDS),
        *(f"graph.{kind}_mean_scaled_log_value" for kind in _KINDS),
        "graph.series_loss_count",
        "graph.series_loss_mean_log1p_ohm",
        "graph.source_degree",
        "graph.load_degree",
    )
    rows = [_flat_graph_row(*_graph_parts(_named_graph(graphs, graph_id))) for graph_id in graph_ids]
    return FlatGraphFeatures(np.asarray(rows, dtype=float), tuple(names))


def _flat_graph_row(
    components: list[Mapping[str, Any]],
    nets: list[str],
    ports: list[Mapping[str, str]],
) -> list[float]:
    by_kind: dict[str, list[float]] = {kind: [] for kind in _KINDS}
    losses: list[float] = []
    degrees = dict.fromkeys(nets, 0)
    for component in components:
        kind = _kind(component.get("kind"))
        by_kind[kind].append(_scaled_log_value(kind, component.get("value")))
        loss = component.get("series_resistance_ohm")
        if loss is not None:
            losses.append(math.log1p(_finite_nonnegative(loss, "series_resistance_ohm")))
        for terminal in _terminals(component):
            degrees[terminal["net"]] += 1
    role_nets = {port["role"]: port["net"] for port in ports}
    return [
        float(len(components)),
        float(len(nets)),
        *(float(len(by_kind[kind])) for kind in _KINDS),
        *(_mean_or_zero(by_kind[kind]) for kind in _KINDS),
        float(len(losses)),
        _mean_or_zero(losses),
        float(degrees[role_nets["source"]]),
        float(degrees[role_nets["load"]]),
    ]


def _named_graph(graphs: Mapping[str, Mapping[str, Any]], graph_id: str) -> Mapping[str, Any]:
    try:
        graph = graphs[graph_id]
    except KeyError as exc:
        raise ValueError(f"sample references unknown graph_id {graph_id!r}") from exc
    if graph.get("graph_id") != graph_id or graph.get("schema") != "circuit_graph.v1":
        raise ValueError(f"invalid circuit graph record for {graph_id!r}")
    return graph


def _graph_parts(
    graph: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[str], list[Mapping[str, str]]]:
    components, component_nets = _parse_components(graph.get("components"))
    ports, port_nets = _parse_ports(graph.get("ports"))
    return components, sorted(component_nets | port_nets), ports


def _parse_components(raw_components: object) -> tuple[list[Mapping[str, Any]], set[str]]:
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("circuit graph components must be a non-empty list")
    components: list[Mapping[str, Any]] = []
    nets: set[str] = set()
    for component in raw_components:
        if not isinstance(component, Mapping):
            raise ValueError("circuit graph component must be a mapping")
        components.append(component)
        nets.update(terminal["net"] for terminal in _terminals(component))
    return sorted(components, key=lambda item: str(item.get("reference", ""))), nets


def _parse_ports(raw_ports: object) -> tuple[list[Mapping[str, str]], set[str]]:
    if not isinstance(raw_ports, list):
        raise ValueError("circuit graph ports must be a list")
    ports: list[Mapping[str, str]] = []
    nets: set[str] = set()
    for port in raw_ports:
        if not isinstance(port, Mapping) or port.get("role") not in _PORT_ROLES or not isinstance(port.get("net"), str):
            raise ValueError("circuit graph has an invalid port")
        ports.append({"role": str(port["role"]), "net": str(port["net"])})
        nets.add(str(port["net"]))
    roles = {port["role"] for port in ports}
    if roles != set(_PORT_ROLES):
        raise ValueError("circuit graph must declare source, load, and ground ports")
    return ports, nets


def _terminals(component: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = component.get("terminals")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("circuit graph component must have two terminals")
    terminals: list[dict[str, str]] = []
    for terminal in raw:
        if not isinstance(terminal, Mapping) or terminal.get("name") not in {"p", "n"}:
            raise ValueError("circuit graph terminal must be named p or n")
        net = terminal.get("net")
        if not isinstance(net, str) or not net:
            raise ValueError("circuit graph terminal net must be non-empty text")
        terminals.append({"name": str(terminal["name"]), "net": net})
    if {terminal["name"] for terminal in terminals} != {"p", "n"}:
        raise ValueError("circuit graph component must have one p and one n terminal")
    return terminals


def _encode_one_graph(
    components: list[Mapping[str, Any]],
    nets: list[str],
    ports: list[Mapping[str, str]],
    features: np.ndarray,
    adjacency: np.ndarray,
    node_mask: np.ndarray,
    component_mask: np.ndarray,
    port_masks: Mapping[str, np.ndarray],
    row: int,
) -> None:
    net_index = {net: len(components) + index for index, net in enumerate(nets)}
    node_mask[: len(components) + len(nets)] = 1.0
    component_mask[: len(components)] = 1.0
    for index, component in enumerate(components):
        kind = _kind(component.get("kind"))
        features[index, 0] = 1.0
        features[index, 2 + _KINDS.index(kind)] = 1.0
        features[index, 6] = _scaled_log_value(kind, component.get("value"))
        loss = component.get("series_resistance_ohm")
        features[index, 7] = 0.0 if loss is None else math.log1p(_finite_nonnegative(loss, "series_resistance_ohm"))
        for terminal in _terminals(component):
            target = net_index[terminal["net"]]
            adjacency[_RELATIONS[(terminal["name"], False)], target, index] = 1.0
            adjacency[_RELATIONS[(terminal["name"], True)], index, target] = 1.0
    for net, index in net_index.items():
        features[index, 1] = 1.0
        roles = [port["role"] for port in ports if port["net"] == net]
        for role in roles:
            features[index, 8 + _PORT_ROLES.index(role)] = 1.0
            port_masks[role][row, index] = 1.0
        if not roles:
            features[index, 11] = 1.0


def _normalize_adjacency(adjacency: np.ndarray) -> None:
    degree = adjacency.sum(axis=3, keepdims=True)
    np.divide(adjacency, degree, out=adjacency, where=degree > 0.0)


def _kind(value: object) -> str:
    text = str(value).strip().lower()
    return text if text in _KINDS[:-1] else "other"


def _scaled_log_value(kind: str, value: object) -> float:
    number = _finite_positive(value, "component value")
    offsets = {"capacitor": 12.0, "inductor": 6.0, "resistor": 0.0, "other": 0.0}
    return math.log10(number) + offsets[kind]


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
