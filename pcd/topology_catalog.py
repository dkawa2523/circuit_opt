"""Verified RF matching templates shared by input planning and netlist generation.

The catalog contains connectivity and component roles only.  Values, search
ranges, losses, operating frequency, and engineering limits remain explicit in
each case because the topology name alone cannot qualify those quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

_SOURCE = "source"
_LOAD = "load"
_GROUND = "ground"


@dataclass(frozen=True, slots=True)
class TopologyElement:
    """One named component and its electrical role in a template."""

    reference: str
    component_type: str
    role: str
    n1: str
    n2: str


@dataclass(frozen=True, slots=True)
class TopologyTemplate:
    """A reviewed lumped matching-network connection template."""

    name: str
    elements: tuple[TopologyElement, ...]
    purpose: str = "lumped RF impedance matching"

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(element.reference for element in self.elements)

    def connections(self, output_node: str = "electrode") -> tuple[tuple[str, str, str], ...]:
        nodes = {_SOURCE: "src", _LOAD: output_node, _GROUND: "0"}
        return tuple(
            (element.reference, nodes.get(element.n1, element.n1), nodes.get(element.n2, element.n2))
            for element in self.elements
        )


MATCHING_TOPOLOGIES: dict[str, TopologyTemplate] = {
    "l_match": TopologyTemplate(
        "l_match",
        (
            TopologyElement("L1", "inductor", "series_match", _SOURCE, _LOAD),
            TopologyElement("C1", "capacitor", "load_shunt", _LOAD, _GROUND),
        ),
    ),
    "pi_match": TopologyTemplate(
        "pi_match",
        (
            TopologyElement("C1", "capacitor", "source_shunt", _SOURCE, _GROUND),
            TopologyElement("L1", "inductor", "series_match", _SOURCE, _LOAD),
            TopologyElement("C2", "capacitor", "load_shunt", _LOAD, _GROUND),
        ),
    ),
    "pi_match_harmonic": TopologyTemplate(
        "pi_match_harmonic",
        (
            TopologyElement("C1", "capacitor", "source_shunt", _SOURCE, _GROUND),
            TopologyElement("L1", "inductor", "series_match", _SOURCE, _LOAD),
            TopologyElement("C2", "capacitor", "load_shunt", _LOAD, _GROUND),
            TopologyElement("Lh", "inductor", "harmonic_trap", _LOAD, "harmonic_mid"),
            TopologyElement("Ch", "capacitor", "harmonic_trap", "harmonic_mid", _GROUND),
        ),
    ),
}


def matching_topology(name: str) -> TopologyTemplate:
    """Return one reviewed topology or reject an unregistered connection."""

    try:
        return MATCHING_TOPOLOGIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown matching topology {name!r}") from exc
