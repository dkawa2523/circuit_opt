"""Small, explicit non-ideal two-terminal component conventions.

The platform does not own a component database or a thermal solver.  A case
may attach one effective series resistance to a declared L/C and ask for its
terminal voltage/current to be observed.  This covers capacitor ESR and
inductor DCR/core loss at the qualified operating point without inventing a
frequency-dependent vendor model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .case import Case
from .spice import resolve_value


def component_id(reference: str) -> str:
    """A stable identifier safe for generated SPICE names and metric keys."""

    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(reference)).strip("_")
    if not safe:
        raise ValueError("component reference must contain a letter, digit, or underscore")
    return safe


def meter_reference(reference: str) -> str:
    return f"Vobserve_{component_id(reference)}"


def meter_node(reference: str) -> str:
    return f"observe_{component_id(reference)}_meter"


def core_node(reference: str) -> str:
    return f"observe_{component_id(reference)}_core"


def loss_reference(reference: str) -> str:
    return f"Rloss_{component_id(reference)}"


def series_resistance_ohm(item: dict[str, Any], params: dict[str, Any] | None = None) -> float | None:
    """Resolve an optional effective ESR/DCR and reject active/non-finite data."""

    if "series_resistance_ohm" not in item:
        return None
    raw = resolve_value(item["series_resistance_ohm"], params or {})
    try:
        resistance = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"series_resistance_ohm for {item.get('ref', '?')} must resolve to a number") from exc
    if not math.isfinite(resistance) or resistance < 0:
        raise ValueError(f"series_resistance_ohm for {item.get('ref', '?')} must be finite and non-negative")
    return resistance


@dataclass(frozen=True)
class ComponentObservation:
    """How one declared component is probed and reported."""

    reference: str
    p: str
    n: str
    series_resistance_ohm: float | None = None

    @property
    def metric_id(self) -> str:
        return component_id(self.reference)

    @property
    def voltage_column(self) -> str:
        return f"component_{self.metric_id}_voltage_V"

    @property
    def current_column(self) -> str:
        return f"component_{self.metric_id}_current_A"

    @property
    def voltage_vector(self) -> str:
        return f"v({self.p},{self.n})" if self.n != "0" else f"v({self.p})"

    @property
    def current_vector(self) -> str:
        return f"i({meter_reference(self.reference)})"


def observed_components(case: Case, params: dict[str, Any] | None = None) -> tuple[ComponentObservation, ...]:
    """Observations declared on structured ``from_yaml`` components."""

    circuit = case.data.get("circuit", {}) or {}
    if str(circuit.get("builder", "from_yaml")) != "from_yaml":
        return ()
    observations: list[ComponentObservation] = []
    seen: set[str] = set()
    for item in circuit.get("components", []) or []:
        if not isinstance(item, dict) or not item.get("observe"):
            continue
        if "raw" in item:
            raise ValueError("raw circuit components cannot use observe; declare ref/n1/n2/value explicitly")
        reference = str(item["ref"])
        metric_id = component_id(reference)
        if metric_id in seen:
            raise ValueError(f"observed component identifiers collide after normalization: {reference}")
        seen.add(metric_id)
        resistance: float | None
        raw_resistance = item.get("series_resistance_ohm")
        if params is None and isinstance(raw_resistance, str):
            try:
                float(raw_resistance)
            except ValueError:
                resistance = None  # probe naming does not need the runtime parameter value
            else:
                resistance = series_resistance_ohm(item)
        else:
            resistance = series_resistance_ohm(item, params)
        observations.append(
            ComponentObservation(
                reference=reference,
                p=str(item["n1"]),
                n=str(item["n2"]),
                series_resistance_ohm=resistance,
            )
        )
    return tuple(observations)
