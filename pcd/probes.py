"""Small typed probe declarations shared by netlist and solver adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .simulation import (
    AC_LOAD_VOLTAGE_COLUMN,
    CURRENT_COLUMN,
    LOAD_CURRENT_COLUMN,
    SOURCE_VOLTAGE_COLUMN,
    TIME_COLUMN,
    VOLTAGE_COLUMN,
)

LOAD_AMMETER = "Vload_meter"

RESERVED_PROBE_COLUMNS = frozenset(
    {
        TIME_COLUMN,
        VOLTAGE_COLUMN,
        CURRENT_COLUMN,
        SOURCE_VOLTAGE_COLUMN,
        LOAD_CURRENT_COLUMN,
        AC_LOAD_VOLTAGE_COLUMN,
        "frequency_Hz",
        # Frequency-domain vectors expand to ``<name>_re``/``<name>_im``.
        "voltage",
        "current",
    }
)


def component_id(reference: str) -> str:
    """Return the observed component's stable metric/probe identifier."""

    safe = re.sub(r"[^A-Za-z0-9_]+", "_", reference).strip("_")
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


@dataclass(frozen=True)
class NamedProbe:
    vector: str
    column: str


@dataclass(frozen=True)
class ProbePlan:
    """Case observations shared by netlist rendering and result parsing."""

    source_name: str
    source_voltage_vector: str
    load_current_column: str | None
    transient: tuple[NamedProbe, ...]
    ac: tuple[NamedProbe, ...]

    @property
    def transient_vectors(self) -> tuple[str, ...]:
        return tuple(probe.vector for probe in self.transient)

    @property
    def transient_columns(self) -> tuple[str, ...]:
        return tuple(probe.column for probe in self.transient)

    @property
    def ac_vectors(self) -> tuple[str, ...]:
        return tuple(probe.vector for probe in self.ac)

    @property
    def ac_columns(self) -> tuple[str, ...]:
        return tuple(probe.column for probe in self.ac)
