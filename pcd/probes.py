"""Small typed probe declarations shared by netlist and solver adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .simulation import (
    CURRENT_COLUMN,
    LOAD_CURRENT_COLUMN,
    LOAD_VOLTAGE_COLUMN,
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
        LOAD_VOLTAGE_COLUMN,
        "frequency_Hz",
        # Frequency-domain vectors expand to ``<name>_re``/``<name>_im``.
        "voltage",
        "current",
        "source_delivered_current_A",
        "source_power_W",
        "load_power_W",
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
    unit: str | None = None
    label: str | None = None

    @property
    def is_current(self) -> bool:
        """Recognize branch/device current vectors without guessing from a column name."""

        return re.fullmatch(r"[+-]?(?:i\([^()]+\)|@[^\[\]\s]+\[i\])", self.vector.strip(), re.IGNORECASE) is not None

    @property
    def resolved_unit(self) -> str | None:
        if self.is_current:
            return "A"
        if re.fullmatch(r"[+-]?v\([^()]+\)", self.vector.strip(), re.IGNORECASE):
            return "V"
        return self.unit


@dataclass(frozen=True)
class ProbePlan:
    """Case observations shared by netlist rendering and result parsing."""

    source_name: str
    source_voltage_vector: str
    load_current_column: str | None
    transient: tuple[NamedProbe, ...]
    ac: tuple[NamedProbe, ...]
    source_current_vector: str = ""

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


def signal_catalog(
    plan: ProbePlan,
    measurement: Mapping[str, Any],
    *,
    transient_columns: tuple[str, ...] = (),
    ac_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe columns that actually exist; never manufacture missing observations."""

    ports = measurement.get("load_ports") or {}
    p, n = str(ports.get("p", "out")), str(ports.get("n", "0"))
    load_vector = f"v({p},{n})" if n != "0" else f"v({p})"
    declarations = [
        (
            NamedProbe(
                str(measurement.get("output_voltage_vector", load_vector)), VOLTAGE_COLUMN, "V", "observed output"
            ),
            VOLTAGE_COLUMN,
            "as_declared",
        ),
        (NamedProbe(plan.source_voltage_vector, SOURCE_VOLTAGE_COLUMN, "V", "source port"), "voltage", "as_declared"),
        (
            NamedProbe(
                plan.source_current_vector or f"i({plan.source_name})", CURRENT_COLUMN, "A", "current into source"
            ),
            "current",
            "into_source_positive_terminal",
        ),
        (NamedProbe(load_vector, LOAD_VOLTAGE_COLUMN, "V", "load port"), LOAD_VOLTAGE_COLUMN, "as_declared"),
        *(
            (probe, probe.column, "into_load" if probe.column == plan.load_current_column else "as_declared")
            for probe in plan.ac
        ),
    ]
    channels: dict[str, dict[str, Any]] = {}
    for probe, ac_name, direction in declarations:
        columns: dict[str, list[str]] = {}
        if probe.column in transient_columns:
            columns["transient"] = [probe.column]
        pair = [f"{ac_name}_re", f"{ac_name}_im"]
        if probe.column == VOLTAGE_COLUMN and not all(name in ac_columns for name in pair):
            # Older archives saved the same output under the load/source name.
            if probe.vector == load_vector:
                pair = [f"{LOAD_VOLTAGE_COLUMN}_re", f"{LOAD_VOLTAGE_COLUMN}_im"]
            elif probe.vector == plan.source_voltage_vector:
                pair = ["voltage_re", "voltage_im"]
        if all(name in ac_columns for name in pair):
            columns["ac"] = pair
        channels[probe.column] = {
            "vector": probe.vector,
            "unit": probe.resolved_unit,
            "label": probe.label or probe.column,
            "positive_direction": direction,
            "columns": columns,
            "ac_convention": "peak_phasor" if "ac" in columns else None,
        }
        voltage = re.fullmatch(r"[+-]?v\(([^(),]+)(?:,([^()]+))?\)", probe.vector.strip(), re.IGNORECASE)
        if voltage:
            channels[probe.column]["terminals"] = {"p": voltage[1], "n": voltage[2] or "0"}
    return {
        "schema": "signal_catalog.v1",
        "axes": {"transient": {"column": "time_s", "unit": "s"}, "ac": {"column": "frequency_Hz", "unit": "Hz"}},
        "signals": channels,
        "drive_basis": measurement.get("drive_basis", "declared_or_imported"),
        "load_ports": dict(ports),
        "load_reference_plane": measurement.get("load_reference_plane", "load_ports"),
        "impedance_reference_plane": "source_port",
    }
