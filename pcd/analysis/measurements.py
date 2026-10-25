"""Electrical operating-point measurements shared by studies and reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.probes import ComponentObservation
from pcd.simulation import LOAD_VOLTAGE_COLUMN

from .ac import DEFAULT_Z0, ac_power_flow, at_frequency, input_impedance
from .stress import ac_component_metrics, component_loss_balance, transient_component_metrics
from .transient import rf_port_metrics


def ac_operating_metrics(
    ac: pd.DataFrame,
    frequency_hz: float,
    reference_impedance_ohm: float = DEFAULT_Z0,
    load_current_column: str | None = None,
    components: tuple[ComponentObservation, ...] = (),
) -> dict[str, Any]:
    """Interpolate complex observations first, then calculate every AC metric."""

    required = {"frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"}
    probes = [column for component in components for column in (component.voltage_column, component.current_column)]
    if load_current_column:
        probes.extend((LOAD_VOLTAGE_COLUMN, load_current_column))
    required.update(f"{name}_{part}" for name in probes for part in ("re", "im"))
    if missing := sorted(required - set(ac)):
        raise ValueError(f"canonical AC response is missing columns {missing}")
    selected = at_frequency(ac, frequency_hz)
    row = input_impedance(pd.DataFrame([selected]), reference_impedance_ohm).iloc[0]
    metrics: dict[str, Any] = {
        name: _finite_or_none(row[name])
        for name in (
            "resistance_ohm",
            "reactance_ohm",
            "magnitude_ohm",
            "reflection_db",
            "vswr",
        )
    }
    reflection = float(row["reflection_magnitude"])
    metrics.update(
        impedance_reference_plane="source_port",
        frequency_Hz=float(row["frequency_Hz"]),
        reference_impedance_ohm=reference_impedance_ohm,
        phase_deg=_finite_or_none(np.degrees(np.angle(complex(row["resistance_ohm"], row["reactance_ohm"])))),
        reflection_magnitude=reflection,
        reflected_power_fraction=reflection**2,
    )
    metrics.update(ac_power_flow(selected, load_current_column, reference_impedance_ohm))
    metrics.update(ac_component_metrics(selected, components))
    metrics.update(component_loss_balance(metrics))
    return metrics


def periodic_operating_metrics(
    waveform: pd.DataFrame,
    frequency_hz: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    options: Mapping[str, int | float],
) -> dict[str, Any]:
    """Use the same native-sample measurement convention for every consumer."""

    periodic_cycles = int(options["periodic_cycles"])
    settling_comparisons = int(options["settling_comparisons"])
    settling_tolerance = float(options["settling_tolerance"])
    metrics: dict[str, Any] = {}
    if load_current_column:
        metrics.update(
            rf_port_metrics(
                waveform,
                frequency_hz,
                load_current_column,
                periodic_cycles=periodic_cycles,
                settling_comparisons=settling_comparisons,
                settling_tolerance=settling_tolerance,
                harmonic_count=int(options["harmonic_count"]),
            )
        )
    metrics.update(
        transient_component_metrics(
            waveform,
            frequency_hz,
            components,
            periodic_cycles=periodic_cycles,
            settling_comparisons=settling_comparisons,
            settling_tolerance=settling_tolerance,
        )
    )
    metrics.update(component_loss_balance(metrics))
    return metrics


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
