"""Pure component terminal-stress and modeled-loss measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.probes import ComponentObservation
from pcd.signals import time_average

from .ac import phasor_from_row
from .transient import (
    DEFAULT_PERIODIC_CYCLES,
    DEFAULT_SETTLING_COMPARISONS,
    DEFAULT_SETTLING_TOLERANCE,
    final_periodic_cycles,
    require_periodic_settle,
)


def ac_component_metrics(row: pd.Series, components: tuple[ComponentObservation, ...]) -> dict[str, float]:
    """Peak/RMS terminal stress and effective series-resistance loss."""

    out: dict[str, float] = {}
    losses: list[float] = []
    for component in components:
        voltage = phasor_from_row(row, component.voltage_column)
        current = phasor_from_row(row, component.current_column)
        if voltage is None or current is None:
            raise ValueError(f"AC response is missing probes for observed component {component.reference}")
        prefix = f"component_{component.metric_id}"
        voltage_peak, current_peak = abs(voltage), abs(current)
        voltage_rms, current_rms = voltage_peak / np.sqrt(2.0), current_peak / np.sqrt(2.0)
        out.update(
            {
                f"{prefix}_voltage_peak_V": voltage_peak,
                f"{prefix}_voltage_rms_V": voltage_rms,
                f"{prefix}_current_peak_A": current_peak,
                f"{prefix}_current_rms_A": current_rms,
            }
        )
        if component.series_resistance_ohm is not None:
            loss = current_rms**2 * component.series_resistance_ohm
            out[f"{prefix}_loss_W"] = loss
            losses.append(loss)
    if losses:
        out["modeled_component_loss_W"] = float(sum(losses))
    return out


def transient_component_metrics(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    components: tuple[ComponentObservation, ...],
    *,
    require_settled: bool = True,
    periodic_cycles: int = DEFAULT_PERIODIC_CYCLES,
    settling_comparisons: int = DEFAULT_SETTLING_COMPARISONS,
    settling_tolerance: float = DEFAULT_SETTLING_TOLERANCE,
) -> dict[str, float]:
    """Component stress/loss over the RF port's final-cycle convention."""

    if waveform.empty or not components:
        return {}
    signal_columns = tuple(
        column for component in components for column in (component.voltage_column, component.current_column)
    )
    measured, evidence = final_periodic_cycles(
        waveform,
        fundamental_hz,
        signal_columns,
        periodic_cycles=periodic_cycles,
        settling_comparisons=settling_comparisons,
        settling_tolerance=settling_tolerance,
    )
    if require_settled:
        require_periodic_settle(evidence)
    time_s = measured["time_s"].to_numpy(float)
    out: dict[str, float] = {}
    losses: list[float] = []
    for component in components:
        if component.voltage_column not in measured or component.current_column not in measured:
            raise ValueError(f"waveform is missing probes for observed component {component.reference}")
        voltage = measured[component.voltage_column].to_numpy(float)
        current = measured[component.current_column].to_numpy(float)
        voltage_rms = float(np.sqrt(max(time_average(voltage**2, time_s), 0.0)))
        current_rms = float(np.sqrt(max(time_average(current**2, time_s), 0.0)))
        prefix = f"component_{component.metric_id}"
        out.update(
            {
                f"{prefix}_voltage_peak_V": float(np.nanmax(np.abs(voltage))),
                f"{prefix}_voltage_rms_V": voltage_rms,
                f"{prefix}_current_peak_A": float(np.nanmax(np.abs(current))),
                f"{prefix}_current_rms_A": current_rms,
            }
        )
        if component.series_resistance_ohm is not None:
            loss = current_rms**2 * component.series_resistance_ohm
            out[f"{prefix}_loss_W"] = loss
            losses.append(loss)
    if losses:
        out["modeled_component_loss_W"] = float(sum(losses))
    return out


def component_loss_balance(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Compare summed explicit ESR/DCR loss with source-to-load network loss."""

    if "modeled_component_loss_W" not in metrics or "network_loss_W" not in metrics:
        return {}
    residual = float(metrics["network_loss_W"]) - float(metrics["modeled_component_loss_W"])
    source = abs(float(metrics.get("source_real_power_W", 0.0)))
    return {
        "component_loss_balance_residual_W": residual,
        "component_loss_balance_fraction_of_source": abs(residual) / max(source, 1e-30),
    }
