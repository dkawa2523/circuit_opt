"""Pure component terminal-stress and modeled-loss measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.probes import ComponentObservation
from pcd.signals import time_average
from pcd.signals.power import power_ratio

from .ac import phasor_from_row
from .transient import (
    DEFAULT_PERIODIC_CYCLES,
    DEFAULT_SETTLING_COMPARISONS,
    DEFAULT_SETTLING_TOLERANCE,
    final_periodic_cycles,
    require_periodic_settle,
)


def ac_component_metrics(row: pd.Series, components: tuple[ComponentObservation, ...]) -> dict[str, Any]:
    """Peak/RMS terminal stress and effective series-resistance loss."""

    out: dict[str, Any] = {}
    losses: list[float | None] = []
    for component in components:
        voltage = phasor_from_row(row, component.voltage_column)
        current = phasor_from_row(row, component.current_column)
        if voltage is None or current is None:
            raise ValueError(f"AC response is missing probes for observed component {component.reference}")
        prefix = f"component_{component.metric_id}"
        _ac_peak_rms(out, prefix, "voltage", "V", voltage)
        current_rms = _ac_peak_rms(out, prefix, "current", "A", current)
        if component.series_resistance_ohm is not None:
            loss = None if current_rms is None else current_rms * current_rms * component.series_resistance_ohm
            if loss is None or not np.isfinite(loss):
                loss = None
                out[f"{prefix}_loss_W_unavailable_reason"] = "finite component current and modeled loss are required"
            out[f"{prefix}_loss_W"] = loss
            losses.append(loss)
    if losses:
        if None in losses:
            out["modeled_component_loss_W"] = None
            out["modeled_component_loss_W_unavailable_reason"] = "a modeled component loss is unavailable"
        else:
            out["modeled_component_loss_W"] = sum(loss for loss in losses if loss is not None)
    return out


def _ac_peak_rms(out: dict[str, Any], prefix: str, signal: str, unit: str, phasor: complex) -> float | None:
    peak_name, rms_name = f"{prefix}_{signal}_peak_{unit}", f"{prefix}_{signal}_rms_{unit}"
    peak = float(np.abs(phasor))
    if not np.isfinite(peak):
        for name in (peak_name, rms_name):
            out[name] = None
            out[f"{name}_unavailable_reason"] = f"a finite component {signal} phasor is required"
        return None
    rms = float(peak / np.sqrt(2.0))
    out[peak_name], out[rms_name] = peak, rms
    return rms


def component_signal_columns(components: tuple[ComponentObservation, ...]) -> tuple[str, ...]:
    return tuple(column for item in components for column in (item.voltage_column, item.current_column))


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
    measured, evidence, _window = final_periodic_cycles(
        waveform,
        fundamental_hz,
        component_signal_columns(components),
        periodic_cycles=periodic_cycles,
        settling_comparisons=settling_comparisons,
        settling_tolerance=settling_tolerance,
    )
    if require_settled:
        require_periodic_settle(evidence)
    return measure_component_stress(measured, components)


def measure_component_stress(measured: pd.DataFrame, components: tuple[ComponentObservation, ...]) -> dict[str, float]:
    """Calculate transient component values on an already selected interval."""

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


def component_loss_balance(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Compare summed explicit ESR/DCR loss with source-to-load network loss."""

    if "modeled_component_loss_W" not in metrics or "network_loss_W" not in metrics:
        return {}
    network, modeled = metrics["network_loss_W"], metrics["modeled_component_loss_W"]
    residual = None if network is None or modeled is None else float(network) - float(modeled)
    source = metrics.get("source_real_power_W")
    result: dict[str, Any] = {
        "component_loss_balance_residual_W": residual,
        "component_loss_balance_fraction_of_source": (
            None if residual is None or source is None else power_ratio(abs(residual), abs(float(source)))
        ),
    }
    if residual is None:
        result["component_loss_balance_residual_W_unavailable_reason"] = "both network and modeled loss are required"
    if result["component_loss_balance_fraction_of_source"] is None:
        result["component_loss_balance_fraction_of_source_unavailable_reason"] = (
            "a finite residual and a finite ratio to nonzero source real power are required"
        )
    return result
