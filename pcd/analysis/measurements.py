"""Electrical operating-point measurements shared by studies and reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.probes import ComponentObservation
from pcd.simulation import LOAD_VOLTAGE_COLUMN

from .ac import DEFAULT_Z0, ac_frequency_sample, ac_power_flow, input_impedance
from .stress import ac_component_metrics, component_loss_balance, component_signal_columns, measure_component_stress
from .transient import final_periodic_cycles, measure_rf_port, require_periodic_settle, rf_port_signal_columns


def ac_operating_metrics(
    ac: pd.DataFrame,
    frequency_hz: float,
    reference_impedance_ohm: float = DEFAULT_Z0,
    load_current_column: str | None = None,
    components: tuple[ComponentObservation, ...] = (),
    *,
    driven_port: bool = True,
    require_simulated: bool = False,
    power_balance_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Calculate AC metrics from a solved sample or a labelled interpolated point."""

    selected, frequency_evidence = ac_frequency_sample(ac, frequency_hz, require_simulated=require_simulated)
    return {
        **ac_sample_metrics(
            selected,
            reference_impedance_ohm,
            load_current_column,
            components,
            driven_port=driven_port,
            power_balance_unavailable_reason=power_balance_unavailable_reason,
        ),
        **frequency_evidence,
    }


def ac_sample_metrics(
    selected: pd.Series,
    reference_impedance_ohm: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    *,
    driven_port: bool,
    power_balance_unavailable_reason: str | None,
) -> dict[str, Any]:
    """Measure an AC row at its already selected frequency."""

    required = {"frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"}
    probes = [column for component in components for column in (component.voltage_column, component.current_column)]
    if load_current_column:
        probes.extend((LOAD_VOLTAGE_COLUMN, load_current_column))
    required.update(f"{name}_{part}" for name in probes for part in ("re", "im"))
    if missing := sorted(required - set(selected.index)):
        raise ValueError(f"canonical AC response is missing columns {missing}")
    metrics: dict[str, Any] = {"frequency_Hz": float(selected["frequency_Hz"])}
    if driven_port:
        metrics.update(_source_impedance_metrics(selected, reference_impedance_ohm))
    metrics.update(
        ac_power_flow(
            selected,
            load_current_column,
            reference_impedance_ohm,
            driven_port=driven_port,
            power_balance_unavailable_reason=power_balance_unavailable_reason,
        )
    )
    metrics.update(ac_component_metrics(selected, components))
    metrics.update(component_loss_balance(metrics))
    return metrics


def _source_impedance_metrics(selected: pd.Series, z0: float) -> dict[str, Any]:
    row = input_impedance(pd.DataFrame([selected]), z0).iloc[0]
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
    reflection = _finite_or_none(row["reflection_magnitude"])
    metrics.update(
        impedance_reference_plane="source_port",
        reference_impedance_ohm=z0,
        phase_deg=_finite_or_none(np.degrees(np.angle(complex(row["resistance_ohm"], row["reactance_ohm"])))),
        reflection_magnitude=reflection,
        reflected_power_fraction=None if reflection is None else reflection**2,
    )
    if reflection is None:
        metrics["reflection_magnitude_unavailable_reason"] = "the source-port incident wave is zero or undefined"
        metrics["reflected_power_fraction_unavailable_reason"] = metrics["reflection_magnitude_unavailable_reason"]
    return metrics


def periodic_operating_metrics(
    waveform: pd.DataFrame,
    frequency_hz: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    options: Mapping[str, int | float],
    *,
    power_balance_unavailable_reason: str | None = None,
    source_port_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Measure port and component values on one shared, settled time window."""

    port_columns = rf_port_signal_columns(waveform, load_current_column) if load_current_column else ()
    signals = (*port_columns, *component_signal_columns(components))
    if not signals:
        return {}
    measured, evidence, window = final_periodic_cycles(
        waveform,
        frequency_hz,
        signals,
        periodic_cycles=int(options["periodic_cycles"]),
        settling_comparisons=int(options["settling_comparisons"]),
        settling_tolerance=float(options["settling_tolerance"]),
        observe_source=bool(port_columns) and source_port_unavailable_reason is None,
    )
    require_periodic_settle(evidence)
    metrics = {"fundamental_Hz": frequency_hz, **evidence}
    if port_columns:
        metrics.update(
            measure_rf_port(
                measured,
                window,
                frequency_hz,
                port_columns[0],
                port_columns[1],
                int(options["harmonic_count"]),
                source_unavailable_reason=source_port_unavailable_reason
                or evidence.get("source_port_unavailable_reason"),
            )
        )
        if source_port_unavailable_reason:
            metrics["source_port_unavailable_reason"] = source_port_unavailable_reason
        if power_balance_unavailable_reason:
            for name in ("network_loss_W", "transfer_efficiency"):
                metrics[name] = None
                metrics.setdefault(f"{name}_unavailable_reason", power_balance_unavailable_reason)
    metrics.update(measure_component_stress(measured, components))
    metrics.update(component_loss_balance(metrics))
    return metrics


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
