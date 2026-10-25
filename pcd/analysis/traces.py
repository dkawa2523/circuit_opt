"""Reusable tables behind waveform, transfer and harmonic figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pcd.probes import ComponentObservation
from pcd.signals import harmonic_phasors

from .ac import ac_power_flow, input_impedance
from .stress import ac_component_metrics, component_loss_balance
from .transient import final_periodic_cycles, require_periodic_settle, rf_measurement_options


def phasor_values(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if len(columns) != 2 or any(name not in frame for name in columns):
        raise ValueError("signal has no saved complex AC observation")
    return frame[columns[0]].to_numpy(float) + 1j * frame[columns[1]].to_numpy(float)


def magnitude_phase(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(values)
    valid = np.isfinite(values) & (magnitude > 0)
    phase = np.full(len(values), np.nan)
    # Unwrap each available interval independently; never bridge a missing pole/zero.
    for indices in np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) != 1) + 1):
        if len(indices):
            phase[indices] = np.degrees(np.unwrap(np.angle(values[indices])))
    return magnitude, phase


def ac_trace(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    values = phasor_values(frame, columns)
    magnitude, phase = magnitude_phase(values)
    return pd.DataFrame(
        {
            "frequency_Hz": frame["frequency_Hz"],
            "real": values.real,
            "imag": values.imag,
            "magnitude": magnitude,
            "phase_deg": phase,
        }
    )


def transfer_trace(frame: pd.DataFrame, numerator: Sequence[str], denominator: Sequence[str]) -> pd.DataFrame:
    top, bottom = phasor_values(frame, numerator), phasor_values(frame, denominator)
    available = np.isfinite(top) & np.isfinite(bottom) & (np.abs(bottom) > 0)
    ratio = np.full(len(frame), complex(np.nan, np.nan))
    np.divide(top, bottom, out=ratio, where=available)
    magnitude, phase = magnitude_phase(ratio)
    gain_db = np.full(len(frame), np.nan)
    np.log10(magnitude, out=gain_db, where=np.isfinite(magnitude) & (magnitude > 0))
    gain_db *= 20
    gain_db[np.isfinite(magnitude) & (magnitude == 0)] = -np.inf
    return pd.DataFrame(
        {
            "frequency_Hz": frame["frequency_Hz"],
            "ratio_re": ratio.real,
            "ratio_im": ratio.imag,
            "magnitude": magnitude,
            "gain_dB": gain_db,
            "phase_deg": phase,
        }
    )


def frequency_metrics(
    frame: pd.DataFrame, z0: float, load_current: str | None, components: tuple[ComponentObservation, ...]
) -> pd.DataFrame:
    result = input_impedance(frame, z0)
    result["return_loss_dB"] = -result["reflection_db"]
    result["impedance_phase_deg"] = magnitude_phase(
        result["resistance_ohm"].to_numpy() + 1j * result["reactance_ohm"].to_numpy()
    )[1]
    rows = []
    for _, row in frame.iterrows():
        metrics = {**ac_power_flow(row, load_current, z0), **ac_component_metrics(row, components)}
        metrics.update(component_loss_balance(metrics))
        rows.append(metrics)
    metrics = pd.DataFrame(rows)
    for name in map(str, metrics):
        result[name] = metrics[name].to_numpy()
    return result


def relative_harmonic_levels(amplitude: np.ndarray) -> np.ndarray:
    levels = np.full(len(amplitude), np.nan)
    if len(amplitude) and amplitude[0] > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            levels = 20 * np.log10(amplitude / amplitude[0])
    return levels


def instantaneous_power(waveform: pd.DataFrame, load_current: str | None = None) -> pd.DataFrame:
    result = pd.DataFrame({"time_s": waveform["time_s"]})
    if {"current_A", "source_voltage_V"} <= set(waveform):
        result["source_delivered_current_A"] = -waveform["current_A"]
        result["source_power_W"] = -waveform["source_voltage_V"] * waveform["current_A"]
    if load_current and load_current in waveform:
        voltage = "load_voltage_V" if "load_voltage_V" in waveform else "voltage_V"
        result["load_power_W"] = waveform[voltage] * waveform[load_current]
    return result


def harmonic_trace(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    count: int | None = None,
    *,
    measurement: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    options = rf_measurement_options(measurement)
    count = int(options["harmonic_count"]) if count is None else count
    measured, evidence, window = final_periodic_cycles(
        pd.DataFrame({"time_s": time_s, "value": values}),
        frequency_hz,
        ("value",),
        periodic_cycles=int(options["periodic_cycles"]),
        settling_comparisons=int(options["settling_comparisons"]),
        settling_tolerance=float(options["settling_tolerance"]),
    )
    if window is None:
        raise ValueError("harmonic plot has no complete periodic measurement window")
    require_periodic_settle(evidence)
    harmonics = harmonic_phasors(
        measured["time_s"].to_numpy(float),
        measured["value"].to_numpy(float),
        frequency_hz,
        range(1, count + 1),
        window=window,
    )
    if len(harmonics) != count:
        raise ValueError("not enough independent samples for the requested harmonics")
    amplitude = np.asarray([abs(harmonics[index]) for index in range(1, count + 1)])
    if amplitude[0] == 0:
        raise ValueError("relative dBc is undefined for a zero fundamental")
    levels = relative_harmonic_levels(amplitude)
    return pd.DataFrame(
        {
            "harmonic": np.arange(1, count + 1),
            "frequency_Hz": frequency_hz * np.arange(1, count + 1),
            "amplitude": amplitude,
            "relative_dBc": levels,
            "window_start_s": window.start_s,
            "window_end_s": window.end_s,
            "measurement_cycles": window.cycles,
            "settling_tolerance": float(options["settling_tolerance"]),
        }
    )
