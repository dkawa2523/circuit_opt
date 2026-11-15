"""Pure periodic-transient and RF load-port measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.core.models import MeasurementUnavailableError, UnsettledMeasurementError
from pcd.signals import PeriodicWindow, harmonic_phasors, periodic_window, time_average
from pcd.signals.power import cumulative_energy, power_ratio
from pcd.signals.windows import required_periodic_cycles
from pcd.simulation import LOAD_VOLTAGE_COLUMN

DEFAULT_PERIODIC_CYCLES = 3
DEFAULT_SETTLING_COMPARISONS = 2
DEFAULT_SETTLING_TOLERANCE = 1e-3
DEFAULT_HARMONIC_COUNT = 3

_SOURCE_SIGNALS = ("source_voltage_V", "current_A")


def _positive_integer(config: Mapping[str, Any], name: str, default: int) -> int:
    try:
        number = float(config.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement.{name} must be a positive integer") from exc
    if not np.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"measurement.{name} must be a positive integer")
    return int(number)


def rf_measurement_options(measurement: Mapping[str, Any] | None = None) -> dict[str, int | float]:
    """Read the small set of configurable periodic RF measurement controls."""

    config = measurement or {}
    try:
        tolerance = float(config.get("settling_tolerance", DEFAULT_SETTLING_TOLERANCE))
    except (TypeError, ValueError) as exc:
        raise ValueError("measurement.settling_tolerance must be positive and finite") from exc
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("measurement.settling_tolerance must be positive and finite")
    return {
        "periodic_cycles": _positive_integer(config, "periodic_cycles", DEFAULT_PERIODIC_CYCLES),
        "settling_comparisons": _positive_integer(config, "settling_comparisons", DEFAULT_SETTLING_COMPARISONS),
        "settling_tolerance": tolerance,
        "harmonic_count": _positive_integer(config, "harmonic_count", DEFAULT_HARMONIC_COUNT),
    }


def harmonic_spectrum(
    time_s: np.ndarray,
    values: np.ndarray,
    fundamental_hz: float,
    count: int = 3,
    *,
    window: PeriodicWindow | None = None,
) -> np.ndarray | None:
    """Complex amplitudes at exact harmonics of the fundamental.

    A time-weighted least-squares fit works directly on adaptive solver
    timesteps; it does not require a tone to coincide with an FFT bin.
    """

    if len(time_s) < 4 or fundamental_hz <= 0:
        return None
    fitted = harmonic_phasors(time_s, values, fundamental_hz, range(1, count + 1), window=window)
    if len(fitted) != count:
        return None
    return np.asarray([fitted[h] for h in range(1, count + 1)], dtype=complex)


def has_finite_samples(waveform: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """Whether at least two time-aligned samples can support a measurement."""

    required = ("time_s", *columns)
    if any(column not in waveform for column in required):
        return False
    values = waveform.loc[:, required].to_numpy(dtype=float)
    return int(np.all(np.isfinite(values), axis=1).sum()) >= 2


def _port_signals(waveform: pd.DataFrame, load_current: str | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ports: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if {"source_voltage_V", "current_A"} <= set(waveform):
        ports["source"] = (waveform["source_voltage_V"].to_numpy(float), -waveform["current_A"].to_numpy(float))
    voltage = LOAD_VOLTAGE_COLUMN if LOAD_VOLTAGE_COLUMN in waveform else "voltage_V"
    if load_current and {voltage, load_current} <= set(waveform):
        ports["load"] = (waveform[voltage].to_numpy(float), waveform[load_current].to_numpy(float))
    return ports


def transient_power_trace(waveform: pd.DataFrame, load_current: str | None = None) -> pd.DataFrame:
    """Signed instantaneous power and its integral at each observed port."""

    result = waveform[["time_s"]].copy()
    time = result["time_s"].to_numpy(float)
    for port, (voltage, current) in _port_signals(waveform, load_current).items():
        power = voltage * current
        if port == "source":
            result["source_delivered_current_A"] = current
        result[f"{port}_power_W"] = power
        result[f"{port}_net_energy_J"] = cumulative_energy(time, power)
    return result


def power_flow(waveform: pd.DataFrame, load_current: str | None = None) -> dict[str, Any]:
    """Real power delivered by the source and accepted at the load port.

    Imported port data may omit the source term.  Load power requires the
    current actually entering the load; source current is never substituted.
    """

    if waveform.empty:
        return {}
    time_s = waveform["time_s"].to_numpy(float)
    out: dict[str, Any] = {}
    for port, (voltage, current) in _port_signals(waveform, load_current).items():
        if np.count_nonzero(np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current)) < 2:
            continue
        key = f"{port}_power_W"
        power = time_average(voltage * current, time_s)
        out[key] = power if np.isfinite(power) else None
        if out[key] is None:
            out[f"{key}_unavailable_reason"] = (
                "an observed, finite voltage/current pair is required across the interval"
            )
        elif port == "source":
            voltage_rms = float(np.sqrt(time_average(voltage**2, time_s)))
            current_rms = float(np.sqrt(time_average(current**2, time_s)))
            out.update(source_current_rms_A=current_rms, source_apparent_power_VA=voltage_rms * current_rms)
    source_power, load_power = out.get("source_power_W"), out.get("load_power_W")
    if source_power is not None and load_power is not None:
        out["network_loss_W"] = source_power - load_power
        out["transfer_efficiency"] = power_ratio(load_power, source_power)
        if out["transfer_efficiency"] is None:
            out["transfer_efficiency_unavailable_reason"] = (
                "finite load power and a finite ratio to nonzero source real power are required"
            )
    return out


def _periodic_windows(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    columns: tuple[str, ...],
    periodic_cycles: int,
    settling_comparisons: int,
    settling_tolerance: float,
) -> dict[str, PeriodicWindow | None]:
    # Anchor all channels to the same saved tail. Missing current cannot shift
    # the window or discard a native voltage sample from the settling check.
    time_s = waveform["time_s"].to_numpy(float)
    return {
        column: periodic_window(
            time_s,
            waveform[column].to_numpy(float),
            fundamental_hz,
            measure_cycles=periodic_cycles,
            consecutive=settling_comparisons,
            tolerance=settling_tolerance,
        )
        for column in columns
    }


def _periodic_evidence(
    windows: Mapping[str, PeriodicWindow | None],
    available: tuple[PeriodicWindow, ...],
    periodic_cycles: int,
    settling_comparisons: int,
    settling_tolerance: float,
) -> dict[str, Any]:
    residuals = {name: None if window is None else window.residual for name, window in windows.items()}
    if not available or len(available) != len(windows):
        return {
            "periodic_settled": False,
            "periodic_residual": None,
            "periodic_residuals": residuals,
            "measurement_cycles": 0,
            "available_cycles": 0,
            "required_cycles": required_periodic_cycles(periodic_cycles, settling_comparisons),
        }

    finite_residuals = [value for value in residuals.values() if value is not None]
    unobserved = [name for name, window in windows.items() if window is not None and not window.observed]
    return {
        "periodic_settled": all(window.settled for window in available),
        "periodic_residual": max(finite_residuals) if finite_residuals else None,
        "periodic_residuals": residuals,
        "measurement_cycles": available[0].cycles,
        "available_cycles": min(window.available_cycles for window in available),
        "required_cycles": max(window.required_cycles for window in available),
        "measurement_start_s": available[0].start_s,
        "measurement_end_s": available[0].end_s,
        **(
            {
                "periodic_unavailable_reason": (
                    "periodic measurement requires finite samples spaced by less than half the fundamental period "
                    f"throughout the measurement and settling interval: {', '.join(unobserved)}"
                )
            }
            if unobserved and not any(value > settling_tolerance for value in finite_residuals)
            else {}
        ),
    }


def _slice_interval(waveform: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    ordered = waveform.sort_values("time_s", kind="stable")
    time_s = ordered["time_s"].to_numpy(float)
    selected = ordered.loc[(time_s >= start_s) & (time_s <= end_s)]
    boundaries = []
    for boundary in (start_s, end_s):
        if not np.any(time_s == boundary):
            row = {
                column: float(np.interp(boundary, time_s, ordered[column].to_numpy(float)))
                for column in ordered
                if column != "time_s"
            }
            boundaries.append({"time_s": boundary, **row})
    if boundaries:
        selected = pd.concat([selected, pd.DataFrame(boundaries)], ignore_index=True)
    # Optional unavailable probes stay missing without discarding valid load
    # observations. Native duplicate/event rows also retain their peak values.
    return selected.sort_values("time_s", kind="stable").reset_index(drop=True)


def final_periodic_cycles(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    signal_columns: tuple[str, ...] = ("voltage_V",),
    *,
    periodic_cycles: int = DEFAULT_PERIODIC_CYCLES,
    settling_comparisons: int = DEFAULT_SETTLING_COMPARISONS,
    settling_tolerance: float = DEFAULT_SETTLING_TOLERANCE,
    observe_source: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], PeriodicWindow | None]:
    """Return native samples, settling evidence, and their exact common window."""

    if waveform.empty:
        return waveform, {}, None
    columns = tuple(dict.fromkeys(signal_columns))
    missing = [column for column in columns if column not in waveform]
    if missing:
        raise MeasurementUnavailableError(f"waveform is missing periodic measurement signals: {missing}")

    source = _SOURCE_SIGNALS if observe_source and all(column in waveform for column in _SOURCE_SIGNALS) else ()
    windows = _periodic_windows(
        waveform,
        fundamental_hz,
        (*columns, *source),
        periodic_cycles,
        settling_comparisons,
        settling_tolerance,
    )
    source_unavailable = any((window := windows[column]) is None or not window.observed for column in source)
    if source_unavailable:
        windows = {column: windows[column] for column in columns}
    available = tuple(window for window in windows.values() if window is not None)
    evidence = _periodic_evidence(windows, available, periodic_cycles, settling_comparisons, settling_tolerance)
    if source_unavailable:
        evidence["source_port_unavailable_reason"] = (
            "finite source voltage and current are required across the periodic measurement and settling interval"
        )
    if len(available) != len(windows) or len(waveform) < 4:
        return waveform, evidence, None

    window = available[0]
    return _slice_interval(waveform, window.start_s, window.end_s), evidence, window


def require_periodic_settle(evidence: Mapping[str, Any]) -> None:
    """Raise when periodic measurement evidence is missing or unsettled."""
    if reason := evidence.get("periodic_unavailable_reason"):
        raise MeasurementUnavailableError(str(reason))

    if evidence.get("periodic_settled"):
        return
    residuals = evidence.get("periodic_residuals") or {}
    detail = ", ".join(f"{name}={value!r}" for name, value in residuals.items())
    cycles = f"available={evidence.get('available_cycles', 0)}, required={evidence.get('required_cycles', 0)}"
    raise UnsettledMeasurementError(
        f"periodic measurement signals have not settled ({cycles}; {detail or 'insufficient cycles'})"
    )


def _harmonic_amplitudes(spectrum: np.ndarray | None) -> dict[str, float]:
    if spectrum is None:
        return {}
    return {f"h{index}": float(abs(value)) for index, value in enumerate(spectrum, start=1)}


def rf_port_signal_columns(waveform: pd.DataFrame, load_current: str | None) -> tuple[str, str]:
    if waveform.empty:
        raise MeasurementUnavailableError("waveform is empty; RF-port metrics are unavailable")
    if not load_current:
        raise ValueError("RF-port metrics require measurement.load_current (use 'auto' for built-in loads)")
    voltage = LOAD_VOLTAGE_COLUMN if LOAD_VOLTAGE_COLUMN in waveform else "voltage_V"
    required = (voltage, load_current)
    if not has_finite_samples(waveform, required):
        raise MeasurementUnavailableError("RF-port metrics require finite load voltage/current samples")
    return required


def rf_port_metrics(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    load_current_column: str | None,
    *,
    require_settled: bool = True,
    periodic_cycles: int = DEFAULT_PERIODIC_CYCLES,
    settling_comparisons: int = DEFAULT_SETTLING_COMPARISONS,
    settling_tolerance: float = DEFAULT_SETTLING_TOLERANCE,
    harmonic_count: int = DEFAULT_HARMONIC_COUNT,
) -> dict[str, Any]:
    """Measure the electrical RF-load port without inferring plasma physics.

    ``load_real_power_W`` is accepted power at the electrical reference plane.
    The load current must come from a probe physically in series with the load.
    """

    signals = rf_port_signal_columns(waveform, load_current_column)
    if fundamental_hz <= 0:
        raise ValueError("a positive source fundamental frequency is required for RF-port metrics")

    measured, evidence, window = final_periodic_cycles(
        waveform,
        fundamental_hz,
        signals,
        periodic_cycles=periodic_cycles,
        settling_comparisons=settling_comparisons,
        settling_tolerance=settling_tolerance,
        observe_source=True,
    )
    if require_settled:
        require_periodic_settle(evidence)
    port = measure_rf_port(
        measured,
        window,
        fundamental_hz,
        *signals,
        harmonic_count,
        source_unavailable_reason=evidence.get("source_port_unavailable_reason"),
    )
    return {**port, **evidence}


def measure_rf_port(
    measured: pd.DataFrame,
    window: PeriodicWindow | None,
    fundamental_hz: float,
    voltage_column: str,
    load_current_column: str,
    harmonic_count: int,
    *,
    source_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Calculate RF quantities on an already selected measurement interval."""

    time_s = measured["time_s"].to_numpy(float)
    voltage = measured[voltage_column].to_numpy(float)
    current = measured[load_current_column].to_numpy(float)
    power_signals = (
        measured.drop(columns="source_voltage_V", errors="ignore") if source_unavailable_reason else measured
    )
    flow = power_flow(power_signals, load_current_column)
    voltage_harmonics = (
        harmonic_spectrum(time_s, voltage, fundamental_hz, harmonic_count, window=window)
        if window is not None
        else None
    )
    current_harmonics = (
        harmonic_spectrum(time_s, current, fundamental_hz, harmonic_count, window=window)
        if window is not None
        else None
    )
    impedance: complex | None = None
    if voltage_harmonics is not None and current_harmonics is not None and abs(current_harmonics[0]) > 1e-30:
        impedance = complex(voltage_harmonics[0] / current_harmonics[0])

    result: dict[str, Any] = {
        "fundamental_Hz": fundamental_hz,
        "load_v_peak_V": float(np.nanmax(np.abs(voltage))),
        "load_v_rms_V": float(np.sqrt(max(time_average(voltage**2, time_s), 0.0))),
        "load_v_dc_V": time_average(voltage, time_s),
        "load_i_rms_A": float(np.sqrt(max(time_average(current**2, time_s), 0.0))),
        "load_real_power_W": flow.get("load_power_W"),
        "voltage_harmonic_amplitude_V": _harmonic_amplitudes(voltage_harmonics),
        "current_harmonic_amplitude_A": _harmonic_amplitudes(current_harmonics),
    }
    if impedance is None or not np.isfinite(impedance):
        reason = (
            "the requested harmonics could not be resolved for both load voltage and current"
            if voltage_harmonics is None or current_harmonics is None
            else "finite load voltage and nonzero current phasors at the fundamental are required"
        )
        for name in ("load_fundamental_resistance_ohm", "load_fundamental_reactance_ohm"):
            result[name] = None
            result[f"{name}_unavailable_reason"] = reason
    else:
        result["load_fundamental_resistance_ohm"] = impedance.real
        result["load_fundamental_reactance_ohm"] = impedance.imag
    aliases = {"source_power_W": "source_real_power_W", "load_power_W": "load_real_power_W"}
    for name, value in flow.items():
        signal = name.removesuffix("_unavailable_reason")
        output = aliases.get(signal, signal) + name[len(signal) :]
        result[output] = value
    if source_unavailable_reason:
        for name in (
            "source_real_power_W",
            "source_current_rms_A",
            "source_apparent_power_VA",
            "network_loss_W",
            "transfer_efficiency",
        ):
            result[name] = None
            result[f"{name}_unavailable_reason"] = source_unavailable_reason
    return result
