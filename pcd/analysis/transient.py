"""Pure periodic-transient and RF load-port measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.core.models import UnsettledMeasurementError
from pcd.signals import PeriodicWindow, harmonic_phasors, periodic_window, time_average

DEFAULT_PERIODIC_CYCLES = 3
DEFAULT_SETTLING_COMPARISONS = 2
DEFAULT_SETTLING_TOLERANCE = 1e-3
DEFAULT_HARMONIC_COUNT = 3


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
    time_s: np.ndarray, values: np.ndarray, fundamental_hz: float, count: int = 3
) -> np.ndarray | None:
    """Complex amplitudes at exact harmonics of the fundamental.

    A time-weighted least-squares fit works directly on adaptive solver
    timesteps; it does not require a tone to coincide with an FFT bin.
    """

    if len(time_s) < 4 or fundamental_hz <= 0:
        return None
    fitted = harmonic_phasors(time_s, values, fundamental_hz, range(1, count + 1))
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


def power_flow(waveform: pd.DataFrame, load_current: str | None = None) -> dict[str, float]:
    """Real power delivered by the source and accepted at the load port.

    Imported port data may omit the source term.  Load power requires the
    current actually entering the load; source current is never substituted.
    """

    if waveform.empty:
        return {}
    time_s = waveform["time_s"].to_numpy(float)
    out: dict[str, float] = {}
    if has_finite_samples(waveform, ("source_voltage_V", "current_A")):
        source_voltage = waveform["source_voltage_V"].to_numpy(float)
        source_current = waveform["current_A"].to_numpy(float)
        voltage_rms = float(np.sqrt(max(time_average(source_voltage**2, time_s), 0.0)))
        current_rms = float(np.sqrt(max(time_average(source_current**2, time_s), 0.0)))
        out.update(
            {
                "source_power_W": time_average(-source_voltage * source_current, time_s),
                "source_current_rms_A": current_rms,
                "source_apparent_power_VA": voltage_rms * current_rms,
            }
        )
    if load_current and has_finite_samples(waveform, ("voltage_V", load_current)):
        load_voltage = waveform["voltage_V"].to_numpy(float)
        load_current_values = waveform[load_current].to_numpy(float)
        load_power = time_average(load_voltage * load_current_values, time_s)
        out["load_power_W"] = load_power
        if "source_power_W" in out:
            source_power = out["source_power_W"]
            out["network_loss_W"] = source_power - load_power
            out["transfer_efficiency"] = load_power / source_power if abs(source_power) > 1e-30 else 0.0
    return out


def _periodic_windows(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    columns: tuple[str, ...],
    periodic_cycles: int,
    settling_comparisons: int,
    settling_tolerance: float,
) -> dict[str, PeriodicWindow | None]:
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
) -> dict[str, Any]:
    residuals = {name: None if window is None else window.residual for name, window in windows.items()}
    if not available or len(available) != len(windows):
        return {
            "periodic_settled": False,
            "periodic_residual": None,
            "periodic_residuals": residuals,
            "measurement_cycles": 0,
            "available_cycles": 0,
            "required_cycles": max(periodic_cycles, settling_comparisons + 1),
        }

    finite_residuals = [value for value in residuals.values() if value is not None]
    return {
        "periodic_settled": all(window.settled for window in available),
        "periodic_residual": max(finite_residuals) if finite_residuals else None,
        "periodic_residuals": residuals,
        "measurement_cycles": available[0].cycles,
        "available_cycles": min(window.available_cycles for window in available),
        "required_cycles": max(window.required_cycles for window in available),
    }


def _resample_interval(waveform: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    time_s = waveform["time_s"].to_numpy(float)
    grid = np.linspace(start_s, end_s, max(len(waveform), 16))
    columns: dict[str, np.ndarray] = {"time_s": grid}
    for column in waveform.columns:
        if column != "time_s":
            columns[column] = np.asarray(
                np.interp(grid, time_s, waveform[column].to_numpy(float)),
                dtype=float,
            )
    return pd.DataFrame(columns)


def final_periodic_cycles(
    waveform: pd.DataFrame,
    fundamental_hz: float,
    signal_columns: tuple[str, ...] = ("voltage_V",),
    *,
    periodic_cycles: int = DEFAULT_PERIODIC_CYCLES,
    settling_comparisons: int = DEFAULT_SETTLING_COMPARISONS,
    settling_tolerance: float = DEFAULT_SETTLING_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the final whole-cycle waveform and settling evidence."""

    if waveform.empty:
        return waveform, {}
    columns = tuple(dict.fromkeys(signal_columns))
    missing = [column for column in columns if column not in waveform]
    if missing:
        raise ValueError(f"waveform is missing periodic measurement signals: {missing}")

    windows = _periodic_windows(
        waveform,
        fundamental_hz,
        columns,
        periodic_cycles,
        settling_comparisons,
        settling_tolerance,
    )
    available = tuple(window for window in windows.values() if window is not None)
    evidence = _periodic_evidence(windows, available, periodic_cycles, settling_comparisons)
    if len(available) != len(windows) or len(waveform) < 4:
        return waveform, evidence

    window = available[0]
    return _resample_interval(waveform, window.start_s, window.end_s), evidence


def require_periodic_settle(evidence: Mapping[str, Any]) -> None:
    """Raise when periodic measurement evidence is missing or unsettled."""

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

    if waveform.empty:
        raise ValueError("waveform is empty; RF-port metrics are unavailable")
    if not load_current_column or load_current_column not in waveform:
        raise ValueError("RF-port metrics require measurement.load_current (use 'auto' for built-in loads)")
    if not has_finite_samples(waveform, ("voltage_V", load_current_column)):
        raise ValueError("RF-port metrics require finite load voltage/current samples")
    if fundamental_hz <= 0:
        raise ValueError("a positive source fundamental frequency is required for RF-port metrics")

    signals = ["voltage_V", load_current_column]
    if has_finite_samples(waveform, ("source_voltage_V", "current_A")):
        signals.extend(("source_voltage_V", "current_A"))
    measured, evidence = final_periodic_cycles(
        waveform,
        fundamental_hz,
        tuple(signals),
        periodic_cycles=periodic_cycles,
        settling_comparisons=settling_comparisons,
        settling_tolerance=settling_tolerance,
    )
    if require_settled:
        require_periodic_settle(evidence)
    time_s = measured["time_s"].to_numpy(float)
    voltage = measured["voltage_V"].to_numpy(float)
    current = measured[load_current_column].to_numpy(float)
    flow = power_flow(measured, load_current_column)
    voltage_harmonics = harmonic_spectrum(time_s, voltage, fundamental_hz, harmonic_count)
    current_harmonics = harmonic_spectrum(time_s, current, fundamental_hz, harmonic_count)
    impedance: complex | None = None
    if voltage_harmonics is not None and current_harmonics is not None and abs(current_harmonics[0]) > 1e-30:
        impedance = complex(voltage_harmonics[0] / current_harmonics[0])

    result: dict[str, Any] = {
        "fundamental_Hz": fundamental_hz,
        "load_v_peak_V": float(np.nanmax(np.abs(voltage))),
        "load_v_rms_V": float(np.sqrt(max(time_average(voltage**2, time_s), 0.0))),
        "load_v_dc_V": time_average(voltage, time_s),
        "load_i_rms_A": float(np.sqrt(max(time_average(current**2, time_s), 0.0))),
        "load_real_power_W": flow["load_power_W"],
        "voltage_harmonic_amplitude_V": _harmonic_amplitudes(voltage_harmonics),
        "current_harmonic_amplitude_A": _harmonic_amplitudes(current_harmonics),
        **evidence,
    }
    if impedance is not None:
        result.update(
            {
                "load_fundamental_resistance_ohm": impedance.real,
                "load_fundamental_reactance_ohm": impedance.imag,
            }
        )
    if "source_power_W" in flow:
        result.update(
            {
                "source_real_power_W": flow["source_power_W"],
                "source_current_rms_A": flow["source_current_rms_A"],
                "source_apparent_power_VA": flow["source_apparent_power_VA"],
                "network_loss_W": flow["network_loss_W"],
                "transfer_efficiency": flow["transfer_efficiency"],
            }
        )
    return result
