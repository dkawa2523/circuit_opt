"""Pure frequency-domain measurements over canonical AC phasors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pcd.core.models import MeasurementUnavailableError
from pcd.signals.power import power_ratio
from pcd.simulation import LOAD_VOLTAGE_COLUMN, same_frequency_hz

#: Reference impedance for reflection coefficient / VSWR, in ohms.
DEFAULT_Z0 = 50.0


def reference_impedance_ohm(measurement: Mapping[str, Any] | None = None) -> float:
    """Resolve the case's positive real reference for reflection and wave power."""

    try:
        value = float((measurement or {}).get("reference_impedance_ohm", DEFAULT_Z0))
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_impedance_ohm must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("reference_impedance_ohm must be positive and finite")
    return value


def input_impedance(ac: pd.DataFrame, z0: float = DEFAULT_Z0) -> pd.DataFrame:
    """Impedance seen by the source, plus reflection coefficient and VSWR.

    ngspice reports ``i(Vsrc)`` flowing *into* the source's positive terminal,
    so the current delivered to the network is its negative.
    """

    voltage = ac["voltage_re"].to_numpy() + 1j * ac["voltage_im"].to_numpy()
    delivered = -(ac["current_re"].to_numpy() + 1j * ac["current_im"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        z = voltage / delivered
        # Written in V and I so an open circuit remains gamma=1 instead of
        # producing inf/inf and silently poisoning the objective with NaN.
        gamma = (voltage - z0 * delivered) / (voltage + z0 * delivered)
    magnitude = np.abs(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        vswr = np.where(magnitude < 1.0, (1 + magnitude) / (1 - magnitude), np.inf)
    return ac.assign(
        resistance_ohm=np.real(z),
        reactance_ohm=np.imag(z),
        magnitude_ohm=np.abs(z),
        reflection_re=np.real(gamma),
        reflection_im=np.imag(gamma),
        reflection_magnitude=magnitude,
        reflection_db=20.0 * np.log10(np.maximum(magnitude, 1e-30)),
        vswr=vswr,
    )


def phasor_from_row(row: pd.Series, column: str) -> complex | None:
    """Read one canonical rectangular phasor, or ``None`` when absent."""

    real, imag = f"{column}_re", f"{column}_im"
    if real not in row or imag not in row:
        return None
    return complex(float(row[real]), float(row[imag]))


def ac_power_flow(
    row: pd.Series,
    load_current_column: str | None = None,
    reference_impedance_ohm: float = DEFAULT_Z0,
    *,
    driven_port: bool = True,
    power_balance_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Electrical port power from peak AC phasors.

    ``load_real_power_W`` is accepted power at the named load reference plane;
    it is not a plasma-species or heating-path allocation.
    """

    out = _source_power_flow(row, reference_impedance_ohm) if driven_port else {}
    load_voltage = phasor_from_row(row, LOAD_VOLTAGE_COLUMN)
    load_current_phasor = phasor_from_row(row, load_current_column) if load_current_column else None
    if load_voltage is None or load_current_phasor is None:
        return out

    with np.errstate(over="ignore", invalid="ignore"):
        load_power = 0.5 * float(np.real(load_voltage * np.conj(load_current_phasor)))
    if not np.isfinite(load_power):
        names = (
            ("load_real_power_W", "network_loss_W", "transfer_efficiency") if driven_port else ("load_real_power_W",)
        )
        for name in names:
            out[name] = None
            out[f"{name}_unavailable_reason"] = "finite load-port voltage, current and real power are required"
        return out
    out["load_real_power_W"] = load_power
    if not driven_port:
        return out
    if power_balance_unavailable_reason:
        for name in ("network_loss_W", "transfer_efficiency"):
            out[name] = None
            out[f"{name}_unavailable_reason"] = power_balance_unavailable_reason
        return out
    source_power = out["source_real_power_W"]
    out["network_loss_W"] = source_power - load_power
    out["transfer_efficiency"] = power_ratio(load_power, source_power)
    if out["transfer_efficiency"] is None:
        out["transfer_efficiency_unavailable_reason"] = (
            "finite load power and a finite ratio to nonzero source real power are required"
        )
    return out


def _source_power_flow(row: pd.Series, z0: float) -> dict[str, Any]:
    if not np.isfinite(z0) or z0 <= 0:
        raise ValueError("reference_impedance_ohm must be positive and finite")
    source_voltage = complex(float(row["voltage_re"]), float(row["voltage_im"]))
    source_current = -complex(float(row["current_re"]), float(row["current_im"]))
    source_power = 0.5 * float(np.real(source_voltage * np.conj(source_current)))
    source_current_rms = abs(source_current) / np.sqrt(2.0)
    source_voltage_rms = abs(source_voltage) / np.sqrt(2.0)
    forward_voltage = 0.5 * (source_voltage + z0 * source_current)
    reflected_voltage = 0.5 * (source_voltage - z0 * source_current)
    forward_power = abs(forward_voltage) ** 2 / (2.0 * z0)
    reflected_power = abs(reflected_voltage) ** 2 / (2.0 * z0)
    return {
        "source_real_power_W": source_power,
        "source_voltage_rms_V": source_voltage_rms,
        "source_current_rms_A": source_current_rms,
        "source_apparent_power_VA": source_voltage_rms * source_current_rms,
        "forward_power_W": forward_power,
        "reflected_power_W": reflected_power,
    }


def half_power_bandwidth(
    frequency_hz: np.ndarray,
    power_response: np.ndarray,
) -> dict[str, float] | None:
    """Measure the dominant sampled resonance and its -3 dB bandwidth.

    ``power_response`` must be a linear power quantity, not a voltage or dB
    magnitude.  The result is a loaded, response-based Q.  It is unavailable
    when the sweep does not bracket both half-power crossings.
    """

    frequency, response = _validated_power_sweep(frequency_hz, power_response)
    if len(frequency) < 3:
        return None
    peak_index = int(np.argmax(response))
    peak = float(response[peak_index])
    if peak <= 0 or peak_index == 0 or peak_index == len(response) - 1:
        return None
    lower, upper = _half_power_crossings(frequency, response, peak_index, peak)
    if lower is None or upper is None or upper <= lower:
        return None
    return _bandwidth_result(float(frequency[peak_index]), peak, lower, upper)


def _validated_power_sweep(
    frequency_hz: np.ndarray,
    power_response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency_hz, dtype=float)
    response = np.asarray(power_response, dtype=float)
    if frequency.ndim != 1 or response.ndim != 1 or len(frequency) != len(response):
        raise ValueError("frequency and power response must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(response)):
        raise ValueError("frequency and power response must be finite")
    if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequency must be positive and strictly increasing")
    return frequency, response


def _half_power_crossings(
    frequency: np.ndarray,
    response: np.ndarray,
    peak_index: int,
    peak: float,
) -> tuple[float | None, float | None]:
    level = peak / 2.0
    lower = _nearest_level_crossing(frequency, response, level, peak_index, -1)
    upper = _nearest_level_crossing(frequency, response, level, peak_index, 1)
    return lower, upper


def _bandwidth_result(resonance: float, peak: float, lower: float, upper: float) -> dict[str, float]:
    bandwidth = upper - lower
    return {
        "resonant_frequency_Hz": resonance,
        "peak_response": peak,
        "lower_frequency_Hz": lower,
        "upper_frequency_Hz": upper,
        "bandwidth_Hz": bandwidth,
        "fractional_bandwidth": bandwidth / resonance,
        "loaded_quality_factor": resonance / bandwidth,
    }


def frequency_sweep_metrics(
    ac: pd.DataFrame,
    z0: float = DEFAULT_Z0,
    load_current_column: str | None = None,
) -> dict[str, Any] | None:
    """Summarize a solver-produced multi-point AC sweep.

    A declared load probe selects load real power. With no declared load
    probe, the basis is the accepted-power fraction ``1 - |Gamma|^2``.
    """

    if len(ac) < 2:
        return None
    if not np.isfinite(z0) or z0 <= 0:
        raise ValueError("reference impedance must be positive and finite")
    return summarize_impedance_sweep(input_impedance(ac, z0), load_current_column)


def summarize_impedance_sweep(impedance: pd.DataFrame, load_current_column: str | None) -> dict[str, Any] | None:
    """Summarize the computed impedance sweep shared by tables and figures."""

    if len(impedance) < 2:
        return None
    impedance = impedance.sort_values("frequency_Hz")
    frequency = impedance["frequency_Hz"].to_numpy(float)
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise ValueError("AC sweep frequency must be finite, positive, and unique")
    reflection = impedance["reflection_magnitude"].to_numpy(float)
    available = np.flatnonzero(np.isfinite(reflection))
    best_index = int(available[np.argmin(reflection[available])]) if len(available) else None
    basis, response = _power_response(impedance, load_current_column)
    reason = None
    resonance = None
    if response is None or not np.all(np.isfinite(response)):
        reason = f"{basis} requires finite observations at every sweep frequency"
    elif band := half_power_bandwidth(frequency, response):
        resonance = {"basis": basis, **band}
    else:
        reason = f"the sweep does not bracket a half-power band on the {basis} basis"
    return {
        "sample_count": len(impedance),
        "start_frequency_Hz": float(frequency[0]),
        "stop_frequency_Hz": float(frequency[-1]),
        "reflection_sample_count": len(available),
        "sampled_best_match_frequency_Hz": None if best_index is None else float(frequency[best_index]),
        "minimum_reflection_magnitude": None if best_index is None else float(reflection[best_index]),
        "half_power_resonance": resonance,
        **({"half_power_unavailable_reason": reason} if reason is not None else {}),
    }


def _power_response(ac: pd.DataFrame, load_current_column: str | None) -> tuple[str, np.ndarray | None]:
    if not load_current_column:
        reflection = ac["reflection_magnitude"].to_numpy(float)
        return "accepted_power_fraction", 1.0 - reflection**2
    load_voltage = _phasor_array(ac, LOAD_VOLTAGE_COLUMN)
    load_current = _phasor_array(ac, load_current_column)
    if load_voltage is None or load_current is None:
        return "load_real_power_W", None
    return "load_real_power_W", 0.5 * np.real(load_voltage * np.conj(load_current))


def _phasor_array(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    real, imag = f"{column}_re", f"{column}_im"
    if real not in frame or imag not in frame:
        return None
    return frame[real].to_numpy(float) + 1j * frame[imag].to_numpy(float)


def _nearest_level_crossing(
    frequency: np.ndarray,
    response: np.ndarray,
    level: float,
    peak_index: int,
    direction: int,
) -> float | None:
    indices = range(peak_index - 1, -1, -1) if direction < 0 else range(peak_index, len(response) - 1)
    for index in indices:
        crossing = _linear_level_crossing(
            float(frequency[index]),
            float(response[index]),
            float(frequency[index + 1]),
            float(response[index + 1]),
            level,
        )
        if crossing is not None:
            return crossing
    return None


def _linear_level_crossing(f0: float, y0: float, f1: float, y1: float, level: float) -> float | None:
    below0, below1 = y0 - level, y1 - level
    if below0 == 0.0:
        return f0
    if below1 == 0.0:
        return f1
    if below0 * below1 > 0.0 or y1 == y0:
        return None
    return f0 + (level - y0) * (f1 - f0) / (y1 - y0)


def at_frequency(ac: pd.DataFrame, frequency_hz: float) -> pd.Series:
    """Read a requested point from a simulated AC sweep without extrapolation.

    Only solver-produced phasors are interpolated here. Independent measured
    load-table rows are never interpolated into new electrical conditions.
    """

    return ac_frequency_sample(ac, frequency_hz)[0]


def ac_frequency_sample(
    ac: pd.DataFrame, frequency_hz: float, *, require_simulated: bool = False
) -> tuple[pd.Series, dict[str, Any]]:
    """Resolve a solved frequency or identify the bounds of an interpolated point."""

    if "frequency_Hz" not in ac:
        raise ValueError("canonical AC response is missing columns ['frequency_Hz']")
    if ac.empty:
        raise ValueError("frequency response is empty")
    target = frequency_hz
    if not np.isfinite(target) or target <= 0:
        raise ValueError("requested frequency must be positive and finite")
    ordered = ac.sort_values("frequency_Hz").reset_index(drop=True)
    frequencies = ordered["frequency_Hz"].to_numpy(float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("AC frequencies must be finite, positive, and unique")
    closest = int(np.argmin(np.abs(frequencies - target)))
    if same_frequency_hz(frequencies[closest], target):
        return ordered.iloc[closest], {"frequency_basis": "simulated"}
    if require_simulated:
        raise MeasurementUnavailableError(
            f"AC decision requires a solved sample at {target:g} Hz; "
            "use solver.ac.frequency_Hz or include the operating point in the sweep"
        )
    if target < frequencies[0] or target > frequencies[-1]:
        raise ValueError(
            f"frequency {target:g} Hz is outside the simulated sweep [{frequencies[0]:g}, {frequencies[-1]:g}] Hz"
        )
    upper = int(np.searchsorted(frequencies, target, side="right"))
    lower = upper - 1
    fraction = (target - frequencies[lower]) / (frequencies[upper] - frequencies[lower])
    values: dict[str, Any] = {"frequency_Hz": target}
    for column in ordered.columns:
        if column == "frequency_Hz":
            continue
        lo, hi = ordered.iloc[lower][column], ordered.iloc[upper][column]
        if pd.api.types.is_numeric_dtype(ordered[column]):
            values[column] = float(lo) + fraction * (float(hi) - float(lo))
        else:
            values[column] = lo if fraction <= 0.5 else hi
    return pd.Series(values), {
        "frequency_basis": "interpolated",
        "frequency_bracket_Hz": [float(frequencies[lower]), float(frequencies[upper])],
    }
