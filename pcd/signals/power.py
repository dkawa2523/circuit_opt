"""Time-weighted power measurements shared by every application."""

from __future__ import annotations

import numpy as np

from .series import ordered_samples


def power_ratio(numerator_W: float, denominator_W: float) -> float | None:
    """Return a finite power ratio without substituting a numerical floor."""

    if not np.isfinite(numerator_W) or not np.isfinite(denominator_W) or denominator_W == 0:
        return None
    ratio = numerator_W / denominator_W
    return float(ratio) if np.isfinite(ratio) else None


def time_average(values: np.ndarray, time_s: np.ndarray) -> float:
    """Average over native intervals, with NaN for an unobserved interval."""

    time, signal = ordered_samples(time_s, values)
    if len(time) == 0:
        return 0.0
    if not np.all(np.isfinite(time)):
        return float("nan")
    duration = time[-1] - time[0]
    if duration == 0:
        finite = signal[np.isfinite(signal)]
        return float(np.mean(finite)) if len(finite) else float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        average = np.sum(_interval_energy(time, signal)) / duration
    return float(average) if np.isfinite(average) else float("nan")


def real_power(time_s: np.ndarray, voltage_V: np.ndarray, current_A: np.ndarray) -> float:
    voltage, current = np.asarray(voltage_V, dtype=float), np.asarray(current_A, dtype=float)
    if voltage.shape != current.shape:
        raise ValueError("voltage and current arrays must have equal shapes")
    return time_average(voltage * current, time_s)


def cumulative_energy(time_s: np.ndarray, power_W: np.ndarray) -> np.ndarray:
    """Integrate signed power on the supplied time grid, starting at its first sample.

    Equal-time event samples add no energy. An unobserved interval makes the
    cumulative value unavailable from that interval onward.
    """

    time = np.asarray(time_s, dtype=float)
    power = np.asarray(power_W, dtype=float)
    if time.ndim != 1 or time.shape != power.shape:
        raise ValueError("energy requires one-dimensional, aligned time and power samples")
    if not len(time):
        return time.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.cumsum(np.r_[0.0, _interval_energy(time, power)])
    return np.where(np.isfinite(energy), energy, np.nan)


def _interval_energy(time: np.ndarray, power: np.ndarray) -> np.ndarray:
    step = np.diff(time)
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(step)) or np.any(step < 0):
        raise ValueError("energy requires finite samples in time order")
    with np.errstate(over="ignore", invalid="ignore"):
        increments = np.where(step == 0, 0.0, (0.5 * power[:-1] + 0.5 * power[1:]) * step)
    return np.where(np.isfinite(increments), increments, np.nan)
