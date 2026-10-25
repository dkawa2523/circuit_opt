"""Time-weighted power measurements shared by every application."""

from __future__ import annotations

import numpy as np

from .series import clean_series, trapezoid_weights


def power_ratio(numerator_W: float, denominator_W: float) -> float | None:
    """Return a finite power ratio without substituting a numerical floor."""

    if not np.isfinite(numerator_W) or not np.isfinite(denominator_W) or denominator_W == 0:
        return None
    ratio = numerator_W / denominator_W
    return float(ratio) if np.isfinite(ratio) else None


def time_average(values: np.ndarray, time_s: np.ndarray) -> float:
    time, signal = clean_series(time_s, values)
    if len(time) == 0:
        return 0.0
    if len(time) == 1 or time[-1] <= time[0]:
        return float(np.mean(signal))
    return float(np.dot(signal, trapezoid_weights(time)) / (time[-1] - time[0]))


def real_power(time_s: np.ndarray, voltage_V: np.ndarray, current_A: np.ndarray) -> float:
    time, voltage, current = clean_series(time_s, voltage_V, current_A)
    return time_average(voltage * current, time)
