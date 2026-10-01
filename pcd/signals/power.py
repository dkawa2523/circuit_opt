"""Time-weighted power measurements shared by every application."""

from __future__ import annotations

import numpy as np

from .series import clean_series


def time_average(values: np.ndarray, time_s: np.ndarray) -> float:
    time, signal = clean_series(time_s, values)
    if len(time) == 0:
        return 0.0
    if len(time) == 1 or time[-1] <= time[0]:
        return float(np.mean(signal))
    return float(np.trapezoid(signal, time) / (time[-1] - time[0]))


def real_power(time_s: np.ndarray, voltage_V: np.ndarray, current_A: np.ndarray) -> float:
    time, voltage, current = clean_series(time_s, voltage_V, current_A)
    return time_average(voltage * current, time)
