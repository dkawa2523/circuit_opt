"""Canonical handling of non-uniform simulator time series."""

from __future__ import annotations

import numpy as np


def ordered_samples(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align arrays and order equal-time events by their original sample order."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    signal = np.asarray(values, dtype=float).reshape(-1)
    if time.shape != signal.shape:
        raise ValueError("time and signal arrays must have equal lengths")
    if np.any(np.diff(time) < 0):
        order = np.argsort(time, kind="stable")
        return time[order], signal[order]
    return time, signal


def clean_series(time_s: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop non-finite rows, sort time, and average duplicate timestamps."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    if any(len(value) != len(time) for value in arrays):
        raise ValueError("time and signal arrays must have equal lengths")
    if not len(time):
        return (time, *(np.asarray([], dtype=float) for _value in arrays))

    finite = np.isfinite(time)
    for value in arrays:
        finite &= np.isfinite(value)
    time = time[finite]
    arrays = [value[finite] for value in arrays]
    if not len(time):
        return (time, *(np.asarray([], dtype=float) for _value in arrays))

    order = np.argsort(time, kind="stable")
    time = time[order]
    arrays = [value[order] for value in arrays]
    unique, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    if len(unique) == len(time):
        return (time, *arrays)

    reduced: list[np.ndarray] = []
    for value in arrays:
        sums = np.bincount(inverse, weights=value, minlength=len(unique))
        reduced.append(sums / counts)
    return (unique, *reduced)


def trapezoid_weights(time_s: np.ndarray) -> np.ndarray:
    """Per-sample integration weights for an irregular monotonic grid."""

    time = np.asarray(time_s, dtype=float)
    if len(time) == 0:
        return np.asarray([], dtype=float)
    if len(time) == 1:
        return np.ones(1, dtype=float)
    delta = np.diff(time)
    if np.any(delta < 0):
        raise ValueError("time must be nondecreasing")
    weights = np.empty(len(time), dtype=float)
    weights[0] = delta[0] / 2.0
    weights[-1] = delta[-1] / 2.0
    if len(time) > 2:
        weights[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    return weights


def interval_observed(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    max_step_s: float | None = None,
) -> bool:
    """Check finite endpoints and, when requested, the spacing of used intervals."""

    time, signal = ordered_samples(time_s, values)
    if len(time) < 2 or not np.all(np.isfinite(time)):
        return False
    tolerance = 64 * np.finfo(float).eps * max(abs(float(time[0])), abs(float(time[-1])))
    if start_s >= end_s or start_s < time[0] - tolerance or end_s > time[-1] + tolerance:
        return False
    step = np.diff(time)
    used = (step > 0) & (time[1:] > start_s) & (time[:-1] < end_s)
    if max_step_s is not None and np.any(step[used] >= max_step_s):
        return False
    finite = np.isfinite(signal)
    return bool(np.any(used) and np.all((finite[:-1] & finite[1:])[used]))


def interpolate_sides(time: np.ndarray, signal: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sorted samples, returning both limits at equal-time events."""

    # Snap roundoff from cycle shifts back to the native knot before sampling.
    upper = np.minimum(np.searchsorted(time, query), len(time) - 1)
    lower = np.maximum(upper - 1, 0)
    nearest = np.where(np.abs(query - time[lower]) < np.abs(query - time[upper]), time[lower], time[upper])
    tolerance = 8 * np.finfo(float).eps * max(abs(float(time[0])), abs(float(time[-1])))
    query = np.clip(np.where(np.abs(query - nearest) <= tolerance, nearest, query), time[0], time[-1])
    first = np.searchsorted(time, query, side="left")
    last = np.searchsorted(time, query, side="right") - 1
    at_knot = time[first] == query
    before = np.maximum(first - 1, 0)
    fraction = np.divide(query - time[before], time[first] - time[before], out=np.zeros_like(query), where=~at_knot)
    interpolated = (1 - fraction) * signal[before] + fraction * signal[first]
    return np.where(at_knot, signal[first], interpolated), np.where(at_knot, signal[last], interpolated)


def interval_series(
    time_s: np.ndarray, values: np.ndarray, *, start_s: float, end_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep native samples in an interval, interpolating only its boundaries.

    Adaptive solver knots carry narrow peaks and switching events. Replacing
    them with a uniform plotting grid changes both stress and power evidence.
    """

    time, signal = ordered_samples(time_s, values)
    if not len(time):
        return time, signal
    first = int(np.searchsorted(time, start_s, side="left"))
    last = int(np.searchsorted(time, end_s, side="right"))
    grid, observed = time[first:last], signal[first:last]
    if not len(grid) or grid[0] != start_s:
        _, boundary = interpolate_sides(time, signal, np.array([start_s]))
        grid, observed = np.r_[start_s, grid], np.r_[boundary, observed]
    if grid[-1] != end_s:
        boundary, _ = interpolate_sides(time, signal, np.array([end_s]))
        grid, observed = np.r_[grid, end_s], np.r_[observed, boundary]
    return grid, observed
