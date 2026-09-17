"""Canonical handling of non-uniform simulator time series."""

from __future__ import annotations

import numpy as np


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
    if np.any(delta <= 0):
        raise ValueError("time must be strictly increasing")
    weights = np.empty(len(time), dtype=float)
    weights[0] = delta[0] / 2.0
    weights[-1] = delta[-1] / 2.0
    if len(time) > 2:
        weights[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    return weights
