"""Exact-tone phasor estimation on uniform or irregular time grids."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .series import clean_series, trapezoid_weights
from .windows import PeriodicWindow, periodic_window


def harmonic_phasors(
    time_s: np.ndarray,
    values: np.ndarray,
    fundamental_hz: float,
    harmonics: Iterable[int] = (1,),
    *,
    window: PeriodicWindow | None = None,
    measure_cycles: int = 3,
) -> dict[int, complex]:
    """Fit requested harmonics at their exact frequencies by weighted LS.

    The complex convention matches an FFT: ``A*sin(wt)`` is ``-j*A`` and
    ``A*cos(wt)`` is ``A``.  Trapezoidal sample weights prevent adaptive solver
    timesteps from biasing the fit toward densely sampled intervals.
    """

    requested = tuple(dict.fromkeys(int(item) for item in harmonics))
    if not requested or any(item < 1 for item in requested) or fundamental_hz <= 0:
        return {}
    time, signal = clean_series(time_s, values)
    if len(time) < 2 * len(requested) + 1:
        return {}
    selected = window or periodic_window(
        time,
        signal,
        fundamental_hz,
        measure_cycles=measure_cycles,
        consecutive=1,
        tolerance=float("inf"),
    )
    if selected is None:
        return {}
    mask = (time >= selected.start_s) & (time <= selected.end_s)
    time = time[mask]
    signal = signal[mask]
    if len(time) < 2 * len(requested) + 1:
        return {}

    relative = time - selected.start_s
    columns = [np.ones(len(time), dtype=float)]
    for harmonic in requested:
        angle = 2.0 * np.pi * harmonic * fundamental_hz * relative
        columns.extend((np.cos(angle), np.sin(angle)))
    matrix = np.column_stack(columns)
    root_weight = np.sqrt(trapezoid_weights(time))
    weighted_matrix = matrix * root_weight[:, None]
    weighted_signal = signal * root_weight
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(weighted_matrix, weighted_signal, rcond=None)
    if rank < matrix.shape[1]:
        return {}
    return {
        harmonic: complex(coefficients[1 + 2 * index], -coefficients[2 + 2 * index])
        for index, harmonic in enumerate(requested)
    }
