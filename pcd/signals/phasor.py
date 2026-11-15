"""Exact-tone phasor estimation on uniform or irregular time grids."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .series import interval_observed, interval_series, trapezoid_weights
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
    if not requested or any(item < 1 for item in requested) or not np.isfinite(fundamental_hz) or fundamental_hz <= 0:
        return {}
    selected = window or periodic_window(
        time_s,
        values,
        fundamental_hz,
        measure_cycles=measure_cycles,
        consecutive=1,
        tolerance=float("inf"),
    )
    max_step_s = 0.5 / max(requested) / fundamental_hz
    if selected is None or not interval_observed(
        time_s, values, selected.start_s, selected.end_s, max_step_s=max_step_s
    ):
        return {}
    time, signal = interval_series(time_s, values, start_s=selected.start_s, end_s=selected.end_s)
    weights = trapezoid_weights(time)
    usable = weights > 0
    if np.count_nonzero(usable) < 2 * len(requested) + 1:
        return {}

    relative = time[usable] - selected.start_s
    columns = [np.ones(len(relative), dtype=float)]
    for harmonic in requested:
        angle = 2.0 * np.pi * harmonic * fundamental_hz * relative
        columns.extend((np.cos(angle), np.sin(angle)))
    matrix = np.column_stack(columns)
    root_weight = np.sqrt(weights[usable])
    weighted_matrix = matrix * root_weight[:, None]
    weighted_signal = signal[usable] * root_weight
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(weighted_matrix, weighted_signal, rcond=None)
    if rank < matrix.shape[1]:
        return {}
    return {
        harmonic: complex(coefficients[1 + 2 * index], -coefficients[2 + 2 * index])
        for index, harmonic in enumerate(requested)
    }
