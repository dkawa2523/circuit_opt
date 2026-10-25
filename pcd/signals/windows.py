"""Periodic steady-state detection and measurement-window selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .power import time_average
from .series import clean_series


@dataclass(frozen=True, slots=True)
class PeriodicWindow:
    start_s: float
    end_s: float
    cycles: int
    settled: bool
    residual: float | None
    compared_cycles: int
    available_cycles: int
    required_cycles: int


def _cycle_residual(
    time_s: np.ndarray,
    values: np.ndarray,
    newer_end: float,
    period: float,
) -> float:
    older_start, newer_start = newer_end - 2 * period, newer_end - period
    # Compare at every knot from either cycle. A fixed phase grid can miss
    # precisely the short pulse whose cycle-to-cycle change matters most.
    older_knots = time_s[(time_s > older_start) & (time_s < newer_start)] - older_start
    newer_knots = time_s[(time_s > newer_start) & (time_s < newer_end)] - newer_start
    phase = np.unique(np.r_[0.0, older_knots, newer_knots, period])
    older = np.asarray(np.interp(older_start + phase, time_s, values), dtype=float)
    newer = np.asarray(np.interp(newer_start + phase, time_s, values), dtype=float)
    scale = max(float(np.sqrt(time_average(newer**2, phase))), float(np.ptp(newer)), 1e-15)
    return float(np.sqrt(time_average((newer - older) ** 2, phase)) / scale)


def periodic_window(
    time_s: np.ndarray,
    values: np.ndarray,
    fundamental_hz: float,
    *,
    measure_cycles: int = 3,
    consecutive: int = 2,
    tolerance: float = 1e-3,
) -> PeriodicWindow | None:
    """Select final whole cycles and report whether adjacent cycles agree.

    ``settled`` is true only when the requested measurement history exists and
    each of the final ``consecutive`` adjacent cycle pairs has a normalized RMS
    difference no greater than ``tolerance``. The returned window is still
    useful for diagnostics when it is not settled.
    """

    if fundamental_hz <= 0 or measure_cycles < 1 or consecutive < 1:
        return None
    time, signal = clean_series(time_s, values)
    if len(time) < 4:
        return None
    period = 1.0 / float(fundamental_hz)
    span = float(time[-1] - time[0])
    full_cycles = int(np.floor(span / period + 1e-12))
    if full_cycles < 1:
        return None

    end = float(time[-1])
    cycles = min(measure_cycles, full_cycles)
    required_cycles = max(measure_cycles, consecutive + 1)
    compared = min(consecutive, max(0, full_cycles - 1))
    residuals = [_cycle_residual(time, signal, end - offset * period, period) for offset in range(compared)]
    residual = max(residuals) if residuals else None
    settled = (
        full_cycles >= required_cycles and compared == consecutive and residual is not None and residual <= tolerance
    )
    return PeriodicWindow(
        start_s=end - cycles * period,
        end_s=end,
        cycles=cycles,
        settled=settled,
        residual=residual,
        compared_cycles=compared,
        available_cycles=full_cycles,
        required_cycles=required_cycles,
    )
