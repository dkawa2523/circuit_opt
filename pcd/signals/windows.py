"""Periodic steady-state detection and measurement-window selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .power import time_average
from .series import interpolate_sides, interval_observed, ordered_samples


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
    observed: bool = True


def required_periodic_cycles(measure_cycles: int, consecutive: int) -> int:
    return max(measure_cycles, consecutive + 1)


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
    older = np.column_stack(interpolate_sides(time_s, values, older_start + phase)).ravel()
    newer = np.column_stack(interpolate_sides(time_s, values, newer_start + phase)).ravel()
    grid = np.repeat(phase, 2)
    scale = max(float(np.sqrt(time_average(newer**2, grid))), float(np.ptp(newer[1:-1])), 1e-15)
    return float(np.sqrt(time_average((newer - older) ** 2, grid)) / scale)


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
    every adjacent pair within the measurement window, and at least the final
    ``consecutive`` pairs, agrees within ``tolerance``. Sampling must resolve the
    fundamental in the compared and measured cycles. The returned window is still
    useful for diagnostics when it is not settled.
    """

    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0 or measure_cycles < 1 or consecutive < 1:
        return None
    time, signal = ordered_samples(time_s, values)
    if not np.all(np.isfinite(time)) or np.count_nonzero(np.diff(time) > 0) < 3:
        return None
    period = 1.0 / float(fundamental_hz)
    start, end = float(time[0]), float(time[-1])
    span = end - start
    full_cycles = int(np.floor(span / period + 1e-12))
    if full_cycles < 1:
        return None

    cycles = min(measure_cycles, full_cycles)
    required_cycles = required_periodic_cycles(measure_cycles, consecutive)
    required_pairs = required_cycles - 1
    compared = min(required_pairs, full_cycles - 1)
    observed_start = end - (compared + 1) * period
    observed = interval_observed(time, signal, observed_start, end, max_step_s=0.5 * period)
    residual = (
        max(_cycle_residual(time, signal, end - offset * period, period) for offset in range(compared))
        if compared and observed
        else None
    )
    settled = compared == required_pairs and residual is not None and residual <= tolerance
    return PeriodicWindow(
        start_s=end - cycles * period,
        end_s=end,
        cycles=cycles,
        settled=settled,
        residual=residual,
        compared_cycles=compared,
        available_cycles=full_cycles,
        required_cycles=required_cycles,
        observed=observed,
    )
