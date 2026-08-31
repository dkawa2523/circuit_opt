"""Periodic steady-state detection and measurement-window selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .series import clean_series


@dataclass(frozen=True, slots=True)
class PeriodicWindow:
    start_s: float
    end_s: float
    cycles: int
    settled: bool
    residual: float | None
    compared_cycles: int


def _cycle_residual(
    time_s: np.ndarray,
    values: np.ndarray,
    newer_end: float,
    period: float,
    samples_per_cycle: int,
) -> float:
    phase = np.linspace(0.0, period, samples_per_cycle, endpoint=False)
    older = np.interp(newer_end - 2 * period + phase, time_s, values)
    newer = np.interp(newer_end - period + phase, time_s, values)
    scale = max(float(np.sqrt(np.mean(newer**2))), float(np.ptp(newer)), 1e-15)
    return float(np.sqrt(np.mean((newer - older) ** 2)) / scale)


def periodic_window(
    time_s: np.ndarray,
    values: np.ndarray,
    fundamental_hz: float,
    *,
    measure_cycles: int = 3,
    consecutive: int = 2,
    tolerance: float = 1e-3,
    samples_per_cycle: int = 256,
) -> PeriodicWindow | None:
    """Select final whole cycles and report whether adjacent cycles agree.

    ``settled`` is true only when each of the final ``consecutive`` adjacent
    cycle pairs has a normalized RMS difference no greater than ``tolerance``.
    The returned window is still useful for diagnostics when it is not settled.
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
    compared = min(consecutive, max(0, full_cycles - 1))
    residuals = [
        _cycle_residual(time, signal, end - offset * period, period, samples_per_cycle) for offset in range(compared)
    ]
    residual = max(residuals) if residuals else None
    settled = compared == consecutive and residual is not None and residual <= tolerance
    return PeriodicWindow(
        start_s=end - cycles * period,
        end_s=end,
        cycles=cycles,
        settled=settled,
        residual=residual,
        compared_cycles=compared,
    )
