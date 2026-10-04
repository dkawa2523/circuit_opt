"""Solver-neutral requests, result columns, and in-memory results.

This module is the boundary between circuit/netlist preparation and a solver
adapter.  It deliberately contains no case parsing, ngspice syntax, metric
calculation, persistence, or study orchestration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ngspice writes these temporary files in the solver working directory.  The
# simulation persistence layer converts them to the stable paths below and
# removes the temporary copies.
AC_FILE = "ac.csv"
WAVEFORM_FILE = "waveform.csv"

# Stable, user-facing numerical artifacts.  Directory layout belongs to the
# persistence boundary rather than to a solver adapter or a netlist renderer.
AC_ARTIFACT = "data/ac.csv"
TRANSIENT_ARTIFACT = "data/transient.csv"

TIME_COLUMN = "time_s"
VOLTAGE_COLUMN = "voltage_V"
CURRENT_COLUMN = "current_A"
SOURCE_VOLTAGE_COLUMN = "source_voltage_V"
LOAD_CURRENT_COLUMN = "load_current_A"
AC_LOAD_VOLTAGE_COLUMN = "load_voltage_V"


def _resolve(value: Any, params: Mapping[str, Any] | None) -> Any:
    """Resolve a bare or ``$`` parameter reference in an analysis request."""

    if isinstance(value, str) and params is not None:
        key = value[1:] if value.startswith("$") else value
        if key in params:
            return params[key]
    return value


def _integer(value: Any, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(number)


@dataclass(frozen=True)
class TransientAnalysis:
    """One transient request, expressed in SI seconds."""

    step_s: float = 1e-9
    stop_s: float = 1e-6

    def __post_init__(self) -> None:
        step = self.step_s
        stop = self.stop_s
        if not math.isfinite(step) or not math.isfinite(stop) or step <= 0 or stop <= 0:
            raise ValueError("transient step_s and stop_s must be positive and finite")
        if step > stop:
            raise ValueError("transient step_s must not exceed stop_s")
        object.__setattr__(self, "step_s", step)
        object.__setattr__(self, "stop_s", stop)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> TransientAnalysis:
        return cls(
            step_s=float(_resolve(config.get("step_s", 1e-9), params)),
            stop_s=float(_resolve(config.get("stop_s", 1e-6), params)),
        )


@dataclass(frozen=True)
class AcSweep:
    """One ngspice-compatible frequency sweep."""

    sweep: str = "dec"
    points: int = 20
    start_hz: float = 1e6
    stop_hz: float = 1e8

    def __post_init__(self) -> None:
        sweep = self.sweep.strip().lower()
        if sweep not in {"lin", "dec", "oct"}:
            raise ValueError("AC sweep must be lin, dec, or oct")
        if self.points <= 0:
            raise ValueError("AC points must be a positive integer")
        start = self.start_hz
        stop = self.stop_hz
        if not math.isfinite(start) or not math.isfinite(stop) or start <= 0 or stop <= 0 or stop < start:
            raise ValueError("AC frequencies must be positive and finite, with stop_Hz >= start_Hz")
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> AcSweep:
        if "frequency_Hz" in config:
            conflicting = sorted({"sweep", "points", "start_Hz", "stop_Hz"} & set(config))
            if conflicting:
                raise ValueError(f"AC frequency_Hz cannot be combined with {conflicting}")
            frequency = float(_resolve(config["frequency_Hz"], params))
            return cls(sweep="lin", points=1, start_hz=frequency, stop_hz=frequency)
        return cls(
            sweep=str(config.get("sweep", "dec")),
            points=_integer(config.get("points", 20), "AC points"),
            start_hz=float(_resolve(config.get("start_Hz", 1e6), params)),
            stop_hz=float(_resolve(config.get("stop_Hz", 1e8), params)),
        )


@dataclass(frozen=True)
class AnalysisRequest:
    """The analyses one solver invocation must produce."""

    transient: TransientAnalysis | None
    ac: AcSweep | None

    def __post_init__(self) -> None:
        if self.transient is None and self.ac is None:
            raise ValueError("at least one transient or AC analysis is required")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
        params: Mapping[str, Any] | None = None,
    ) -> AnalysisRequest:
        solver = config or {}
        ac_config = solver.get("ac")
        if "ac" in solver and not isinstance(ac_config, Mapping):
            raise TypeError("solver.ac must be a mapping")
        ac = AcSweep.from_config(ac_config or {}, params) if "ac" in solver else None

        # Preserve the established default: a case with no explicit analysis
        # runs transient, while an explicit AC-only case does not.
        transient_config = solver.get("tran")
        if "tran" in solver and not isinstance(transient_config, Mapping):
            raise TypeError("solver.tran must be a mapping")
        transient = (
            TransientAnalysis.from_config(transient_config or {}, params)
            if "tran" in solver or "ac" not in solver
            else None
        )
        return cls(transient=transient, ac=ac)


@dataclass
class SimulationResult:
    """Canonical in-memory output returned by every solver adapter."""

    time_s: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray | None = None
    status: str = "ok"
    log: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    frequency_response: pd.DataFrame | None = None
    probes: dict[str, np.ndarray] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        """Return the canonical transient table, preserving missing current."""

        current = (
            self.current_A
            if self.current_A is not None
            else np.full(np.asarray(self.time_s).shape, np.nan, dtype=float)
        )
        frame = pd.DataFrame(
            {
                TIME_COLUMN: self.time_s,
                VOLTAGE_COLUMN: self.voltage_V,
                CURRENT_COLUMN: current,
            }
        )
        for name, values in self.probes.items():
            frame[name] = values
        return frame
