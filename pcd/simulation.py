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

from .inputs import integer, reject_unknown

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
LOAD_VOLTAGE_COLUMN = "load_voltage_V"


def _resolve(value: Any, params: Mapping[str, Any] | None) -> Any:
    """Resolve a bare or ``$`` parameter reference in an analysis request."""

    if isinstance(value, str) and params is not None:
        key = value[1:] if value.startswith("$") else value
        if key in params:
            return params[key]
    return value


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
        reject_unknown(config, {"step_s", "stop_s"}, "solver.tran")
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
    source: str | None = None

    def __post_init__(self) -> None:
        sweep = self.sweep.strip().lower()
        if sweep not in {"lin", "dec", "oct"}:
            raise ValueError("AC sweep must be lin, dec, or oct")
        points = integer(self.points, "AC points", minimum=1)
        start = self.start_hz
        stop = self.stop_hz
        if not math.isfinite(start) or not math.isfinite(stop) or start <= 0 or stop <= 0 or stop < start:
            raise ValueError("AC frequencies must be positive and finite, with stop_Hz >= start_Hz")
        if sweep == "lin" and ((points == 1) != (start == stop)):
            raise ValueError(
                "single-point linear AC requires frequency_Hz or equal endpoints; a range needs >= 2 points"
            )
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> AcSweep:
        reject_unknown(config, {"frequency_Hz", "sweep", "points", "start_Hz", "stop_Hz", "source"}, "solver.ac")
        if "frequency_Hz" in config:
            conflicting = sorted({"sweep", "points", "start_Hz", "stop_Hz"} & set(config))
            if conflicting:
                raise ValueError(f"AC frequency_Hz cannot be combined with {conflicting}")
            frequency = float(_resolve(config["frequency_Hz"], params))
            return cls(sweep="lin", points=1, start_hz=frequency, stop_hz=frequency, source=config.get("source"))
        return cls(
            sweep=str(config.get("sweep", "dec")),
            points=integer(config.get("points", 20), "AC points", minimum=1),
            start_hz=float(_resolve(config.get("start_Hz", 1e6), params)),
            stop_hz=float(_resolve(config.get("stop_Hz", 1e8), params)),
            source=config.get("source"),
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
        reject_unknown(solver, {"name", "executable", "timeout_s", "options", "tran", "ac"}, "solver")
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

    def validate_for(self, analysis: AnalysisRequest) -> None:
        """A successful response must contain the requested numerical domain."""

        if self.status != "ok":
            return
        if analysis.transient is not None:
            frame = self.as_frame()
            time = frame[TIME_COLUMN].to_numpy(float)
            voltage = frame[VOLTAGE_COLUMN].to_numpy(float)
            stop = analysis.transient.stop_s
            tolerance = stop * 1e-9
            if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(voltage)):
                raise ValueError("successful transient requires finite aligned time/voltage observations")
            if np.any(np.diff(time) < 0) or abs(time[0]) > tolerance or abs(time[-1] - stop) > tolerance:
                raise ValueError("transient result must cover [0, requested stop_s] in time order")
        if analysis.ac is not None:
            frame = self.frequency_response
            required = ["frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"]
            if frame is None or frame.empty or not set(required) <= set(frame):
                raise ValueError("successful AC analysis requires frequency and complex port observations")
            if not np.all(np.isfinite(frame[required].to_numpy(float))):
                raise ValueError("successful AC response must be finite")
            sweep = analysis.ac
            if sweep.sweep == "lin":
                expected = np.linspace(sweep.start_hz, sweep.stop_hz, sweep.points)
            else:
                base = 10.0 if sweep.sweep == "dec" else 2.0
                count = 1 + math.floor(math.log(sweep.stop_hz / sweep.start_hz, base) * sweep.points + 1e-9)
                expected = sweep.start_hz * base ** (np.arange(count) / sweep.points)
            actual = frame["frequency_Hz"].to_numpy(float)
            if len(actual) != len(expected) or not np.allclose(actual, expected, rtol=1e-8, atol=0):
                raise ValueError("AC result frequencies differ from the requested sweep")

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
