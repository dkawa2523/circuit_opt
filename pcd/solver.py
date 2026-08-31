"""Running a netlist and reading back a waveform.

This is the second half of the simulation pipeline:

    netlist text -> solver -> SimulationResult

A solver never raises for a simulation problem: it returns a failed
:class:`SimulationResult` carrying the reason in ``diagnostics``, so an
optimizer keeps collecting observations instead of aborting the run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import AC_FILE, AC_LOAD_VOLTAGE, ac_probe_plan, ac_sweep, probe_plan, read_ac, transient_requested
from .case import Case

DEFAULT_TIMEOUT_S = 300.0


@dataclass
class SimulationResult:
    """What every solver returns."""

    time_s: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray | None = None
    status: str = "ok"
    log: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Frequency response, when the case asked for an AC sweep.
    frequency_response: pd.DataFrame | None = None
    #: Extra vectors the case asked for, keyed by the column name to store them under.
    probes: dict[str, np.ndarray] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        """The boundary artifact.

        The first three columns are the contract every consumer relies on;
        probes are appended under their own names, so recording one more cannot
        break a reader that selects by name.
        """

        current = self.current_A if self.current_A is not None else np.zeros_like(self.time_s)
        frame = pd.DataFrame({"time_s": self.time_s, "voltage_V": self.voltage_V, "current_A": current})
        for name, values in self.probes.items():
            frame[name] = values
        return frame


def _failed(log: str, diagnostics: dict[str, Any]) -> SimulationResult:
    """A failure still has the waveform shape, so downstream code stays uniform."""

    nan = np.array([np.nan])
    return SimulationResult(np.array([0.0]), nan, nan, status="failed", log=log, diagnostics=diagnostics)


# -----------------------------------------------------------------------------
# Solver environment
# -----------------------------------------------------------------------------


def default_ngspice_executable() -> str:
    """Prefer the console build on Windows so batch runs open no window."""

    if sys.platform == "win32" and shutil.which("ngspice_con.exe"):
        return "ngspice_con.exe"
    return "ngspice"


def solver_timeout_s(case: Case) -> float:
    raw = (case.data.get("solver", {}) or {}).get("timeout_s", DEFAULT_TIMEOUT_S)
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return timeout if timeout > 0 else DEFAULT_TIMEOUT_S


@lru_cache(maxsize=16)
def solver_version(executable: str) -> str | None:
    if shutil.which(executable) is None:
        return None
    try:
        completed = subprocess.run([executable, "--version"], text=True, capture_output=True, check=False, timeout=5)
    except Exception:
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0] if text else None


def solver_identity(case: Case, solver_name: str | None = None) -> dict[str, Any]:
    """Return the solver facts that affect execution and cache reuse."""

    cfg = case.data.get("solver", {}) or {}
    name = str(solver_name or cfg.get("name", "ngspice_cli"))
    if "executable" in cfg:
        executable = str(cfg["executable"])
    elif name == "ngspice_cli":
        executable = default_ngspice_executable()
    else:
        executable = None
    resolved = shutil.which(executable) if executable else None
    return {
        "name": name,
        "executable": executable,
        "resolved_executable": resolved,
        "version": solver_version(executable) if executable and resolved else None,
        "timeout_s": solver_timeout_s(case),
    }


def diagnose_solver(
    solver_name: str = "ngspice_cli", executable: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> dict[str, Any]:
    """Report whether a solver can actually run here, before a long batch."""

    report = {
        "schema": "solver_diagnostic.v1",
        "solver": str(solver_name),
        "executable": executable,
        "resolved_executable": shutil.which(str(executable)) if executable else None,
        "version": None,
        "timeout_s": float(timeout_s),
        "batch_runnable": False,
        "windows_prefers_console_binary": False,
        "notes": [],
    }
    if solver_name != "ngspice_cli":
        return {**report, "notes": [f"no built-in diagnostic for solver '{solver_name}'"]}
    return {**report, **_ngspice_diagnostic(executable)}


def _ngspice_diagnostic(executable: str | None) -> dict[str, Any]:
    """Can ngspice actually be run here, and which binary would be used?"""

    default_exe = default_ngspice_executable()
    exe = str(executable or default_exe)
    resolved = shutil.which(exe)

    notes = [] if resolved else [f"executable not found on PATH: {exe}"]
    if sys.platform == "win32" and not executable and exe == "ngspice":
        notes.append("ngspice_con.exe was not found on PATH; a GUI-capable ngspice.exe may open a window")
    return {
        "executable": exe,
        "resolved_executable": resolved,
        "version": solver_version(exe) if resolved else None,
        "batch_runnable": bool(resolved),
        "windows_prefers_console_binary": bool(sys.platform == "win32" and default_exe == "ngspice_con.exe"),
        "batch_command": [exe, "-b", "-o", "solver.log", "netlist.cir"],
        "notes": notes,
    }


#: Older ngspice releases wrote a different number of columns for the same
#: three vectors, so the standard layouts stay pinned by column count.
_WRDATA_COLUMNS = {6: (0, 3, 5), 5: (0, 2, 4), 4: (0, 1, 3), 3: (0, 1, 2)}


def parse_wrdata(path: str | Path, probe_names: list[str] | None = None) -> SimulationResult:
    """Read an ngspice transient `wrdata` file.

    ngspice writes every vector as a ``(scale, value)`` pair, so a file with
    probes has two columns per vector and the values sit at the odd indices.
    """

    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    width = arr.shape[1]
    if width == 2:
        return SimulationResult(time_s=arr[:, 0], voltage_V=arr[:, 1], current_A=None)

    names = probe_names or []
    if names:
        values = arr[:, 1::2]
        expected = 3 + len(names)
        if values.shape[1] < expected:
            raise ValueError(f"expected {expected} vectors for probes {names}, found {values.shape[1]}: {path}")
        return SimulationResult(
            time_s=values[:, 0],
            voltage_V=values[:, 1],
            current_A=values[:, 2],
            probes={name: values[:, 3 + k] for k, name in enumerate(names)},
        )

    layout = _WRDATA_COLUMNS.get(min(width, 6))
    if layout is None:
        raise ValueError(f"cannot parse wrdata output: {path}")
    t, v, i = layout
    return SimulationResult(time_s=arr[:, t], voltage_V=arr[:, v], current_A=arr[:, i])


def _as_text(value: str | bytes | None) -> str:
    """``subprocess`` types stdout/stderr as bytes on TimeoutExpired even in text mode."""

    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def ngspice_cli(netlist_path: str | Path, run_dir: str | Path, case: Case, params: dict[str, Any]) -> SimulationResult:
    run_dir = Path(run_dir)
    exe = str(case.data.get("solver", {}).get("executable") or default_ngspice_executable())
    timeout = solver_timeout_s(case)
    diagnostics: dict[str, Any] = {"executable": exe, "timeout_s": timeout}

    if shutil.which(exe) is None:
        return _failed(f"ngspice executable not found: {exe}", {**diagnostics, "missing_executable": True})

    log_path = run_dir / "solver.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        completed = subprocess.run(
            [exe, "-b", "-o", str(log_path), str(netlist_path)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        diagnostics |= {"timed_out": True, "timeout_command": list(exc.cmd) if exc.cmd else [exe]}
        log = f"ngspice timed out after {timeout:g}s\nSTDOUT:\n{_as_text(exc.stdout)}\nSTDERR:\n{_as_text(exc.stderr)}"
        return _failed(log, diagnostics)

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log += f"\nSTDOUT:\n{_as_text(completed.stdout)}\nSTDERR:\n{_as_text(completed.stderr)}"
    diagnostics["returncode"] = completed.returncode

    solver_cfg = case.data.get("solver", {}) or {}
    waveform = run_dir / "waveform.csv"
    ac_path = run_dir / AC_FILE
    missing_waveform = transient_requested(solver_cfg) and not waveform.exists()
    missing_ac = ac_sweep(solver_cfg, params) is not None and not ac_path.exists()
    if completed.returncode != 0 or missing_waveform or missing_ac:
        return _failed(
            log,
            {
                **diagnostics,
                "missing_waveform": missing_waveform,
                "missing_frequency_response": missing_ac,
            },
        )
    try:
        if waveform.exists() and transient_requested(solver_cfg):
            result = parse_wrdata(waveform, probe_plan(case)[1])
        else:
            empty = np.array([], dtype=float)
            result = SimulationResult(time_s=empty, voltage_V=empty, current_A=empty)
        if ac_path.exists() and ac_sweep(solver_cfg, params) is not None:
            ac_columns = ac_probe_plan(case)[1]
            extras = [AC_LOAD_VOLTAGE, *ac_columns]
            result.frequency_response = read_ac(ac_path, extras)
    except Exception as exc:
        return _failed(log, {**diagnostics, "parse_error": f"{type(exc).__name__}: {exc}"})
    result.log = log
    result.diagnostics.update(diagnostics)
    return result
