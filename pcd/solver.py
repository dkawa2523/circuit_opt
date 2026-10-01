"""Running a netlist and reading back a waveform.

This is the second half of the simulation pipeline:

    netlist text -> solver -> SimulationResult

A solver never raises for a simulation problem: it returns a failed
:class:`SimulationResult` carrying the reason in ``diagnostics``, so an
optimizer keeps collecting observations instead of aborting the run.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .ngspice_io import read_frequency_response, read_transient_output
from .simulation import AC_FILE, AC_LOAD_VOLTAGE_COLUMN, WAVEFORM_FILE, SimulationResult, TransientAnalysis
from .simulation_input import DEFAULT_SOLVER_TIMEOUT_S, SolverRunRequest, SolverSettings


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


@lru_cache(maxsize=16)
def _solver_version_cached(resolved_executable: str, size: int, mtime_ns: int) -> str | None:
    """Read a version once for a particular installed solver binary."""

    del size, mtime_ns  # They are cache-key material, not command arguments.
    try:
        completed = subprocess.run(
            [resolved_executable, "--version"], text=True, capture_output=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return None
    lines = [line.strip().strip("*").strip() for line in text.splitlines()]
    meaningful = [line for line in lines if line]
    for line in meaningful:
        if line.lower().startswith("ngspice"):
            return line.split(":", 1)[0].strip()
    return meaningful[0] if meaningful else None


def solver_version(executable: str) -> str | None:
    """Return the executable version, refreshing when the binary changes."""

    resolved = shutil.which(executable)
    if resolved is None:
        return None
    try:
        stat = Path(resolved).stat()
        signature = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        # Test doubles and PATH shims need not name a local, stat-able file.
        signature = (0, 0)
    return _solver_version_cached(resolved, *signature)


def clear_solver_version_cache() -> None:
    """Clear cached solver metadata (primarily useful to environment tests)."""

    _solver_version_cached.cache_clear()
    _solver_binary_sha256_cached.cache_clear()


@lru_cache(maxsize=16)
def _solver_binary_sha256_cached(resolved_executable: str, size: int, mtime_ns: int) -> str | None:
    """Hash a solver binary once for a particular filesystem identity."""

    del size, mtime_ns
    path = Path(resolved_executable)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _solver_binary_identity(resolved: str | None) -> dict[str, Any]:
    if not resolved:
        return {"executable_size": None, "executable_mtime_ns": None, "executable_sha256": None}
    try:
        stat = Path(resolved).stat()
    except OSError:
        return {"executable_size": None, "executable_mtime_ns": None, "executable_sha256": None}
    return {
        "executable_size": stat.st_size,
        "executable_mtime_ns": stat.st_mtime_ns,
        "executable_sha256": _solver_binary_sha256_cached(resolved, stat.st_size, stat.st_mtime_ns),
    }


def solver_identity(settings: SolverSettings) -> dict[str, Any]:
    """Return the solver facts that affect execution and cache reuse."""

    if settings.executable:
        executable = settings.executable
    elif settings.name == "ngspice_cli":
        executable = default_ngspice_executable()
    else:
        executable = None
    resolved = shutil.which(executable) if executable else None
    return {
        "name": settings.name,
        "executable": executable,
        "resolved_executable": resolved,
        "version": solver_version(executable) if executable and resolved else None,
        "timeout_s": settings.timeout_s,
        **_solver_binary_identity(resolved),
    }


def diagnose_solver(
    solver_name: str = "ngspice_cli",
    executable: str | None = None,
    timeout_s: float = DEFAULT_SOLVER_TIMEOUT_S,
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


def _as_text(value: str | bytes | None) -> str:
    """``subprocess`` types stdout/stderr as bytes on TimeoutExpired even in text mode."""

    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def _transient_completion_failure(
    result: SimulationResult,
    requested: TransientAnalysis,
) -> dict[str, Any] | None:
    """Describe a partial transient left behind by an otherwise clean process exit."""

    finite_time = np.asarray(result.time_s, dtype=float)
    finite_time = finite_time[np.isfinite(finite_time)]
    last_time_s = float(np.max(finite_time)) if finite_time.size else None
    tolerance_s = max(requested.step_s * 1.01, requested.stop_s * 1e-9)
    if last_time_s is not None and last_time_s >= requested.stop_s - tolerance_s:
        return None
    return {
        "incomplete_transient": True,
        "last_time_s": last_time_s,
        "requested_stop_s": requested.stop_s,
    }


def ngspice_cli(request: SolverRunRequest) -> SimulationResult:
    """Execute ngspice from an already resolved, solver-facing request."""

    run_dir = request.run_dir
    netlist_path = request.netlist_path
    simulation = request.simulation
    exe = simulation.solver.executable or default_ngspice_executable()
    timeout = simulation.solver.timeout_s
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

    analysis = simulation.analysis
    probes = simulation.probes
    waveform = run_dir / WAVEFORM_FILE
    ac_path = run_dir / AC_FILE
    missing_waveform = analysis.transient is not None and not waveform.exists()
    missing_ac = analysis.ac is not None and not ac_path.exists()
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
        if waveform.exists() and analysis.transient is not None:
            result = read_transient_output(waveform, probes.transient_columns)
            incomplete = _transient_completion_failure(result, analysis.transient)
            if incomplete is not None:
                detail = (
                    "ngspice transient ended before the requested stop time: "
                    f"last={incomplete['last_time_s']!r} s, requested={analysis.transient.stop_s:g} s"
                )
                return _failed(
                    f"{log}\nPCD: {detail}\n",
                    {**diagnostics, **incomplete},
                )
        else:
            empty = np.array([], dtype=float)
            result = SimulationResult(time_s=empty, voltage_V=empty, current_A=empty)
        if ac_path.exists() and analysis.ac is not None:
            extras = [AC_LOAD_VOLTAGE_COLUMN, *probes.ac_columns]
            result.frequency_response = read_frequency_response(ac_path, extras)
    except (OSError, ValueError) as exc:
        return _failed(log, {**diagnostics, "parse_error": f"{type(exc).__name__}: {exc}"})
    result.log = log
    result.diagnostics.update(diagnostics)
    return result
