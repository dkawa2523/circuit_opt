"""Running a netlist and reading back a waveform.

This is the second half of the simulation pipeline:

    netlist text -> solver -> SimulationResult

A solver never raises for a simulation problem: it returns a failed
:class:`SimulationResult` carrying the reason in ``diagnostics``, so an
optimizer keeps collecting observations instead of aborting the run.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .ngspice_io import read_frequency_response, read_transient_output
from .simulation import AC_FILE, LOAD_VOLTAGE_COLUMN, WAVEFORM_FILE, SimulationResult
from .simulation_input import DEFAULT_SOLVER_TIMEOUT_S, SolverRunRequest, SolverSettings


def _failed(log: str, diagnostics: dict[str, Any], reason: str) -> SimulationResult:
    match = re.search(r"(?im)^\s*((?:error|fatal error):[^\n]+)", log)
    nan = np.array([np.nan])
    return SimulationResult(
        np.array([0.0]),
        nan,
        nan,
        status="failed",
        log=log,
        diagnostics={**diagnostics, "error": match[1].strip() if match else reason},
    )


# -----------------------------------------------------------------------------
# Solver environment
# -----------------------------------------------------------------------------


def default_ngspice_executable() -> str:
    """Prefer the console build on Windows so batch runs open no window."""

    if sys.platform == "win32" and shutil.which("ngspice_con.exe"):
        return "ngspice_con.exe"
    return "ngspice"


@lru_cache(maxsize=16)
def _solver_version_cached(
    resolved_executable: str, size: int, mtime_ns: int, environment: tuple[tuple[str, str], ...] | None = None
) -> str | None:
    """Read a version once for a particular installed solver binary."""

    del size, mtime_ns  # They are cache-key material, not command arguments.
    try:
        completed = subprocess.run(
            [resolved_executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
            env=dict(environment) if environment is not None else None,
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


def _resolved_solver_version(resolved: str, environment: tuple[tuple[str, str], ...] | None) -> str | None:
    try:
        stat = Path(resolved).stat()
        signature = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        # Test doubles and PATH shims need not name a local, stat-able file.
        signature = (0, 0)
    return _solver_version_cached(resolved, *signature, environment)


@lru_cache(maxsize=64)
def _solver_file_sha256_cached(resolved: str, size: int, mtime_ns: int, ctime_ns: int) -> str | None:
    """Hash an installed executable or model for a particular filesystem identity."""

    del size, mtime_ns, ctime_ns
    path = Path(resolved)
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
        "executable_sha256": _solver_file_sha256_cached(resolved, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    }


def _startup_fingerprint(executable: str | None, environment: dict[str, str]) -> dict[str, Any]:
    """Record known ngspice initialization inputs without saving the process environment."""

    roots = {Path(environment[key]) for key in ("SPICE_LIB_DIR", "SPICE_SCRIPTS") if environment.get(key)}
    folder = Path(executable).parent if executable else None
    if folder is not None:
        roots.update((folder, folder.parent / "share" / "ngspice"))
    if sys.platform != "win32":
        roots.update((Path("/usr/share/ngspice"), Path("/usr/local/share/ngspice")))
    files = {candidate for root in roots for candidate in (root / "spinit", root / "scripts" / "spinit")}
    user_init_names = (".spiceinit", "spice.rc") if sys.platform == "win32" else (".spiceinit",)
    for key in ("SPICE_USERINIT_DIR", "HOME", "USERPROFILE"):
        if directory := environment.get(key):
            files.update(Path(directory) / name for name in user_init_names)
    hashes, unresolved = _startup_dependencies(files, folder)
    affecting = {
        key: value
        for key, value in environment.items()
        if key.startswith(("SPICE_", "NGSPICE_", "LC_"))
        or key in {"LANG", "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH"}
    }
    return {
        "configuration_sha256": hashlib.sha256(repr(sorted(affecting.items())).encode()).hexdigest(),
        "startup_files_sha256": hashes,
        **({"unresolved_sources": unresolved} if unresolved else {}),
    }


def _startup_dependencies(files: set[Path], executable_dir: Path | None) -> tuple[dict[str, str], list[str]]:
    """Follow literal absolute source scripts and record installed code models."""

    pending = [path.resolve() for path in sorted(files) if path.is_file()]
    seen: set[Path] = set()
    hashes: dict[str, str] = {}
    unresolved: list[str] = []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        for line_number, line in enumerate(content.decode(errors="replace").splitlines(), 1):
            match = re.match(r"\s*(source|\.include|\.inc|\.lib|codemodel|osdi)\s+(.+)", line, re.IGNORECASE)
            if not match:
                continue
            command = match[1].lower()
            arguments = re.findall(r""""[^"\r\n]+"|'[^'\r\n]+'|[^\s;]+""", match[2].split(";", 1)[0])
            if command == ".lib" and len(arguments) < 2:
                continue  # A local library section declaration.
            for argument in arguments if command == "source" else arguments[:1]:
                declared = Path(argument.strip("\"'"))
                if command not in {"codemodel", "osdi"}:
                    if declared.is_absolute() and declared.is_file():
                        pending.append(declared.resolve())
                    else:
                        unresolved.append(f"{path}:{line_number}")
                    continue
                if "$" in argument or str(declared).startswith("~"):
                    unresolved.append(f"{path}:{line_number}")
                    continue
                candidates = [declared] if declared.is_absolute() else [path.parent / declared]
                if not declared.is_absolute() and executable_dir is not None:
                    candidates.append(executable_dir / declared)
                for dependency in candidates:
                    try:
                        resolved = dependency.resolve(strict=True)
                        stat = resolved.stat()
                    except (FileNotFoundError, NotADirectoryError):
                        continue
                    if digest := _solver_file_sha256_cached(
                        str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
                    ):
                        hashes[str(resolved)] = digest
    return hashes, sorted(set(unresolved))


def bind_solver_settings(settings: SolverSettings) -> SolverSettings:
    """Bind the executable and environment used for all evaluations of a case."""

    if settings.name != "ngspice_cli" or settings.environment is not None:
        return settings
    executable = settings.executable or default_ngspice_executable()
    environment = dict(os.environ)
    found = shutil.which(executable)
    resolved = str(Path(found).resolve()) if found and Path(found).is_file() else found
    return replace(
        settings,
        executable=executable,
        resolved_executable=resolved,
        environment=tuple(sorted(environment.items())),
        executable_identity=_solver_binary_identity(resolved),
        startup_fingerprint=_startup_fingerprint(resolved, environment),
    )


def require_bound_solver_inputs(settings: SolverSettings | None) -> None:
    """Reject changes to the executable and startup inputs bound to a run."""
    if settings is None or settings.startup_fingerprint is None or settings.environment is None:
        return
    if settings.executable_identity is not None and settings.executable_identity != _solver_binary_identity(
        settings.resolved_executable
    ):
        raise ValueError("ngspice executable changed during execution; start a new run with the updated solver")
    try:
        current = _startup_fingerprint(settings.resolved_executable, dict(settings.environment))
    except OSError as exc:
        raise ValueError(f"could not verify bound ngspice startup inputs: {exc}") from exc
    if current != settings.startup_fingerprint:
        raise ValueError("ngspice startup inputs changed during execution; start a new run with the updated solver")


def solver_identity(settings: SolverSettings) -> dict[str, Any]:
    """Return the solver facts that affect execution and cache reuse."""

    settings = bind_solver_settings(settings)
    executable = settings.executable
    resolved = settings.resolved_executable
    if settings.name != "ngspice_cli" and executable:
        resolved = shutil.which(executable)
    return {
        "name": settings.name,
        "executable": executable,
        "resolved_executable": resolved,
        "version": _resolved_solver_version(resolved, settings.environment) if resolved else None,
        "timeout_s": settings.timeout_s,
        **(settings.executable_identity or _solver_binary_identity(resolved)),
        **({"startup": settings.startup_fingerprint} if settings.startup_fingerprint is not None else {}),
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
    return {**report, **_ngspice_diagnostic(SolverSettings(executable=executable, timeout_s=timeout_s))}


def _ngspice_diagnostic(declared: SolverSettings) -> dict[str, Any]:
    """Can ngspice actually be run here, and which binary would be used?"""

    settings = bind_solver_settings(declared)
    exe, resolved = settings.executable, settings.resolved_executable
    notes = [] if resolved else [f"executable not found on PATH: {exe}"]
    if sys.platform == "win32" and not declared.executable and exe == "ngspice":
        notes.append("ngspice_con.exe was not found on PATH; a GUI-capable ngspice.exe may open a window")
    version = _resolved_solver_version(resolved, settings.environment) if resolved else None
    if resolved:
        problem = (
            _batch_probe(settings)
            if version and version.lower().startswith("ngspice")
            else "executable did not identify itself as ngspice"
        )
        if problem:
            notes.append(problem)
    else:
        problem = "missing executable"
    return {
        "executable": exe,
        "resolved_executable": resolved,
        "version": version,
        "batch_runnable": not problem,
        "windows_prefers_console_binary": bool(sys.platform == "win32" and exe == "ngspice_con.exe"),
        "batch_command": [resolved or exe, "-b", "-o", "solver.log", "netlist.cir"],
        "notes": notes,
    }


def _run_ngspice(
    settings: SolverSettings, netlist_path: Path, log_path: Path, run_dir: Path, timeout: float
) -> subprocess.CompletedProcess[str]:
    executable = settings.resolved_executable
    if executable is None:
        raise FileNotFoundError(settings.executable)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    require_bound_solver_inputs(settings)
    try:
        return subprocess.run(
            [executable, "-b", "-o", str(log_path), str(netlist_path)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            creationflags=creationflags,
            env=dict(settings.environment) if settings.environment is not None else None,
        )
    finally:
        require_bound_solver_inputs(settings)


def _batch_probe(settings: SolverSettings) -> str | None:
    with tempfile.TemporaryDirectory(prefix="pcd-spice-") as staging:
        root = Path(staging)
        deck, log = root / "probe.cir", root / "probe.log"
        deck.write_text(
            "PCD solver probe\nV1 in 0 AC 1\nR1 in 0 50\n"
            ".control\nset numdgt=15\nac lin 1 1meg 1meg\nwrdata probe.csv v(in) i(V1)\nquit\n.endc\n.end\n",
            encoding="ascii",
        )
        try:
            completed = _run_ngspice(settings, deck, log, root, min(settings.timeout_s, 10.0))
            messages = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            match = re.search(r"(?im)^\s*((?:error|fatal error):[^\n]+)", messages + "\n" + completed.stderr)
            if match or completed.returncode != 0:
                return f"ngspice batch probe failed: {match[1] if match else f'exit {completed.returncode}'}"
            response = read_frequency_response(root / "probe.csv", ())
            row = response.iloc[0]
            if len(response) != 1 or not np.allclose(
                row[["frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"]], [1e6, 1, 0, -0.02, 0]
            ):
                return "ngspice batch probe produced an unexpected AC response"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return f"ngspice batch probe failed: {exc}"
    return None


def _as_text(value: str | bytes | None) -> str:
    """``subprocess`` types stdout/stderr as bytes on TimeoutExpired even in text mode."""

    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def ngspice_cli(request: SolverRunRequest) -> SimulationResult:
    """Execute ngspice from an already resolved, solver-facing request."""

    run_dir = request.run_dir
    netlist_path = request.netlist_path
    simulation = request.simulation
    settings = bind_solver_settings(simulation.solver)
    exe = settings.executable or "ngspice"
    timeout = settings.timeout_s
    diagnostics: dict[str, Any] = {"executable": exe, "timeout_s": timeout}

    if settings.resolved_executable is None:
        message = f"ngspice executable not found: {exe}"
        return _failed(message, diagnostics, message)

    log_path = run_dir / "solver.log"
    try:
        completed = _run_ngspice(settings, netlist_path, log_path, run_dir, timeout)
    except subprocess.TimeoutExpired as exc:
        message = f"ngspice timed out after {timeout:g}s"
        log = f"{message}\nSTDOUT:\n{_as_text(exc.stdout)}\nSTDERR:\n{_as_text(exc.stderr)}"
        return _failed(log, diagnostics, message)
    except OSError as exc:
        message = f"ngspice could not start: {exc}"
        return _failed(message, diagnostics, message)

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log += f"\nSTDOUT:\n{_as_text(completed.stdout)}\nSTDERR:\n{_as_text(completed.stderr)}"
    diagnostics["returncode"] = completed.returncode
    if completed.returncode != 0:
        return _failed(log, diagnostics, f"ngspice exited with status {completed.returncode}")

    analysis = simulation.analysis
    probes = simulation.probes
    try:
        if analysis.transient is not None:
            result = read_transient_output(
                run_dir / WAVEFORM_FILE,
                (*probes.transient_columns, LOAD_VOLTAGE_COLUMN),
                has_source_current=probes.source_current is not None,
            )
        else:
            empty = np.array([], dtype=float)
            result = SimulationResult(time_s=empty, voltage_V=empty, current_A=empty)
        if analysis.ac is not None:
            extras = [LOAD_VOLTAGE_COLUMN, "voltage_V", *probes.ac_columns]
            result.frequency_response = read_frequency_response(run_dir / AC_FILE, extras)
        result.validate_for(analysis)
    except (OSError, ValueError) as exc:
        return _failed(log, diagnostics, f"invalid ngspice output: {exc}")
    result.log = log
    result.diagnostics.update(diagnostics)
    return result
