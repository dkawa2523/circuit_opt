"""Readers for persisted simulation artifacts.

Simulation owns artifact creation. This module intentionally only exposes the
small, stable read boundary used by metrics and downstream analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import read_json
from .ngspice_io import read_frequency_response
from .simulation import CURRENT_COLUMN, TIME_COLUMN, VOLTAGE_COLUMN, SimulationResult

_DEBUG_MANIFEST = Path("debug/manifest.json")


def read_sim_record(record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(record_or_path, dict):
        rec = dict(record_or_path)
        if rec.get("schema") == "simulation_summary.v1":
            run_dir = Path(str(rec.get("run_dir", ".")))
            declared = (rec.get("artifacts") or {}).get("debug_manifest", _DEBUG_MANIFEST.as_posix())
            rec = _read_manifest(run_dir / str(declared))
    else:
        path = Path(record_or_path)
        if path.is_dir():
            path = path / _DEBUG_MANIFEST
        path = path.resolve()
        rec = _read_record_file(path)
    if "run_dir" not in rec and (manifest := rec.get("manifest_path")):
        rec["run_dir"] = str(Path(manifest).parent.parent)
    return rec


def _read_record_file(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") == "simulation_summary.v1":
        declared = (payload.get("artifacts") or {}).get("debug_manifest", _DEBUG_MANIFEST.as_posix())
        manifest = Path(str(declared))
        path = manifest if manifest.is_absolute() else path.parent / manifest
    return _read_manifest(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    rec = read_json(path)
    rec["manifest_path"] = str(path)
    # The manifest location is authoritative after a run is copied or moved;
    # an embedded original absolute run_dir is only provenance.
    rec["run_dir"] = str(path.parent.parent)
    return rec


def artifact_path(
    record_or_path: dict[str, Any] | str | Path,
    name: str,
) -> Path | None:
    """Resolve one artifact declared by a simulation-record manifest."""

    rec = read_sim_record(record_or_path)
    declared = (rec.get("artifacts") or {}).get(name)
    if not isinstance(declared, str) or not declared.strip():
        return None
    path = Path(declared)
    return path if path.is_absolute() else Path(rec["run_dir"]) / path


def waveform_path(record: dict[str, Any] | str | Path) -> Path:
    path = artifact_path(record, "waveform")
    if path is None:
        raise ValueError("simulation manifest does not declare a waveform artifact")
    return path


def frequency_response_path(record: dict[str, Any] | str | Path) -> Path | None:
    return artifact_path(record, "frequency_response")


def load_waveform(record: dict[str, Any] | str | Path) -> pd.DataFrame:
    if isinstance(record, (str, Path)) and Path(record).is_file() and Path(record).suffix.lower() == ".csv":
        return pd.read_csv(record)
    return pd.read_csv(waveform_path(read_sim_record(record)))


def load_frequency_response(
    record: dict[str, Any] | str | Path,
    extra_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the canonical frequency response declared by a run manifest."""

    path = frequency_response_path(record)
    if path is None:
        raise ValueError("simulation manifest does not declare a frequency-response artifact")
    return read_frequency_response(path, extra_columns)


def load_simulation_result(record_or_path: dict[str, Any] | str | Path) -> SimulationResult:
    """Restore the canonical in-memory solver response from saved artifacts."""

    record = read_sim_record(record_or_path)
    numeric = _numeric_waveform(load_waveform(record))
    frequency = load_frequency_response(record) if frequency_response_path(record) is not None else None
    return SimulationResult(
        time_s=numeric[TIME_COLUMN],
        voltage_V=numeric[VOLTAGE_COLUMN],
        current_A=_available_current(numeric.get(CURRENT_COLUMN)),
        status=str(record.get("status", "failed")),
        log=_solver_log(record),
        diagnostics=dict(record.get("diagnostics") or {}),
        frequency_response=frequency,
        probes={
            name: values
            for name, values in numeric.items()
            if name not in {TIME_COLUMN, VOLTAGE_COLUMN, CURRENT_COLUMN}
        },
    )


def _numeric_waveform(waveform: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {TIME_COLUMN, VOLTAGE_COLUMN}
    if missing := sorted(required - set(waveform)):
        raise ValueError(f"canonical waveform is missing columns {missing}")
    return {name: pd.to_numeric(waveform[name], errors="raise").to_numpy(float) for name in waveform.columns}


def _available_current(current: np.ndarray | None) -> np.ndarray | None:
    return None if current is None or np.isnan(current).all() else current


def _solver_log(record: dict[str, Any]) -> str:
    path = artifact_path(record, "solver_log")
    return path.read_text(encoding="utf-8", errors="replace") if path and path.is_file() else ""
