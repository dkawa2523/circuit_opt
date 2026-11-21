"""Readers for persisted simulation artifacts.

Simulation owns artifact creation. This module intentionally only exposes the
small, stable read boundary used by metrics and downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import resolve_artifact_path
from .simulation import CURRENT_COLUMN, TIME_COLUMN, VOLTAGE_COLUMN, SimulationResult


def read_sim_record(record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(record_or_path, dict):
        rec = dict(record_or_path)
        if rec.get("schema") == "simulation_summary.v1":
            run_dir = Path(str(rec.get("run_dir", ".")))
            rec = _summary_manifest(rec, run_dir)
    else:
        path = Path(record_or_path)
        if path.is_dir():
            path = path / "summary.json"
        rec = _read_record_file(path.resolve())
    if rec.get("schema") != "simulation_record.v2":
        raise ValueError("simulation record must use simulation_record.v2")
    return rec


def _read_record_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    payload = _read_record_object(path, data)
    if payload.get("schema") == "simulation_summary.v1":
        return _summary_manifest(payload, path.parent)
    run_dir = path.parent.parent
    summary = run_dir / "summary.json"
    _verify_manifest(_read_record_object(summary), path, data)
    return _located_manifest(payload, path, run_dir)


def _summary_manifest(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    declared = _artifact_links(summary).get("debug_manifest")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("simulation summary must declare a valid debug manifest path")
    manifest = resolve_artifact_path(run_dir, declared).resolve()
    data = manifest.read_bytes()
    payload = _read_record_object(manifest, data)
    _verify_manifest(summary, manifest, data)
    return _located_manifest(payload, manifest, run_dir.resolve())


def _verify_manifest(summary: dict[str, Any], manifest: Path, data: bytes) -> None:
    if summary.get("schema") != "simulation_summary.v1":
        raise ValueError("simulation summary must use simulation_summary.v1")
    if hashlib.sha256(data).hexdigest() != summary.get("debug_manifest_sha256"):
        raise ValueError(f"saved simulation manifest differs from its public summary: {manifest}")


def _read_record_object(path: Path, data: bytes | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_bytes() if data is None else data)
    if not isinstance(payload, dict):
        raise ValueError(f"simulation record must be a JSON object: {path}")
    return payload


def _located_manifest(rec: dict[str, Any], path: Path, run_dir: Path) -> dict[str, Any]:
    rec["manifest_path"] = str(path)
    # Use the record location when a run was copied or moved.
    rec["run_dir"] = str(run_dir)
    return rec


def _artifact_links(record: dict[str, Any]) -> dict[str, Any]:
    links = record.get("artifacts")
    if links is None:
        return {}
    if not isinstance(links, dict):
        raise ValueError("simulation record artifacts must be a JSON object")
    return links


def artifact_path(
    record_or_path: dict[str, Any] | str | Path,
    name: str,
) -> Path | None:
    """Resolve one artifact declared by a simulation-record manifest."""

    rec = read_sim_record(record_or_path)
    declared = _artifact_links(rec).get(name)
    if not isinstance(declared, str) or not declared.strip():
        return None
    return resolve_artifact_path(Path(rec["run_dir"]), declared)


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
    return pd.read_csv(waveform_path(record))


def load_frequency_response(
    record: dict[str, Any] | str | Path,
    extra_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the canonical frequency response declared by a run manifest."""

    path = frequency_response_path(record)
    if path is None:
        raise ValueError("simulation manifest does not declare a frequency-response artifact")
    return _canonical_frequency_response(path, extra_columns)


def _canonical_frequency_response(path: Path, extra_columns: Sequence[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"]
    required.extend(f"{name}_{part}" for name in (extra_columns or ()) for part in ("re", "im"))
    if missing := [name for name in required if name not in frame]:
        raise ValueError(f"canonical AC output is missing columns {missing}: {path}")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return frame


def load_simulation_result(
    record_or_path: dict[str, Any] | str | Path, *, ac_columns: Sequence[str] | None = None
) -> SimulationResult:
    """Restore the canonical in-memory solver response from saved artifacts."""

    record = read_sim_record(record_or_path)
    numeric = _numeric_waveform(load_waveform(record))
    ac_path = frequency_response_path(record)
    return SimulationResult(
        time_s=numeric[TIME_COLUMN],
        voltage_V=numeric[VOLTAGE_COLUMN],
        current_A=_available_current(numeric.get(CURRENT_COLUMN)),
        status=str(record.get("status", "failed")),
        log=_solver_log(record),
        diagnostics=dict(record.get("diagnostics") or {}),
        frequency_response=_canonical_frequency_response(ac_path, ac_columns) if ac_path is not None else None,
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
