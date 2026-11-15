"""Readers for persisted simulation artifacts.

Simulation owns artifact creation. This module intentionally only exposes the
small, stable read boundary used by metrics and downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import evidence_digest, resolve_artifact_path
from .ngspice_io import read_frequency_response
from .simulation import CURRENT_COLUMN, LOAD_CURRENT_COLUMN, TIME_COLUMN, VOLTAGE_COLUMN, SimulationResult

_DEBUG_MANIFEST = Path("debug/manifest.json")


def read_sim_record(record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(record_or_path, dict):
        rec = dict(record_or_path)
        if rec.get("schema") == "simulation_summary.v1":
            run_dir = Path(str(rec.get("run_dir", ".")))
            rec = _summary_manifest(rec, run_dir)
    else:
        path = Path(record_or_path)
        if path.is_dir():
            manifest, summary = path / _DEBUG_MANIFEST, path / "summary.json"
            path = summary if not manifest.is_file() and summary.is_file() else manifest
        rec = _read_record_file(path.resolve())
    if "run_dir" not in rec and (manifest := rec.get("manifest_path")):
        rec["run_dir"] = str(Path(manifest).parent.parent)
    return rec


def _read_record_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    payload = _read_record_object(path, data)
    if payload.get("schema") == "simulation_summary.v1":
        return _summary_manifest(payload, path.parent)
    run_dir = path.parent.parent
    summary = run_dir / "summary.json"
    verified = summary.is_file() and _verified_manifest(_read_record_object(summary), path, data)
    return _located_manifest(payload, path, run_dir, verified)


def _summary_manifest(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    declared = _artifact_links(summary).get("debug_manifest", _DEBUG_MANIFEST.as_posix())
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("simulation summary must declare a valid debug manifest path")
    manifest = resolve_artifact_path(run_dir, declared).resolve()
    data = manifest.read_bytes()
    payload = _read_record_object(manifest, data)
    return _located_manifest(payload, manifest, run_dir.resolve(), _verified_manifest(summary, manifest, data))


def _verified_manifest(summary: dict[str, Any], manifest: Path, data: bytes) -> bool:
    if "debug_manifest_sha256" not in summary:
        return False
    if hashlib.sha256(data).hexdigest() != summary["debug_manifest_sha256"]:
        raise ValueError(f"saved simulation manifest differs from its public summary: {manifest}")
    return True


def _read_record_object(path: Path, data: bytes | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_bytes() if data is None else data)
    if not isinstance(payload, dict):
        raise ValueError(f"simulation record must be a JSON object: {path}")
    return payload


def _located_manifest(rec: dict[str, Any], path: Path, run_dir: Path, verified: bool) -> dict[str, Any]:
    rec["manifest_path"] = str(path)
    rec.pop("manifest_sha256_verified", None)
    if verified:
        rec["manifest_sha256_verified"] = True
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
    resolved = read_sim_record(record)
    frame = pd.read_csv(waveform_path(resolved))
    measurement = resolved.get("measurement") or {}
    ports = measurement.get("load_ports") or {}
    if (
        not frame.empty
        and measurement.get("load_current")
        and "load_voltage_V" not in frame
        and (measurement.get("voltage_node") != ports.get("p") or str(ports.get("n", "0")) != "0")
    ):
        raise ValueError("saved waveform has no load-port voltage; rerun this case to measure load power")
    return frame


def load_frequency_response(
    record: dict[str, Any] | str | Path,
    extra_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the canonical frequency response declared by a run manifest."""

    path = frequency_response_path(record)
    if path is None:
        raise ValueError("simulation manifest does not declare a frequency-response artifact")
    return read_frequency_response(path, extra_columns)


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
        diagnostics=_observation_diagnostics(record),
        frequency_response=read_frequency_response(ac_path, ac_columns) if ac_path is not None else None,
        probes={
            name: values
            for name, values in numeric.items()
            if name not in {TIME_COLUMN, VOLTAGE_COLUMN, CURRENT_COLUMN}
        },
    )


def _observation_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(record.get("diagnostics") or {})
    if "load_current_column" in diagnostics:
        return diagnostics
    column = (record.get("measurement") or {}).get("load_current")
    if not column:
        return diagnostics
    if column != LOAD_CURRENT_COLUMN and record.get("load") not in {None, "none"}:
        path = artifact_path(record, "netlist")
        if path is not None and path.is_file():
            evidence_digest(record, "netlist", path)
            # The compiler emits this top-level instance only for an attached load.
            attached = (
                re.search(r"(?im)^Xload[ \t]+\S+[ \t]+\S+[ \t]+load_model[ \t]*$", path.read_text(encoding="utf-8"))
                is not None
            )
        else:
            attached = True
        if attached:
            diagnostics["load_power_unavailable_reason"] = (
                "the saved run did not observe the attached load's own current; rerun to measure it"
            )
            column = None
    diagnostics["load_current_column"] = column
    return diagnostics


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
