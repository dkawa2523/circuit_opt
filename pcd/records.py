"""Readers for persisted simulation artifacts.

Simulation owns artifact creation. This module intentionally only exposes the
small, stable read boundary used by metrics and downstream analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import read_json


def read_sim_record(record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(record_or_path, dict):
        rec = dict(record_or_path)
    else:
        path = Path(record_or_path)
        if path.is_dir():
            path = path / "sim_manifest.json"
        rec = read_json(path)
        rec["manifest_path"] = str(path)
    if "run_dir" not in rec:
        manifest = rec.get("manifest_path")
        if manifest:
            rec["run_dir"] = str(Path(manifest).parent)
    return rec


def waveform_path(record: dict[str, Any]) -> Path:
    rec = read_sim_record(record)
    artifacts = rec.get("artifacts", {}) or {}
    name = artifacts.get("waveform") or rec.get("waveform_file") or "waveform.csv"
    return Path(rec["run_dir"]) / str(name)


def frequency_response_path(record: dict[str, Any] | str | Path) -> Path | None:
    rec = read_sim_record(record)
    name = (rec.get("artifacts") or {}).get("frequency_response")
    return Path(rec["run_dir"]) / str(name) if name else None


def load_waveform(record: dict[str, Any] | str | Path) -> pd.DataFrame:
    if isinstance(record, (str, Path)) and Path(record).is_file() and Path(record).name != "sim_manifest.json":
        return pd.read_csv(record)
    return pd.read_csv(waveform_path(read_sim_record(record)))
