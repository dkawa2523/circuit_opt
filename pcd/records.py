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
        path = path.resolve()
        rec = read_json(path)
        rec["manifest_path"] = str(path)
        # The manifest's directory is authoritative when reading a persisted
        # record.  This keeps a copied or renamed run self-contained even when
        # an older manifest embeds its original absolute run_dir.
        rec["run_dir"] = str(path.parent)
    if "run_dir" not in rec and (manifest := rec.get("manifest_path")):
        rec["run_dir"] = str(Path(manifest).parent)
    return rec


def artifact_path(
    record_or_path: dict[str, Any] | str | Path,
    name: str,
    legacy: str | None = None,
) -> Path | None:
    """Resolve one declared run artifact, including the v1 flat-key fallback."""

    rec = read_sim_record(record_or_path)
    declared = (rec.get("artifacts") or {}).get(name)
    if not isinstance(declared, str) or not declared.strip():
        old = rec.get(f"{name}_file")
        declared = old if isinstance(old, str) and old.strip() else legacy
    if declared is None:
        return None
    path = Path(declared)
    return path if path.is_absolute() else Path(rec["run_dir"]) / path


def waveform_path(record: dict[str, Any] | str | Path) -> Path:
    path = artifact_path(record, "waveform", "waveform.csv")
    if path is None:  # pragma: no cover - the explicit fallback above is invariant
        raise ValueError("waveform artifact is unavailable")
    return path


def frequency_response_path(record: dict[str, Any] | str | Path) -> Path | None:
    return artifact_path(record, "frequency_response")


def load_waveform(record: dict[str, Any] | str | Path) -> pd.DataFrame:
    if isinstance(record, (str, Path)) and Path(record).is_file() and Path(record).name != "sim_manifest.json":
        return pd.read_csv(record)
    return pd.read_csv(waveform_path(read_sim_record(record)))
