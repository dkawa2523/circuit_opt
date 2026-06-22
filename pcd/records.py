from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .common import Case, read_json, utc_now, write_json


def find_sim_records(run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root)
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("sim_manifest.json")):
        out.append(read_sim_record(path))
    return out


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
    return Path(rec["run_dir"]) / artifacts.get("waveform", rec.get("waveform_file", "waveform.csv"))


def load_waveform(record: dict[str, Any] | str | Path) -> pd.DataFrame:
    if isinstance(record, (str, Path)) and Path(record).is_file() and Path(record).name != "sim_manifest.json":
        return pd.read_csv(record)
    return pd.read_csv(waveform_path(read_sim_record(record)))


# Backward-friendly alias.
read_waveform = load_waveform


def read_metrics(record: dict[str, Any] | str | Path) -> dict[str, Any]:
    rec = read_sim_record(record)
    path = Path(rec["run_dir"]) / "metrics.json"
    return read_json(path) if path.exists() else {}


def save_metrics(record: dict[str, Any] | str | Path, metrics: dict[str, Any]) -> None:
    rec = read_sim_record(record)
    write_json(Path(rec["run_dir"]) / "metrics.json", metrics)


# Backward-friendly alias.
write_metrics = save_metrics


def summary_dataframe(run_root: str | Path, include_metrics: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in find_sim_records(run_root):
        row = {
            "case_id": rec.get("case_id"),
            "schema": rec.get("schema"),
            "status": rec.get("status"),
            "circuit": rec.get("circuit"),
            "load": rec.get("load"),
            "solver": rec.get("solver"),
            "created_at": rec.get("created_at"),
            "run_seconds": rec.get("run_seconds"),
            "error": rec.get("error"),
            "run_dir": rec.get("run_dir"),
        }
        for key, val in (rec.get("params", {}) or {}).items():
            row[f"param.{key}"] = val
        if include_metrics:
            metrics = read_metrics(rec)
            for key, val in metrics.items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    row[f"metric.{key}"] = val
                if key == "loss":
                    row["loss"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "loss" in df.columns:
        df = df.sort_values("loss", na_position="last").reset_index(drop=True)
    return df


def _first_existing_col(df: pd.DataFrame, name: str) -> str | None:
    for col in (name, f"metric.{name}"):
        if col in df.columns:
            return col
    return None


def _finite_series(df: pd.DataFrame, name: str) -> pd.Series:
    col = _first_existing_col(df, name)
    if col is None:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _status_failed(df: pd.DataFrame) -> pd.Series:
    failed = pd.Series(False, index=df.index)
    if "status" in df.columns:
        failed = failed | df["status"].astype(str).ne("ok")
    metric_status = _first_existing_col(df, "status")
    if metric_status and metric_status != "status":
        failed = failed | df[metric_status].astype(str).eq("failed")
    return failed


def metric_summary_dataframe(
    df: pd.DataFrame,
    loss_col: str = "loss",
    constraint_col: str = "metric.constraint_penalty",
) -> pd.DataFrame:
    """Summarize loss distributions with a generic feasible/infeasible split."""

    if df.empty:
        return pd.DataFrame([{
            "count": 0,
            "failed_count": 0,
            "loss_min": None,
            "loss_p10": None,
            "loss_p25": None,
            "loss_median": None,
            "loss_p75": None,
            "loss_p90": None,
            "loss_max": None,
            "feasible_count": 0,
            "infeasible_count": 0,
            "feasible_median": None,
            "infeasible_median": None,
        }])
    loss = _finite_series(df, loss_col)
    constraint = _finite_series(df, constraint_col).fillna(0.0)
    finite_loss = loss[np.isfinite(loss)]
    feasible_mask = constraint <= 0.0
    failed = _status_failed(df)

    def stat(series: pd.Series, op: str) -> float | None:
        vals = pd.to_numeric(series, errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            return None
        if op == "min":
            return float(vals.min())
        if op == "max":
            return float(vals.max())
        if op == "mean":
            return float(vals.mean())
        if op == "median":
            return float(vals.median())
        if op.startswith("p"):
            return float(vals.quantile(float(op[1:]) / 100.0))
        raise ValueError(op)

    row = {
        "count": int(len(df)),
        "failed_count": int(failed.sum()),
        "loss_min": stat(finite_loss, "min"),
        "loss_p10": stat(finite_loss, "p10"),
        "loss_p25": stat(finite_loss, "p25"),
        "loss_median": stat(finite_loss, "median"),
        "loss_p75": stat(finite_loss, "p75"),
        "loss_p90": stat(finite_loss, "p90"),
        "loss_max": stat(finite_loss, "max"),
        "feasible_count": int(feasible_mask.sum()),
        "infeasible_count": int((~feasible_mask).sum()),
        "feasible_best": stat(loss[feasible_mask], "min"),
        "feasible_median": stat(loss[feasible_mask], "median"),
        "feasible_mean": stat(loss[feasible_mask], "mean"),
        "infeasible_best": stat(loss[~feasible_mask], "min"),
        "infeasible_median": stat(loss[~feasible_mask], "median"),
        "infeasible_mean": stat(loss[~feasible_mask], "mean"),
    }
    return pd.DataFrame([row])


def save_summary(run_root: str | Path, out: str | Path | None = None) -> pd.DataFrame:
    df = summary_dataframe(run_root)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def import_external_waveform(
    case: Case,
    waveform_csv: str | Path,
    run_root: str | Path,
    params: dict[str, Any] | None = None,
    time_col: str | None = None,
    voltage_col: str | None = None,
    current_col: str | None = None,
) -> dict[str, Any]:
    """Create a simulation-record-compatible artifact from measured/external data."""

    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"external_{case.case_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    idx = 1
    base = run_dir
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{idx:03d}")
        idx += 1
    run_dir.mkdir(parents=True)

    src = Path(waveform_csv)
    df = pd.read_csv(src)
    tcol = time_col or ("time_s" if "time_s" in df.columns else "t")
    vcol = voltage_col or ("voltage_V" if "voltage_V" in df.columns else "v")
    if tcol not in df.columns or vcol not in df.columns:
        raise ValueError("external waveform must provide time/voltage columns; use time_col and voltage_col if needed")
    out = pd.DataFrame({"time_s": df[tcol], "voltage_V": df[vcol]})
    ccol = current_col or ("current_A" if "current_A" in df.columns else None)
    out["current_A"] = df[ccol] if ccol and ccol in df.columns else 0.0
    out.to_csv(run_dir / "waveform.csv", index=False)
    shutil.copy2(src, run_dir / "external_waveform_source.csv")
    write_json(run_dir / "params.json", params or {})
    manifest = {
        "schema": "simulation_record.v2",
        "case_id": case.case_id,
        "run_dir": str(run_dir),
        "status": "ok",
        "created_at": utc_now(),
        "params": params or {},
        "circuit": (case.data.get("circuit", {}) or {}).get("builder", "external"),
        "load": (case.data.get("load", {}) or {}).get("name", "external"),
        "solver": "external",
        "measurement": case.data.get("measurement", {}) or {},
        "artifacts": {"waveform": "waveform.csv", "solver_log": "solver.log"},
        "waveform_file": "waveform.csv",
        "solver_log_file": "solver.log",
        "warnings": ["external waveform imported; no netlist was generated"],
    }
    (run_dir / "solver.log").write_text("external waveform import\n", encoding="utf-8")
    write_json(run_dir / "sim_manifest.json", manifest)
    return read_sim_record(run_dir)
