from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import Case, default_params, resolve_path, sample_param, variable_specs, write_json
from .ml_registry import get as get_ml_method, load_plugins
from .records import find_sim_records, load_waveform, read_metrics, read_sim_record, save_metrics


@dataclass
class BaseOptimizer:
    case: Case
    seed: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def ask(self) -> dict[str, Any]:
        raise NotImplementedError

    def tell(self, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.history.append({"params": dict(params), "metrics": dict(metrics)})

    def state(self) -> dict[str, Any]:
        best = None
        for item in self.history:
            loss = item.get("metrics", {}).get("loss")
            if loss is None:
                continue
            if best is None or float(loss) < float(best.get("metrics", {}).get("loss", float("inf"))):
                best = item
        return {"type": type(self).__name__, "n_observations": len(self.history), "best": best}


def create_optimizer(case: Case, optimizer_name: str | None = None, seed: int | None = None) -> BaseOptimizer:
    load_plugins(case.data.get("plugins"), case.base_dir)
    cfg = case.data.get("optimizer", {}) or {}
    name = optimizer_name or str(cfg.get("name", "random"))
    factory = get_ml_method("optimizer", name)
    return factory(case, seed=seed)


def propose_candidates(case: Case, n: int, optimizer_name: str | None = None, seed: int | None = None) -> pd.DataFrame:
    opt = create_optimizer(case, optimizer_name=optimizer_name, seed=seed)
    return pd.DataFrame([opt.ask() for _ in range(int(n))])


def save_candidates(df: pd.DataFrame, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        df.to_json(out, orient="records", indent=2, force_ascii=False)
    else:
        df.to_csv(out, index=False)


def score_record(
    case: Case,
    record_or_path: dict[str, Any] | str | Path,
    allow_failed_penalty: bool = True,
    failed_loss: float = 1e30,
) -> dict[str, Any]:
    load_plugins(case.data.get("plugins"), case.base_dir)
    record = read_sim_record(record_or_path)
    obj_name = str((case.data.get("target", {}) or {}).get("objective", "waveform_l2"))
    if record.get("status") != "ok":
        if not allow_failed_penalty:
            raise ValueError(f"cannot score failed simulation record: {record.get('run_dir')}")
        metrics = {
            "loss": float(failed_loss),
            "status": "failed",
            "reason": "simulation_failed",
            "record_status": record.get("status"),
            "objective": obj_name,
        }
        save_metrics(record, metrics)
        return metrics
    objective = get_ml_method("objective", obj_name)
    waveform = load_waveform(record)
    metrics = objective(case, record, waveform)
    if "loss" not in metrics:
        raise ValueError(f"objective '{obj_name}' must return a metrics dict containing loss")
    save_metrics(record, metrics)
    return metrics


def score_run_root(
    case: Case,
    run_root: str | Path,
    allow_failed_penalty: bool = True,
    failed_loss: float = 1e30,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in find_sim_records(run_root):
        metrics = score_record(case, rec, allow_failed_penalty=allow_failed_penalty, failed_loss=failed_loss)
        rows.append({"run_dir": rec.get("run_dir"), "status": rec.get("status"), **metrics})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(Path(run_root) / "scores.csv", index=False)
    return df


def build_learning_table(run_root: str | Path, include_failed: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in find_sim_records(run_root):
        metrics = read_metrics(rec)
        if not metrics:
            continue
        if not include_failed and (rec.get("status") != "ok" or metrics.get("status") == "failed"):
            continue
        row: dict[str, Any] = {"run_dir": rec.get("run_dir"), "status": rec.get("status")}
        row.update({f"param.{k}": v for k, v in (rec.get("params") or {}).items()})
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_ridge_surrogate(
    table: pd.DataFrame,
    target_col: str = "loss",
    alpha: float = 1e-6,
    exclude_infeasible: bool = False,
    constraint_col: str = "constraint_penalty",
    target_transform: str = "none",
    clip_target_quantile: float | None = None,
) -> dict[str, Any]:
    if table.empty:
        raise ValueError("learning table is empty")
    target_col = _resolve_table_col(table, target_col)
    if target_col is None:
        raise ValueError(f"target column not found: {target_col}")
    target_transform = str(target_transform)
    if target_transform not in {"none", "log1p"}:
        raise ValueError("target_transform must be 'none' or 'log1p'")
    feature_cols = [c for c in table.columns if c.startswith("param.")]
    if not feature_cols:
        raise ValueError("no param.* columns found")
    feature_source = table[feature_cols]
    Xdf = _candidate_feature_frame(feature_source)
    feature_schema = _feature_schema(feature_source, Xdf)
    y = pd.to_numeric(table[target_col], errors="coerce")
    mask = y.notna() & np.isfinite(y.to_numpy(float))
    constraint_resolved_col = _resolve_table_col(table, constraint_col)
    if exclude_infeasible:
        if constraint_resolved_col is None:
            raise ValueError(f"constraint column not found: {constraint_col}")
        constraint = pd.to_numeric(table[constraint_resolved_col], errors="coerce").fillna(0.0)
        mask = mask & (constraint <= 0.0)
    if target_transform == "log1p":
        mask = mask & (y >= -1.0)
    X = Xdf.loc[mask].to_numpy(float)
    yv_original = y.loc[mask].to_numpy(float)
    if len(yv_original) == 0:
        raise ValueError("no finite target values")
    yv_fit = yv_original.copy()
    clipped_rows = 0
    clip_value = None
    if clip_target_quantile is not None:
        q = float(clip_target_quantile)
        if not 0.0 < q <= 1.0:
            raise ValueError("clip_target_quantile must be in (0, 1]")
        clip_value = float(np.quantile(yv_fit, q))
        clipped_rows = int((yv_fit > clip_value).sum())
        yv_fit = np.minimum(yv_fit, clip_value)
    y_model = _transform_target(yv_fit, target_transform)
    x_mean, x_std, weights = _fit_ridge_arrays(X, y_model, alpha=float(alpha))
    Xn = (X - x_mean) / x_std
    X1 = np.column_stack([np.ones(len(Xn)), Xn])
    pred_model = X1 @ weights
    pred = _inverse_transform_target(pred_model, target_transform)
    rmse = float(np.sqrt(np.mean((pred - yv_fit) ** 2)))
    mae = float(np.mean(np.abs(pred - yv_fit)))
    denom = float(np.sum((yv_fit - yv_fit.mean()) ** 2))
    r2 = float(1.0 - np.sum((pred - yv_fit) ** 2) / denom) if denom > 0 else 1.0
    cv_rmse = _cross_validated_rmse(X, yv_fit, alpha=float(alpha), target_transform=target_transform)
    return {
        "schema": "ridge_surrogate.v2",
        "target_col": target_col,
        "constraint_col": constraint_resolved_col,
        "exclude_infeasible": bool(exclude_infeasible),
        "target_transform": target_transform,
        "clip_target_quantile": clip_target_quantile,
        "clip_target_value": clip_value,
        "feature_columns": list(Xdf.columns),
        "feature_schema": feature_schema,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "weights": weights.tolist(),
        "training_rmse": rmse,
        "training_mae": mae,
        "training_r2": r2,
        "cv_rmse": cv_rmse,
        "n_train": int(len(yv_fit)),
        "n_samples": int(len(yv_fit)),
        "n_features": int(X.shape[1]),
        "dropped_rows": int(len(table) - len(yv_fit)),
        "clipped_rows": clipped_rows,
        "target_min": float(np.min(yv_fit)),
        "target_median": float(np.median(yv_fit)),
        "target_max": float(np.max(yv_fit)),
    }


def predict_candidates_with_surrogate(model: dict[str, Any], candidates: pd.DataFrame, strict_schema: bool = False) -> pd.DataFrame:
    df = candidates.copy()
    feature_source = df.copy()
    if not any(c.startswith("param.") for c in feature_source.columns):
        feature_source = feature_source.rename(columns={c: f"param.{c}" for c in feature_source.columns})
    feature_source = feature_source[[c for c in feature_source.columns if c.startswith("param.")]]
    warnings = _schema_warnings(model, feature_source)
    if strict_schema and warnings:
        raise ValueError("; ".join(warnings))
    Xdf = _candidate_feature_frame(feature_source)
    for col in model["feature_columns"]:
        if col not in Xdf.columns:
            Xdf[col] = 0.0
    Xdf = Xdf[model["feature_columns"]]
    X = Xdf.to_numpy(float)
    x_mean = np.asarray(model["x_mean"], dtype=float)
    x_std = np.asarray(model["x_std"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    Xn = (X - x_mean) / x_std
    pred_model = np.column_stack([np.ones(len(Xn)), Xn]) @ weights
    df["predicted_loss_model_space"] = pred_model
    df["predicted_loss"] = _inverse_transform_target(pred_model, str(model.get("target_transform", "none")))
    if warnings:
        df["prediction_warning"] = "; ".join(warnings)
    return df.sort_values("predicted_loss").reset_index(drop=True)


def _resolve_table_col(table: pd.DataFrame, col: str) -> str | None:
    if col in table.columns:
        return col
    alt = "metric." + col
    if alt in table.columns:
        return alt
    if col.startswith("metric."):
        raw = col[len("metric."):]
        if raw in table.columns:
            return raw
    return None


def _transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log1p":
        return np.log1p(np.maximum(y, -1.0))
    raise ValueError(transform)


def _inverse_transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(transform)


def _candidate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric columns and one-hot encode categorical design choices.
    out = pd.get_dummies(df, dummy_na=False)
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _feature_schema(source: pd.DataFrame, encoded: pd.DataFrame) -> dict[str, Any]:
    categorical: dict[str, list[str]] = {}
    numeric: list[str] = []
    for col in source.columns:
        numeric_values = pd.to_numeric(source[col], errors="coerce")
        if numeric_values.notna().all():
            numeric.append(col)
            continue
        vals = sorted(str(v) for v in source[col].dropna().unique())
        categorical[col] = vals
    return {
        "raw_columns": list(source.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "encoded_columns": list(encoded.columns),
    }


def _schema_warnings(model: dict[str, Any], feature_source: pd.DataFrame) -> list[str]:
    schema = model.get("feature_schema") or {}
    raw_columns = set(schema.get("raw_columns") or [])
    warnings: list[str] = []
    if raw_columns:
        extra = sorted(set(feature_source.columns) - raw_columns)
        missing = sorted(raw_columns - set(feature_source.columns))
        if extra:
            warnings.append(f"extra candidate param columns ignored: {extra}")
        if missing:
            warnings.append(f"candidate param columns missing and filled with zero/default encoding: {missing}")
    for col, known in (schema.get("categorical_columns") or {}).items():
        if col not in feature_source.columns:
            continue
        observed = set(str(v) for v in feature_source[col].dropna().unique())
        unknown = sorted(observed - set(str(v) for v in known))
        if unknown:
            warnings.append(f"unknown categories for {col}: {unknown}")
    return warnings


def _fit_ridge_arrays(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    Xn = (X - x_mean) / x_std
    X1 = np.column_stack([np.ones(len(Xn)), Xn])
    reg = float(alpha) * np.eye(X1.shape[1])
    reg[0, 0] = 0.0
    weights = np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)
    return x_mean, x_std, weights


def _cross_validated_rmse(X: np.ndarray, y: np.ndarray, alpha: float, target_transform: str = "none") -> float | None:
    if len(y) < 3:
        return None
    k = min(5, len(y))
    preds = np.full(len(y), np.nan, dtype=float)
    for fold in np.array_split(np.arange(len(y)), k):
        train = np.setdiff1d(np.arange(len(y)), fold)
        if len(train) == 0 or len(fold) == 0:
            continue
        x_mean, x_std, weights = _fit_ridge_arrays(X[train], _transform_target(y[train], target_transform), alpha=alpha)
        Xn = (X[fold] - x_mean) / x_std
        preds[fold] = _inverse_transform_target(np.column_stack([np.ones(len(Xn)), Xn]) @ weights, target_transform)
    mask = np.isfinite(preds)
    if not mask.any():
        return None
    return float(np.sqrt(np.mean((preds[mask] - y[mask]) ** 2)))


# Objective utilities ----------------------------------------------------------


def load_target_waveform(case: Case) -> pd.DataFrame:
    cfg = case.data.get("target", {}) or {}
    path = cfg.get("waveform_file")
    if path is None:
        raise ValueError("target.waveform_file is required for waveform objectives")
    df = pd.read_csv(resolve_path(case, path))
    if not {"time_s", "voltage_V"}.issubset(df.columns):
        raise ValueError("target waveform must contain time_s and voltage_V")
    return df


def interpolate_to_target(target: pd.DataFrame, waveform: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tt, vt = _clean_time_value(target, "time_s", "voltage_V")
    if len(tt) == 0:
        return tt, vt, np.asarray([], dtype=float)
    if waveform.empty or "time_s" not in waveform or "voltage_V" not in waveform:
        return tt, vt, np.full_like(tt, np.nan, dtype=float)
    time, value = _clean_time_value(waveform, "time_s", "voltage_V")
    if len(time) == 0:
        return tt, vt, np.full_like(tt, np.nan, dtype=float)
    return tt, vt, np.interp(tt, time, value, left=value[0], right=value[-1])


def _clean_time_value(df: pd.DataFrame, time_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    if time_col not in df.columns or value_col not in df.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    tmp = pd.DataFrame({
        "time": pd.to_numeric(df[time_col], errors="coerce"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    })
    mask = np.isfinite(tmp["time"].to_numpy(float)) & np.isfinite(tmp["value"].to_numpy(float))
    tmp = tmp.loc[mask]
    if tmp.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    tmp = tmp.groupby("time", as_index=False, sort=True)["value"].mean()
    return tmp["time"].to_numpy(float), tmp["value"].to_numpy(float)


def constraint_penalty(case: Case, voltage: np.ndarray) -> tuple[float, dict[str, Any]]:
    cfg = (case.data.get("target", {}) or {}).get("constraints", {}) or {}
    penalty = 0.0
    info: dict[str, Any] = {}
    vmax = cfg.get("max_abs_voltage_V")
    if vmax is not None and np.isfinite(voltage).any():
        peak = float(np.nanmax(np.abs(voltage)))
        info["peak_abs_voltage_V"] = peak
        if peak > float(vmax):
            penalty += ((peak - float(vmax)) / max(float(vmax), 1e-12)) ** 2
    return penalty, info


def random_params(case: Case, seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out = default_params(case)
    for name, spec in variable_specs(case).items():
        out[name] = sample_param(rng, spec)
    return out
