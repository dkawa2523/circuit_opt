from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcd.ml_core import fit_ridge_surrogate
from pcd.records import metric_summary_dataframe


DEFAULT_PROFILE_FILE = PACK_DIR / "ngspice_benchmark_profiles.json"

CASE_META = {
    "level2_timevarying_plasma": {
        "title": "Level 2 time-varying plasma",
        "target_file": "target_gec_tailored.csv",
    },
    "level3_topology_load_choice": {
        "title": "Level 3 topology/load choice",
        "target_file": "target_gec_tailored.csv",
    },
}


def _load_profiles(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _metric_col(df: pd.DataFrame, name: str) -> str | None:
    for col in (name, f"metric.{name}"):
        if col in df.columns:
            return col
    return None


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    col = _metric_col(df, name)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _load_summary(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")
    return pd.read_csv(path)


def _finite_values(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals[np.isfinite(vals)]


def _stat(series: pd.Series, op: str) -> float | None:
    vals = _finite_values(series)
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


def _case_stats(df: pd.DataFrame) -> dict[str, Any]:
    loss = _numeric(df, "loss")
    penalty = _numeric(df, "constraint_penalty").fillna(0.0)
    v_peak = _numeric(df, "v_peak_abs_V")
    i_rms = _numeric(df, "i_rms_A")
    summary = metric_summary_dataframe(df).iloc[0].to_dict()
    count = int(summary["count"] or 0)
    penalty_count = int((penalty > 0.0).sum())
    return {
        "n": count,
        "failed_trials": int(summary["failed_count"] or 0),
        "best_loss": summary.get("loss_min"),
        "p10_loss": summary.get("loss_p10"),
        "p25_loss": summary.get("loss_p25"),
        "median_loss": summary.get("loss_median"),
        "p75_loss": summary.get("loss_p75"),
        "p90_loss": summary.get("loss_p90"),
        "mean_loss": _stat(loss, "mean"),
        "max_loss": summary.get("loss_max"),
        "penalty_count": penalty_count,
        "penalty_rate": float(penalty_count / count) if count else None,
        "feasible_count": int(summary["feasible_count"] or 0),
        "feasible_median": summary.get("feasible_median"),
        "infeasible_count": int(summary["infeasible_count"] or 0),
        "infeasible_median": summary.get("infeasible_median"),
        "v_peak_gt_1000_count": int((v_peak > 1000.0).sum()),
        "i_rms_gt_20_count": int((i_rms > 20.0).sum()),
        "i_rms_gt_25_count": int((i_rms > 25.0).sum()),
        "loss_lt_1_count": int((loss < 1.0).sum()),
        "loss_lt_2_count": int((loss < 2.0).sum()),
    }


def _feasibility_rows(case_key: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    loss = _numeric(df, "loss")
    penalty = _numeric(df, "constraint_penalty").fillna(0.0)
    v_peak = _numeric(df, "v_peak_abs_V")
    i_rms = _numeric(df, "i_rms_A")
    rows: list[dict[str, Any]] = []
    for bucket, mask in [
        ("all", pd.Series(True, index=df.index)),
        ("feasible", penalty <= 0.0),
        ("infeasible", penalty > 0.0),
    ]:
        sub_loss = loss[mask]
        count = int(mask.sum())
        rows.append({
            "case": case_key,
            "bucket": bucket,
            "count": count,
            "best_loss": _stat(sub_loss, "min"),
            "median_loss": _stat(sub_loss, "median"),
            "mean_loss": _stat(sub_loss, "mean"),
            "p90_loss": _stat(sub_loss, "p90"),
            "max_loss": _stat(sub_loss, "max"),
            "v_peak_max": _stat(v_peak[mask], "max"),
            "i_rms_max": _stat(i_rms[mask], "max"),
            "penalty_rate": float((penalty[mask] > 0.0).sum() / count) if count else None,
        })
    return rows


def _category_stats(df: pd.DataFrame, case_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("topology", ["param.topology_choice"]),
        ("load", ["param.load_model"]),
        ("topology_load", ["param.topology_choice", "param.load_model"]),
    ]
    loss = _numeric(df, "loss")
    penalty = _numeric(df, "constraint_penalty").fillna(0.0)
    if loss.isna().all():
        return pd.DataFrame()
    for group_type, cols in groups:
        if not all(c in df.columns for c in cols):
            continue
        for keys, sub in df.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = sub.index
            sub_loss = loss.loc[idx]
            count = int(len(sub))
            rows.append({
                "case": case_key,
                "group_type": group_type,
                "group": " + ".join(str(k) for k in keys),
                "count": count,
                "min_loss": _stat(sub_loss, "min"),
                "median_loss": _stat(sub_loss, "median"),
                "mean_loss": _stat(sub_loss, "mean"),
                "p75_loss": _stat(sub_loss, "p75"),
                "p90_loss": _stat(sub_loss, "p90"),
                "max_loss": _stat(sub_loss, "max"),
                "penalty_rate": float((penalty.loc[idx] > 0.0).sum() / count) if count else None,
            })
    return pd.DataFrame(rows)


def _correlations(df: pd.DataFrame, case_key: str) -> pd.DataFrame:
    loss_col = _metric_col(df, "loss")
    if loss_col is None:
        return pd.DataFrame()
    loss = pd.to_numeric(df[loss_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        if col in {loss_col, "loss", "metric.loss"} or not (col.startswith("metric.") or col.startswith("param.")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = np.isfinite(values) & np.isfinite(loss)
        if valid.sum() < 3:
            continue
        corr = values[valid].corr(loss[valid], method="spearman")
        if corr is None or not math.isfinite(float(corr)):
            continue
        rows.append({"case": case_key, "feature": col, "spearman_loss": float(corr)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.assign(abs_corr=out["spearman_loss"].abs()).sort_values(["case", "abs_corr"], ascending=[True, False])
    return out


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    loss = _numeric(df, "loss")
    finite = loss[np.isfinite(loss)]
    if finite.empty:
        return None
    return df.loc[finite.idxmin()]


def _wrap_phase_delta_deg(delta: float) -> float:
    return float((delta + 180.0) % 360.0 - 180.0)


def _waveform_stats(path: Path, harmonics: list[int], fundamental_hz: float = 13.56e6) -> dict[str, Any]:
    df = pd.read_csv(path)
    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df["voltage_V"], errors="coerce").to_numpy(float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    order = np.argsort(t)
    t, v = t[order], v[order]
    if len(t) < 3:
        return {}
    dt = float(np.median(np.diff(t)))
    freq = np.fft.rfftfreq(len(t), d=dt)
    spectrum = np.fft.rfft(v)
    out: dict[str, Any] = {
        "mean_V": float(np.mean(v)),
        "rms_V": float(np.sqrt(np.mean(v ** 2))),
        "v_peak_abs_V": float(np.max(np.abs(v))),
    }
    for h in harmonics:
        idx = int(np.argmin(np.abs(freq - h * fundamental_hz)))
        coeff = spectrum[idx]
        out[f"A{h}_V"] = float(2.0 * abs(coeff) / len(v))
        out[f"P{h}_deg"] = float(np.degrees(np.angle(coeff)))
    return out


def _harmonic_rows(case_key: str, df: pd.DataFrame, harmonics: list[int]) -> list[dict[str, Any]]:
    meta = CASE_META[case_key]
    target_stats = _waveform_stats(PACK_DIR / meta["target_file"], harmonics)
    best = _best_row(df)
    if best is None:
        return []
    best_stats = _waveform_stats(Path(str(best["run_dir"])) / "waveform.csv", harmonics)
    rows = []
    for source, stats in [("target", target_stats), ("ngspice_best", best_stats)]:
        row = {"case": case_key, "source": source, **stats}
        if source == "ngspice_best":
            for h in harmonics:
                target_amp = target_stats.get(f"A{h}_V")
                amp = stats.get(f"A{h}_V")
                row[f"A{h}_target_ratio"] = float(amp / target_amp) if target_amp else None
                target_phase = target_stats.get(f"P{h}_deg")
                phase = stats.get(f"P{h}_deg")
                row[f"P{h}_error_deg"] = _wrap_phase_delta_deg(float(phase) - float(target_phase)) if phase is not None and target_phase is not None else None
        rows.append(row)
    return rows


def _top_candidates(case_key: str, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    loss = _numeric(df, "loss")
    penalty = _numeric(df, "constraint_penalty").fillna(0.0)
    work = df.copy()
    work["_loss"] = loss
    work["_penalty"] = penalty
    frames = [
        ("feasible", work[work["_penalty"] <= 0.0].sort_values("_loss").head(n)),
        ("risky_low_loss", work[work["_penalty"] > 0.0].sort_values("_loss").head(n)),
    ]
    rows: list[dict[str, Any]] = []
    for bucket, sub in frames:
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            run_dir = str(row.get("run_dir", ""))
            rows.append({
                "case": case_key,
                "bucket": bucket,
                "rank": rank,
                "trial": Path(run_dir).name,
                "topology": row.get("param.topology_choice"),
                "load": row.get("param.load_model"),
                "loss": row.get("loss", row.get("metric.loss")),
                "normalized_rmse": row.get("metric.normalized_rmse"),
                "harmonic_error": row.get("metric.harmonic_error"),
                "v_peak_abs_V": row.get("metric.v_peak_abs_V"),
                "i_rms_A": row.get("metric.i_rms_A"),
                "constraint_penalty": row.get("metric.constraint_penalty"),
                "run_dir": run_dir,
            })
    return pd.DataFrame(rows)


def _surrogate_rows(case_key: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    variants = [
        ("all", {}),
        ("feasible_only", {"exclude_infeasible": True, "constraint_col": "metric.constraint_penalty"}),
        ("log1p_clip_p90", {"target_transform": "log1p", "clip_target_quantile": 0.9}),
    ]
    rows: list[dict[str, Any]] = []
    for name, kwargs in variants:
        row: dict[str, Any] = {"case": case_key, "variant": name}
        try:
            model = fit_ridge_surrogate(df, target_col="loss", **kwargs)
            row.update({
                "schema": model.get("schema"),
                "n_train": model.get("n_train"),
                "n_features": model.get("n_features"),
                "training_rmse": model.get("training_rmse"),
                "training_r2": model.get("training_r2"),
                "cv_rmse": model.get("cv_rmse"),
                "dropped_rows": model.get("dropped_rows"),
                "clipped_rows": model.get("clipped_rows"),
                "target_transform": model.get("target_transform"),
            })
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return rows


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6g}")
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])


def _write_markdown(
    out_path: Path,
    run_root: Path,
    comparison: pd.DataFrame,
    feasibility: pd.DataFrame,
    categories: pd.DataFrame,
    correlations: pd.DataFrame,
    harmonics: pd.DataFrame,
    surrogates: pd.DataFrame,
) -> None:
    top_corr = correlations
    if not top_corr.empty:
        top_corr = top_corr.sort_values("abs_corr", ascending=False).groupby("case").head(8).drop(columns=["abs_corr"])
    lines = [
        "# ngspice benchmark analysis v2",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Interpretation Rules",
        "",
        "- Treat best loss as a secondary indicator.",
        "- Prefer feasible median, penalty rate, p90/max, and topology/load risk profile.",
        "- Do not call waveform tailoring successful when A2/A3 target ratios are low.",
        "- Treat surrogate results as diagnostics until feasible-only CV metrics are acceptable.",
        "",
        "## Dummy vs ngspice summary",
        "",
        _markdown_table(comparison),
        "",
        "## Feasibility summary",
        "",
        _markdown_table(feasibility),
        "",
        "## Category risk",
        "",
        _markdown_table(categories),
        "",
        "## Top Spearman correlations",
        "",
        _markdown_table(top_corr),
        "",
        "## Harmonic amplitudes and phase",
        "",
        _markdown_table(harmonics),
        "",
        "## Surrogate diagnostics",
        "",
        _markdown_table(surrogates),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _load_profiles(Path(args.profile_file))
    analysis_cfg = profiles.get("analysis", {}) or {}
    run_root = _resolve(args.ngspice_run_root)
    dummy_root = _resolve(args.dummy_root or analysis_cfg.get("dummy_root", "runs/ccp_benchmark_eval"))
    out_dir = Path(args.out_dir) if args.out_dir else _resolve(analysis_cfg.get("curated_result_root", "ccp_benchmark_pack/results")) / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    harmonics = [int(v) for v in analysis_cfg.get("harmonics", [1, 2, 3])]

    comparison_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    corr_frames: list[pd.DataFrame] = []
    harmonic_rows: list[dict[str, Any]] = []
    surrogate_rows: list[dict[str, Any]] = []
    top_candidate_frames: list[pd.DataFrame] = []

    for case_key, spec in profiles["cases"].items():
        case_dir = run_root / spec.get("output_dir", case_key)
        if not case_dir.exists():
            continue
        ng_df = _load_summary(case_dir)
        ng_stats = _case_stats(ng_df)
        dummy_case_dir = dummy_root / spec.get("output_dir", case_key)
        dummy_stats = _case_stats(_load_summary(dummy_case_dir)) if dummy_case_dir.exists() else {}
        comparison_rows.append({
            "case": case_key,
            "dummy_best_loss": dummy_stats.get("best_loss"),
            "dummy_median_loss": dummy_stats.get("median_loss"),
            "dummy_p90_loss": dummy_stats.get("p90_loss"),
            "dummy_penalty_rate": dummy_stats.get("penalty_rate"),
            "ngspice_best_loss": ng_stats.get("best_loss"),
            "ngspice_median_loss": ng_stats.get("median_loss"),
            "ngspice_p90_loss": ng_stats.get("p90_loss"),
            "ngspice_mean_loss": ng_stats.get("mean_loss"),
            "ngspice_max_loss": ng_stats.get("max_loss"),
            "ngspice_failed": ng_stats.get("failed_trials"),
            "ngspice_penalty_rate": ng_stats.get("penalty_rate"),
            "feasible_median_loss": ng_stats.get("feasible_median"),
            "infeasible_median_loss": ng_stats.get("infeasible_median"),
            "v_peak_gt_1000": ng_stats.get("v_peak_gt_1000_count"),
            "i_rms_gt_20": ng_stats.get("i_rms_gt_20_count"),
            "i_rms_gt_25": ng_stats.get("i_rms_gt_25_count"),
            "loss_lt_1": ng_stats.get("loss_lt_1_count"),
            "loss_lt_2": ng_stats.get("loss_lt_2_count"),
        })
        feasibility_rows.extend(_feasibility_rows(case_key, ng_df))
        category_frames.append(_category_stats(ng_df, case_key))
        corr_frames.append(_correlations(ng_df, case_key))
        harmonic_rows.extend(_harmonic_rows(case_key, ng_df, harmonics))
        surrogate_rows.extend(_surrogate_rows(case_key, ng_df))
        top_candidate_frames.append(_top_candidates(case_key, ng_df))

    comparison = pd.DataFrame(comparison_rows)
    feasibility = pd.DataFrame(feasibility_rows)
    categories = pd.concat([f for f in category_frames if not f.empty], ignore_index=True) if category_frames else pd.DataFrame()
    correlations = pd.concat([f for f in corr_frames if not f.empty], ignore_index=True) if corr_frames else pd.DataFrame()
    harmonics_df = pd.DataFrame(harmonic_rows)
    surrogates = pd.DataFrame(surrogate_rows)
    top_candidates = pd.concat([f for f in top_candidate_frames if not f.empty], ignore_index=True) if top_candidate_frames else pd.DataFrame()

    comparison.to_csv(out_dir / "comparison_summary.csv", index=False)
    feasibility.to_csv(out_dir / "feasibility_summary.csv", index=False)
    categories.to_csv(out_dir / "category_stats.csv", index=False)
    categories.to_csv(out_dir / "level3_category_stats.csv", index=False)
    top_candidates.to_csv(out_dir / "top_candidates.csv", index=False)
    correlations.to_csv(out_dir / "spearman_correlations.csv", index=False)
    harmonics_df.to_csv(out_dir / "harmonic_amplitudes.csv", index=False)
    surrogates.to_csv(out_dir / "surrogate_diagnostics.csv", index=False)
    _write_markdown(out_dir / "analysis.md", run_root, comparison, feasibility, categories, correlations, harmonics_df, surrogates)

    files = [
        "comparison_summary.csv",
        "feasibility_summary.csv",
        "category_stats.csv",
        "level3_category_stats.csv",
        "top_candidates.csv",
        "spearman_correlations.csv",
        "harmonic_amplitudes.csv",
        "surrogate_diagnostics.csv",
        "analysis.md",
    ]
    manifest = {
        "schema": "ccp_ngspice_benchmark_analysis.v2",
        "ngspice_run_root": str(run_root),
        "dummy_root": str(dummy_root),
        "out_dir": str(out_dir),
        "files": files,
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze ngspice CCP benchmark runs without adding benchmark logic to the platform core.")
    parser.add_argument("ngspice_run_root", help="Run root containing benchmark case directories.")
    parser.add_argument("--profile-file", default=str(DEFAULT_PROFILE_FILE))
    parser.add_argument("--dummy-root")
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
