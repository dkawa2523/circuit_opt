"""Read and plot the Optuna comparison for the etch-CCP inverse case.

This module is deliberately read-only with respect to simulation. It follows
completed PCD study artifacts, compares the pinned OptunaHub AutoSampler with
Optuna's standard TPESampler, and evaluates an upper-reflection waveform that
was not used by either optimizer.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from pcd.figures import (
    BLUE,
    INK,
    ORANGE,
    PAGE_SIZE,
    add_figure_footer,
    add_figure_title,
    add_panel_title,
)

STUDY_ID = "etch_ccp_dcs_continuous_inverse"
EVALUATION_BUDGET = 200
Z0_OHM = 50.0
OPTUNAHUB_REGISTRY_REF = "61da9ce3093a6c92da998a127b55720f9d3b8fd7"
PARAMETER_ORDER = ("Rp_on_ohm", "Lp_on_H", "Csu_on_F", "Csw_on_F")
PARAMETER_LABELS = (r"$R_p$", r"$L_p$", r"$C_{s,u}$", r"$C_{s,w}$")
TRUE_PARAMETERS = {
    "Rp_on_ohm": 15.0,
    "Lp_on_H": 1.6e-8,
    "Csu_on_F": 5.2e-10,
    "Csw_on_F": 7.2e-10,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"numeric CSV is empty: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def _artifact_path(base: Path, recorded: str) -> Path:
    return base.joinpath(*PureWindowsPath(recorded).parts).resolve()


def _find_study(root: Path) -> Path:
    matches = [
        path.parent
        for path in root.rglob("study_result.json")
        if _read_json(path).get("study", {}).get("study_id") == STUDY_ID
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {STUDY_ID!r} study under {root}; found {len(matches)}")
    return matches[0]


def _study_sources(root: Path, expected_optimizer: str) -> tuple[int, dict[str, Path]]:
    study_root = _find_study(root)
    study_path = study_root / "study_result.json"
    study = _read_json(study_path)
    execution = study.get("execution", {})
    if execution.get("optimizer") != expected_optimizer:
        raise ValueError(f"{study_path} uses {execution.get('optimizer')!r}, expected {expected_optimizer!r}")
    if execution.get("trials") != EVALUATION_BUDGET:
        raise ValueError(f"{study_path} must declare {EVALUATION_BUDGET} trials")
    if study.get("n_evaluations") != EVALUATION_BUDGET or study.get("n_failed_evaluations") != 0:
        raise ValueError(f"{study_path} must contain {EVALUATION_BUDGET} successful evaluation records")
    seed = int(execution["seed"])
    generation = _artifact_path(study_root, str(study["artifacts"]["generation"]))
    evaluations = generation / "evaluations.csv"
    rows = _read_table(evaluations)
    selected_id = str(study["best"]["candidate"]["candidate_id"])
    selected = next(row for row in rows if row["candidate_id"] == selected_id)
    return seed, {
        "study": study_path,
        "input_manifest": generation / "input_manifest.json",
        "best": generation / "best_candidate.json",
        "history": generation / "study_history.json",
        "evaluations": evaluations,
        "selected_waveform": _artifact_path(study_root, selected["artifact.waveform"]),
        "selected_manifest": _artifact_path(study_root, selected["artifact.manifest"]),
        "selected_netlist": _artifact_path(study_root, selected["artifact.netlist"]),
        "selected_solver_log": _artifact_path(study_root, selected["artifact.solver_log"]),
    }


def collect_optuna_sources(
    case_path: Path,
    auto_roots: tuple[Path, ...],
    tpe_roots: tuple[Path, ...],
) -> dict[str, Path]:
    """Resolve repeated Optuna studies through each active generation."""

    if len(auto_roots) < 3 or len(tpe_roots) < 3:
        raise ValueError("Optuna evidence requires at least three seeds per sampler")
    repository_root = Path(__file__).resolve().parents[2]
    plugin = case_path.parent / "plugins" / "optuna_samplers.py"
    if OPTUNAHUB_REGISTRY_REF not in plugin.read_text(encoding="utf-8"):
        raise ValueError("OptunaHub registry ref differs between evidence reader and sampler plugin")
    sources = {
        "continuous_case": case_path,
        "continuous_evidence_module": Path(__file__).resolve(),
        "continuous_optuna_plugin": plugin,
        "continuous_pyproject": repository_root / "pyproject.toml",
        "continuous_lockfile": repository_root / "uv.lock",
    }
    seeds_by_method: dict[str, set[int]] = {}
    for method, optimizer, roots in (
        ("auto", "optuna_auto", auto_roots),
        ("tpe", "optuna_tpe", tpe_roots),
    ):
        seeds: set[int] = set()
        for root in roots:
            seed, paths = _study_sources(root, optimizer)
            if seed in seeds:
                raise ValueError(f"duplicate {method} seed {seed}")
            seeds.add(seed)
            sources.update({f"continuous_{method}_{seed}_{name}": path for name, path in paths.items()})
        seeds_by_method[method] = seeds
    if seeds_by_method["auto"] != seeds_by_method["tpe"]:
        raise ValueError(f"samplers use different seeds: {seeds_by_method}")
    return sources


def _aligned(reference: dict[str, np.ndarray], observed: dict[str, np.ndarray], column: str) -> np.ndarray:
    return np.asarray(np.interp(reference["time_s"], observed["time_s"], observed[column]), dtype=float)


def _reflected(reference: dict[str, np.ndarray], waveform: dict[str, np.ndarray]) -> np.ndarray:
    return 0.5 * (
        _aligned(reference, waveform, "upper_port_voltage_V") + Z0_OHM * _aligned(reference, waveform, "current_A")
    )


def _errors(target: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = observed - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "rmse_V": rmse,
        "normalized_rmse": rmse / float(np.sqrt(np.mean(target**2))),
        "max_abs_error_V": float(np.max(np.abs(residual))),
    }


def _prefixes(sources: dict[str, Path], method: str) -> list[str]:
    suffix = "_study"
    return sorted(
        (key[: -len(suffix)] for key in sources if key.startswith(f"continuous_{method}_") and key.endswith(suffix)),
        key=lambda prefix: int(prefix.split("_")[2]),
    )


def _run_result(sources: dict[str, Path], prefix: str, target: dict[str, np.ndarray]) -> dict[str, Any]:
    study = _read_json(sources[f"{prefix}_study"])
    history = _read_json(sources[f"{prefix}_history"])
    rows = sorted(_read_table(sources[f"{prefix}_evaluations"]), key=lambda row: int(row["trial"]))
    if len(rows) != EVALUATION_BUDGET or len(history) != EVALUATION_BUDGET:
        raise ValueError(f"{prefix} does not contain {EVALUATION_BUDGET} evaluation and history rows")
    losses = np.asarray([float(row["metric.normalized_rmse"]) for row in rows])
    selected_id = str(study["best"]["candidate"]["candidate_id"])
    selected_row = next(row for row in rows if row["candidate_id"] == selected_id)
    selected_values = {name: float(selected_row[f"design.{name}"]) for name in PARAMETER_ORDER}
    waveform = _read_numeric_csv(sources[f"{prefix}_selected_waveform"])
    cache_hits = sum(row["from_cache"].lower() == "true" for row in rows)
    backend_counts = Counter(str(item.get("proposal", {}).get("backend_sampler", "unknown")) for item in history)
    return {
        "source_prefix": prefix,
        "optimizer": str(study["execution"]["optimizer"]),
        "seed": int(study["execution"]["seed"]),
        "dataset_id": str(study["dataset"]["dataset_id"]),
        "runtime_fingerprint_sha256": str(study["dataset"]["runtime_fingerprint_sha256"]),
        "solver_fingerprint_sha256": str(study["dataset"]["solver_fingerprint_sha256"]),
        "selected_candidate_id": selected_id,
        "selected_trial": int(selected_row["trial"]),
        "selected_values": selected_values,
        "parameter_ratio": {name: selected_values[name] / TRUE_PARAMETERS[name] for name in PARAMETER_ORDER},
        "objective_error": _errors(target["wafer_voltage_V"], _aligned(target, waveform, "wafer_voltage_V")),
        "held_out_error": _errors(target["upper_reflected_voltage_V"], _reflected(target, waveform)),
        "initial_loss": float(losses[0]),
        "selected_loss": float(np.min(losses)),
        "loss_trace": losses.tolist(),
        "best_so_far": np.minimum.accumulate(losses).tolist(),
        "n_trials": int(study["execution"]["trials"]),
        "n_evaluation_records": int(study["n_evaluations"]),
        "n_failed_evaluations": int(study["n_failed_evaluations"]),
        "cache_hits": cache_hits,
        "new_solver_runs": len(rows) - cache_hits,
        "backend_sampler_counts": dict(sorted(backend_counts.items())),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _median_parameter_error_pct(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(np.median([100.0 * abs(float(run["parameter_ratio"][name]) - 1.0) for run in runs]))
        for name in PARAMETER_ORDER
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def build_optuna_evidence(sources: dict[str, Path]) -> dict[str, Any]:
    """Build JSON-ready, case-specific AutoSampler-versus-TPE evidence."""

    target = _read_numeric_csv(sources["target_observables"])
    auto_runs = [_run_result(sources, prefix, target) for prefix in _prefixes(sources, "auto")]
    tpe_runs = [_run_result(sources, prefix, target) for prefix in _prefixes(sources, "tpe")]
    auto_losses = [float(run["selected_loss"]) for run in auto_runs]
    tpe_losses = [float(run["selected_loss"]) for run in tpe_runs]
    auto_holdout = [float(run["held_out_error"]["normalized_rmse"]) for run in auto_runs]
    tpe_holdout = [float(run["held_out_error"]["normalized_rmse"]) for run in tpe_runs]
    best_auto = min(auto_runs, key=lambda run: float(run["selected_loss"]))
    best_tpe = min(tpe_runs, key=lambda run: float(run["selected_loss"]))
    backend_counts = Counter()
    for run in auto_runs:
        backend_counts.update(run["backend_sampler_counts"])
    all_runs = auto_runs + tpe_runs
    return {
        "scope": {
            "claim": "case-specific comparison of two requested Optuna samplers for four bounded on-state values",
            "not_claimed": [
                "general sampler ranking",
                "continuous-domain global optimality",
                "unique identification from the fitted wafer waveform",
                "measured-tool or plasma-process validation",
            ],
        },
        "parameterization": {
            "dimension": 4,
            "truth": TRUE_PARAMETERS,
            "waveform_shape": "known shared periodic envelope; only four on-state values are optimized",
            "fitted_output": "wafer_voltage_V",
            "held_out_output": "upper_reflected_voltage_V",
        },
        "trial_budget_per_seed": EVALUATION_BUDGET,
        "seed_count_per_sampler": len(auto_runs),
        "software": {
            "optuna": _package_version("optuna"),
            "optunahub": _package_version("optunahub"),
            "registry_ref": OPTUNAHUB_REGISTRY_REF,
        },
        "auto_sampler": {
            "label": "OptunaHub AutoSampler",
            "registry_package": "samplers/auto_sampler",
            "selection_reason": "fixed four-dimensional numerical, single-objective space below the 250-trial GP threshold",
            "effective_backend_counts": dict(sorted(backend_counts.items())),
            "runs": auto_runs,
            "final_loss": _distribution(auto_losses),
            "held_out_loss": _distribution(auto_holdout),
            "median_absolute_parameter_error_pct": _median_parameter_error_pct(auto_runs),
        },
        "tpe_sampler": {
            "label": "Optuna TPESampler",
            "runs": tpe_runs,
            "final_loss": _distribution(tpe_losses),
            "held_out_loss": _distribution(tpe_holdout),
            "median_absolute_parameter_error_pct": _median_parameter_error_pct(tpe_runs),
        },
        "best_auto_source_prefix": best_auto["source_prefix"],
        "best_tpe_source_prefix": best_tpe["source_prefix"],
        "comparison": {
            "median_objective_auto_over_tpe": float(np.median(auto_losses) / np.median(tpe_losses)),
            "median_holdout_auto_over_tpe": float(np.median(auto_holdout) / np.median(tpe_holdout)),
            "auto_has_lower_median_fitted_objective": bool(np.median(auto_losses) < np.median(tpe_losses)),
            "tpe_has_lower_median_holdout_error": bool(np.median(tpe_holdout) < np.median(auto_holdout)),
        },
        "assessment": {
            "execution_integrity": {
                "total_trials": sum(int(run["n_trials"]) for run in all_runs),
                "new_solver_runs": sum(int(run["new_solver_runs"]) for run in all_runs),
                "cache_hits": sum(int(run["cache_hits"]) for run in all_runs),
                "failed_evaluations": sum(int(run["n_failed_evaluations"]) for run in all_runs),
            },
            "sampler_behavior": {
                "auto_cache_hits": sum(int(run["cache_hits"]) for run in auto_runs),
                "tpe_cache_hits": sum(int(run["cache_hits"]) for run in tpe_runs),
                "interpretation": "AutoSampler reached lower fitted loss but repeated exact GP proposals, especially for seed 0",
            },
            "generalization": {
                "interpretation": "TPE has the lower median error on the unused reflection output despite its higher fitted loss",
            },
            "parameter_recovery": {
                "interpretation": "low fitted waveform error does not by itself establish the most accurate physical parameters",
            },
            "verdicts": {
                "auto_sampler_suitable_for_small_budget_numerical_search": True,
                "auto_sampler_unconditionally_better_than_tpe": False,
                "unique_parameter_recovery_established": False,
                "continuous_global_optimality_established": False,
            },
        },
    }


def _format_axes(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(True, color="#d9dee3", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def figure_optuna_comparison(sources: dict[str, Path], result: dict[str, Any]) -> plt.Figure:
    """Render the requested AutoSampler-versus-TPE publication page."""

    target = _read_numeric_csv(sources["target_observables"])
    auto_prefix = str(result["best_auto_source_prefix"])
    tpe_prefix = str(result["best_tpe_source_prefix"])
    auto_waveform = _read_numeric_csv(sources[f"{auto_prefix}_selected_waveform"])
    tpe_waveform = _read_numeric_csv(sources[f"{tpe_prefix}_selected_waveform"])
    time_us = 1e6 * (target["time_s"] - target["time_s"][0])
    auto_runs = result["auto_sampler"]["runs"]
    tpe_runs = result["tpe_sampler"]["runs"]
    trials = np.arange(1, EVALUATION_BUDGET + 1)
    auto_traces = np.asarray([run["best_so_far"] for run in auto_runs])
    tpe_traces = np.asarray([run["best_so_far"] for run in tpe_runs])
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)

    add_panel_title(axes[0, 0], "(a)", "Fitted wafer-voltage objective")
    axes[0, 0].plot(time_us, target["wafer_voltage_V"], color=INK, lw=1.15, ls=(0, (4, 2)))
    axes[0, 0].plot(time_us, _aligned(target, auto_waveform, "wafer_voltage_V"), color=BLUE, lw=0.8)
    axes[0, 0].plot(time_us, _aligned(target, tpe_waveform, "wafer_voltage_V"), color=ORANGE, lw=0.75)
    axes[0, 0].set_ylabel(r"$V_W$ (V)")

    add_panel_title(axes[0, 1], "(b)", "Unused upper-reflection output")
    axes[0, 1].plot(time_us, target["upper_reflected_voltage_V"], color=INK, lw=1.15, ls=(0, (4, 2)))
    axes[0, 1].plot(time_us, _reflected(target, auto_waveform), color=BLUE, lw=0.8)
    axes[0, 1].plot(time_us, _reflected(target, tpe_waveform), color=ORANGE, lw=0.75)
    axes[0, 1].set_ylabel(r"$V_-$ (V)")

    add_panel_title(axes[1, 0], "(c)", "Cumulative best loss (three seeds)")
    for trace in auto_traces:
        axes[1, 0].semilogy(trials, trace, color=BLUE, lw=0.45, alpha=0.24)
    for trace in tpe_traces:
        axes[1, 0].semilogy(trials, trace, color=ORANGE, lw=0.45, alpha=0.22)
    axes[1, 0].semilogy(trials, np.median(auto_traces, axis=0), color=BLUE, lw=1.45, label="Auto/GP median")
    axes[1, 0].semilogy(trials, np.median(tpe_traces, axis=0), color=ORANGE, lw=1.35, label="TPE median")
    axes[1, 0].set_xlim(1.0, EVALUATION_BUDGET)
    axes[1, 0].set_xticks([1, 50, 100, 150, 200])
    axes[1, 0].set_xlabel("Optuna trial")
    axes[1, 0].set_ylabel("Best wafer nRMSE")
    axes[1, 0].yaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].legend(loc="upper right", fontsize=5.2, frameon=True)

    add_panel_title(axes[1, 1], "(d)", "Final fitted loss by seed")
    seeds = np.asarray([run["seed"] for run in auto_runs])
    auto_final = np.asarray([run["selected_loss"] for run in auto_runs])
    tpe_final = np.asarray([run["selected_loss"] for run in tpe_runs])
    axes[1, 1].semilogy(seeds - 0.08, auto_final, "o", color=BLUE, ms=4.2, label="Auto/GP")
    axes[1, 1].semilogy(seeds + 0.08, tpe_final, "^", color=ORANGE, ms=4.2, label="TPE")
    axes[1, 1].plot([-0.25, 2.25], [np.median(auto_final)] * 2, color=BLUE, lw=0.75, alpha=0.7)
    axes[1, 1].plot([-0.25, 2.25], [np.median(tpe_final)] * 2, color=ORANGE, lw=0.75, alpha=0.7)
    axes[1, 1].set_xlim(-0.3, 2.3)
    axes[1, 1].set_ylim(0.8e-4, 2.4e-4)
    axes[1, 1].set_xticks(seeds)
    axes[1, 1].set_yticks([1.0e-4, 1.5e-4, 2.0e-4], ["1.0e-4", "1.5e-4", "2.0e-4"])
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Final wafer nRMSE")
    axes[1, 1].yaxis.set_minor_formatter(NullFormatter())
    axes[1, 1].legend(loc="upper right", fontsize=5.2, frameon=True)

    add_panel_title(axes[2, 0], "(e)", "Recovered value / known truth")
    x = np.arange(len(PARAMETER_ORDER), dtype=float)
    offsets = np.linspace(-0.08, 0.08, len(auto_runs))
    for offset, run in zip(offsets, auto_runs, strict=True):
        axes[2, 0].scatter(
            x - 0.13 + offset,
            [run["parameter_ratio"][name] for name in PARAMETER_ORDER],
            color=BLUE,
            s=12,
            edgecolors="none",
        )
    for offset, run in zip(offsets, tpe_runs, strict=True):
        axes[2, 0].scatter(
            x + 0.13 + offset,
            [run["parameter_ratio"][name] for name in PARAMETER_ORDER],
            color=ORANGE,
            marker="^",
            s=13,
            edgecolors="none",
        )
    axes[2, 0].axhline(1.0, color=INK, lw=0.9, ls=(0, (4, 2)))
    axes[2, 0].set_xticks(x, PARAMETER_LABELS)
    axes[2, 0].set_ylabel("Estimate / truth")
    axes[2, 0].set_ylim(0.975, 1.025)
    axes[2, 0].text(0.02, 0.92, "circles: Auto/GP   triangles: TPE", transform=axes[2, 0].transAxes, fontsize=5.1)

    add_panel_title(axes[2, 1], "(f)", "Fitted objective versus unused output")
    for run in auto_runs:
        axes[2, 1].loglog(
            run["objective_error"]["normalized_rmse"],
            run["held_out_error"]["normalized_rmse"],
            "o",
            color=BLUE,
            ms=4.2,
        )
    for run in tpe_runs:
        axes[2, 1].loglog(
            run["objective_error"]["normalized_rmse"],
            run["held_out_error"]["normalized_rmse"],
            "^",
            color=ORANGE,
            ms=4.2,
        )
    axes[2, 1].set_xlabel("Wafer objective nRMSE")
    axes[2, 1].set_ylabel("Reflection hold-out nRMSE")
    axes[2, 1].set_xlim(0.9e-4, 2.3e-4)
    axes[2, 1].set_ylim(2.2e-3, 3.9e-3)
    axes[2, 1].set_xticks([1.0e-4, 1.5e-4, 2.0e-4], ["1.0e-4", "1.5e-4", "2.0e-4"])
    axes[2, 1].set_yticks([2.5e-3, 3.0e-3, 3.5e-3], ["2.5e-3", "3.0e-3", "3.5e-3"])
    axes[2, 1].xaxis.set_minor_formatter(NullFormatter())
    axes[2, 1].yaxis.set_minor_formatter(NullFormatter())
    axes[2, 1].text(0.97, 0.08, "lower-left is better", transform=axes[2, 1].transAxes, ha="right", fontsize=5.1)

    for axis in axes.flat:
        _format_axes(axis)
    for axis in axes[0, :]:
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        axis.set_xlabel("Time in final pulse (us)")
    target_handle = plt.Line2D([], [], color=INK, lw=1.15, ls=(0, (4, 2)))
    auto_handle = plt.Line2D([], [], color=BLUE, lw=1.0)
    tpe_handle = plt.Line2D([], [], color=ORANGE, lw=1.0)
    fig.legend(
        [target_handle, auto_handle, tpe_handle],
        ["independent truth / target", "best AutoSampler run", "best TPESampler run"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=3,
        frameon=False,
        fontsize=5.9,
    )
    auto_stats = result["auto_sampler"]["final_loss"]
    tpe_stats = result["tpe_sampler"]["final_loss"]
    auto_holdout = result["auto_sampler"]["held_out_loss"]
    tpe_holdout = result["tpe_sampler"]["held_out_loss"]
    integrity = result["assessment"]["execution_integrity"]
    add_figure_title(
        fig,
        "OptunaHub AutoSampler (GP) versus standard TPE at 200 trials",
        "Three matched seeds; the wafer waveform is fitted and upper reflection is retained as an unused output.",
    )
    add_figure_footer(
        fig,
        f"Fitted waveform: median nRMSE Auto/GP {auto_stats['median']:.3e}, TPE {tpe_stats['median']:.3e} "
        f"(ratio {result['comparison']['median_objective_auto_over_tpe']:.3f}).\n"
        f"Unused reflection: median nRMSE Auto/GP {auto_holdout['median']:.3e}, TPE {tpe_holdout['median']:.3e}. "
        f"Execution: {integrity['total_trials']} trials, {integrity['new_solver_runs']} new solves, "
        f"{integrity['cache_hits']} cache hits, {integrity['failed_evaluations']} failures.\n"
        "Assessment: Auto/GP improves the fitted objective, but repeated proposals and worse hold-out prevent a general superiority claim.",
    )
    fig.subplots_adjust(left=0.10, right=0.965, top=0.72, bottom=0.19, hspace=1.05, wspace=0.48)
    return fig
