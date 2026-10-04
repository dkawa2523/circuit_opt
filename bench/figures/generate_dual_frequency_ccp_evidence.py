"""Render the dual-frequency CCP forward and inverse-problem evidence.

The generator is read-only with respect to simulation: it never invokes
ngspice or a search method. Curves come from the archived independent target,
one completed forward simulation, and two completed exact-grid studies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pcd.figures import (
    BLUE,
    BLUE_LIGHT,
    INK,
    LIGHT,
    MUTED,
    ORANGE,
    ORANGE_LIGHT,
    PAGE_SIZE,
    WHITE,
    add_figure_footer,
    add_figure_title,
    add_panel_title,
    assert_text_inside_canvas,
    configure_publication_style,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / "cases"
DEFAULT_FORWARD_CASE = CASES / "dual_frequency_ccp_forward.yaml"
DEFAULT_WAFER_CASE = CASES / "dual_frequency_ccp_inverse_wafer.yaml"
DEFAULT_REFLECTION_CASE = CASES / "dual_frequency_ccp_inverse_reflection.yaml"
DEFAULT_FORWARD_ROOT = ROOT / "runs" / "dual_frequency_ccp_forward_v2_20261001"
DEFAULT_WAFER_ROOT = ROOT / "runs" / "dual_frequency_ccp_inverse_wafer_v2_20261001"
DEFAULT_REFLECTION_ROOT = ROOT / "runs" / "dual_frequency_ccp_inverse_reflection_v2_20261001"
DEFAULT_OUTPUT = HERE / "dual_frequency_ccp"
DEFAULT_PDF_OUTPUT = ROOT / "output" / "pdf" / "dual-frequency-ccp-pulsed-bias-study.pdf"
SVG_HASHSALT = "pcd-dual-frequency-ccp-v2"
BIAS_PERIOD_S = 1.25e-6
PLOT_START_S = 3.75e-6
PLOT_STOP_S = 5.0e-6
TRUE_PARAMETERS = {
    "R_off_ohm": 70.0,
    "R_edge_ohm": 24.0,
    "R_plateau_ohm": 30.0,
    "R_recovery_ohm": 65.0,
}
PARAMETER_ORDER = tuple(TRUE_PARAMETERS)
PARAMETER_LABELS = (r"$R_{off}$", r"$R_{edge}$", r"$R_{plateau}$", r"$R_{recovery}$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or reader.fieldnames is None:
        raise ValueError(f"numeric CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in reader.fieldnames
        if name is not None
    }


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _source_record(path: Path) -> dict[str, Any]:
    return {"path": _repository_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _artifact_path(base: Path, recorded: str) -> Path:
    return base.joinpath(*PureWindowsPath(recorded).parts).resolve()


def _find_simulation(root: Path, case_id: str) -> Path:
    matches = [path.parent for path in root.glob("sim_*/summary.json") if _read_json(path).get("case_id") == case_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {case_id!r} simulation under {root}; found {len(matches)}")
    return matches[0]


def _find_study(root: Path, study_id: str) -> Path:
    matches = []
    for path in root.rglob("study_result.json"):
        payload = _read_json(path)
        if payload.get("study", {}).get("study_id") == study_id:
            matches.append(path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {study_id!r} study under {root}; found {len(matches)}")
    return matches[0]


def _study_sources(root: Path, study_id: str) -> dict[str, Path]:
    study_root = _find_study(root, study_id)
    study_path = study_root / "study_result.json"
    study = _read_json(study_path)
    if study.get("n_evaluations") != 81 or study.get("n_failed_evaluations") != 0:
        raise ValueError(f"{study_id} must contain 81 successful evaluations")
    generation = _artifact_path(study_root, str(study["artifacts"]["generation"]))
    evaluations_path = generation / "evaluations.csv"
    rows = _read_table(evaluations_path)
    best_path = generation / "best_candidate.json"
    best = _read_json(best_path)
    selected_id = str(best["candidate"]["candidate_id"])
    selected = next(row for row in rows if row["candidate_id"] == selected_id)
    initial = next(row for row in rows if int(row["trial"]) == 0)
    return {
        "study": study_path,
        "input_manifest": generation / "input_manifest.json",
        "best": best_path,
        "history": generation / "study_history.json",
        "evaluations": evaluations_path,
        "initial_waveform": _artifact_path(study_root, initial["artifact.waveform"]),
        "selected_waveform": _artifact_path(study_root, selected["artifact.waveform"]),
        "selected_netlist": _artifact_path(study_root, selected["artifact.netlist"]),
    }


def collect_sources(
    forward_case: Path,
    wafer_case: Path,
    reflection_case: Path,
    forward_root: Path,
    wafer_root: Path,
    reflection_root: Path,
) -> dict[str, Path]:
    simulation = _find_simulation(forward_root, "dual_frequency_ccp_forward")
    summary_path = simulation / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "ok" or summary.get("warnings"):
        raise ValueError(f"forward simulation is not clean: {summary_path}")
    sources = {
        "forward_case": forward_case,
        "wafer_case": wafer_case,
        "reflection_case": reflection_case,
        "profile": CASES / "dual_frequency_ccp_plasma_resistance.csv",
        "target_observables": CASES / "dual_frequency_ccp_target_observables.csv",
        "target_wafer": CASES / "dual_frequency_ccp_target_wafer.csv",
        "target_reflection": CASES / "dual_frequency_ccp_target_reflection.csv",
        "target_generator": HERE / "generate_dual_frequency_ccp_target.py",
        "forward_summary": summary_path,
        "forward_manifest": simulation / "debug" / "manifest.json",
        "forward_netlist": simulation / "debug" / "netlist.cir",
        "forward_waveform": simulation / str(summary["artifacts"]["waveform"]),
    }
    sources.update(
        {f"wafer_{name}": path for name, path in _study_sources(wafer_root, "dual_frequency_ccp_inverse_wafer").items()}
    )
    sources.update(
        {
            f"reflection_{name}": path
            for name, path in _study_sources(reflection_root, "dual_frequency_ccp_inverse_reflection").items()
        }
    )
    return sources


def _aligned(reference: dict[str, np.ndarray], observed: dict[str, np.ndarray], column: str) -> np.ndarray:
    return np.asarray(np.interp(reference["time_s"], observed["time_s"], observed[column]), dtype=float)


def _errors(target: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = observed - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.sqrt(np.mean(target**2)))
    return {
        "rmse_V": rmse,
        "normalized_rmse": rmse / scale,
        "max_abs_error_V": float(np.max(np.abs(residual))),
    }


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "trial": int(row["trial"]),
            "candidate_id": row["candidate_id"],
            "loss": float(row["metric.normalized_rmse"]),
            "rmse_V": float(row["metric.rmse_V"]),
            "status": row["status"],
            "from_cache": row["from_cache"].lower() == "true",
            "values": {name: float(row[f"design.{name}"]) for name in PARAMETER_ORDER},
        }
        for row in _read_table(path)
    ]


def _profile(time_s: np.ndarray, values: dict[str, float]) -> np.ndarray:
    points_s = np.asarray([0.0, 5.0e-8, 1.05e-6, 1.10e-6, BIAS_PERIOD_S])
    points_r = np.asarray(
        [
            values["R_off_ohm"],
            values["R_edge_ohm"],
            values["R_plateau_ohm"],
            values["R_recovery_ohm"],
            values["R_off_ohm"],
        ]
    )
    phase = time_s - np.floor(time_s / BIAS_PERIOD_S) * BIAS_PERIOD_S
    return np.asarray(np.interp(phase, points_s, points_r), dtype=np.float64)


def _inverse_result(sources: dict[str, Path], prefix: str, objective: str, held_out: str) -> dict[str, Any]:
    target = _read_numeric_csv(sources["target_observables"])
    selected_waveform = _read_numeric_csv(sources[f"{prefix}_selected_waveform"])
    best = _read_json(sources[f"{prefix}_best"])
    rows = _candidate_rows(sources[f"{prefix}_evaluations"])
    selected_values = {name: float(best["candidate"]["values"][name]) for name in PARAMETER_ORDER}
    initial_values = next(row["values"] for row in rows if row["trial"] == 0)
    objective_target = target[objective]
    objective_observed = _aligned(target, selected_waveform, "voltage_V")
    held_out_observed = _aligned(target, selected_waveform, held_out)
    true_profile = target["plasma_resistance_ohm"]
    selected_profile = _profile(target["time_s"], selected_values)
    ordered_losses = sorted(row["loss"] for row in rows)
    study = _read_json(sources[f"{prefix}_study"])
    return {
        "objective_column": objective,
        "held_out_column": held_out,
        "selected_candidate_id": best["candidate"]["candidate_id"],
        "selected_values_ohm": selected_values,
        "initial_values_ohm": initial_values,
        "truth_values_ohm": TRUE_PARAMETERS,
        "objective": _errors(objective_target, objective_observed),
        "held_out": _errors(target[held_out], held_out_observed),
        "profile_rmse_ohm": float(np.sqrt(np.mean((selected_profile - true_profile) ** 2))),
        "n_candidates": int(study["n_candidates"]),
        "n_evaluations": int(study["n_evaluations"]),
        "n_failed_evaluations": int(study["n_failed_evaluations"]),
        "cache_hits": sum(row["from_cache"] for row in rows),
        "initial_loss": next(row["loss"] for row in rows if row["trial"] == 0),
        "selected_loss": ordered_losses[0],
        "second_best_loss": ordered_losses[1],
        "search_completeness": study["best"]["search_completeness"],
        "candidates": rows,
    }


def build_figure_data(sources: dict[str, Path]) -> dict[str, Any]:
    target = _read_numeric_csv(sources["target_observables"])
    forward = _read_numeric_csv(sources["forward_waveform"])
    wafer_observed = _aligned(target, forward, "voltage_V")
    reflection_observed = _aligned(target, forward, "upper_reflected_voltage_V")
    reflected_identity = 0.5 * (
        _aligned(target, forward, "upper_port_voltage_V") - 50.0 * _aligned(target, forward, "load_current_A")
    )
    return {
        "schema": "dual_frequency_ccp_figure_data.v2",
        "scope": {
            "claim": "circuit-level forward conformance and bounded plasma-resistance identification",
            "not_claimed": [
                "arbitrary time-function reconstruction from one waveform",
                "self-consistent plasma density, chemistry, or sheath motion",
                "ion-energy distribution or process qualification",
                "directional-coupler calibration at a production tool",
            ],
        },
        "problem": {
            "upper_excitation_Hz": 40.0e6,
            "lower_rectangular_bias_Hz": 0.8e6,
            "lower_negative_duty": 0.8,
            "edge_time_s": 50.0e-9,
            "upper_reference_impedance_ohm": 50.0,
            "fixed_sheath_capacitance_F": 1.0e-9,
            "blocking_capacitance_F": 2.0e-9,
            "bias_path_resistance_ohm": 50.0,
            "plasma_profile_parameterization": {
                "times_s": [0.0, 5.0e-8, 1.05e-6, 1.10e-6, BIAS_PERIOD_S],
                "truth_values_ohm": TRUE_PARAMETERS,
            },
            "observables": {
                "wafer_voltage_V": "lower electrode/wafer chuck relative to grounded chamber",
                "upper_reflected_voltage_V": "V-=(Vport-Z0*Iport)/2 at the upper 50-ohm reference plane",
            },
        },
        "forward": {
            "samples": int(target["time_s"].size),
            "time_range_s": [float(target["time_s"][0]), float(target["time_s"][-1])],
            "independent_integrator": "fixed-step RK4",
            "independent_step_s": 6.25e-11,
            "ngspice_max_step_s": 2.5e-10,
            "wafer_voltage": _errors(target["wafer_voltage_V"], wafer_observed),
            "upper_reflected_voltage": _errors(target["upper_reflected_voltage_V"], reflection_observed),
            "reflected_monitor_identity_max_abs_V": float(np.max(np.abs(reflection_observed - reflected_identity))),
        },
        "inverse_wafer_only": _inverse_result(
            sources,
            "wafer",
            "wafer_voltage_V",
            "upper_reflected_voltage_V",
        ),
        "inverse_reflection_only": _inverse_result(
            sources,
            "reflection",
            "upper_reflected_voltage_V",
            "wafer_voltage_V",
        ),
        "software": {
            "python": platform.python_version(),
            "ngspice": _read_json(sources["forward_summary"])["solver_version"],
            "matplotlib": mpl.__version__,
            "page_inches": list(PAGE_SIZE),
            "png_dpi": 300,
        },
        "sources": {name: _source_record(path) for name, path in sources.items()},
    }


def _format_axes(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(True, color="#d9dee3", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = MUTED,
    fontsize: float = 6.6,
) -> None:
    patch = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.text(x + 0.5 * width, y + 0.5 * height, text, ha="center", va="center", fontsize=fontsize)


def _ground(axis: plt.Axes, x: float, y: float) -> None:
    axis.plot([x, x], [y, y + 0.025], color=INK, lw=1.0)
    axis.plot([x - 0.025, x + 0.025], [y, y], color=INK, lw=1.0)
    axis.plot([x - 0.017, x + 0.017], [y - 0.012, y - 0.012], color=INK, lw=0.9)
    axis.plot([x - 0.009, x + 0.009], [y - 0.023, y - 0.023], color=INK, lw=0.8)


def _resistor_h(axis: plt.Axes, x0: float, x1: float, y: float) -> None:
    lead = 0.08 * (x1 - x0)
    axis.plot([x0, x0 + lead], [y, y], color=INK, lw=1.15)
    axis.plot([x1 - lead, x1], [y, y], color=INK, lw=1.15)
    xs = np.linspace(x0 + lead, x1 - lead, 9)
    ys = np.asarray([y, y + 0.018, y - 0.018, y + 0.018, y - 0.018, y + 0.018, y - 0.018, y + 0.018, y])
    axis.plot(xs, ys, color=INK, lw=1.15)


def _resistor_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    lead = 0.08 * (y1 - y0)
    axis.plot([x, x], [y0, y0 + lead], color=INK, lw=1.15)
    axis.plot([x, x], [y1 - lead, y1], color=INK, lw=1.15)
    ys = np.linspace(y0 + lead, y1 - lead, 9)
    xs = np.asarray([x, x + 0.018, x - 0.018, x + 0.018, x - 0.018, x + 0.018, x - 0.018, x + 0.018, x])
    axis.plot(xs, ys, color=INK, lw=1.15)


def _capacitor_h(axis: plt.Axes, x0: float, x1: float, y: float) -> None:
    middle = 0.5 * (x0 + x1)
    axis.plot([x0, middle - 0.012], [y, y], color=INK, lw=1.15)
    axis.plot([middle + 0.012, x1], [y, y], color=INK, lw=1.15)
    axis.plot([middle - 0.012, middle - 0.012], [y - 0.035, y + 0.035], color=INK, lw=1.15)
    axis.plot([middle + 0.012, middle + 0.012], [y - 0.035, y + 0.035], color=INK, lw=1.15)


def _capacitor_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    middle = 0.5 * (y0 + y1)
    axis.plot([x, x], [y0, middle - 0.012], color=INK, lw=1.15)
    axis.plot([x, x], [middle + 0.012, y1], color=INK, lw=1.15)
    axis.plot([x - 0.035, x + 0.035], [middle - 0.012, middle - 0.012], color=INK, lw=1.15)
    axis.plot([x - 0.035, x + 0.035], [middle + 0.012, middle + 0.012], color=INK, lw=1.15)


def _source_h(axis: plt.Axes, x: float, y: float, *, square: bool) -> None:
    circle = mpatches.Circle((x, y), 0.05, facecolor=WHITE, edgecolor=INK, linewidth=1.05)
    axis.add_patch(circle)
    if square:
        axis.plot(
            [x - 0.028, x - 0.012, x - 0.012, x + 0.012, x + 0.012, x + 0.028],
            [y - 0.012, y - 0.012, y + 0.015, y + 0.015, y - 0.012, y - 0.012],
            color=INK,
            lw=0.9,
        )
    else:
        xx = np.linspace(x - 0.03, x + 0.03, 60)
        axis.plot(xx, y + 0.018 * np.sin((xx - x) / 0.03 * math.pi), color=INK, lw=0.9)


def _apparatus(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    chamber = mpatches.FancyBboxPatch(
        (0.28, 0.10),
        0.64,
        0.78,
        boxstyle="round,pad=0.012",
        facecolor="#fbfcfd",
        edgecolor=INK,
        linewidth=1.15,
    )
    axis.add_patch(chamber)
    axis.text(0.60, 0.905, "grounded chamber wall", ha="center", va="center", color=MUTED, fontsize=6.7)
    axis.add_patch(mpatches.Rectangle((0.37, 0.745), 0.46, 0.045, facecolor=MUTED, edgecolor=INK, lw=0.8))
    axis.text(0.60, 0.812, "[1] upper showerhead / P_HF", ha="center", fontsize=6.8)
    axis.add_patch(mpatches.Rectangle((0.39, 0.665), 0.42, 0.032, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.7))
    axis.text(0.60, 0.681, "[2] upper sheath C_s,u", ha="center", va="center", fontsize=6.2, color=BLUE)
    plasma = mpatches.FancyBboxPatch(
        (0.39, 0.36),
        0.42,
        0.27,
        boxstyle="round,pad=0.012",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=1.0,
    )
    axis.add_patch(plasma)
    axis.text(0.60, 0.495, "[3] plasma bulk\nprescribed or identified R_p(t)", ha="center", va="center", fontsize=7.1)
    axis.add_patch(mpatches.Rectangle((0.39, 0.305), 0.42, 0.032, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.7))
    axis.text(0.60, 0.321, "[4] wafer sheath C_s,w", ha="center", va="center", fontsize=6.2, color=BLUE)
    axis.add_patch(mpatches.Rectangle((0.37, 0.225), 0.46, 0.045, facecolor=MUTED, edgecolor=INK, lw=0.8))
    axis.add_patch(mpatches.Rectangle((0.40, 0.278), 0.40, 0.016, facecolor="#cbd2d8", edgecolor=INK, lw=0.5))
    axis.text(0.60, 0.187, "[5] wafer / lower chuck W", ha="center", fontsize=6.8)

    _box(axis, 0.015, 0.725, 0.13, 0.085, "40 MHz\nRF source", facecolor=BLUE_LIGHT, edgecolor=BLUE)
    _box(axis, 0.16, 0.725, 0.10, 0.085, "match /\nfeed", facecolor=LIGHT, fontsize=5.8)
    axis.plot([0.145, 0.16], [0.767, 0.767], color=BLUE, lw=1.15)
    axis.annotate("", xy=(0.37, 0.767), xytext=(0.26, 0.767), arrowprops={"arrowstyle": "->", "color": BLUE})
    _box(axis, 0.015, 0.202, 0.13, 0.09, "800 kHz\npulse bias", facecolor=ORANGE_LIGHT, edgecolor=ORANGE)
    _box(axis, 0.16, 0.202, 0.10, 0.09, "R_B + C_B", facecolor=LIGHT, fontsize=5.8)
    axis.plot([0.145, 0.16], [0.247, 0.247], color=ORANGE, lw=1.15)
    axis.annotate("", xy=(0.37, 0.247), xytext=(0.26, 0.247), arrowprops={"arrowstyle": "->", "color": ORANGE})

    axis.annotate("", xy=(0.91, 0.767), xytext=(0.82, 0.767), arrowprops={"arrowstyle": "->", "color": BLUE})
    axis.text(0.895, 0.855, r"output 2: $V_-(t)$ at P_HF", ha="right", color=BLUE, fontsize=6.2)
    axis.annotate("", xy=(0.91, 0.247), xytext=(0.82, 0.247), arrowprops={"arrowstyle": "->", "color": ORANGE})
    axis.text(0.895, 0.145, r"output 1: $V_W(t)$", ha="right", color=ORANGE, fontsize=6.2)


def _vertical_circuit(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    bus_x = 0.06
    branch_x = 0.60
    top_y = 0.80
    wafer_y = 0.19
    axis.plot([bus_x, bus_x], [wafer_y, top_y], color=INK, lw=1.05)
    _ground(axis, bus_x, 0.12)
    axis.plot([bus_x, bus_x], [0.12, wafer_y], color=INK, lw=1.05)

    axis.plot([bus_x, 0.11], [top_y, top_y], color=INK, lw=1.1)
    _source_h(axis, 0.16, top_y, square=False)
    axis.plot([0.21, 0.25], [top_y, top_y], color=INK, lw=1.1)
    _resistor_h(axis, 0.25, 0.48, top_y)
    axis.plot([0.48, branch_x], [top_y, top_y], color=INK, lw=1.1)
    axis.text(0.16, 0.865, r"$V_{HF}$", ha="center", fontsize=6.6)
    axis.text(0.365, 0.84, r"$R_{HF}=50\,\Omega$", ha="center", fontsize=6.3)
    axis.scatter([branch_x], [top_y], s=18, color=INK, zorder=3)
    axis.text(branch_x + 0.025, top_y + 0.035, "[1] P_HF", fontsize=6.6, ha="left")

    _capacitor_v(axis, branch_x, 0.67, 0.76)
    axis.plot([branch_x, branch_x], [0.76, top_y], color=INK, lw=1.1)
    axis.plot([branch_x, branch_x], [0.62, 0.67], color=INK, lw=1.1)
    axis.text(branch_x + 0.055, 0.715, r"[2] $C_{s,u}=1$ nF", va="center", fontsize=6.4, color=BLUE)
    _resistor_v(axis, branch_x, 0.39, 0.62)
    axis.text(branch_x + 0.055, 0.505, r"[3] $R_p(t)$", va="center", fontsize=7.0, color=ORANGE)
    _capacitor_v(axis, branch_x, 0.25, 0.35)
    axis.plot([branch_x, branch_x], [0.35, 0.39], color=INK, lw=1.1)
    axis.plot([branch_x, branch_x], [wafer_y, 0.25], color=INK, lw=1.1)
    axis.text(branch_x + 0.055, 0.30, r"[4] $C_{s,w}=1$ nF", va="center", fontsize=6.4, color=BLUE)
    axis.scatter([branch_x], [wafer_y], s=18, color=INK, zorder=3)
    axis.text(branch_x + 0.025, 0.125, "[5] W: wafer/chuck", fontsize=6.2, ha="left")

    axis.plot([bus_x, 0.11], [wafer_y, wafer_y], color=INK, lw=1.1)
    _source_h(axis, 0.16, wafer_y, square=True)
    axis.plot([0.21, 0.24], [wafer_y, wafer_y], color=INK, lw=1.1)
    _resistor_h(axis, 0.24, 0.39, wafer_y)
    _capacitor_h(axis, 0.39, 0.54, wafer_y)
    axis.plot([0.54, branch_x], [wafer_y, wafer_y], color=INK, lw=1.1)
    axis.text(0.16, 0.255, r"$V_{bias}$", ha="center", fontsize=6.6)
    axis.text(0.315, 0.145, r"$R_B=50\,\Omega$", ha="center", fontsize=6.1)
    axis.text(0.465, 0.145, r"$C_B=2$ nF", ha="center", fontsize=6.1)

    boundary = mpatches.FancyBboxPatch(
        (0.53, 0.225),
        0.29,
        0.555,
        boxstyle="round,pad=0.012",
        fill=False,
        edgecolor=ORANGE,
        linestyle=(0, (4, 2)),
        linewidth=1.0,
    )
    axis.add_patch(boundary)
    axis.text(
        0.505,
        0.50,
        "plasma terminal model",
        color=ORANGE,
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.2,
    )
    axis.annotate("", xy=(0.89, top_y), xytext=(branch_x, top_y), arrowprops={"arrowstyle": "->", "color": BLUE})
    axis.text(0.90, 0.84, r"$V_-(t)$", color=BLUE, ha="right", fontsize=7.0)
    axis.text(0.90, 0.805, r"$(V_P-Z_0 I_P)/2$", color=MUTED, ha="right", fontsize=6.1)
    axis.annotate("", xy=(0.89, wafer_y), xytext=(branch_x, wafer_y), arrowprops={"arrowstyle": "->", "color": ORANGE})
    axis.text(0.90, 0.245, r"$V_W(t)$", color=ORANGE, ha="right", fontsize=7.0)
    axis.annotate("", xy=(0.58, 0.70), xytext=(0.58, 0.76), arrowprops={"arrowstyle": "->", "color": INK})
    axis.text(0.545, 0.73, r"$I_P$", ha="right", va="center", fontsize=6.1)


def figure_apparatus(data: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE, gridspec_kw={"width_ratios": [1.0, 1.0]})
    add_panel_title(axes[0], "(a)", "Chamber and named measurement planes")
    _apparatus(axes[0])
    add_panel_title(axes[1], "(b)", "Top-to-bottom ngspice equivalent circuit")
    _vertical_circuit(axes[1])
    add_figure_title(
        fig,
        "Dual-frequency CCP: apparatus and circuit use the same vertical order",
        r"Labels [1]-[5] map hardware to nodes and elements; $V_W$ and $V_-$ are distinct outputs.",
    )
    add_figure_footer(
        fig,
        "Outputs: V_W = wafer/chuck to chamber ground; V- = reflected wave at P_HF.\n"
        "Z0 = 50 ohm and current is positive into the chamber; diagram not to scale.",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.12, wspace=0.16)
    return fig


def _cycle_selection(values: dict[str, np.ndarray]) -> np.ndarray:
    return (values["time_s"] >= PLOT_START_S) & (values["time_s"] <= PLOT_STOP_S)


def _cycle_time_us(values: dict[str, np.ndarray], selected: np.ndarray) -> np.ndarray:
    return 1e6 * (values["time_s"][selected] - PLOT_START_S)


def _apply_cycle_axis(axes: np.ndarray[Any, Any]) -> None:
    for axis in axes.flat:
        axis.set_xlim(0.0, 1.25)
        axis.set_xticks(np.arange(0.0, 1.251, 0.25))
        _format_axes(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time within 800 kHz cycle (us)")


def _figure_legend(fig: plt.Figure, handles: list[Any], labels: list[str]) -> None:
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.52, 0.82), ncol=len(labels))


def figure_forward(sources: dict[str, Path], data: dict[str, Any]) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    observed = _read_numeric_csv(sources["forward_waveform"])
    target_mask = _cycle_selection(target)
    observed_mask = _cycle_selection(observed)
    target_time = _cycle_time_us(target, target_mask)
    observed_time = _cycle_time_us(observed, observed_mask)
    target_handle = plt.Line2D([], [], color=ORANGE, lw=1.7, ls=(0, (4, 2)))
    ngspice_handle = plt.Line2D([], [], color=BLUE, lw=0.85)
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)

    add_panel_title(axes[0, 0], "(a)", "Upper RF source: 40 MHz, 400 Vpk")
    axes[0, 0].plot(target_time, target["upper_hf_source_voltage_V"][target_mask], color=BLUE, lw=0.75)
    axes[0, 0].set_ylabel(r"$V_{HF}$ (V)")
    add_panel_title(axes[0, 1], "(b)", "Lower rectangular bias: 800 kHz")
    axes[0, 1].plot(target_time, target["lower_bias_source_voltage_V"][target_mask], color=ORANGE, lw=1.25)
    axes[0, 1].set_ylabel(r"$V_{bias}$ (V)")
    add_panel_title(axes[1, 0], "(c)", r"Known plasma input: $R_p(t)$")
    axes[1, 0].plot(target_time, target["plasma_resistance_ohm"][target_mask], color=ORANGE, lw=1.55)
    axes[1, 0].set_ylabel(r"$R_p(t)$ ($\Omega$)")
    axes[1, 0].set_ylim(15.0, 80.0)
    add_panel_title(axes[1, 1], "(d)", r"Output 1: wafer voltage $V_W$")
    axes[1, 1].plot(
        target_time,
        target["wafer_voltage_V"][target_mask],
        color=ORANGE,
        lw=1.7,
        ls=(0, (4, 2)),
    )
    axes[1, 1].plot(observed_time, observed["voltage_V"][observed_mask], color=BLUE, lw=0.8)
    axes[1, 1].set_ylabel(r"$V_W(t)$ (V)")
    add_panel_title(axes[2, 0], "(e)", r"Output 2: upper reflected wave $V_-$")
    axes[2, 0].plot(
        target_time,
        target["upper_reflected_voltage_V"][target_mask],
        color=ORANGE,
        lw=1.7,
        ls=(0, (4, 2)),
    )
    axes[2, 0].plot(
        observed_time,
        observed["upper_reflected_voltage_V"][observed_mask],
        color=BLUE,
        lw=0.8,
    )
    axes[2, 0].set_ylabel(r"$V_-(t)$ (V)")
    add_panel_title(axes[2, 1], "(f)", r"Errors: orange $V_W$; blue $V_-$")
    wafer_observed = _aligned(target, observed, "voltage_V")
    reflected_observed = _aligned(target, observed, "upper_reflected_voltage_V")
    axes[2, 1].plot(target_time, (wafer_observed - target["wafer_voltage_V"])[target_mask], color=ORANGE, lw=0.9)
    axes[2, 1].plot(
        target_time,
        (reflected_observed - target["upper_reflected_voltage_V"])[target_mask],
        color=BLUE,
        lw=0.9,
    )
    axes[2, 1].set_ylabel("Error (V)")

    _apply_cycle_axis(axes)
    _figure_legend(
        fig,
        [target_handle, ngspice_handle],
        ["output target: independent RK4", "output: ngspice 46"],
    )
    add_figure_title(
        fig,
        "Forward problem: known inputs produce the two intended device observables",
        r"Source waveforms and $R_p(t)$ are inputs; wafer voltage and upper reflected voltage are outputs.",
    )
    forward = data["forward"]
    add_figure_footer(
        fig,
        f"Final two bias cycles; {forward['samples']} target samples. "
        f"Wafer RMSE={forward['wafer_voltage']['rmse_V']:.4f} V, "
        f"nRMSE={forward['wafer_voltage']['normalized_rmse']:.3e}.\n"
        f"Upper reflection RMSE={forward['upper_reflected_voltage']['rmse_V']:.4f} V, "
        f"nRMSE={forward['upper_reflected_voltage']['normalized_rmse']:.3e}.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.16, hspace=0.66, wspace=0.26)
    return fig


def _inverse_figure(
    sources: dict[str, Path],
    data: dict[str, Any],
    *,
    prefix: str,
    result_key: str,
    observed_target_column: str,
    held_out_target_column: str,
    objective_label: str,
    held_out_label: str,
    title: str,
) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    initial = _read_numeric_csv(sources[f"{prefix}_initial_waveform"])
    selected = _read_numeric_csv(sources[f"{prefix}_selected_waveform"])
    result = data[result_key]
    target_mask = _cycle_selection(target)
    initial_mask = _cycle_selection(initial)
    selected_mask = _cycle_selection(selected)
    target_time = _cycle_time_us(target, target_mask)
    initial_time = _cycle_time_us(initial, initial_mask)
    selected_time = _cycle_time_us(selected, selected_mask)
    truth_handle = plt.Line2D([], [], color=ORANGE, lw=1.7, ls=(0, (4, 2)))
    initial_handle = plt.Line2D([], [], color=MUTED, lw=0.75)
    selected_handle = plt.Line2D([], [], color=BLUE, lw=0.9)
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)

    add_panel_title(axes[0, 0], "(a)", f"Observed objective: {objective_label} only")
    axes[0, 0].plot(initial_time, initial["voltage_V"][initial_mask], color=MUTED, lw=0.75)
    axes[0, 0].plot(
        target_time,
        target[observed_target_column][target_mask],
        color=ORANGE,
        lw=1.7,
        ls=(0, (4, 2)),
    )
    axes[0, 0].plot(selected_time, selected["voltage_V"][selected_mask], color=BLUE, lw=0.85)
    axes[0, 0].set_ylabel(objective_label + " (V)")

    add_panel_title(axes[0, 1], "(b)", r"Recovered plasma resistance $R_p(t)$")
    true_profile = target["plasma_resistance_ohm"]
    initial_profile = _profile(target["time_s"], result["initial_values_ohm"])
    selected_profile = _profile(target["time_s"], result["selected_values_ohm"])
    axes[0, 1].plot(target_time, initial_profile[target_mask], color=MUTED, lw=0.75)
    axes[0, 1].plot(target_time, true_profile[target_mask], color=ORANGE, lw=1.7, ls=(0, (4, 2)))
    axes[0, 1].plot(target_time, selected_profile[target_mask], color=BLUE, lw=0.9)
    axes[0, 1].set_ylabel(r"$R_p(t)$ ($\Omega$)")
    axes[0, 1].set_ylim(10.0, 90.0)

    add_panel_title(axes[1, 0], "(c)", f"Held-out output: {held_out_label} (not fitted)")
    axes[1, 0].plot(
        target_time,
        target[held_out_target_column][target_mask],
        color=ORANGE,
        lw=1.7,
        ls=(0, (4, 2)),
    )
    axes[1, 0].plot(
        selected_time,
        selected[held_out_target_column][selected_mask],
        color=BLUE,
        lw=0.85,
    )
    axes[1, 0].set_ylabel(held_out_label + " (V)")

    candidates = result["candidates"]
    trials = np.asarray([row["trial"] + 1 for row in candidates])
    losses = np.asarray([row["loss"] for row in candidates])
    best_so_far = np.minimum.accumulate(losses)
    selected_trial = (
        int(next(row["trial"] for row in candidates if row["candidate_id"] == result["selected_candidate_id"])) + 1
    )
    add_panel_title(axes[1, 1], "(d)", "Loss over all 81 candidates")
    axes[1, 1].semilogy(trials, losses, color=MUTED, marker=".", ms=2.2, lw=0.45)
    axes[1, 1].step(trials, best_so_far, where="post", color=BLUE, lw=1.35)
    axes[1, 1].scatter([selected_trial], [result["selected_loss"]], color=ORANGE, s=24, zorder=4)
    axes[1, 1].set_xlim(1.0, 81.0)
    axes[1, 1].set_xticks([1, 20, 40, 60, 81])
    axes[1, 1].set_ylabel("Normalized RMSE")

    add_panel_title(axes[2, 0], "(e)", r"Four recovered $R_p$ values")
    x = np.arange(len(PARAMETER_ORDER))
    width = 0.24
    axes[2, 0].bar(
        x - width,
        [result["initial_values_ohm"][name] for name in PARAMETER_ORDER],
        width,
        color="#c8ced3",
        edgecolor=MUTED,
        linewidth=0.6,
    )
    axes[2, 0].bar(
        x,
        [result["truth_values_ohm"][name] for name in PARAMETER_ORDER],
        width,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=0.9,
    )
    axes[2, 0].bar(
        x + width,
        [result["selected_values_ohm"][name] for name in PARAMETER_ORDER],
        width,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.9,
    )
    axes[2, 0].set_xticks(x, PARAMETER_LABELS)
    axes[2, 0].set_ylabel("Resistance (ohm)")
    axes[2, 0].set_ylim(0.0, 95.0)

    add_panel_title(axes[2, 1], "(f)", "Sorted candidate loss")
    ranked = np.sort(losses)
    axes[2, 1].semilogy(np.arange(1, len(ranked) + 1), ranked, color=BLUE, lw=1.15)
    axes[2, 1].scatter([1], [ranked[0]], color=ORANGE, s=24, zorder=4)
    axes[2, 1].set_xlim(1.0, 81.0)
    axes[2, 1].set_xticks([1, 20, 40, 60, 81])
    axes[2, 1].set_xlabel("Candidate rank")
    axes[2, 1].set_ylabel("Normalized RMSE")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlim(0.0, 1.25)
        axis.set_xticks(np.arange(0.0, 1.251, 0.25))
    for axis in axes.flat:
        _format_axes(axis)
    _figure_legend(
        fig,
        [truth_handle, initial_handle, selected_handle],
        ["independent truth/target", "first grid candidate", "selected ngspice candidate"],
    )
    add_figure_title(
        fig,
        title,
        "Only the named objective waveform is ranked; the other output is retained as a held-out circuit check.",
    )
    add_figure_footer(
        fig,
        f"Selected values: off={result['selected_values_ohm']['R_off_ohm']:g}, "
        f"edge={result['selected_values_ohm']['R_edge_ohm']:g}, "
        f"plateau={result['selected_values_ohm']['R_plateau_ohm']:g}, "
        f"recovery={result['selected_values_ohm']['R_recovery_ohm']:g} ohm.\n"
        f"Objective nRMSE={result['objective']['normalized_rmse']:.3e}; "
        f"held-out nRMSE={result['held_out']['normalized_rmse']:.3e}; "
        f"profile RMSE={result['profile_rmse_ohm']:.3g} ohm; 81/81 solves succeeded.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.17, hspace=0.78, wspace=0.27)
    return fig


def figure_inverse_wafer(sources: dict[str, Path], data: dict[str, Any]) -> plt.Figure:
    return _inverse_figure(
        sources,
        data,
        prefix="wafer",
        result_key="inverse_wafer_only",
        observed_target_column="wafer_voltage_V",
        held_out_target_column="upper_reflected_voltage_V",
        objective_label=r"$V_W$",
        held_out_label=r"$V_-$",
        title=r"Inverse problem A: wafer voltage alone recovers bounded $R_p(t)$",
    )


def figure_inverse_reflection(sources: dict[str, Path], data: dict[str, Any]) -> plt.Figure:
    return _inverse_figure(
        sources,
        data,
        prefix="reflection",
        result_key="inverse_reflection_only",
        observed_target_column="upper_reflected_voltage_V",
        held_out_target_column="wafer_voltage_V",
        objective_label=r"$V_-$",
        held_out_label=r"$V_W$",
        title=r"Inverse problem B: upper reflection alone recovers bounded $R_p(t)$",
    )


def _export(figure: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    assert_text_inside_canvas(figure)
    svg_path = output / f"{stem}.svg"
    figure.savefig(
        svg_path,
        metadata={
            "Title": stem.replace("-", " "),
            "Creator": "PCD dual-frequency CCP evidence generator",
            "Description": "Dual-frequency CCP forward and inverse electrical evidence",
            "Date": None,
        },
    )
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output / f"{stem}.png", dpi=300, metadata={"Software": "PCD CCP evidence generator"})
    pdf.savefig(figure)
    plt.close(figure)


def generate(
    forward_case: Path,
    wafer_case: Path,
    reflection_case: Path,
    forward_root: Path,
    wafer_root: Path,
    reflection_root: Path,
    output: Path,
    pdf_output: Path,
) -> dict[str, Any]:
    sources = collect_sources(
        forward_case,
        wafer_case,
        reflection_case,
        forward_root,
        wafer_root,
        reflection_root,
    )
    data = build_figure_data(sources)
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT
    figures = [
        ("01-apparatus-and-equivalent-circuit", figure_apparatus(data)),
        ("02-forward-observable-conformance", figure_forward(sources, data)),
        ("03-inverse-from-wafer-voltage", figure_inverse_wafer(sources, data)),
        ("04-inverse-from-upper-reflection", figure_inverse_reflection(sources, data)),
    ]
    with PdfPages(
        pdf_output,
        metadata={
            "Title": "Dual-frequency CCP forward and inverse circuit study",
            "Author": "PCD benchmark generator",
            "Subject": "Wafer-voltage and upper-reflection observables with bounded plasma-resistance identification",
            "Keywords": "CCP, ngspice, pulsed bias, wafer voltage, reflected voltage, inverse problem",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:
        for stem, figure in figures:
            _export(figure, output, stem, pdf)
    data_path = output / "figure_data.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "pdf": str(pdf_output),
        "figures": [stem for stem, _figure in figures],
        "figure_data": str(data_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-case", type=Path, default=DEFAULT_FORWARD_CASE)
    parser.add_argument("--wafer-case", type=Path, default=DEFAULT_WAFER_CASE)
    parser.add_argument("--reflection-case", type=Path, default=DEFAULT_REFLECTION_CASE)
    parser.add_argument("--forward-root", type=Path, default=DEFAULT_FORWARD_ROOT)
    parser.add_argument("--wafer-root", type=Path, default=DEFAULT_WAFER_ROOT)
    parser.add_argument("--reflection-root", type=Path, default=DEFAULT_REFLECTION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    args = parser.parse_args(argv)
    result = generate(
        args.forward_case.resolve(),
        args.wafer_case.resolve(),
        args.reflection_case.resolve(),
        args.forward_root.resolve(),
        args.wafer_root.resolve(),
        args.reflection_root.resolve(),
        args.output.resolve(),
        args.pdf_output.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
