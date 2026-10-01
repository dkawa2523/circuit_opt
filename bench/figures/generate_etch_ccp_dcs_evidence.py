"""Render evidence for the dynamic-impedance etch-CCP benchmark.

The generator is deliberately read-only with respect to simulation.  It uses
an archived, independently integrated charge/flux MNA target, one completed
ngspice forward run, and one completed 3^4 exact-grid identification study.
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
DEFAULT_FORWARD_CASE = CASES / "etch_ccp_dcs_forward.yaml"
DEFAULT_INVERSE_CASE = CASES / "etch_ccp_dcs_inverse.yaml"
DEFAULT_FORWARD_ROOT = ROOT / "runs" / "etch_ccp_dcs_forward_v6_20261001"
DEFAULT_INVERSE_ROOT = ROOT / "runs" / "etch_ccp_dcs_inverse_v2_20261001"
DEFAULT_OUTPUT = HERE / "etch_ccp_dcs"
DEFAULT_PDF_OUTPUT = ROOT / "output" / "pdf" / "etch-ccp-dcs-dynamic-impedance-study.pdf"
SVG_HASHSALT = "pcd-etch-ccp-dcs-v1"

PERIOD_S = 2.5e-6
PLOT_START_S = 5.0e-6
PLOT_STOP_S = 7.5e-6
Z0_OHM = 50.0
TRUE_PARAMETERS = {
    "Rp_on_ohm": 15.0,
    "Lp_on_H": 1.6e-7,
    "Csu_on_F": 5.2e-10,
    "Csw_on_F": 7.2e-10,
}
PARAMETER_ORDER = tuple(TRUE_PARAMETERS)


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
    return {
        "path": _repository_path(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact_path(base: Path, recorded: str) -> Path:
    return base.joinpath(*PureWindowsPath(recorded).parts).resolve()


def _find_simulation(root: Path, case_id: str) -> Path:
    matches = [path.parent for path in root.glob("sim_*/summary.json") if _read_json(path).get("case_id") == case_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {case_id!r} simulation under {root}; found {len(matches)}")
    return matches[0]


def _find_study(root: Path, study_id: str) -> Path:
    matches = [
        path.parent
        for path in root.rglob("study_result.json")
        if _read_json(path).get("study", {}).get("study_id") == study_id
    ]
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
        "selected_solver_log": _artifact_path(study_root, selected["artifact.solver_log"]),
    }


def collect_sources(
    forward_case: Path,
    inverse_case: Path,
    forward_root: Path,
    inverse_root: Path,
) -> dict[str, Path]:
    simulation = _find_simulation(forward_root, "etch_ccp_dcs_forward")
    summary_path = simulation / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "ok" or summary.get("warnings"):
        raise ValueError(f"forward simulation is not clean: {summary_path}")
    sources = {
        "forward_case": forward_case,
        "inverse_case": inverse_case,
        "target_observables": CASES / "etch_ccp_dcs_target_observables.csv",
        "target_wafer": CASES / "etch_ccp_dcs_target_wafer.csv",
        "target_reflection": CASES / "etch_ccp_dcs_target_reflection.csv",
        "target_generator": HERE / "generate_etch_ccp_dcs_target.py",
        "evidence_generator": Path(__file__).resolve(),
        "forward_summary": summary_path,
        "forward_manifest": simulation / "debug" / "manifest.json",
        "forward_netlist": simulation / "debug" / "netlist.cir",
        "forward_solver_log": simulation / "debug" / "solver.log",
        "forward_waveform": simulation / str(summary["artifacts"]["waveform"]),
    }
    sources.update(
        {f"inverse_{name}": path for name, path in _study_sources(inverse_root, "etch_ccp_dcs_inverse").items()}
    )
    return sources


def _aligned(reference: dict[str, np.ndarray], observed: dict[str, np.ndarray], column: str) -> np.ndarray:
    return np.asarray(np.interp(reference["time_s"], observed["time_s"], observed[column]), dtype=float)


def _reflected(reference: dict[str, np.ndarray], waveform: dict[str, np.ndarray]) -> np.ndarray:
    port_voltage = _aligned(reference, waveform, "upper_port_voltage_V")
    # ngspice reports current into the positive terminal of Vhf.  The current
    # entering the chamber network is therefore -i(Vhf), so V-=(V+Z0*i(Vhf))/2.
    source_current = _aligned(reference, waveform, "current_A")
    return 0.5 * (port_voltage + Z0_OHM * source_current)


def _errors(target: np.ndarray, observed: np.ndarray, unit: str) -> dict[str, float]:
    residual = observed - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    rms_scale = float(np.sqrt(np.mean(target**2)))
    span = float(np.ptp(target))
    return {
        f"rmse_{unit}": rmse,
        "normalized_rmse": rmse / rms_scale,
        "span_normalized_rmse": rmse / span,
        f"max_abs_error_{unit}": float(np.max(np.abs(residual))),
    }


def _envelope(time_s: np.ndarray) -> np.ndarray:
    phase = time_s - np.floor(time_s / PERIOD_S) * PERIOD_S
    return np.asarray(
        np.interp(
            phase,
            np.asarray([0.0, 0.25e-6, 0.55e-6, 2.0e-6, 2.4e-6, PERIOD_S]),
            np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        ),
        dtype=float,
    )


def _profiles(time_s: np.ndarray, values: dict[str, float]) -> dict[str, np.ndarray]:
    env = _envelope(time_s)
    return {
        "Rp_ohm": 35.0 + (values["Rp_on_ohm"] - 35.0) * env,
        "Lp_H": 3.5e-7 + (values["Lp_on_H"] - 3.5e-7) * env,
        "Csu_F": 2.6e-10 + (values["Csu_on_F"] - 2.6e-10) * env,
        "Csw_F": 3.6e-10 + (values["Csw_on_F"] - 3.6e-10) * env,
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


def _branch_impedance(profiles: dict[str, np.ndarray], frequency_hz: float) -> np.ndarray:
    omega = 2.0 * math.pi * frequency_hz
    upper_sheath = 1.0 / (1.0 / 1500.0 + 1j * omega * profiles["Csu_F"])
    wafer_sheath = 1.0 / (1.0 / 2000.0 + 1j * omega * profiles["Csw_F"])
    bulk = profiles["Rp_ohm"] + 1j * omega * profiles["Lp_H"]
    return upper_sheath + bulk + wafer_sheath


def build_figure_data(sources: dict[str, Path]) -> dict[str, Any]:
    target = _read_numeric_csv(sources["target_observables"])
    forward = _read_numeric_csv(sources["forward_waveform"])
    selected = _read_numeric_csv(sources["inverse_selected_waveform"])
    best = _read_json(sources["inverse_best"])
    study = _read_json(sources["inverse_study"])
    candidates = _candidate_rows(sources["inverse_evaluations"])
    selected_values = {name: float(best["candidate"]["values"][name]) for name in PARAMETER_ORDER}
    initial_values = next(row["values"] for row in candidates if row["trial"] == 0)

    forward_wafer = _aligned(target, forward, "wafer_voltage_V")
    forward_reflection = _reflected(target, forward)
    forward_bulk_current = _aligned(target, forward, "bulk_current_A")
    forward_upper_charge = target["upper_sheath_capacitance_F"] * _aligned(target, forward, "upper_sheath_voltage_V")
    forward_flux = target["plasma_inductance_H"] * forward_bulk_current
    selected_wafer = _aligned(target, selected, "wafer_voltage_V")
    selected_reflection = _reflected(target, selected)
    true_profiles = _profiles(target["time_s"], TRUE_PARAMETERS)
    selected_profiles = _profiles(target["time_s"], selected_values)
    profile_errors = {
        "Rp_ohm": float(np.sqrt(np.mean((selected_profiles["Rp_ohm"] - true_profiles["Rp_ohm"]) ** 2))),
        "Lp_H": float(np.sqrt(np.mean((selected_profiles["Lp_H"] - true_profiles["Lp_H"]) ** 2))),
        "Csu_F": float(np.sqrt(np.mean((selected_profiles["Csu_F"] - true_profiles["Csu_F"]) ** 2))),
        "Csw_F": float(np.sqrt(np.mean((selected_profiles["Csw_F"] - true_profiles["Csw_F"]) ** 2))),
    }
    ordered_losses = sorted(row["loss"] for row in candidates)
    return {
        "schema": "etch_ccp_dcs_figure_data.v1",
        "scope": {
            "claim": "circuit-level dynamic R/L/sheath-C forward conformance and bounded four-parameter identification",
            "not_claimed": [
                "self-consistent plasma density, chemistry, or surface-reaction prediction",
                "nonlinear kinetic sheath or ion-energy-distribution prediction",
                "unique recovery of arbitrary time functions from one measured waveform",
                "production-tool matching-network or process qualification",
            ],
        },
        "problem": {
            "upper_rf_Hz": 60.0e6,
            "lower_rf_Hz": 2.0e6,
            "dc_pulse_Hz": 400.0e3,
            "dc_level_V": -800.0,
            "reference_impedance_ohm": Z0_OHM,
            "truth": TRUE_PARAMETERS,
            "external_elements": {
                "upper_match_C_F": 1.2e-10,
                "upper_match_L_H": 7.5e-8,
                "upper_feed_R_ohm": 0.6,
                "upper_feed_L_H": 2.5e-8,
                "upper_stray_C_F": 9.0e-11,
                "dc_supply_R_ohm": 250.0,
                "dc_choke_L_H": 2.0e-4,
                "lower_block_C_F": 4.7e-9,
                "lower_match_L_H": 1.35e-6,
                "lower_feed_R_ohm": 0.4,
                "lower_feed_L_H": 6.0e-8,
                "wafer_stray_C_F": 1.4e-10,
                "wall_leak_R_ohm": 5000.0,
            },
        },
        "forward": {
            "samples": target["time_s"].size,
            "time_range_s": [float(target["time_s"][0]), float(target["time_s"][-1])],
            "independent_integrator": "charge/flux MNA, fixed-step RK4",
            "independent_step_s": 1.25e-10,
            "ngspice_max_step_s": 2.5e-10,
            "wafer_voltage": _errors(target["wafer_voltage_V"], forward_wafer, "V"),
            "upper_reflected_voltage": _errors(target["upper_reflected_voltage_V"], forward_reflection, "V"),
            "bulk_current": _errors(target["bulk_current_A"], forward_bulk_current, "A"),
            "upper_sheath_charge": _errors(target["upper_sheath_charge_C"], forward_upper_charge, "C"),
            "plasma_flux": _errors(target["plasma_flux_Wb"], forward_flux, "Wb"),
        },
        "inverse": {
            "objective": "wafer_voltage_V only",
            "held_out": "upper_reflected_voltage_V",
            "selected_candidate_id": best["candidate"]["candidate_id"],
            "initial_values": initial_values,
            "selected_values": selected_values,
            "truth_values": TRUE_PARAMETERS,
            "objective_error": _errors(target["wafer_voltage_V"], selected_wafer, "V"),
            "held_out_error": _errors(target["upper_reflected_voltage_V"], selected_reflection, "V"),
            "profile_rmse": profile_errors,
            "initial_loss": next(row["loss"] for row in candidates if row["trial"] == 0),
            "selected_loss": ordered_losses[0],
            "second_best_loss": ordered_losses[1],
            "n_candidates": int(study["n_candidates"]),
            "n_evaluations": int(study["n_evaluations"]),
            "n_failed_evaluations": int(study["n_failed_evaluations"]),
            "cache_hits": sum(row["from_cache"] for row in candidates),
            "search_completeness": study["best"]["search_completeness"],
            "candidates": candidates,
        },
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


def _ground(axis: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    axis.plot([x, x], [y, y + 0.025 * scale], color=INK, lw=1.0)
    axis.plot([x - 0.025 * scale, x + 0.025 * scale], [y, y], color=INK, lw=1.0)
    axis.plot(
        [x - 0.017 * scale, x + 0.017 * scale],
        [y - 0.012 * scale, y - 0.012 * scale],
        color=INK,
        lw=0.9,
    )
    axis.plot(
        [x - 0.009 * scale, x + 0.009 * scale],
        [y - 0.023 * scale, y - 0.023 * scale],
        color=INK,
        lw=0.8,
    )


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
    fontsize: float = 6.2,
) -> None:
    patch = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.009",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=fontsize)


def _component(axis: plt.Axes, x: float, y: float, text: str, color: str = WHITE) -> None:
    _box(axis, x - 0.075, y - 0.026, 0.15, 0.052, text, facecolor=color, fontsize=5.6)


def _apparatus(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    chamber = mpatches.FancyBboxPatch(
        (0.12, 0.11),
        0.62,
        0.76,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor="#f7f8f9",
        edgecolor=INK,
        linewidth=1.2,
    )
    axis.add_patch(chamber)
    axis.plot([0.19, 0.67], [0.75, 0.75], color=BLUE, lw=4.0, solid_capstyle="round")
    axis.plot([0.19, 0.67], [0.25, 0.25], color=ORANGE, lw=4.0, solid_capstyle="round")
    axis.add_patch(mpatches.Rectangle((0.21, 0.28), 0.44, 0.035, facecolor="#777f86", edgecolor=INK, lw=0.7))
    axis.add_patch(mpatches.Rectangle((0.24, 0.315), 0.38, 0.02, facecolor="#b8bec4", edgecolor=INK, lw=0.6))
    axis.add_patch(mpatches.Rectangle((0.18, 0.68), 0.50, 0.035, facecolor="#9ca6ad", edgecolor=INK, lw=0.6))
    for x in np.linspace(0.22, 0.64, 7):
        axis.plot([x, x], [0.68, 0.655], color=MUTED, lw=0.7)
    axis.add_patch(mpatches.Rectangle((0.20, 0.58), 0.46, 0.055, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.8))
    axis.add_patch(mpatches.Rectangle((0.20, 0.36), 0.46, 0.20, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, lw=0.8))
    axis.add_patch(mpatches.Rectangle((0.20, 0.325), 0.46, 0.035, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.8))
    axis.text(0.43, 0.607, r"upper sheath $C_{s,u}(t)$", ha="center", va="center", fontsize=6.1)
    axis.text(0.43, 0.46, r"bulk $R_p(t), L_p(t)$", ha="center", va="center", fontsize=6.5)
    axis.text(0.43, 0.342, r"wafer sheath $C_{s,w}(t)$", ha="center", va="center", fontsize=6.1)
    axis.text(0.43, 0.795, "powered showerhead / upper electrode", ha="center", fontsize=6.1)
    axis.text(0.43, 0.205, "300 mm wafer + ESC / lower electrode", ha="center", fontsize=6.1)
    axis.annotate(
        "60 MHz RF",
        xy=(0.19, 0.75),
        xytext=(0.015, 0.82),
        arrowprops={"arrowstyle": "->", "color": BLUE},
        color=BLUE,
        fontsize=6.2,
    )
    axis.annotate(
        "negative DC\npulse",
        xy=(0.20, 0.71),
        xytext=(0.015, 0.63),
        arrowprops={"arrowstyle": "->", "color": INK},
        fontsize=6.2,
    )
    axis.annotate(
        "2 MHz RF",
        xy=(0.19, 0.25),
        xytext=(0.015, 0.17),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
        fontsize=6.2,
    )
    axis.plot([0.74, 0.90], [0.47, 0.47], color=INK, lw=1.0)
    _ground(axis, 0.90, 0.45, 0.8)
    axis.text(0.82, 0.50, "grounded wall", ha="center", fontsize=6.0)
    axis.annotate(
        "held-out: $V_-(t)$ at P_HF",
        xy=(0.70, 0.78),
        xytext=(0.96, 0.96),
        ha="right",
        arrowprops={"arrowstyle": "->", "color": BLUE},
        color=BLUE,
        fontsize=6.0,
    )
    axis.annotate(
        "objective: wafer $V_W(t)$",
        xy=(0.66, 0.25),
        xytext=(0.96, 0.12),
        ha="right",
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
        fontsize=6.0,
    )


def _equivalent_circuit(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    x = 0.64
    ground_x = 0.94
    upper_y = 0.72
    bulk_top_y = 0.56
    bulk_bottom_y = 0.36
    wafer_y = 0.20

    # Main vertical branch, kept in the same top-to-bottom order as the tool.
    axis.plot([x, x], [0.78, upper_y], color=INK, lw=1.0)
    axis.scatter([x], [upper_y], s=18, color=INK, zorder=3)
    _component(axis, x, 0.64, r"$C_{s,u}(t)\parallel1.5$ k$\Omega$", BLUE_LIGHT)
    axis.plot([x, x], [upper_y, 0.666], color=INK, lw=1.0)
    axis.plot([x, x], [0.614, bulk_top_y], color=INK, lw=1.0)
    axis.scatter([x], [bulk_top_y], s=13, color=INK, zorder=3)
    _component(axis, x, 0.49, r"$L_p(t)$", ORANGE_LIGHT)
    axis.plot([x, x], [bulk_top_y, 0.516], color=INK, lw=1.0)
    axis.plot([x, x], [0.464, 0.455], color=INK, lw=1.0)
    _component(axis, x, 0.42, r"$R_p(t)$", ORANGE_LIGHT)
    axis.plot([x, x], [0.394, bulk_bottom_y], color=INK, lw=1.0)
    axis.scatter([x], [bulk_bottom_y], s=13, color=INK, zorder=3)
    _component(axis, x, 0.28, r"$C_{s,w}(t)\parallel2$ k$\Omega$", BLUE_LIGHT)
    axis.plot([x, x], [bulk_bottom_y, 0.306], color=INK, lw=1.0)
    axis.plot([x, x], [0.254, wafer_y], color=INK, lw=1.0)
    axis.scatter([x], [wafer_y], s=18, color=INK, zorder=3)
    axis.text(x + 0.025, upper_y + 0.018, "U: upper electrode", ha="left", fontsize=5.8)
    axis.text(x + 0.025, wafer_y - 0.045, "W: wafer / chuck", ha="left", fontsize=5.8)

    # Upper RF feed and matching network.
    axis.plot([0.03, 0.08], [0.82, 0.82], color=INK, lw=1.0)
    _component(axis, 0.13, 0.82, "60 MHz\n220 Vpk", BLUE_LIGHT)
    _component(axis, 0.29, 0.82, r"50 $\Omega$")
    _component(axis, 0.45, 0.82, "120 pF\n75 nH")
    _component(axis, 0.58, 0.82, "0.6 Ω\n25 nH")
    axis.plot([0.205, 0.215], [0.82, 0.82], color=INK, lw=1.0)
    axis.plot([0.365, 0.375], [0.82, 0.82], color=INK, lw=1.0)
    axis.plot([0.525, 0.505], [0.82, 0.82], color=INK, lw=1.0)
    axis.plot([0.655, x], [0.82, 0.78], color=INK, lw=1.0)
    axis.text(0.30, 0.875, "P_HF reference plane", ha="center", fontsize=5.6, color=BLUE)

    # DC branch joins the upper electrode through an RF choke.
    _component(axis, 0.14, 0.70, "0 to -800 V\n400 kHz", LIGHT)
    _component(axis, 0.32, 0.70, r"250 $\Omega$")
    _component(axis, 0.49, 0.70, r"200 $\mu$H choke")
    axis.plot([0.215, 0.245], [0.70, 0.70], color=INK, lw=1.0)
    axis.plot([0.395, 0.415], [0.70, 0.70], color=INK, lw=1.0)
    axis.plot([0.565, x], [0.70, upper_y], color=INK, lw=1.0)

    # Lower RF feed.
    _component(axis, 0.13, 0.10, "2 MHz\n450 Vpk", ORANGE_LIGHT)
    _component(axis, 0.29, 0.10, "50 Ω\n4.7 nF")
    _component(axis, 0.45, 0.10, "1.35 µH\n0.4 Ω")
    _component(axis, 0.58, 0.10, "60 nH")
    axis.plot([0.205, 0.215], [0.10, 0.10], color=INK, lw=1.0)
    axis.plot([0.365, 0.375], [0.10, 0.10], color=INK, lw=1.0)
    axis.plot([0.525, 0.505], [0.10, 0.10], color=INK, lw=1.0)
    axis.plot([0.655, x], [0.10, wafer_y], color=INK, lw=1.0)

    # Chamber/fixture parasitics and wall loss.
    axis.plot([ground_x, ground_x], [0.12, 0.76], color=MUTED, lw=0.8)
    _ground(axis, ground_x, 0.10, 0.8)
    axis.plot([x, 0.80], [upper_y, upper_y], color=INK, lw=0.9)
    _component(axis, 0.86, upper_y, "90 pF")
    axis.plot([0.935, ground_x], [upper_y, upper_y], color=INK, lw=0.9)
    axis.plot([x, 0.80], [bulk_bottom_y, bulk_bottom_y], color=INK, lw=0.9)
    _component(axis, 0.86, bulk_bottom_y, "5 kΩ")
    axis.plot([0.935, ground_x], [bulk_bottom_y, bulk_bottom_y], color=INK, lw=0.9)
    axis.plot([x, 0.80], [wafer_y, wafer_y], color=INK, lw=0.9)
    _component(axis, 0.86, wafer_y, "140 pF")
    axis.plot([0.935, ground_x], [wafer_y, wafer_y], color=INK, lw=0.9)
    axis.text(ground_x, 0.79, "chamber ground", ha="center", fontsize=5.5, color=MUTED)


def figure_apparatus() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE, gridspec_kw={"width_ratios": [0.92, 1.28]})
    add_panel_title(axes[0], "(a)", "300 mm etch-CCP problem and outputs")
    _apparatus(axes[0])
    add_panel_title(axes[1], "(b)", "Top-to-bottom external + plasma equivalent circuit")
    _equivalent_circuit(axes[1])
    add_figure_title(
        fig,
        "Dual-frequency + DC-superposed etch CCP with dynamic plasma impedance",
        "Hardware and circuit share the same vertical order; the matching, feed, choke, blocking and stray elements remain explicit.",
    )
    add_figure_footer(
        fig,
        "Circuit-level benchmark, not a plasma-chemistry model. Blue: sheath/electrode coupling; orange: bulk/wafer path. Diagram not to scale.",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.10, wspace=0.08)
    return fig


def _target_time_us(target: dict[str, np.ndarray]) -> np.ndarray:
    return 1e6 * (target["time_s"] - PLOT_START_S)


def _figure_legend(fig: plt.Figure, handles: list[Any], labels: list[str], columns: int | None = None) -> None:
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.815),
        ncol=columns or len(labels),
    )


def _apply_time_axes(axes: np.ndarray[Any, Any]) -> None:
    for axis in axes.flat:
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        _format_axes(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time within 400 kHz pulse period (µs)")


def figure_inputs(sources: dict[str, Path]) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    time_us = _target_time_us(target)
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)

    zoom = time_us <= 0.20
    add_panel_title(axes[0, 0], "(a)", "Upper input: 60 MHz RF (200 ns zoom)")
    axes[0, 0].plot(time_us[zoom], target["upper_hf_source_voltage_V"][zoom], color=BLUE, lw=0.9)
    axes[0, 0].set_xlim(0.0, 0.20)
    axes[0, 0].set_ylabel(r"$V_{HF}$ (V)")
    axes[0, 0].set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])

    add_panel_title(axes[0, 1], "(b)", "Lower input: 2 MHz RF")
    axes[0, 1].plot(time_us, target["lower_lf_source_voltage_V"], color=ORANGE, lw=0.9)
    axes[0, 1].set_ylabel(r"$V_{LF}$ (V)")

    add_panel_title(axes[1, 0], "(c)", "Upper negative-DC pulse")
    axes[1, 0].plot(time_us, target["upper_dc_source_voltage_V"], color=INK, lw=1.2)
    axes[1, 0].set_ylabel(r"$V_{DC}$ (V)")

    add_panel_title(axes[1, 1], "(d)", "Prescribed plasma-state envelope")
    axes[1, 1].plot(time_us, target["plasma_envelope"], color=BLUE, lw=1.5)
    axes[1, 1].set_ylabel(r"$s(t)$")
    axes[1, 1].set_ylim(-0.08, 1.08)

    add_panel_title(axes[2, 0], "(e)", "Bulk resistance and inertia")
    axes[2, 0].plot(time_us, target["plasma_resistance_ohm"], color=ORANGE, lw=1.4, label=r"$R_p$ (Ω)")
    ax_l = axes[2, 0].twinx()
    ax_l.plot(time_us, 1e9 * target["plasma_inductance_H"], color=BLUE, lw=1.2, label=r"$L_p$ (nH)")
    axes[2, 0].set_ylabel(r"$R_p$ (Ω)")
    ax_l.tick_params(axis="y", colors=BLUE)
    ax_l.spines["top"].set_visible(False)
    handles_left, labels_left = axes[2, 0].get_legend_handles_labels()
    handles_right, labels_right = ax_l.get_legend_handles_labels()
    axes[2, 0].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        loc="center right",
    )

    add_panel_title(axes[2, 1], "(f)", "Upper and wafer sheath capacitance")
    axes[2, 1].plot(time_us, 1e12 * target["upper_sheath_capacitance_F"], color=BLUE, lw=1.4, label=r"$C_{s,u}$")
    axes[2, 1].plot(time_us, 1e12 * target["wafer_sheath_capacitance_F"], color=ORANGE, lw=1.4, label=r"$C_{s,w}$")
    axes[2, 1].set_ylabel("Capacitance (pF)")
    axes[2, 1].legend(frameon=False, loc="upper right")

    for axis in axes.flat:
        _format_axes(axis)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]):
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
    fig.text(0.50, 0.095, "Time within 400 kHz pulse period (µs)", ha="center", va="center", fontsize=8.4)
    add_figure_title(
        fig,
        "Known excitations and prescribed dynamic plasma elements",
        r"The envelope moves $R_p$: 35→15 Ω, $L_p$: 350→160 nH, $C_{s,u}$: 260→520 pF and $C_{s,w}$: 360→720 pF.",
    )
    add_figure_footer(
        fig,
        r"Constitutive laws used by both solvers: $q_s=C_s(t)v_s$ and $\phi_p=L_p(t)i_p$; therefore $i_s=dq_s/dt$ and $v_L=d\phi_p/dt$.",
    )
    fig.subplots_adjust(left=0.08, right=0.94, top=0.80, bottom=0.15, hspace=0.72, wspace=0.42)
    return fig


def figure_forward(sources: dict[str, Path], data: dict[str, Any]) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    observed = _read_numeric_csv(sources["forward_waveform"])
    time_us = _target_time_us(target)
    wafer = _aligned(target, observed, "wafer_voltage_V")
    reflection = _reflected(target, observed)
    current = _aligned(target, observed, "bulk_current_A")
    charge = target["upper_sheath_capacitance_F"] * _aligned(target, observed, "upper_sheath_voltage_V")
    flux = target["plasma_inductance_H"] * current
    truth_handle = plt.Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (4, 2)))
    ngspice_handle = plt.Line2D([], [], color=BLUE, lw=0.8)
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)
    series = [
        ("Wafer voltage objective", target["wafer_voltage_V"], wafer, r"$V_W$ (V)"),
        ("Upper reflected wave (held-out observable)", target["upper_reflected_voltage_V"], reflection, r"$V_-$ (V)"),
        ("Plasma bulk current", target["bulk_current_A"], current, r"$I_p$ (A)"),
        ("Upper-sheath charge", 1e9 * target["upper_sheath_charge_C"], 1e9 * charge, r"$q_{s,u}$ (nC)"),
        ("Plasma-inductor flux linkage", 1e6 * target["plasma_flux_Wb"], 1e6 * flux, r"$\phi_p$ (µWb)"),
    ]
    for index, (title, truth, result, ylabel) in enumerate(series):
        axis = axes.flat[index]
        add_panel_title(axis, f"({chr(97 + index)})", title)
        axis.plot(time_us, truth, color=ORANGE, lw=1.6, ls=(0, (4, 2)))
        axis.plot(time_us, result, color=BLUE, lw=0.75)
        axis.set_ylabel(ylabel)

    add_panel_title(axes[2, 1], "(f)", "Residuals: orange wafer; blue reflection")
    axes[2, 1].plot(time_us, wafer - target["wafer_voltage_V"], color=ORANGE, lw=0.8, label=r"$V_W$")
    axes[2, 1].plot(time_us, reflection - target["upper_reflected_voltage_V"], color=BLUE, lw=0.8, label=r"$V_-$")
    axes[2, 1].set_ylabel("Error (V)")
    _apply_time_axes(axes)
    _figure_legend(fig, [truth_handle, ngspice_handle], ["independent charge/flux MNA", "ngspice 46"])
    add_figure_title(
        fig,
        "Forward validity: independent state equations and ngspice agree",
        "Voltage, current, charge and flux are compared directly over one complete 400 kHz pulse period.",
    )
    forward = data["forward"]
    add_figure_footer(
        fig,
        f"Wafer RMSE={forward['wafer_voltage']['rmse_V']:.3f} V (nRMSE={forward['wafer_voltage']['normalized_rmse']:.3e}); "
        f"reflection RMSE={forward['upper_reflected_voltage']['rmse_V']:.3f} V "
        f"(nRMSE={forward['upper_reflected_voltage']['normalized_rmse']:.3e}).\n"
        f"Bulk-current RMSE={forward['bulk_current']['rmse_A']:.3e} A; all curves use archived simulation artifacts.",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.75, bottom=0.16, hspace=0.74, wspace=0.26)
    return fig


def figure_impedance(sources: dict[str, Path]) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    time_us = _target_time_us(target)
    profiles = {
        "Rp_ohm": target["plasma_resistance_ohm"],
        "Lp_H": target["plasma_inductance_H"],
        "Csu_F": target["upper_sheath_capacitance_F"],
        "Csw_F": target["wafer_sheath_capacitance_F"],
    }
    z_2m = _branch_impedance(profiles, 2.0e6)
    z_60m = _branch_impedance(profiles, 60.0e6)
    fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE)
    for index, (axis, impedance, frequency) in enumerate(((axes[0, 0], z_2m, "2 MHz"), (axes[0, 1], z_60m, "60 MHz"))):
        add_panel_title(axis, f"({chr(97 + index)})", f"Main-branch impedance at {frequency}")
        axis.plot(time_us, impedance.real, color=ORANGE, lw=1.35, label="Re(Z)")
        axis.plot(time_us, impedance.imag, color=BLUE, lw=1.25, label="Im(Z)")
        axis.axhline(0.0, color=MUTED, lw=0.6)
        axis.set_ylabel("Impedance (Ω)")
        axis.legend(frameon=False, loc="upper right")
    add_panel_title(axes[1, 0], "(c)", "2 MHz impedance trajectory")
    axes[1, 0].plot(z_2m.real, z_2m.imag, color=BLUE, lw=1.3)
    axes[1, 0].scatter(
        [z_2m.real[0], z_2m.real[len(z_2m) // 2]],
        [z_2m.imag[0], z_2m.imag[len(z_2m) // 2]],
        color=[MUTED, ORANGE],
        s=25,
        zorder=3,
    )
    axes[1, 0].set_xlabel("Re(Z) (Ω)")
    axes[1, 0].set_ylabel("Im(Z) (Ω)")
    axes[1, 0].set_xlim(30.0, 125.0)
    axes[1, 0].set_xticks([40.0, 60.0, 80.0, 100.0, 120.0])
    add_panel_title(axes[1, 1], "(d)", "60 MHz impedance trajectory")
    axes[1, 1].plot(z_60m.real, z_60m.imag, color=BLUE, lw=1.3)
    axes[1, 1].scatter(
        [z_60m.real[0], z_60m.real[len(z_60m) // 2]],
        [z_60m.imag[0], z_60m.imag[len(z_60m) // 2]],
        color=[MUTED, ORANGE],
        s=25,
        zorder=3,
    )
    axes[1, 1].set_xlabel("Re(Z) (Ω)")
    axes[1, 1].set_ylabel("Im(Z) (Ω)")
    axes[1, 1].set_xlim(14.0, 37.0)
    axes[1, 1].set_xticks([15.0, 20.0, 25.0, 30.0, 35.0])
    for axis in axes.flat:
        _format_axes(axis)
    for axis in axes[0, :]:
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        axis.set_xlabel("Time within pulse period (µs)")
    add_figure_title(
        fig,
        "Electrical interpretation of the time-varying plasma branch",
        r"$Z_{main}=Z_{s,u}+R_p+j\omega L_p+Z_{s,w}$ with $Z_s=(1/R_s+j\omega C_s)^{-1}$; external matching is excluded here.",
    )
    add_figure_footer(
        fig,
        "Gray marker: pulse start (low-density state); orange marker: on-state.\n"
        "Frozen-time impedance explains RF loading but is not a kinetic plasma solution.",
    )
    fig.subplots_adjust(left=0.09, right=0.955, top=0.80, bottom=0.15, hspace=0.52, wspace=0.30)
    return fig


def _scaled_parameter_values(values: dict[str, float]) -> list[float]:
    return [values["Rp_on_ohm"], 1e9 * values["Lp_on_H"], 1e12 * values["Csu_on_F"], 1e12 * values["Csw_on_F"]]


def figure_inverse(sources: dict[str, Path], data: dict[str, Any]) -> plt.Figure:
    target = _read_numeric_csv(sources["target_observables"])
    initial = _read_numeric_csv(sources["inverse_initial_waveform"])
    selected = _read_numeric_csv(sources["inverse_selected_waveform"])
    time_us = _target_time_us(target)
    initial_wafer = _aligned(target, initial, "wafer_voltage_V")
    selected_wafer = _aligned(target, selected, "wafer_voltage_V")
    selected_reflection = _reflected(target, selected)
    result = data["inverse"]
    true_profiles = _profiles(target["time_s"], result["truth_values"])
    selected_profiles = _profiles(target["time_s"], result["selected_values"])
    candidates = result["candidates"]
    trials = np.asarray([row["trial"] + 1 for row in candidates])
    losses = np.asarray([row["loss"] for row in candidates])
    best_so_far = np.minimum.accumulate(losses)
    selected_trial = 1 + next(
        row["trial"] for row in candidates if row["candidate_id"] == result["selected_candidate_id"]
    )
    fig, axes = plt.subplots(3, 2, figsize=PAGE_SIZE)
    truth_handle = plt.Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (4, 2)))
    initial_handle = plt.Line2D([], [], color=MUTED, lw=0.7)
    selected_handle = plt.Line2D([], [], color=BLUE, lw=0.85)

    add_panel_title(axes[0, 0], "(a)", "Fitted objective: wafer voltage only")
    axes[0, 0].plot(time_us, initial_wafer, color=MUTED, lw=0.65)
    axes[0, 0].plot(time_us, target["wafer_voltage_V"], color=ORANGE, lw=1.6, ls=(0, (4, 2)))
    axes[0, 0].plot(time_us, selected_wafer, color=BLUE, lw=0.75)
    axes[0, 0].set_ylabel(r"$V_W$ (V)")

    add_panel_title(axes[0, 1], "(b)", "Held-out check: upper reflected wave")
    axes[0, 1].plot(time_us, target["upper_reflected_voltage_V"], color=ORANGE, lw=1.6, ls=(0, (4, 2)))
    axes[0, 1].plot(time_us, selected_reflection, color=BLUE, lw=0.75)
    axes[0, 1].set_ylabel(r"$V_-$ (V)")

    add_panel_title(axes[1, 0], "(c)", "Recovered bulk profiles")
    axes[1, 0].plot(time_us, true_profiles["Rp_ohm"], color=ORANGE, lw=1.5, ls=(0, (4, 2)))
    axes[1, 0].plot(time_us, selected_profiles["Rp_ohm"], color=BLUE, lw=0.8)
    axes[1, 0].set_ylabel(r"$R_p$ (Ω)")
    ax_lp = axes[1, 0].twinx()
    ax_lp.plot(time_us, 1e9 * true_profiles["Lp_H"], color=ORANGE, lw=1.3, ls=(0, (4, 2)))
    ax_lp.plot(time_us, 1e9 * selected_profiles["Lp_H"], color=BLUE, lw=0.7)
    ax_lp.tick_params(axis="y", colors=BLUE)
    ax_lp.spines["top"].set_visible(False)
    axes[1, 0].text(
        0.98,
        0.08,
        r"right scale: $L_p$ (nH)",
        transform=axes[1, 0].transAxes,
        ha="right",
        color=BLUE,
        fontsize=6.0,
    )

    add_panel_title(axes[1, 1], "(d)", "Recovered sheath-capacitance profiles")
    axes[1, 1].plot(
        time_us, 1e12 * true_profiles["Csu_F"], color=ORANGE, lw=1.5, ls=(0, (4, 2)), label=r"truth $C_{s,u}$"
    )
    axes[1, 1].plot(time_us, 1e12 * selected_profiles["Csu_F"], color=BLUE, lw=0.8, label=r"fit $C_{s,u}$")
    axes[1, 1].plot(
        time_us, 1e12 * true_profiles["Csw_F"], color=ORANGE, lw=1.1, ls=(0, (1, 2)), label=r"truth $C_{s,w}$"
    )
    axes[1, 1].plot(
        time_us, 1e12 * selected_profiles["Csw_F"], color=BLUE, lw=0.7, ls=(0, (1, 2)), label=r"fit $C_{s,w}$"
    )
    axes[1, 1].set_ylabel("Capacitance (pF)")

    add_panel_title(axes[2, 0], "(e)", "Exact-grid loss history (81/81 solved)")
    axes[2, 0].semilogy(trials, losses, color=MUTED, marker=".", ms=2.2, lw=0.45)
    axes[2, 0].step(trials, best_so_far, where="post", color=BLUE, lw=1.25)
    axes[2, 0].scatter([selected_trial], [result["selected_loss"]], color=ORANGE, s=24, zorder=4)
    axes[2, 0].set_xlim(1.0, 81.0)
    axes[2, 0].set_xticks([1, 20, 40, 60, 81])
    axes[2, 0].set_xlabel("Evaluated candidate")
    axes[2, 0].set_ylabel("Normalized RMSE")

    add_panel_title(axes[2, 1], "(f)", "On-state values: initial, truth, selected")
    x = np.arange(4)
    width = 0.24
    axes[2, 1].bar(
        x - width,
        _scaled_parameter_values(result["initial_values"]),
        width,
        color="#c8ced3",
        edgecolor=MUTED,
        linewidth=0.6,
    )
    axes[2, 1].bar(
        x,
        _scaled_parameter_values(result["truth_values"]),
        width,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=0.8,
    )
    axes[2, 1].bar(
        x + width,
        _scaled_parameter_values(result["selected_values"]),
        width,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.8,
    )
    axes[2, 1].set_xticks(x, ["$R_p$\nΩ", "$L_p$\nnH", "$C_{s,u}$\npF", "$C_{s,w}$\npF"])
    axes[2, 1].set_ylabel("Value in displayed unit")

    for axis in axes.flat:
        _format_axes(axis)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
    _figure_legend(
        fig,
        [truth_handle, initial_handle, selected_handle],
        ["independent truth / target", "first grid candidate", "selected ngspice candidate"],
    )
    add_figure_title(
        fig,
        "Inverse validity: one measured waveform recovers four bounded on-state values",
        "Only wafer voltage contributes to loss; upper reflection is retained as a genuinely unused circuit-level validation output.",
    )
    add_figure_footer(
        fig,
        f"Selected: Rp={result['selected_values']['Rp_on_ohm']:g} Ω, Lp={1e9 * result['selected_values']['Lp_on_H']:.0f} nH, "
        f"Csu={1e12 * result['selected_values']['Csu_on_F']:.0f} pF, Csw={1e12 * result['selected_values']['Csw_on_F']:.0f} pF. "
        f"Objective nRMSE={result['objective_error']['normalized_rmse']:.3e}; held-out nRMSE={result['held_out_error']['normalized_rmse']:.3e}.",
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.75, bottom=0.17, hspace=0.75, wspace=0.34)
    return fig


def _export(figure: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    assert_text_inside_canvas(figure)
    svg_path = output / f"{stem}.svg"
    figure.savefig(
        svg_path,
        metadata={
            "Title": stem.replace("-", " "),
            "Creator": "PCD etch-CCP evidence generator",
            "Description": "Dynamic-impedance dual-RF and DC-superposed etch-CCP evidence",
            "Date": None,
        },
    )
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output / f"{stem}.png", dpi=300, metadata={"Software": "PCD etch-CCP evidence generator"})
    pdf.savefig(figure)
    plt.close(figure)


def generate(
    forward_case: Path,
    inverse_case: Path,
    forward_root: Path,
    inverse_root: Path,
    output: Path,
    pdf_output: Path,
) -> dict[str, Any]:
    sources = collect_sources(forward_case, inverse_case, forward_root, inverse_root)
    data = build_figure_data(sources)
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT
    figures = [
        ("01-apparatus-and-equivalent-circuit", figure_apparatus()),
        ("02-inputs-and-dynamic-elements", figure_inputs(sources)),
        ("03-forward-conformance", figure_forward(sources, data)),
        ("04-dynamic-branch-impedance", figure_impedance(sources)),
        ("05-inverse-identification", figure_inverse(sources, data)),
    ]
    with PdfPages(
        pdf_output,
        metadata={
            "Title": "Dynamic-impedance dual-RF and DC-superposed etch-CCP study",
            "Author": "PCD benchmark generator",
            "Subject": "Independent forward conformance and bounded R-L-sheath-C identification",
            "Keywords": "CCP, ngspice, RF, DC superposition, dynamic impedance, inverse problem",
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
    parser.add_argument("--inverse-case", type=Path, default=DEFAULT_INVERSE_CASE)
    parser.add_argument("--forward-root", type=Path, default=DEFAULT_FORWARD_ROOT)
    parser.add_argument("--inverse-root", type=Path, default=DEFAULT_INVERSE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    args = parser.parse_args(argv)
    result = generate(
        args.forward_case.resolve(),
        args.inverse_case.resolve(),
        args.forward_root.resolve(),
        args.inverse_root.resolve(),
        args.output.resolve(),
        args.pdf_output.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
