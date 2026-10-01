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
DEFAULT_FORWARD_ROOT = ROOT / "runs" / "etch_ccp_dcs_forward_v7_20261001"
DEFAULT_INVERSE_ROOT = ROOT / "runs" / "etch_ccp_dcs_inverse_v3_20261001"
DEFAULT_OUTPUT = HERE / "etch_ccp_dcs"
DEFAULT_PDF_OUTPUT = ROOT / "output" / "pdf" / "etch-ccp-dcs-dynamic-impedance-study.pdf"
SVG_HASHSALT = "pcd-etch-ccp-dcs-v2"

PERIOD_S = 2.5e-6
PLOT_START_S = 5.0e-6
PLOT_STOP_S = 7.5e-6
Z0_OHM = 50.0
TRUE_PARAMETERS = {
    "Rp_on_ohm": 15.0,
    "Lp_on_H": 1.6e-8,
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
        "Lp_H": 3.5e-8 + (values["Lp_on_H"] - 3.5e-8) * env,
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


def _physical_scale_checks() -> dict[str, Any]:
    """Return transparent geometry-based plausibility checks, not calibration."""

    epsilon_0 = 8.8541878128e-12
    electron_mass = 9.1093837139e-31
    elementary_charge = 1.602176634e-19
    area_m2 = math.pi * 0.15**2
    bulk_length_m = 18.0e-3
    capacitances = np.asarray([2.6e-10, 5.2e-10, 3.6e-10, 7.2e-10])
    sheath_mm = 1e3 * epsilon_0 * area_m2 / capacitances
    inductances = np.asarray([35.0e-9, 16.0e-9])
    density_m3 = electron_mass * bulk_length_m / (elementary_charge**2 * area_m2 * inductances)
    return {
        "electrode_diameter_m": 0.300,
        "electrode_area_m2": area_m2,
        "assumed_bulk_length_m": bulk_length_m,
        "sheath_thickness_range_mm_from_C_parallel_plate": [float(np.min(sheath_mm)), float(np.max(sheath_mm))],
        "electron_density_range_m-3_from_L_uniform_bulk": [float(np.min(density_m3)), float(np.max(density_m3))],
        "collision_frequency_s-1_from_R_over_L": [15.0 / 16.0e-9, 35.0 / 35.0e-9],
        "interpretation": "order-of-magnitude consistency check only; geometry and plasma are not self-consistently solved",
    }


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
            "basis": {
                "literature_grounded": [
                    "300 mm, 20 mm gap, upper 60 MHz and lower 2 MHz apparatus class: Kim et al., JJAP 54 01AE07 (2015)",
                    "negative DC on the upper 60 MHz electrode: Yamaguchi et al., J. Phys. D 45 025203 (2012)",
                    "series sheath-C / bulk-L-R interpretation: Kim, Lee, and Hong, Electronics 14 2022 (2025)",
                    "sub-microsecond transient impedance observability: Lee et al., Rev. Sci. Instrum. 86 083505 (2015)",
                    "explicit generator, matcher, and stray coupling: Schmidt et al., Plasma Sources Sci. Technol. 27 105017 (2018)",
                ],
                "benchmark_specific": [
                    "the 60 MHz / 2 MHz / pulsed-DC combination is a synthesis, not one published reactor",
                    "400 kHz DC-pulse envelope and ideal source amplitudes are engineering test inputs",
                    "matching, feedthrough, stray, and leakage values are plausible but not fitted to a production tool",
                    "R/L/C time profiles and the 3x3x3x3 identification grid are synthetic known-truth evidence",
                ],
                "scale_checks": _physical_scale_checks(),
            },
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


def _wire(axis: plt.Axes, x0: float, y0: float, x1: float, y1: float, *, color: str = INK) -> None:
    axis.plot([x0, x1], [y0, y1], color=color, lw=1.15, solid_capstyle="round")


def _node(axis: plt.Axes, x: float, y: float) -> None:
    axis.scatter([x], [y], s=15, color=INK, zorder=5)


def _resistor_h(axis: plt.Axes, x0: float, x1: float, y: float) -> None:
    lead = 0.10 * (x1 - x0)
    _wire(axis, x0, y, x0 + lead, y)
    _wire(axis, x1 - lead, y, x1, y)
    xs = np.linspace(x0 + lead, x1 - lead, 9)
    ys = np.asarray([y, y + 0.015, y - 0.015, y + 0.015, y - 0.015, y + 0.015, y - 0.015, y + 0.015, y])
    axis.plot(xs, ys, color=INK, lw=1.15)


def _resistor_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    lead = 0.10 * (y1 - y0)
    _wire(axis, x, y0, x, y0 + lead)
    _wire(axis, x, y1 - lead, x, y1)
    ys = np.linspace(y0 + lead, y1 - lead, 9)
    xs = np.asarray([x, x + 0.014, x - 0.014, x + 0.014, x - 0.014, x + 0.014, x - 0.014, x + 0.014, x])
    axis.plot(xs, ys, color=INK, lw=1.15)


def _capacitor_h(axis: plt.Axes, x0: float, x1: float, y: float) -> None:
    middle = 0.5 * (x0 + x1)
    _wire(axis, x0, y, middle - 0.010, y)
    _wire(axis, middle + 0.010, y, x1, y)
    axis.plot([middle - 0.010, middle - 0.010], [y - 0.028, y + 0.028], color=INK, lw=1.15)
    axis.plot([middle + 0.010, middle + 0.010], [y - 0.028, y + 0.028], color=INK, lw=1.15)


def _capacitor_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    middle = 0.5 * (y0 + y1)
    _wire(axis, x, y0, x, middle - 0.010)
    _wire(axis, x, middle + 0.010, x, y1)
    axis.plot([x - 0.028, x + 0.028], [middle - 0.010, middle - 0.010], color=INK, lw=1.15)
    axis.plot([x - 0.028, x + 0.028], [middle + 0.010, middle + 0.010], color=INK, lw=1.15)


def _inductor_h(axis: plt.Axes, x0: float, x1: float, y: float) -> None:
    lead = 0.10 * (x1 - x0)
    _wire(axis, x0, y, x0 + lead, y)
    _wire(axis, x1 - lead, y, x1, y)
    xs = np.linspace(x0 + lead, x1 - lead, 120)
    ys = y + 0.014 * np.sin(np.linspace(0.0, 8.0 * math.pi, xs.size))
    axis.plot(xs, ys, color=INK, lw=1.15)


def _inductor_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    lead = 0.10 * (y1 - y0)
    _wire(axis, x, y0, x, y0 + lead)
    _wire(axis, x, y1 - lead, x, y1)
    ys = np.linspace(y0 + lead, y1 - lead, 120)
    xs = x + 0.014 * np.sin(np.linspace(0.0, 8.0 * math.pi, ys.size))
    axis.plot(xs, ys, color=INK, lw=1.15)


def _source(axis: plt.Axes, x: float, y: float, *, pulse: bool = False) -> None:
    axis.add_patch(mpatches.Circle((x, y), 0.030, facecolor=WHITE, edgecolor=INK, linewidth=1.1))
    if pulse:
        axis.plot(
            [x - 0.018, x - 0.008, x - 0.008, x + 0.008, x + 0.008, x + 0.018],
            [y - 0.008, y - 0.008, y + 0.010, y + 0.010, y - 0.008, y - 0.008],
            color=INK,
            lw=0.8,
        )
    else:
        xx = np.linspace(x - 0.018, x + 0.018, 50)
        axis.plot(xx, y + 0.010 * np.sin((xx - x) / 0.018 * math.pi), color=INK, lw=0.8)


def _parallel_rc_v(axis: plt.Axes, x: float, y0: float, y1: float) -> None:
    left = x - 0.027
    right = x + 0.027
    _wire(axis, left, y0, right, y0)
    _wire(axis, left, y1, right, y1)
    _capacitor_v(axis, left, y0, y1)
    _resistor_v(axis, right, y0, y1)


def _apparatus(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    chamber = mpatches.FancyBboxPatch(
        (0.18, 0.09),
        0.62,
        0.80,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor="#fbfcfd",
        edgecolor=INK,
        linewidth=1.25,
    )
    axis.add_patch(chamber)
    axis.text(0.49, 0.925, "grounded process chamber", ha="center", fontsize=6.4, color=MUTED)

    # Gas showerhead and upper powered electrode.
    axis.add_patch(mpatches.Rectangle((0.29, 0.75), 0.40, 0.050, facecolor="#737d85", edgecolor=INK, lw=0.8))
    for x in np.linspace(0.32, 0.66, 7):
        axis.plot([x, x], [0.75, 0.725], color=MUTED, lw=0.75)
    axis.annotate(
        "process gas",
        xy=(0.49, 0.81),
        xytext=(0.49, 0.875),
        ha="center",
        fontsize=5.9,
        arrowprops={"arrowstyle": "->", "color": MUTED},
    )
    axis.text(0.49, 0.705, "Si showerhead / upper electrode", ha="center", fontsize=6.1)

    # The two sheaths and bulk preserve the physical top-to-bottom order.
    axis.add_patch(mpatches.Rectangle((0.29, 0.635), 0.40, 0.035, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.8))
    plasma = mpatches.FancyBboxPatch(
        (0.29, 0.365), 0.40, 0.245, boxstyle="round,pad=0.010", facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.9
    )
    axis.add_patch(plasma)
    axis.add_patch(mpatches.Rectangle((0.29, 0.315), 0.40, 0.035, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.8))
    axis.text(0.49, 0.652, r"upper sheath: $C_{s,u}(t)$", ha="center", va="center", fontsize=5.9, color=BLUE)
    axis.text(0.49, 0.487, r"plasma bulk: $R_p(t)$, $L_p(t)$", ha="center", va="center", fontsize=6.4)
    axis.text(0.49, 0.332, r"wafer sheath: $C_{s,w}(t)$", ha="center", va="center", fontsize=5.9, color=BLUE)

    # Wafer, ESC, cooling plate, and pumping path.
    axis.add_patch(mpatches.Rectangle((0.28, 0.275), 0.42, 0.025, facecolor="#c5ccd2", edgecolor=INK, lw=0.65))
    axis.add_patch(mpatches.Rectangle((0.26, 0.225), 0.46, 0.045, facecolor="#7b848c", edgecolor=INK, lw=0.8))
    axis.add_patch(mpatches.Rectangle((0.30, 0.175), 0.38, 0.040, facecolor="#dbe0e4", edgecolor=INK, lw=0.65))
    axis.text(0.49, 0.290, "300 mm wafer", ha="center", va="center", fontsize=5.8)
    axis.text(0.49, 0.245, "ESC / lower electrode", ha="center", va="center", fontsize=5.8, color=WHITE)
    axis.text(0.49, 0.195, "He backside + coolant", ha="center", va="center", fontsize=5.6)
    axis.plot([0.80, 0.88], [0.18, 0.18], color=INK, lw=1.0)
    axis.annotate(
        "turbo pump / throttle",
        xy=(0.88, 0.18),
        xytext=(0.98, 0.12),
        ha="right",
        fontsize=5.6,
        arrowprops={"arrowstyle": "->", "color": MUTED},
        color=MUTED,
    )

    # Literature-derived electrode spacing.
    axis.annotate(
        "", xy=(0.735, 0.315), xytext=(0.735, 0.75), arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 0.8}
    )
    axis.text(0.748, 0.535, "20 mm gap [R1]", rotation=90, va="center", fontsize=5.7, color=MUTED)

    # Power paths and measurement planes.
    _box(axis, 0.015, 0.765, 0.12, 0.070, "60 MHz RF\n220 Vpk", facecolor=BLUE_LIGHT, edgecolor=BLUE, fontsize=5.7)
    _box(axis, 0.015, 0.655, 0.12, 0.070, "-800 V DC\npulse", facecolor=LIGHT, edgecolor=MUTED, fontsize=5.7)
    _box(axis, 0.015, 0.215, 0.12, 0.070, "2 MHz RF\n450 Vpk", facecolor=ORANGE_LIGHT, edgecolor=ORANGE, fontsize=5.7)
    axis.annotate("", xy=(0.29, 0.775), xytext=(0.135, 0.800), arrowprops={"arrowstyle": "->", "color": BLUE})
    axis.annotate("", xy=(0.29, 0.755), xytext=(0.135, 0.690), arrowprops={"arrowstyle": "->", "color": MUTED})
    axis.annotate("", xy=(0.26, 0.247), xytext=(0.135, 0.250), arrowprops={"arrowstyle": "->", "color": ORANGE})
    axis.text(0.145, 0.850, "matcher + feed", fontsize=5.5, color=BLUE)
    axis.text(0.145, 0.635, "bias tee + choke", fontsize=5.5, color=MUTED)
    axis.text(0.145, 0.205, "matcher + blocking C", fontsize=5.5, color=ORANGE)
    axis.annotate(
        r"held-out $V_-(t)$ at $P_{HF}$",
        xy=(0.69, 0.775),
        xytext=(0.98, 0.845),
        ha="right",
        arrowprops={"arrowstyle": "->", "color": BLUE},
        color=BLUE,
        fontsize=5.8,
    )
    axis.annotate(
        r"objective $V_W(t)$",
        xy=(0.70, 0.290),
        xytext=(0.98, 0.285),
        ha="right",
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
        fontsize=5.8,
    )
    axis.text(
        0.82,
        0.56,
        "hybrid benchmark:\n[R1] 60/2 MHz geometry\n[R2] upper negative DC",
        fontsize=5.6,
        va="center",
        color=MUTED,
    )


def _equivalent_circuit(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    x = 0.70
    ground_x = 0.965
    upper_y = 0.70
    bulk_top_y = 0.55
    bulk_mid_y = 0.42
    bulk_bottom_y = 0.31
    wafer_y = 0.17

    # Explicit upper 60 MHz source, 50 ohm port, matcher, and feedthrough.
    y_hf = 0.86
    _wire(axis, 0.01, y_hf, 0.035, y_hf)
    _source(axis, 0.065, y_hf)
    _wire(axis, 0.095, y_hf, 0.115, y_hf)
    _resistor_h(axis, 0.115, 0.215, y_hf)
    axis.axvline(0.225, ymin=0.80, ymax=0.94, color=BLUE, lw=0.8, ls=(0, (3, 2)))
    _capacitor_h(axis, 0.235, 0.315, y_hf)
    _inductor_h(axis, 0.330, 0.435, y_hf)
    _resistor_h(axis, 0.450, 0.525, y_hf)
    _inductor_h(axis, 0.540, 0.635, y_hf)
    _wire(axis, 0.635, y_hf, x, upper_y)
    axis.text(0.065, 0.915, "60 MHz\n220 Vpk", ha="center", va="bottom", fontsize=5.5)
    axis.text(0.165, 0.900, r"$R_{g,H}=50\,\Omega$", ha="center", fontsize=5.4)
    axis.text(0.275, 0.815, "120 pF", ha="center", fontsize=5.2)
    axis.text(0.383, 0.900, "75 nH", ha="center", fontsize=5.2)
    axis.text(0.488, 0.815, r"0.6 $\Omega$", ha="center", fontsize=5.2)
    axis.text(0.588, 0.900, "25 nH", ha="center", fontsize=5.2)
    axis.text(0.225, 0.945, r"$P_{HF}$ / $V_-$", ha="center", fontsize=5.5, color=BLUE)

    # Negative-DC branch and RF choke join the upper electrode.
    y_dc = 0.755
    _wire(axis, 0.01, y_dc, 0.035, y_dc)
    _source(axis, 0.065, y_dc, pulse=True)
    _wire(axis, 0.095, y_dc, 0.125, y_dc)
    _resistor_h(axis, 0.125, 0.235, y_dc)
    _inductor_h(axis, 0.260, 0.475, y_dc)
    _wire(axis, 0.475, y_dc, x, upper_y)
    axis.text(0.065, 0.705, "0 to -800 V\n400 kHz", ha="center", va="top", fontsize=5.4)
    axis.text(0.180, 0.715, r"$R_{DC}=250\,\Omega$", ha="center", fontsize=5.2)
    axis.text(0.368, 0.715, r"$L_{choke}=200\,\mu H$", ha="center", fontsize=5.2)

    # Dynamic plasma branch in the same order as the chamber cross-section.
    _node(axis, x, upper_y)
    _parallel_rc_v(axis, x, bulk_top_y, upper_y)
    axis.text(0.744, 0.625, r"$C_{s,u}(t)$", fontsize=5.5, color=BLUE, va="center")
    axis.text(0.744, 0.596, r"$R_{s,u}=1.5$ k$\Omega$", fontsize=5.0, va="center")
    _node(axis, x, bulk_top_y)
    _inductor_v(axis, x, bulk_mid_y, bulk_top_y)
    axis.text(0.744, 0.486, r"$L_p(t)$", fontsize=5.6, color=ORANGE, va="center")
    _resistor_v(axis, x, bulk_bottom_y, bulk_mid_y)
    axis.text(0.744, 0.365, r"$R_p(t)$", fontsize=5.6, color=ORANGE, va="center")
    _node(axis, x, bulk_bottom_y)
    _parallel_rc_v(axis, x, wafer_y, bulk_bottom_y)
    axis.text(0.744, 0.250, r"$C_{s,w}(t)$", fontsize=5.5, color=BLUE, va="center")
    axis.text(0.744, 0.221, r"$R_{s,w}=2$ k$\Omega$", fontsize=5.0, va="center")
    _node(axis, x, wafer_y)
    axis.text(0.718, 0.724, "U", fontsize=5.6, fontweight="bold")
    axis.text(0.718, 0.137, "W / wafer", fontsize=5.6, fontweight="bold")

    # Lower 2 MHz source, blocking capacitor, matching inductor, and feed.
    y_lf = 0.075
    _wire(axis, 0.01, y_lf, 0.035, y_lf)
    _source(axis, 0.065, y_lf)
    _wire(axis, 0.095, y_lf, 0.115, y_lf)
    _resistor_h(axis, 0.115, 0.205, y_lf)
    _capacitor_h(axis, 0.220, 0.300, y_lf)
    _inductor_h(axis, 0.315, 0.430, y_lf)
    _resistor_h(axis, 0.445, 0.520, y_lf)
    _inductor_h(axis, 0.535, 0.630, y_lf)
    _wire(axis, 0.630, y_lf, x, wafer_y)
    axis.text(0.065, 0.020, "2 MHz\n450 Vpk", ha="center", va="top", fontsize=5.5)
    axis.text(0.160, 0.118, r"50 $\Omega$", ha="center", fontsize=5.2)
    axis.text(0.260, 0.025, "4.7 nF", ha="center", fontsize=5.2)
    axis.text(0.373, 0.118, r"1.35 $\mu$H", ha="center", fontsize=5.2)
    axis.text(0.483, 0.025, r"0.4 $\Omega$", ha="center", fontsize=5.2)
    axis.text(0.583, 0.118, "60 nH", ha="center", fontsize=5.2)

    # Chamber strays and wall leakage to the grounded enclosure.
    _wire(axis, ground_x, 0.09, ground_x, 0.74, color=MUTED)
    _ground(axis, ground_x, 0.065, 0.72)
    axis.text(0.925, 0.770, "chamber ground", ha="center", fontsize=5.1, color=MUTED)
    _wire(axis, x, upper_y, 0.825, upper_y)
    _capacitor_h(axis, 0.825, 0.915, upper_y)
    _wire(axis, 0.915, upper_y, ground_x, upper_y)
    axis.text(0.870, 0.732, "90 pF", ha="center", fontsize=5.1)
    _wire(axis, x, bulk_bottom_y, 0.825, bulk_bottom_y)
    _resistor_h(axis, 0.825, 0.915, bulk_bottom_y)
    _wire(axis, 0.915, bulk_bottom_y, ground_x, bulk_bottom_y)
    axis.text(0.870, 0.342, r"5 k$\Omega$", ha="center", fontsize=5.1)
    _wire(axis, x, wafer_y, 0.825, wafer_y)
    _capacitor_h(axis, 0.825, 0.915, wafer_y)
    _wire(axis, 0.915, wafer_y, ground_x, wafer_y)
    axis.text(0.870, 0.202, "140 pF", ha="center", fontsize=5.1)

    boundary = mpatches.FancyBboxPatch(
        (0.646, 0.145),
        0.165,
        0.585,
        boxstyle="round,pad=0.008",
        fill=False,
        edgecolor=ORANGE,
        linestyle=(0, (4, 2)),
        linewidth=0.85,
    )
    axis.add_patch(boundary)
    axis.text(0.634, 0.44, "dynamic plasma branch", rotation=90, va="center", ha="right", fontsize=5.2, color=ORANGE)


def figure_apparatus() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE, gridspec_kw={"width_ratios": [1.0, 1.38]})
    add_panel_title(axes[0], "(a)", "Chamber section and measurement planes")
    _apparatus(axes[0])
    add_panel_title(axes[1], "(b)", "Solved ngspice electrical schematic")
    _equivalent_circuit(axes[1])
    add_figure_title(
        fig,
        "Literature-grounded etch-CCP topology and the solved electrical network",
        "The chamber and schematic share U-to-W vertical order; this is a hybrid benchmark, not a drawing copied from one reactor.",
    )
    add_figure_footer(
        fig,
        "[R1] Kim et al., JJAP 54, 01AE07 (300 mm, 20 mm, upper 60 MHz/lower 2 MHz).\n"
        "[R2] Yamaguchi et al., J. Phys. D 45, 025203 (negative DC on upper VHF electrode).\n"
        "See p. 6 for source details, benchmark-specific assumptions, and the model boundary.",
    )
    fig.subplots_adjust(left=0.020, right=0.992, top=0.80, bottom=0.105, wspace=0.055)
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
    axes[0, 0].plot(time_us[zoom], target["upper_hf_source_voltage_V"][zoom], color=BLUE, lw=1.05)
    axes[0, 0].set_xlim(0.0, 0.20)
    axes[0, 0].set_ylabel(r"$V_{HF}$ (V)")
    axes[0, 0].set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])

    add_panel_title(axes[0, 1], "(b)", "Lower input: 2 MHz RF")
    axes[0, 1].plot(time_us, target["lower_lf_source_voltage_V"], color=ORANGE, lw=1.05)
    axes[0, 1].set_ylabel(r"$V_{LF}$ (V)")

    add_panel_title(axes[1, 0], "(c)", "Upper negative-DC pulse")
    axes[1, 0].plot(time_us, target["upper_dc_source_voltage_V"], color=INK, lw=1.2)
    axes[1, 0].set_ylabel(r"$V_{DC}$ (V)")

    add_panel_title(axes[1, 1], "(d)", "Prescribed plasma-state envelope")
    axes[1, 1].plot(time_us, target["plasma_envelope"], color=BLUE, lw=1.5)
    axes[1, 1].set_ylabel(r"$s(t)$")
    axes[1, 1].set_ylim(-0.08, 1.08)

    add_panel_title(axes[2, 0], "(e)", "Bulk resistance and inertia")
    axes[2, 0].plot(time_us, target["plasma_resistance_ohm"], color=ORANGE, lw=1.45)
    ax_l = axes[2, 0].twinx()
    ax_l.plot(time_us, 1e9 * target["plasma_inductance_H"], color=BLUE, lw=1.25)
    ax_l.set_ylabel(r"$L_p$ (nH)", color=BLUE)
    axes[2, 0].set_ylabel(r"$R_p$ (Ω)")
    ax_l.tick_params(axis="y", colors=BLUE)
    ax_l.spines["top"].set_visible(False)
    axes[2, 0].text(0.96, 0.86, r"$R_p$", transform=axes[2, 0].transAxes, ha="right", color=ORANGE, fontsize=7.0)
    axes[2, 0].text(0.96, 0.68, r"$L_p$", transform=axes[2, 0].transAxes, ha="right", color=BLUE, fontsize=7.0)

    add_panel_title(axes[2, 1], "(f)", "Upper and wafer sheath capacitance")
    axes[2, 1].plot(time_us, 1e12 * target["upper_sheath_capacitance_F"], color=BLUE, lw=1.45)
    axes[2, 1].plot(time_us, 1e12 * target["wafer_sheath_capacitance_F"], color=ORANGE, lw=1.45)
    axes[2, 1].set_ylabel("Capacitance (pF)")
    axes[2, 1].text(0.96, 0.78, r"$C_{s,w}$", transform=axes[2, 1].transAxes, ha="right", color=ORANGE, fontsize=7.0)
    axes[2, 1].text(0.96, 0.58, r"$C_{s,u}$", transform=axes[2, 1].transAxes, ha="right", color=BLUE, fontsize=7.0)

    for axis in axes.flat:
        _format_axes(axis)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]):
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
    fig.text(0.50, 0.095, "Time within 400 kHz pulse period (µs)", ha="center", va="center", fontsize=8.4)
    add_figure_title(
        fig,
        "Known excitations and prescribed dynamic plasma elements",
        r"The prescribed envelope moves $R_p$: 35→15 Ω, $L_p$: 35→16 nH, $C_{s,u}$: 260→520 pF and $C_{s,w}$: 360→720 pF.",
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
    truth_handle = plt.Line2D([], [], color=ORANGE, lw=1.55, ls=(0, (4, 2)))
    ngspice_handle = plt.Line2D([], [], color=BLUE, lw=1.0)
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
        axis.plot(time_us, result, color=BLUE, lw=1.0)
        axis.plot(time_us, truth, color=ORANGE, lw=1.55, ls=(0, (4, 2)))
        axis.set_ylabel(ylabel)

    add_panel_title(axes[2, 1], "(f)", "RMS-normalized error against 1% criterion")
    forward = data["forward"]
    error_labels = ["wafer V", "reflection", "bulk I", "sheath q", "plasma flux"]
    error_percent = 100.0 * np.asarray(
        [
            forward["wafer_voltage"]["normalized_rmse"],
            forward["upper_reflected_voltage"]["normalized_rmse"],
            forward["bulk_current"]["normalized_rmse"],
            forward["upper_sheath_charge"]["normalized_rmse"],
            forward["plasma_flux"]["normalized_rmse"],
        ]
    )
    y_positions = np.arange(len(error_labels))
    axes[2, 1].hlines(y_positions, 0.003, error_percent, color=BLUE, linewidth=1.1)
    axes[2, 1].scatter(error_percent, y_positions, color=BLUE, s=22, zorder=3)
    axes[2, 1].axvline(1.0, color=INK, lw=1.0, ls=(0, (4, 2)))
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_xlim(0.003, 2.0)
    axes[2, 1].set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1"])
    axes[2, 1].set_yticks(y_positions, error_labels)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlabel("nRMSE (%)")
    for y_position, value in zip(y_positions, error_percent, strict=True):
        axes[2, 1].text(value * 1.16, y_position, f"{value:.3f}%", va="center", fontsize=5.8, color=INK)
    for axis in list(axes.flat)[:5]:
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        _format_axes(axis)
    for axis in (axes[2, 0],):
        axis.set_xlabel("Time within 400 kHz pulse period (µs)")
    _format_axes(axes[2, 1])
    _figure_legend(fig, [truth_handle, ngspice_handle], ["independent charge/flux MNA", "ngspice 46"])
    add_figure_title(
        fig,
        "Forward validity: independent state equations and ngspice agree",
        "Voltage, current, charge and flux are compared directly over one complete 400 kHz pulse period.",
    )
    add_figure_footer(
        fig,
        f"Benchmark rule: all five RMS-normalized errors <1% and a complete ngspice transient. "
        f"Wafer={100 * forward['wafer_voltage']['normalized_rmse']:.3f}%; "
        f"reflection={100 * forward['upper_reflected_voltage']['normalized_rmse']:.3f}%; all five pass.\n"
        "This establishes circuit-equation conformance, not chamber or plasma-chemistry validation.",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.75, bottom=0.16, hspace=0.74, wspace=0.26)
    return fig


def _reflection_coefficient(impedance: np.ndarray) -> np.ndarray:
    return (impedance - Z0_OHM) / (impedance + Z0_OHM)


def _smith_grid(axis: plt.Axes) -> None:
    angle = np.linspace(0.0, 2.0 * math.pi, 500)
    axis.plot(np.cos(angle), np.sin(angle), color=INK, lw=0.9)
    axis.axhline(0.0, color="#c7cdd2", lw=0.55)
    axis.axvline(0.0, color="#e1e5e8", lw=0.45)
    reactance = np.linspace(-12.0, 12.0, 600)
    for resistance in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0):
        normalized = resistance + 1j * reactance
        gamma = (normalized - 1.0) / (normalized + 1.0)
        axis.plot(gamma.real, gamma.imag, color="#d6dbe0", lw=0.45)
    resistance = np.geomspace(1.0e-3, 1.0e3, 600)
    for reactance_value in (-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0):
        normalized = resistance + 1j * reactance_value
        gamma = (normalized - 1.0) / (normalized + 1.0)
        axis.plot(gamma.real, gamma.imag, color="#d6dbe0", lw=0.45)
    axis.scatter([0.0], [0.0], marker="+", s=35, color=INK, linewidths=0.9, zorder=4)
    axis.text(0.03, 0.04, r"50+$j$0 $\Omega$", fontsize=5.3, color=MUTED)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Re($\Gamma$)")
    axis.set_ylabel(r"Im($\Gamma$)")
    axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axis.grid(False)


def _plot_smith_trajectory(
    axis: plt.Axes,
    impedance: np.ndarray,
    envelope: np.ndarray,
    *,
    frequency: str,
) -> None:
    gamma = _reflection_coefficient(impedance)
    off_index = 0
    on_index = int(np.argmax(envelope))
    axis.plot(gamma.real, gamma.imag, color=BLUE, lw=1.45, zorder=3)
    axis.scatter([gamma.real[off_index]], [gamma.imag[off_index]], s=30, color=MUTED, zorder=5)
    axis.scatter([gamma.real[on_index]], [gamma.imag[on_index]], s=35, marker="s", color=ORANGE, zorder=5)
    axis.annotate(
        "",
        xy=(gamma.real[on_index], gamma.imag[on_index]),
        xytext=(gamma.real[off_index], gamma.imag[off_index]),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.9},
    )
    off_reflected = abs(gamma[off_index]) ** 2
    on_reflected = abs(gamma[on_index]) ** 2
    axis.text(
        0.03,
        0.97,
        f"off circle: |Γ|²={off_reflected:.3f}\non square: |Γ|²={on_reflected:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=INK,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axis.text(0.97, 0.03, frequency, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.4, fontweight="bold")


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
    envelope = target["plasma_envelope"]
    fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE)
    for index, (axis, impedance, frequency) in enumerate(((axes[0, 0], z_2m, "2 MHz"), (axes[0, 1], z_60m, "60 MHz"))):
        add_panel_title(axis, f"({chr(97 + index)})", f"Frozen-state load impedance at {frequency}")
        axis.plot(time_us, impedance.real, color=ORANGE, lw=1.45)
        axis.plot(time_us, impedance.imag, color=BLUE, lw=1.35)
        axis.axhline(0.0, color=MUTED, lw=0.6)
        axis.set_ylabel("Impedance (Ω)")
        axis.text(0.97, 0.91, r"Re($Z$)", transform=axis.transAxes, ha="right", color=ORANGE, fontsize=6.5)
        axis.text(0.97, 0.76, r"Im($Z$)", transform=axis.transAxes, ha="right", color=BLUE, fontsize=6.5)
        _format_axes(axis)
    add_panel_title(axes[1, 0], "(c)", "2 MHz Smith chart")
    _smith_grid(axes[1, 0])
    _plot_smith_trajectory(axes[1, 0], z_2m, envelope, frequency="2 MHz")
    add_panel_title(axes[1, 1], "(d)", "60 MHz Smith chart")
    _smith_grid(axes[1, 1])
    _plot_smith_trajectory(axes[1, 1], z_60m, envelope, frequency="60 MHz")
    for axis in axes[0, :]:
        axis.set_xlim(0.0, 2.5)
        axis.set_xticks(np.arange(0.0, 2.51, 0.5))
        axis.set_xlabel("Time within pulse period (µs)")
    add_figure_title(
        fig,
        "Dynamic plasma load: impedance and 50 Ω Smith-chart interpretation",
        r"$Z_{load}=Z_{s,u}+R_p+j\omega L_p+Z_{s,w}$ and $\Gamma=(Z_{load}-50)/(Z_{load}+50)$; the external matcher is excluded.",
    )
    add_figure_footer(
        fig,
        "Smith center = 50+j0 Ω (no reflection); outer circle = complete reflection.\n"
        "Lower half = capacitive; gray circle is pulse-off and orange square is on-state.\n"
        "These are frozen-state plasma-load values before matching, not measured source-plane S11 and not a kinetic plasma solution.",
    )
    fig.subplots_adjust(left=0.09, right=0.965, top=0.79, bottom=0.17, hspace=0.58, wspace=0.30)
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
    truth_handle = plt.Line2D([], [], color=ORANGE, lw=1.55, ls=(0, (4, 2)))
    initial_handle = plt.Line2D([], [], color=MUTED, lw=0.9)
    selected_handle = plt.Line2D([], [], color=BLUE, lw=1.0)

    add_panel_title(axes[0, 0], "(a)", "Fitted objective: wafer voltage only")
    axes[0, 0].plot(time_us, initial_wafer, color=MUTED, lw=0.9)
    axes[0, 0].plot(time_us, selected_wafer, color=BLUE, lw=1.0)
    axes[0, 0].plot(time_us, target["wafer_voltage_V"], color=ORANGE, lw=1.55, ls=(0, (4, 2)))
    axes[0, 0].set_ylabel(r"$V_W$ (V)")

    add_panel_title(axes[0, 1], "(b)", "Held-out check: upper reflected wave")
    axes[0, 1].plot(time_us, selected_reflection, color=BLUE, lw=1.0)
    axes[0, 1].plot(time_us, target["upper_reflected_voltage_V"], color=ORANGE, lw=1.55, ls=(0, (4, 2)))
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

    add_panel_title(axes[2, 0], "(e)", "Complete 81-candidate grid")
    axes[2, 0].semilogy(trials, losses, color=MUTED, marker=".", ms=2.2, lw=0.45)
    axes[2, 0].step(trials, best_so_far, where="post", color=BLUE, lw=1.25)
    axes[2, 0].scatter([selected_trial], [result["selected_loss"]], color=ORANGE, s=24, zorder=4)
    axes[2, 0].set_xlim(1.0, 81.0)
    axes[2, 0].set_xticks([1, 20, 40, 60, 81])
    axes[2, 0].set_xlabel("Evaluated candidate")
    axes[2, 0].set_ylabel("Normalized RMSE")
    axes[2, 0].text(
        0.98,
        0.92,
        f"best {result['selected_loss']:.3e}\n2nd {result['second_best_loss']:.3e}",
        transform=axes[2, 0].transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )

    add_panel_title(axes[2, 1], "(f)", "Parameter recovery / truth")
    x = np.arange(4)
    width = 0.32
    truth_values = np.asarray(_scaled_parameter_values(result["truth_values"]))
    initial_ratio = np.asarray(_scaled_parameter_values(result["initial_values"])) / truth_values
    selected_ratio = np.asarray(_scaled_parameter_values(result["selected_values"])) / truth_values
    axes[2, 1].bar(
        x - width / 2.0,
        initial_ratio,
        width,
        color="#c8ced3",
        edgecolor=MUTED,
        linewidth=0.7,
    )
    axes[2, 1].bar(
        x + width / 2.0,
        selected_ratio,
        width,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.8,
    )
    axes[2, 1].axhline(1.0, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    axes[2, 1].set_ylim(0.72, 1.06)
    axes[2, 1].set_xticks(x, ["$R_p$", "$L_p$", "$C_{s,u}$", "$C_{s,w}$"])
    axes[2, 1].set_ylabel("estimated / truth")
    axes[2, 1].text(0.02, 0.91, "orange line = exact truth", transform=axes[2, 1].transAxes, fontsize=5.7, color=ORANGE)

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
        f"Objective nRMSE={result['objective_error']['normalized_rmse']:.3e}; held-out nRMSE={result['held_out_error']['normalized_rmse']:.3e}; "
        f"second-best / best loss={result['second_best_loss'] / result['selected_loss']:.2f}.",
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.75, bottom=0.17, hspace=0.75, wspace=0.34)
    return fig


def figure_basis_and_scope(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    axis = fig.add_axes((0.035, 0.145, 0.93, 0.67))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    axis.text(0.00, 0.98, "(a)  Primary-source basis", fontsize=8.3, fontweight="bold", va="top")
    references = [
        (
            "[R1] H. J. Kim et al., JJAP 54, 01AE07 (2015)",
            "300 mm DF-CCP; 20 mm gap; upper 13.56-60 MHz; lower 2 MHz.\ndoi:10.7567/JJAP.54.01AE07",
        ),
        (
            "[R2] T. Yamaguchi et al., J. Phys. D 45, 025203 (2012)",
            "Negative DC above about 800 V on the upper 60 MHz electrode;\nlower electrode was 13.56 MHz. doi:10.1088/0022-3727/45/2/025203",
        ),
        (
            "[R3] J. Kim et al., Electronics 14, 2022 (2025)",
            "CCP load interpreted as sheath capacitance plus plasma L and R;\n50 ohm matching-network reference plane. doi:10.3390/electronics14102022",
        ),
        (
            "[R4] H. Lee et al., Rev. Sci. Instrum. 86, 083505 (2015)",
            "Sub-microsecond V/I impedance observation in pulsed dual-frequency CCP.\ndoi:10.1063/1.4928121",
        ),
        (
            "[R5] F. Schmidt et al., Plasma Sources Sci. Technol. 27, 105017 (2018)",
            "Generator, matcher, discharge and stray elements must be coupled explicitly.\ndoi:10.1088/1361-6595/aae429",
        ),
    ]
    for index, (heading, detail) in enumerate(references):
        y = 0.91 - 0.14 * index
        axis.text(0.00, y, heading, fontsize=5.9, fontweight="bold", va="top")
        axis.text(0.00, y - 0.047, detail, fontsize=5.25, color=MUTED, va="top", linespacing=1.25)

    axis.plot([0.505, 0.505], [0.21, 0.98], color="#d6dbe0", lw=0.8)
    axis.text(0.535, 0.98, "(b)  What the numerical values mean", fontsize=8.3, fontweight="bold", va="top")
    grounded = [
        (
            "Literature-grounded",
            "300 mm geometry, 20 mm gap, 60/2 MHz arrangement, upper negative DC, R-L-C plasma interpretation",
        ),
        (
            "Scale-grounded",
            "Hundreds-of-volts RF/DC excitation, millimeter-order sheath thickness, 50 ohm generator reference",
        ),
        (
            "Benchmark-specific",
            "400 kHz DC-pulse envelope, ideal source amplitudes, matcher/feed/stray values, exact R/L/C trajectories",
        ),
        (
            "Identification-specific",
            "Three values for each of four on-state parameters; complete 3^4 = 81 candidate enumeration",
        ),
        (
            "Not established",
            "Gas chemistry, electron-energy distribution, ion-energy distribution, etch rate, uniformity, or a calibrated production recipe",
        ),
    ]
    for index, (heading, detail) in enumerate(grounded):
        y = 0.90 - 0.145 * index
        face = BLUE_LIGHT if index < 2 else ORANGE_LIGHT if index < 4 else LIGHT
        axis.add_patch(
            mpatches.FancyBboxPatch(
                (0.535, y - 0.095),
                0.445,
                0.105,
                boxstyle="round,pad=0.008",
                facecolor=face,
                edgecolor="#c6cdd2",
                linewidth=0.65,
            )
        )
        axis.text(0.550, y - 0.010, heading, fontsize=5.9, fontweight="bold", va="top")
        axis.text(0.550, y - 0.044, detail, fontsize=5.15, color=INK, va="top", wrap=True)

    scale = data["problem"]["basis"]["scale_checks"]
    axis.text(0.00, 0.185, "(c)  Transparent order-of-magnitude checks", fontsize=8.0, fontweight="bold", va="top")
    cards = [
        (
            "Sheath C -> thickness",
            f"{scale['sheath_thickness_range_mm_from_C_parallel_plate'][0]:.2f}-"
            f"{scale['sheath_thickness_range_mm_from_C_parallel_plate'][1]:.2f} mm",
            r"$d_s=\epsilon_0 A/C_s$, 300 mm plate",
        ),
        (
            "Plasma L -> density",
            f"{scale['electron_density_range_m-3_from_L_uniform_bulk'][0]:.2e}-"
            f"{scale['electron_density_range_m-3_from_L_uniform_bulk'][1]:.2e} m$^{{-3}}$",
            r"$L_p=m_e l/(n_e e^2 A)$, $l=18$ mm",
        ),
        (
            "R/L -> collision scale",
            f"{scale['collision_frequency_s-1_from_R_over_L'][0]:.2e}-"
            f"{scale['collision_frequency_s-1_from_R_over_L'][1]:.2e} s$^{{-1}}$",
            r"$\nu_m\approx R_p/L_p$",
        ),
    ]
    for index, (heading, value, formula) in enumerate(cards):
        x = 0.00 + 0.335 * index
        axis.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.020),
                0.315,
                0.125,
                boxstyle="round,pad=0.008",
                facecolor=WHITE,
                edgecolor="#bfc7cd",
                linewidth=0.7,
            )
        )
        axis.text(x + 0.012, 0.117, heading, fontsize=5.8, fontweight="bold")
        axis.text(x + 0.012, 0.073, value, fontsize=6.2, color=BLUE)
        axis.text(x + 0.012, 0.035, formula, fontsize=4.9, color=MUTED)

    add_figure_title(
        fig,
        "Evidence basis, realism assessment, and model boundary",
        "Published apparatus topology and scale checks are separated from synthetic benchmark inputs.",
    )
    add_figure_footer(
        fig,
        "Assessment: suitable for circuit-solver and bounded-identification verification.\n"
        "Not established: quantitative chamber calibration or process prediction.",
    )
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
        ("06-literature-basis-and-scope", figure_basis_and_scope(data)),
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
