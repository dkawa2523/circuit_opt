"""Generate direct calculation-evidence figures from completed ngspice runs.

The generator is deliberately read-only: it never runs ngspice or an
optimizer.  Every curve comes from a saved canonical waveform, an independent
closed-form expectation, or a declared target waveform.
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
import schemdraw
import schemdraw.elements as elm
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from pcd.figures import (
    BLUE,
    BLUE_LIGHT,
    INK,
    LIGHT,
    MUTED,
    ORANGE,
    PAGE_SIZE,
    CircuitDiagram,
    CircuitViewport,
    add_figure_footer,
    add_figure_title,
    add_panel_title,
    assert_text_inside_canvas,
    configure_publication_style,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_BENCHMARK_RESULT = ROOT / "runs" / "benchmark_suite" / "benchmark_result.json"
DEFAULT_VARYING_ROOT = ROOT / "runs" / "p4_release_final2_20260930" / "cases" / "time_varying_resistor"
DEFAULT_NETLIST_ROOT = ROOT / "runs" / "calculation_evidence_20261001_verified"
DEFAULT_OPTIMIZATION_ROOT = ROOT / "runs" / "calculation_evidence_20261001"
DEFAULT_OUTPUT = HERE / "evidence"
DEFAULT_PDF_OUTPUT = ROOT / "output" / "pdf" / "calculation-evidence-pack.pdf"
SVG_HASHSALT = "pcd-calculation-evidence-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


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
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence must stay inside the repository: {resolved}") from exc


def _recorded_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    parts = list(PureWindowsPath(value).parts)
    root_index = next((index for index, part in enumerate(parts) if part.casefold() == ROOT.name.casefold()), None)
    if root_index is None:
        raise FileNotFoundError(value)
    reconstructed = ROOT.joinpath(*parts[root_index + 1 :])
    if not reconstructed.exists():
        raise FileNotFoundError(reconstructed)
    return reconstructed.resolve()


def _artifact_path(base: Path, value: str) -> Path:
    return base.joinpath(*PureWindowsPath(value).parts).resolve()


def _source_record(path: Path) -> dict[str, Any]:
    return {"path": _repository_path(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _find_simulation(root: Path, case_id: str) -> Path:
    matches = []
    for summary_path in root.glob("sim_*/summary.json"):
        if _read_json(summary_path).get("case_id") == case_id:
            matches.append(summary_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {case_id!r} simulation under {root}; found {len(matches)}")
    return matches[0]


def _find_study(root: Path, study_id: str) -> Path:
    matches = []
    for study_path in root.glob("*/study_result.json"):
        if (_read_json(study_path).get("study") or {}).get("study_id") == study_id:
            matches.append(study_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {study_id!r} study under {root}; found {len(matches)}")
    return matches[0]


def _benchmark_case(payload: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    match = next((item for item in payload["cases"] if item["benchmark_id"] == benchmark_id), None)
    if match is None or not match.get("passed"):
        raise ValueError(f"benchmark case is absent or failed: {benchmark_id}")
    return dict(match)


def _archived_input(manifest_path: Path) -> Path:
    manifest = _read_json(manifest_path)
    inputs = manifest.get("inputs") or []
    if len(inputs) != 1:
        raise ValueError(f"expected one archived input in {manifest_path}")
    return _artifact_path(manifest_path.parent, str(inputs[0]["artifact"]))


def _simulation_sources(run_dir: Path) -> dict[str, Path]:
    summary = run_dir / "summary.json"
    record = run_dir / "debug" / "manifest.json"
    payload = _read_json(summary)
    if payload.get("status") != "ok" or payload.get("warnings"):
        raise ValueError(f"simulation is not clean: {summary}")
    waveform = _artifact_path(run_dir, str(payload["artifacts"]["waveform"]))
    return {"summary": summary, "record": record, "waveform": waveform}


def _plasma_rows(case: dict[str, Any]) -> list[dict[str, float | str]]:
    case_path = _recorded_path(str(case["case_path"]))
    source = _read_yaml(case_path)
    frequencies = {str(item["id"]): float(item["frequency_Hz"]) for item in source["conditions"]}
    observed = {str(item["scenario_id"]): item for item in case["scenarios"]}
    rows = []
    for scenario_id, expected in case["expected"]["input_impedance_ohm"].items():
        actual = observed[scenario_id]
        rows.append(
            {
                "scenario_id": scenario_id,
                "frequency_Hz": frequencies[scenario_id],
                "expected_resistance_ohm": float(expected[0]),
                "expected_reactance_ohm": float(expected[1]),
                "ngspice_resistance_ohm": float(actual["input_resistance_ohm"]),
                "ngspice_reactance_ohm": float(actual["input_reactance_ohm"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["frequency_Hz"]))


def _max_impedance_error(rows: list[dict[str, float | str]]) -> float:
    return max(
        max(
            abs(float(row["ngspice_resistance_ohm"]) - float(row["expected_resistance_ohm"])),
            abs(float(row["ngspice_reactance_ohm"]) - float(row["expected_reactance_ohm"])),
        )
        for row in rows
    )


def _optimization_sources(root: Path) -> dict[str, Path]:
    study_root = _find_study(root, "figure_target_waveform_conformance")
    study_path = study_root / "study_result.json"
    study = _read_json(study_path)
    generation = _artifact_path(study_root, str(study["artifacts"]["generation"]))
    case_path = generation / "case.yaml"
    case = _read_yaml(case_path)
    target = _artifact_path(generation, str(case["target"]["waveform_file"]))
    return {
        "root": study_root,
        "study": study_path,
        "generation": generation,
        "case": case_path,
        "target": target,
        "best": generation / "best_candidate.json",
        "history": generation / "study_history.json",
        "evaluations": generation / "evaluations.csv",
    }


def collect_sources(
    benchmark_result: Path,
    varying_root: Path,
    netlist_root: Path,
    optimization_root: Path,
) -> dict[str, Any]:
    external_run = _find_simulation(netlist_root, "figure_external_netlist_rc")
    external = _simulation_sources(external_run)
    external.update(
        {
            "case": ROOT / "bench" / "figures" / "cases" / "external_netlist_rc.yaml",
            "fragment": ROOT / "bench" / "figures" / "cases" / "external_netlist_rc.cir",
            "resolved_netlist": netlist_root / "external_netlist_resolved.cir",
        }
    )
    varying_run = _find_simulation(varying_root, "time_varying_resistor")
    varying = _simulation_sources(varying_run)
    varying.update(
        {
            "case": varying_run / "debug" / "case.yaml",
            "profile": _archived_input(varying_run / "debug" / "input_manifest.json"),
        }
    )
    return {
        "benchmark_result": benchmark_result,
        "external": external,
        "varying": varying,
        "optimization": _optimization_sources(optimization_root),
    }


def _external_comparison(paths: dict[str, Path]) -> dict[str, Any]:
    waveform = _read_numeric_csv(paths["waveform"])
    time_s = waveform["time_s"]
    resistance_ohm = 1000.0
    capacitance_f = 100e-9
    frequency_hz = 1000.0
    omega = 2.0 * math.pi * frequency_hz
    transfer = 1.0 / complex(1.0, omega * resistance_ohm * capacitance_f)
    expected = abs(transfer) * np.sin(omega * time_s + np.angle(transfer))
    compare = time_s >= time_s[-1] - 2.0 / frequency_hz
    error = waveform["voltage_V"][compare] - expected[compare]
    return {
        "R_ohm": resistance_ohm,
        "C_F": capacitance_f,
        "frequency_Hz": frequency_hz,
        "expected_gain": abs(transfer),
        "expected_phase_deg": float(np.degrees(np.angle(transfer))),
        "comparison_start_s": float(time_s[compare][0]),
        "rmse_V": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_V": float(np.max(np.abs(error))),
        "samples": time_s.size,
    }


def _varying_comparison(paths: dict[str, Path]) -> dict[str, Any]:
    waveform = _read_numeric_csv(paths["waveform"])
    profile = _read_numeric_csv(paths["profile"])
    resistance = np.interp(waveform["time_s"], profile["time_s"], profile["resistance_ohm"])
    expected_voltage = 10.0 * resistance / (50.0 + resistance)
    expected_current = 10.0 / (50.0 + resistance)
    return {
        "source_voltage_V": 10.0,
        "source_resistance_ohm": 50.0,
        "profile_points": profile["time_s"].size,
        "samples": waveform["time_s"].size,
        "max_abs_voltage_error_V": float(np.max(np.abs(waveform["voltage_V"] - expected_voltage))),
        "max_abs_current_error_A": float(np.max(np.abs(waveform["component_Rchamber_current_A"] - expected_current))),
    }


def _optimization_data(paths: dict[str, Path]) -> dict[str, Any]:
    study = _read_json(paths["study"])
    best = _read_json(paths["best"])
    rows = _read_table(paths["evaluations"])
    if study.get("n_failed_evaluations") != 0:
        raise ValueError("optimization evidence contains failed evaluations")
    selected = best["candidate"]
    selected_values = selected["values"]
    if not math.isclose(float(selected_values["R1"]), 1000.0) or not math.isclose(float(selected_values["C1"]), 3e-10):
        raise ValueError(f"known RC target was not recovered: {selected_values}")
    evaluations = [
        {
            "trial": int(row["trial"]),
            "candidate_id": row["candidate_id"],
            "R1_ohm": float(row["design.R1"]),
            "C1_F": float(row["design.C1"]),
            "normalized_rmse": float(row["metric.normalized_rmse"]),
            "waveform": _repository_path(_artifact_path(paths["root"], row["artifact.waveform"])),
        }
        for row in rows
    ]
    return {
        "optimizer": study["execution"]["optimizer"],
        "n_evaluations": int(study["n_evaluations"]),
        "n_failed_evaluations": int(study["n_failed_evaluations"]),
        "selected_candidate_id": selected["candidate_id"],
        "selected_R1_ohm": float(selected_values["R1"]),
        "selected_C1_F": float(selected_values["C1"]),
        "selected_time_constant_s": float(selected_values["R1"]) * float(selected_values["C1"]),
        "selected_normalized_rmse": float(best["aggregates"]["normalized_rmse"]),
        "evaluations": evaluations,
    }


def build_figure_data(sources: dict[str, Any]) -> dict[str, Any]:
    benchmark = _read_json(sources["benchmark_result"])
    ccp = _benchmark_case(benchmark, "A4_ccp_lumped_frequency_conformance")
    icp = _benchmark_case(benchmark, "A5_icp_transformer_frequency_conformance")
    ccp_rows = _plasma_rows(ccp)
    icp_rows = _plasma_rows(icp)
    source_files: dict[str, dict[str, Any]] = {
        "benchmark_result": _source_record(sources["benchmark_result"]),
    }
    for group_name in ("external", "varying", "optimization"):
        for name, path in sources[group_name].items():
            if path.is_file():
                source_files[f"{group_name}.{name}"] = _source_record(path)
    return {
        "schema": "calculation_evidence_figures.v1",
        "sources": source_files,
        "renderer": {
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "schemdraw": schemdraw.__version__,
            "page_inches": list(PAGE_SIZE),
            "png_dpi": 300,
        },
        "external_netlist": _external_comparison(sources["external"]),
        "time_varying_element": _varying_comparison(sources["varying"]),
        "effective_plasma_loads": {
            "scope": "terminal-equivalent circuit conformance; not plasma-state validation",
            "ccp": ccp_rows,
            "icp": icp_rows,
            "ccp_max_abs_impedance_error_ohm": _max_impedance_error(ccp_rows),
            "icp_max_abs_impedance_error_ohm": _max_impedance_error(icp_rows),
        },
        "optimization": _optimization_data(sources["optimization"]),
    }


def _format_axes(axis: plt.Axes) -> None:
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)


def _draw_rc(axis: plt.Axes, resistor_label: str, capacitor_label: str) -> None:
    circuit = CircuitDiagram(axis)
    source = circuit.anchor("source", (0.0, 0.0))
    output = circuit.anchor("output", (2.35, 0.0))
    ground = circuit.anchor("ground", (2.35, -1.55))
    port = circuit.anchor("port", (3.05, 0.0))
    circuit.port(source, r"$V_{in}$", "left")
    circuit.component(elm.Resistor, source, output, label=resistor_label, name="R1")
    circuit.node(output)
    circuit.wire(output, port)
    circuit.port(port, r"$V_{out}$", "right")
    circuit.component(elm.Capacitor, output, ground, name="C1")
    circuit.ground(ground)
    circuit.finish(CircuitViewport(xlim=(-0.45, 3.45), ylim=(-2.15, 0.65)))
    axis.text(2.63, -0.82, capacitor_label, ha="left", va="center")


def _draw_varying_divider(axis: plt.Axes) -> None:
    circuit = CircuitDiagram(axis)
    source = circuit.anchor("source", (0.0, 0.0))
    output = circuit.anchor("output", (2.45, 0.0))
    port = circuit.anchor("port", (3.05, 0.0))
    ground = circuit.anchor("ground", (2.45, -1.65))
    circuit.port(source, "10 V DC", "left")
    circuit.component(elm.Resistor, source, output, label=r"$R_s=50\,\Omega$", name="Rsource")
    circuit.node(output)
    circuit.wire(output, port)
    circuit.port(port, r"$V_{out}$", "right")
    circuit.component(
        elm.Resistor,
        output,
        ground,
        name="Rchamber",
    )
    circuit.ground(ground)
    circuit.finish(CircuitViewport(xlim=(-0.6, 3.55), ylim=(-2.2, 0.7)))
    axis.text(2.78, -0.87, r"$R_{ch}(t)$", ha="left", va="center")


def _draw_ccp(axis: plt.Axes) -> None:
    circuit = CircuitDiagram(axis)
    start = circuit.anchor("p", (0.0, 0.0))
    r_end = circuit.anchor("r_end", (1.35, 0.0))
    l_end = circuit.anchor("l_end", (2.7, 0.0))
    c_end = circuit.anchor("c_end", (4.05, 0.0))
    end = circuit.anchor("n", (4.45, 0.0))
    circuit.port(start, "p", "left")
    circuit.component(elm.Resistor, start, r_end, label=r"$R_{eff}$", name="Reff")
    circuit.component(elm.Inductor, r_end, l_end, label=r"$L_{eff}$", name="Leff")
    circuit.component(elm.Capacitor, l_end, c_end, label=r"$C_{sheath,eq}$", name="Csheath")
    circuit.wire(c_end, end)
    circuit.port(end, "n", "right")
    circuit.finish(CircuitViewport(xlim=(-0.35, 4.85), ylim=(-0.85, 0.75)))


def _draw_icp(axis: plt.Axes) -> None:
    circuit = CircuitDiagram(axis)
    start = circuit.anchor("p", (0.0, 0.0))
    node_a = circuit.anchor("node_a", (0.2, 0.0))
    r_end = circuit.anchor("r_end", (1.45, 0.0))
    l_end = circuit.anchor("l_end", (2.75, 0.0))
    z_end = circuit.anchor("z_end", (4.0, 0.0))
    node_b = circuit.anchor("node_b", (4.35, 0.0))
    end = circuit.anchor("n", (4.65, 0.0))
    shunt_a = circuit.anchor("shunt_a", (0.2, -1.15))
    shunt_b = circuit.anchor("shunt_b", (4.35, -1.15))
    circuit.port(start, "p", "left")
    circuit.wire(start, node_a)
    circuit.node(node_a)
    circuit.component(elm.Resistor, node_a, r_end, label=r"$R_{coil}$", name="Rcoil")
    circuit.component(elm.Inductor, r_end, l_end, label=r"$L_{coil}$", name="Lcoil")
    circuit.component(elm.RBox, l_end, z_end, label=r"$Z_{ref}(\omega)$", name="Zref")
    circuit.wire(z_end, node_b)
    circuit.node(node_b)
    circuit.wire(node_b, end)
    circuit.port(end, "n", "right")
    circuit.wire(node_a, shunt_a)
    circuit.component(elm.Capacitor, shunt_a, shunt_b, label=r"$C_{parallel}$", name="Cparallel")
    circuit.wire(shunt_b, node_b)
    circuit.finish(CircuitViewport(xlim=(-0.35, 5.0), ylim=(-1.65, 0.75)))


def figure_netlist_workflow(sources: dict[str, Any], data: dict[str, Any]) -> plt.Figure:
    waveform = _read_numeric_csv(sources["external"]["waveform"])
    comparison = data["external_netlist"]
    time_s = waveform["time_s"]
    omega = 2.0 * math.pi * float(comparison["frequency_Hz"])
    transfer = 1.0 / complex(1.0, omega * float(comparison["R_ohm"]) * float(comparison["C_F"]))
    expected = abs(transfer) * np.sin(omega * time_s + np.angle(transfer))
    selected = time_s >= float(comparison["comparison_start_s"])

    fig = plt.figure(figsize=PAGE_SIZE)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.72, 2.2], width_ratios=[0.88, 2.12])
    flow = fig.add_subplot(grid[0, :])
    schematic = fig.add_subplot(grid[1, 0])
    wave = fig.add_subplot(grid[1, 1])
    flow.text(0.0, 1.055, "(a)", transform=flow.transAxes, ha="left", va="bottom", fontweight="bold")
    flow.text(0.07, 1.055, "Public calculation path", transform=flow.transAxes, ha="left", va="bottom")
    flow.set_axis_off()
    labels = [
        "Authored .cir\nR1 src out 1k\nC1 out 0 100n",
        "PCD case resolution\nsource + probes + .tran",
        "ngspice 46\nbatch solve",
        "Canonical result\ntransient.csv",
    ]
    x_positions = [0.01, 0.275, 0.54, 0.805]
    for index, (x_position, label) in enumerate(zip(x_positions, labels, strict=True)):
        box = mpatches.FancyBboxPatch(
            (x_position, 0.08),
            0.185,
            0.66,
            boxstyle="round,pad=0.015",
            transform=flow.transAxes,
            facecolor=BLUE_LIGHT if index in (0, 3) else LIGHT,
            edgecolor=BLUE if index in (0, 3) else MUTED,
            linewidth=0.9,
        )
        flow.add_patch(box)
        flow.text(x_position + 0.0925, 0.41, label, transform=flow.transAxes, ha="center", va="center", fontsize=7.0)
        if index < len(labels) - 1:
            flow.annotate(
                "",
                xy=(x_positions[index + 1] - 0.012, 0.41),
                xytext=(x_position + 0.197, 0.41),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0},
            )

    add_panel_title(schematic, "(b)", "Imported RC circuit")
    _draw_rc(schematic, r"$R_1=1\,k\Omega$", r"$C_1=100\,nF$")
    add_panel_title(wave, "(c)", "Steady-state input and output")
    x_ms = 1e3 * time_s[selected]
    wave.plot(x_ms, waveform["source_voltage_V"][selected], color=MUTED, lw=1.1, label=r"$V_{in}$ (ngspice)")
    wave.plot(x_ms, expected[selected], color=ORANGE, lw=1.8, ls=(0, (4, 2)), label="analytic RC output")
    wave.plot(x_ms, waveform["voltage_V"][selected], color=BLUE, lw=1.0, label=r"$V_{out}$ (ngspice)")
    wave.set_xlim(float(x_ms[0]) - 0.05, float(x_ms[-1]) + 0.05)
    wave.set_xticks(np.linspace(math.ceil(float(x_ms[0]) * 2.0) / 2.0, math.floor(float(x_ms[-1]) * 2.0) / 2.0, 5))
    wave.set_xlabel("Time (ms)")
    wave.set_ylabel("Voltage (V)")
    wave.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _format_axes(wave)
    add_figure_title(
        fig,
        "An authored netlist is resolved, solved, and returned as a canonical waveform",
        "Direct 1 kHz RC low-pass example; output is compared with the independent steady-state transfer equation.",
    )
    add_figure_footer(
        fig,
        f"ngspice-46; last two cycles shown; analytic gain={comparison['expected_gain']:.6f}, "
        f"phase={comparison['expected_phase_deg']:.2f} deg; max |error|={1e6 * comparison['max_abs_error_V']:.2f} uV.",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.12, hspace=0.48, wspace=0.28)
    wave_position = wave.get_position()
    wave.set_position((wave_position.x0, wave_position.y0, wave_position.width * 0.64, wave_position.height))
    return fig


def figure_time_varying_element(sources: dict[str, Any], data: dict[str, Any]) -> plt.Figure:
    waveform = _read_numeric_csv(sources["varying"]["waveform"])
    profile = _read_numeric_csv(sources["varying"]["profile"])
    time_us = 1e6 * waveform["time_s"]
    resistance = np.interp(waveform["time_s"], profile["time_s"], profile["resistance_ohm"])
    expected_voltage = 10.0 * resistance / (50.0 + resistance)
    expected_current = 10.0 / (50.0 + resistance)
    comparison = data["time_varying_element"]

    fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE)
    add_panel_title(axes[0, 0], "(a)", "Variable-resistor circuit")
    _draw_varying_divider(axes[0, 0])
    add_panel_title(axes[0, 1], "(b)", r"Prescribed $R_{ch}(t)$")
    axes[0, 1].plot(time_us, resistance, color=ORANGE, lw=1.8)
    axes[0, 1].scatter(
        1e6 * profile["time_s"], profile["resistance_ohm"], color=INK, s=18, zorder=3, label="declared points"
    )
    axes[0, 1].set_xlim(0.0, 2.5)
    axes[0, 1].set_xticks(np.arange(0.0, 2.51, 0.5))
    axes[0, 1].set_ylabel(r"$R_{ch}(t)$ ($\Omega$)")
    axes[0, 1].legend(frameon=False, loc="upper right")
    _format_axes(axes[0, 1])

    add_panel_title(axes[1, 0], "(c)", "Output voltage")
    axes[1, 0].plot(time_us, expected_voltage, color=ORANGE, lw=2.0, ls=(0, (4, 2)), label="divider equation")
    axes[1, 0].plot(time_us, waveform["voltage_V"], color=BLUE, lw=1.0, label="ngspice")
    axes[1, 0].set_xlim(0.0, 2.5)
    axes[1, 0].set_xticks(np.arange(0.0, 2.51, 0.5))
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel(r"$V_{out}$ (V)")
    axes[1, 0].legend(frameon=False, loc="upper right")
    _format_axes(axes[1, 0])

    add_panel_title(axes[1, 1], "(d)", "Chamber-element current")
    axes[1, 1].plot(
        time_us,
        1e3 * expected_current,
        color=ORANGE,
        lw=2.0,
        ls=(0, (4, 2)),
        label="divider equation",
    )
    axes[1, 1].plot(
        time_us,
        1e3 * waveform["component_Rchamber_current_A"],
        color=BLUE,
        lw=1.0,
        label="ngspice",
    )
    axes[1, 1].set_xlim(0.0, 2.5)
    axes[1, 1].set_xticks(np.arange(0.0, 2.51, 0.5))
    axes[1, 1].set_xlabel("Time (us)")
    axes[1, 1].set_ylabel(r"$I_{ch}$ (mA)")
    axes[1, 1].legend(frameon=False, loc="upper right")
    _format_axes(axes[1, 1])
    add_figure_title(
        fig,
        "A prescribed component trajectory is used during the transient solve",
        "The independent divider equation is evaluated at every saved time point; no fitted surrogate is used.",
    )
    add_figure_footer(
        fig,
        f"ngspice-46; {comparison['samples']} saved samples; max voltage error="
        f"{comparison['max_abs_voltage_error_V']:.2e} V; max current error="
        f"{comparison['max_abs_current_error_A']:.2e} A.",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.12, hspace=0.52, wspace=0.3)
    return fig


def _plot_impedance_component(
    axis: plt.Axes,
    rows: list[dict[str, float | str]],
    component: str,
    *,
    show_legend: bool = False,
) -> None:
    x = np.asarray([float(row["frequency_Hz"]) / 1e6 for row in rows])
    expected = np.asarray([float(row[f"expected_{component}_ohm"]) for row in rows])
    observed = np.asarray([float(row[f"ngspice_{component}_ohm"]) for row in rows])
    axis.scatter(x, expected, facecolors="none", edgecolors=ORANGE, s=46, linewidths=1.5, label="analytic expectation")
    axis.scatter(x, observed, color=BLUE, marker="x", s=42, linewidths=1.4, label="ngspice")
    axis.set_xticks(x, [f"{value:g}" for value in x])
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel(r"$R_{in}$ ($\Omega$)" if component == "resistance" else r"$X_{in}$ ($\Omega$)")
    if show_legend:
        axis.legend(frameon=False, loc="best")
    _format_axes(axis)


def figure_effective_plasma_loads(data: dict[str, Any]) -> plt.Figure:
    evidence = data["effective_plasma_loads"]
    fig = plt.figure(figsize=PAGE_SIZE)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.78, 1.0, 1.0])
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)])
    add_panel_title(axes[0, 0], "(a)", "Effective CCP terminal model")
    _draw_ccp(axes[0, 0])
    add_panel_title(axes[0, 1], "(b)", "Reduced ICP terminal model")
    _draw_icp(axes[0, 1])
    add_panel_title(axes[1, 0], "(c)", "CCP input resistance")
    _plot_impedance_component(axes[1, 0], evidence["ccp"], "resistance", show_legend=True)
    add_panel_title(axes[1, 1], "(d)", "ICP input resistance")
    _plot_impedance_component(axes[1, 1], evidence["icp"], "resistance")
    add_panel_title(axes[2, 0], "(e)", "CCP input reactance")
    _plot_impedance_component(axes[2, 0], evidence["ccp"], "reactance")
    add_panel_title(axes[2, 1], "(f)", "ICP input reactance")
    _plot_impedance_component(axes[2, 1], evidence["icp"], "reactance")
    add_figure_title(
        fig,
        "Effective plasma-load circuits propagate correctly through ngspice",
        "Each declared load feeds the fixed benchmark L-match; points compare whole-network input Z, not plasma-state physics.",
    )
    add_figure_footer(
        fig,
        f"Open circles: independent load + L-match equations. Crosses: ngspice extraction. Max |Z-component error|: "
        f"CCP {evidence['ccp_max_abs_impedance_error_ohm']:.2e} ohm; "
        f"ICP {evidence['icp_max_abs_impedance_error_ohm']:.2e} ohm.",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.12, hspace=0.75, wspace=0.3)
    return fig


def figure_target_optimization(sources: dict[str, Any], data: dict[str, Any]) -> plt.Figure:
    optimization = data["optimization"]
    evaluations = optimization["evaluations"]
    target = _read_numeric_csv(sources["optimization"]["target"])
    baseline_path = ROOT / next(item["waveform"] for item in evaluations if item["trial"] == 0)
    selected_path = ROOT / next(
        item["waveform"] for item in evaluations if item["candidate_id"] == optimization["selected_candidate_id"]
    )
    baseline = _read_numeric_csv(baseline_path)
    selected = _read_numeric_csv(selected_path)
    trials = np.asarray([int(item["trial"]) for item in evaluations])
    losses = np.asarray([float(item["normalized_rmse"]) for item in evaluations])
    best_so_far = np.minimum.accumulate(losses)

    fig = plt.figure(figsize=PAGE_SIZE)
    grid = fig.add_gridspec(2, 3, width_ratios=[0.88, 1.0, 1.0])
    schematic = fig.add_subplot(grid[0, 0])
    waveform = fig.add_subplot(grid[0, 1:])
    history = fig.add_subplot(grid[1, 0])
    design = fig.add_subplot(grid[1, 1:])
    add_panel_title(schematic, "(a)", "Optimized RC circuit")
    _draw_rc(schematic, r"$R_1^*=1\,k\Omega$", r"$C_1^*=300\,pF$")

    add_panel_title(waveform, "(b)", "Target-waveform agreement")
    waveform.plot(1e6 * baseline["time_s"], baseline["voltage_V"], color=MUTED, lw=1.0, label="first grid point")
    waveform.plot(1e6 * target["time_s"], target["voltage_V"], color=ORANGE, lw=2.0, ls=(0, (4, 2)), label="target")
    waveform.plot(
        1e6 * selected["time_s"], selected["voltage_V"], color=BLUE, lw=1.1, label="selected ngspice waveform"
    )
    waveform.set_xlim(0.0, 2.0)
    waveform.set_xticks(np.arange(0.0, 2.01, 0.5))
    waveform.set_xlabel("Time (us)")
    waveform.set_ylabel(r"$V_{out}$ (V)")
    waveform.legend(frameon=False, loc="lower right")
    _format_axes(waveform)

    add_panel_title(history, "(c)", "Search loss history")
    history.semilogy(trials + 1, losses, color=MUTED, marker="o", ms=3.8, lw=0.8, label="evaluated")
    history.step(trials + 1, best_so_far, where="post", color=BLUE, lw=1.6, label="best so far")
    history.set_xlabel("Grid evaluation")
    history.set_ylabel("Normalized RMSE")
    history.set_xticks(trials + 1)
    history.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.03, 0.42),
        borderaxespad=0.0,
        handlelength=1.6,
        labelspacing=0.25,
        fontsize=6.8,
    )
    _format_axes(history)

    add_panel_title(design, "(d)", "Complete component-value grid")
    resistances = sorted({float(item["R1_ohm"]) for item in evaluations})
    capacitances = sorted({float(item["C1_F"]) for item in evaluations})
    matrix = np.empty((len(resistances), len(capacitances)))
    for row_index, resistance in enumerate(resistances):
        for column_index, capacitance in enumerate(capacitances):
            match = next(
                item
                for item in evaluations
                if float(item["R1_ohm"]) == resistance and float(item["C1_F"]) == capacitance
            )
            matrix[row_index, column_index] = float(match["normalized_rmse"])
    image = design.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=float(matrix.max()))
    del image
    design.set_xticks(range(len(capacitances)), [f"{1e12 * value:g}" for value in capacitances])
    design.set_yticks(range(len(resistances)), [f"{value / 1e3:g}" for value in resistances])
    design.set_xlabel(r"$C_1$ (pF)")
    design.set_ylabel(r"$R_1$ (k$\Omega$)")
    for row_index, resistance in enumerate(resistances):
        for column_index, capacitance in enumerate(capacitances):
            value = matrix[row_index, column_index]
            design.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.28 else INK,
                fontsize=7.2,
            )
            if math.isclose(resistance, float(optimization["selected_R1_ohm"])) and math.isclose(
                capacitance, float(optimization["selected_C1_F"])
            ):
                design.add_patch(
                    mpatches.Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=BLUE,
                        linewidth=2.2,
                    )
                )
    add_figure_title(
        fig,
        "Element optimization recovers the RC time constant behind the target waveform",
        "A finite 3 x 3 component grid is solved exhaustively; the selected waveform is an ngspice result, not a surrogate prediction.",
    )
    add_figure_footer(
        fig,
        f"Selected R1={optimization['selected_R1_ohm'] / 1e3:g} kOhm, "
        f"C1={1e12 * optimization['selected_C1_F']:g} pF, tau="
        f"{1e9 * optimization['selected_time_constant_s']:.0f} ns; "
        f"nRMSE={optimization['selected_normalized_rmse']:.4f}; "
        f"{optimization['n_evaluations']}/{optimization['n_evaluations']} ngspice evaluations succeeded.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.12, hspace=0.52, wspace=0.45)
    return fig


def _export(figure: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    assert_text_inside_canvas(figure)
    svg_path = output / f"{stem}.svg"
    figure.savefig(
        svg_path,
        metadata={
            "Title": stem.replace("-", " "),
            "Creator": "PCD calculation-evidence generator",
            "Description": "Direct circuit calculation evidence",
            "Date": None,
        },
    )
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output / f"{stem}.png", dpi=300, metadata={"Software": "PCD calculation-evidence generator"})
    pdf.savefig(figure)
    plt.close(figure)


def generate(
    benchmark_result: Path,
    varying_root: Path,
    netlist_root: Path,
    optimization_root: Path,
    output: Path,
    pdf_output: Path,
) -> dict[str, Any]:
    sources = collect_sources(benchmark_result, varying_root, netlist_root, optimization_root)
    data = build_figure_data(sources)
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT
    data_path = output / "figure_data.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figures = [
        ("01-netlist-to-ngspice-waveform", figure_netlist_workflow(sources, data)),
        ("02-time-varying-element-conformance", figure_time_varying_element(sources, data)),
        ("03-effective-plasma-load-conformance", figure_effective_plasma_loads(data)),
        ("04-target-waveform-optimization", figure_target_optimization(sources, data)),
    ]
    with PdfPages(
        pdf_output,
        metadata={
            "Title": "PCD direct calculation evidence",
            "Author": "PCD calculation-evidence generator",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:
        for stem, figure in figures:
            _export(figure, output, stem, pdf)
    return {
        "output": str(output.resolve()),
        "figure_count": len(figures),
        "formats": ["svg", "png", "pdf"],
        "data": str(data_path.resolve()),
        "pdf": str(pdf_output.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-result", type=Path, default=DEFAULT_BENCHMARK_RESULT)
    parser.add_argument("--varying-root", type=Path, default=DEFAULT_VARYING_ROOT)
    parser.add_argument("--netlist-root", type=Path, default=DEFAULT_NETLIST_ROOT)
    parser.add_argument("--optimization-root", type=Path, default=DEFAULT_OPTIMIZATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    args = parser.parse_args(argv)
    result = generate(
        args.benchmark_result.resolve(),
        args.varying_root.resolve(),
        args.netlist_root.resolve(),
        args.optimization_root.resolve(),
        args.output.resolve(),
        args.pdf_output.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
