"""Generate publication-style circuit and benchmark figures.

The figures are derived from one completed core benchmark run.  No curve is
interpolated through independent scenarios, and no corner count is interpreted
as a probability or yield estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from pcd.analysis import AC_LOAD_VOLTAGE, ac_probe_plan, read_ac
from pcd.case import load_case
from pcd.figures import (
    BLUE,
    BLUE_LIGHT,
    GRID,
    INK,
    MUTED,
    ORANGE,
    ORANGE_LIGHT,
    PAGE_SIZE,
    WHITE,
    CircuitDiagram,
    CircuitViewport,
    assert_text_inside_canvas,
)
from pcd.figures import (
    add_figure_footer as _footer,
)
from pcd.figures import (
    add_figure_title as _title,
)
from pcd.figures import (
    add_panel_title as _panel,
)
from pcd.figures import (
    configure_publication_style as _configure_style,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "benchmark_suite"
DEFAULT_OUTPUT = HERE / "generated"
DEFAULT_PDF_OUTPUT = ROOT / "output" / "pdf"

CASE_IDS = {
    "B1": "B1_fixed_nominal",
    "B2": "B2_limited_tuner",
    "B3": "B3_full_tuner",
    "A5": "A5_icp_transformer_frequency_conformance",
    "B5": "B5_high_drive_stress",
    "B8": "B8_component_value_corner_stress",
    "D1": "D1_reference_plane_explicit",
    "D2": "D2_reference_plane_embedded",
    "D3": "D3_reference_plane_double_counted",
}

TOPOLOGY_VIEWPORT = CircuitViewport(xlim=(-0.35, 5.45), ylim=(-3.05, 1.0))
LOAD_VIEWPORT = CircuitViewport(xlim=(-0.35, 5.15), ylim=(-1.7, 0.9))

SCENARIO_ORDER = [
    "high_R_strong_capacitive",
    "nominal",
    "low_R_weak_capacitive",
    "low_R_nominal_X",
    "high_R_nominal_X",
]
SCENARIO_CODES = {
    "high_R_strong_capacitive": "HS",
    "nominal": "N",
    "low_R_weak_capacitive": "LW",
    "low_R_nominal_X": "LN",
    "high_R_nominal_X": "HN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repository_path(path: str | Path) -> str:
    """Return a portable repository-relative path for committed provenance."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"figure provenance must stay inside the repository: {resolved}") from exc


def _suite_case(payload: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    for item in payload["cases"]:
        if item["benchmark_id"] == benchmark_id:
            result = dict(item)
            if result.get("case_path"):
                result["case_path"] = _repository_path(result["case_path"])
            return result
    raise KeyError(f"benchmark result is missing {benchmark_id}")


def _candidate_for_case(run_root: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    for path in sorted(run_root.glob("*/candidates/trial_*.json")):
        payload = _read_json(path)
        scenarios = payload.get("scenarios") or []
        if not scenarios:
            continue
        selected = scenarios[0].get("selected") or {}
        observations = (selected.get("raw") or {}).get("observations") or {}
        if observations.get("case_id") == case_id:
            return payload, path
    raise FileNotFoundError(f"candidate artifact not found for {case_id} under {run_root}")


def _scenario_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["scenario_id"]): dict(item) for item in case["scenarios"]}


def _archived_case_for_case_id(run_root: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    for path in sorted(run_root.glob("*/case.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("case_id") == case_id:
            return dict(payload), path
    raise FileNotFoundError(f"archived case not found for {case_id} under {run_root}")


def _load_scenarios_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in candidate["scenarios"]:
        scenario = item["scenario"]
        values = scenario["values"]
        rows.append(
            {
                "scenario_id": str(scenario["scenario_id"]),
                "resistance_ohm": float(values["load_resistance_ohm"]),
                "reactance_ohm": float(values["load_reactance_ohm"]),
            }
        )
    return rows


def _tuning_grid_from_candidate(candidate: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, set[float]] = {"C1": set(), "C2": set()}
    for scenario in candidate["scenarios"]:
        for trial in scenario["trials"]:
            control = trial["request"]["control"]["values"]
            for name in values:
                values[name].add(float(control[name]))
    return {name: sorted(items) for name, items in values.items()}


def _scenario_scalar_from_candidate(candidate: dict[str, Any], key: str, scale: float = 1.0) -> dict[str, float]:
    return {
        str(item["scenario"]["scenario_id"]): float(item["scenario"]["values"][key]) * scale
        for item in candidate["scenarios"]
    }


def _candidate_artifact(candidate_path: Path, item: dict[str, Any], key: str) -> Path:
    raw = str(item["selected"]["raw"]["artifacts"][key])
    relative = Path(*PureWindowsPath(raw).parts)
    path = candidate_path.parents[1] / relative
    if not path.is_file():
        raise FileNotFoundError(f"candidate artifact is missing: {path}")
    return path


def _phasor_payload(value: complex) -> dict[str, float]:
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "peak": float(abs(value)),
        "phase_deg": float(np.degrees(np.angle(value))),
    }


def _extract_b5(candidate: dict[str, Any], candidate_path: Path) -> dict[str, Any]:
    case = load_case(candidate_path.parents[1] / "case.yaml")
    ac_columns = ac_probe_plan(case)[1]
    extra_columns = [AC_LOAD_VOLTAGE, *ac_columns] if ac_columns else None
    rows = []
    for item in candidate["scenarios"]:
        selected = item["selected"]
        ac_path = _candidate_artifact(candidate_path, item, "frequency_response")
        response = read_ac(ac_path, extra_columns)
        if len(response) != 1:
            raise ValueError(f"B5 waveform figure requires one AC phasor, got {len(response)} rows: {ac_path}")
        ac_row = response.iloc[0]
        source_voltage = complex(float(ac_row["voltage_re"]), float(ac_row["voltage_im"]))
        source_current = -complex(float(ac_row["current_re"]), float(ac_row["current_im"]))
        load_voltage = complex(
            float(ac_row[f"{AC_LOAD_VOLTAGE}_re"]),
            float(ac_row[f"{AC_LOAD_VOLTAGE}_im"]),
        )
        load_current = complex(
            float(ac_row["load_current_A_re"]),
            float(ac_row["load_current_A_im"]),
        )
        source_power = 0.5 * float(np.real(source_voltage * np.conj(source_current)))
        load_power = 0.5 * float(np.real(load_voltage * np.conj(load_current)))
        derived = {
            "source_real_power_W": source_power,
            "load_real_power_W": load_power,
            "network_loss_W": source_power - load_power,
        }
        for metric_name, value in derived.items():
            expected = float(selected["metrics"][metric_name])
            if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"B5 {metric_name} disagrees with archived metrics: {value} != {expected}")
        rows.append(
            {
                "scenario_id": item["scenario"]["scenario_id"],
                "drive_peak_V": float(item["scenario"]["values"]["drive_amplitude_V"]),
                "frequency_Hz": float(ac_row["frequency_Hz"]),
                "reflection_magnitude": float(selected["metrics"]["reflection_magnitude"]),
                "phasors": {
                    "source_voltage_V": _phasor_payload(source_voltage),
                    "load_voltage_V": _phasor_payload(load_voltage),
                    "source_current_A": _phasor_payload(source_current),
                    "load_current_A": _phasor_payload(load_current),
                },
                "ac_sha256": _sha256(ac_path),
                "metrics": {
                    name: float(selected["metrics"][name])
                    for name in (
                        "source_real_power_W",
                        "load_real_power_W",
                        "network_loss_W",
                        "transfer_efficiency",
                        "component_L1_current_rms_A",
                        "component_L1_loss_W",
                        "source_current_rms_A",
                        "source_apparent_power_VA",
                    )
                },
            }
        )
    metric_names = (
        "component_L1_current_rms_A",
        "component_L1_loss_W",
        "source_current_rms_A",
        "source_apparent_power_VA",
    )
    constraint_names = {name: f"max_{name}" for name in metric_names}
    limit_sets: dict[str, set[float]] = {name: set() for name in metric_names}
    for item in candidate["scenarios"]:
        constraints = {row["name"]: row for row in item["selected"]["constraints"]}
        for metric_name, constraint_name in constraint_names.items():
            limit_sets[metric_name].add(float(constraints[constraint_name]["limit"]))
    if any(len(items) != 1 for items in limit_sets.values()):
        raise ValueError("B5 engineering limits differ between scenarios")
    return {
        "scenarios": rows,
        "limits": {name: next(iter(items)) for name, items in limit_sets.items()},
    }


def _extract_b8(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in candidate["scenarios"]:
        selected = item["selected"]
        values = item["scenario"]["values"]
        rows.append(
            {
                "scenario_id": item["scenario"]["scenario_id"],
                "C1_factor": float(values["C1_factor"]),
                "L1_factor": float(values["L1_factor"]),
                "C2_factor": float(values["C2_factor"]),
                "reflection_magnitude": float(selected["metrics"]["reflection_magnitude"]),
                "input_resistance_ohm": float(selected["metrics"]["resistance_ohm"]),
                "input_reactance_ohm": float(selected["metrics"]["reactance_ohm"]),
                "feasible": all(bool(row["satisfied"]) for row in selected["constraints"]),
            }
        )
    return rows


def _component_value(case: dict[str, Any], ref: str) -> float:
    for component in case["circuit"]["components"]:
        if component["ref"] == ref:
            return float(component["value"])
    raise KeyError(f"component {ref} not found in archived case {case['case_id']}")


def _reference_circuit_data(run_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    case_ids = {
        "D1": "benchmark_reference_plane_fixture_explicit",
        "D2": "benchmark_reference_plane_fixture_embedded",
        "D3": "benchmark_reference_plane_fixture_double_counted",
    }
    cases: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, case_id in case_ids.items():
        cases[name], paths[name] = _archived_case_for_case_id(run_root, case_id)

    reference_values = {float(case["measurement"]["reference_impedance_ohm"]) for case in cases.values()}
    if len(reference_values) != 1:
        raise ValueError("D1-D3 use different reference impedances")

    return (
        {
            "reference_impedance_ohm": next(iter(reference_values)),
            "fixture_resistance_ohm": _component_value(cases["D1"], "Rfixture"),
            "fixture_inductance_H": _component_value(cases["D1"], "Lfixture"),
            "plasma_resistance_ohm": float(cases["D1"]["load"]["resistance_ohm"]),
            "plasma_reactance_ohm": float(cases["D1"]["load"]["reactance_ohm"]),
            "embedded_resistance_ohm": float(cases["D2"]["load"]["resistance_ohm"]),
            "embedded_reactance_ohm": float(cases["D2"]["load"]["reactance_ohm"]),
        },
        {name: _sha256(path) for name, path in paths.items()},
    )


def build_figure_data(run_root: Path) -> dict[str, Any]:
    result_path = run_root / "benchmark_result.json"
    suite = _read_json(result_path)
    b2_candidate, b2_path = _candidate_for_case(run_root, "benchmark_match_limited_tuner")
    b3_candidate, b3_path = _candidate_for_case(run_root, "benchmark_match_full_tuner")
    a5_candidate, a5_path = _candidate_for_case(run_root, "benchmark_icp_transformer_frequency_conformance")
    b5_candidate, b5_path = _candidate_for_case(run_root, "benchmark_match_high_drive_stress")
    b8_candidate, b8_path = _candidate_for_case(run_root, "benchmark_component_value_corner_stress")
    reference_circuit, reference_case_hashes = _reference_circuit_data(run_root)
    b8_reference_impedance = float(b8_candidate["scenarios"][0]["selected"]["metrics"]["reference_impedance_ohm"])
    if not math.isclose(
        b8_reference_impedance,
        reference_circuit["reference_impedance_ohm"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("B8 and D1-D3 use different reference impedances")
    return {
        "schema": "pcd.benchmark_figures.v3",
        "rendering": {
            "layout_schema": "pcd.publication.v1",
            "page_size_inches": list(PAGE_SIZE),
            "python": platform.python_version(),
            "matplotlib": mpl.__version__,
            "schemdraw": schemdraw.__version__,
            "symbol_standard": "IEEE",
        },
        "source": {
            "path_style": "repository-relative POSIX",
            "run_root": _repository_path(run_root),
            "platform_version": suite["platform_version"],
            "solver": suite["solver"],
            "generated_at": suite["generated_at"],
            "benchmark_result_sha256": _sha256(result_path),
            "candidate_sha256": {
                "B2": _sha256(b2_path),
                "B3": _sha256(b3_path),
                "A5": _sha256(a5_path),
                "B5": _sha256(b5_path),
                "B8": _sha256(b8_path),
            },
            "reference_case_sha256": reference_case_hashes,
        },
        "acceptance": suite["acceptance"],
        "reference_impedance_ohm": reference_circuit["reference_impedance_ohm"],
        "benchmark_cases": [
            {
                key: item[key]
                for key in (
                    "benchmark_id",
                    "role",
                    "question",
                    "demonstrates",
                    "does_not_establish",
                    "passed",
                    "feasible",
                    "n_evaluations",
                    "n_failed_evaluations",
                )
            }
            for item in suite["cases"]
        ],
        "reference_circuit": reference_circuit,
        "load_scenarios": _load_scenarios_from_candidate(b2_candidate),
        "control_cases": {name: _suite_case(suite, CASE_IDS[name]) for name in ("B1", "B2", "B3")},
        "control_grids": {
            "B2": _tuning_grid_from_candidate(b2_candidate),
            "B3": _tuning_grid_from_candidate(b3_candidate),
        },
        "a5": {
            "result": _suite_case(suite, CASE_IDS["A5"]),
            "frequency_MHz": _scenario_scalar_from_candidate(a5_candidate, "rf_frequency_Hz", 1e-6),
        },
        "b5": _extract_b5(b5_candidate, b5_path),
        "b8": _extract_b8(b8_candidate),
        "reference_plane": {name: _suite_case(suite, CASE_IDS[name]) for name in ("D1", "D2", "D3")},
    }


def _draw_l_match(ax: plt.Axes) -> None:
    circuit = CircuitDiagram(ax)
    source = circuit.anchor("source", (0.0, 0.0))
    l1_start = circuit.anchor("l1_start", (0.7, 0.0))
    junction = circuit.anchor("junction", (3.0, 0.0))
    load = circuit.anchor("load", (4.2, 0.0))
    shunt_ground = circuit.anchor("c1_ground", (3.0, -1.55))
    circuit.port(source, "src", "left")
    circuit.wire(source, l1_start)
    circuit.component(elm.Inductor, l1_start, junction, label=r"$L_1$", name="L1")
    circuit.node(junction)
    circuit.component(
        elm.Capacitor,
        junction,
        shunt_ground,
        label=r"$C_1$",
        label_location="left",
        name="C1",
    )
    circuit.ground(shunt_ground)
    circuit.wire(junction, load)
    circuit.port(load, "load", "right")
    circuit.finish(TOPOLOGY_VIEWPORT)


def _draw_pi_match(ax: plt.Axes, harmonic: bool = False) -> None:
    circuit = CircuitDiagram(ax)
    source = circuit.anchor("source", (0.0, 0.0))
    input_node = circuit.anchor("input_node", (0.7, 0.0))
    output_node = circuit.anchor("output_node", (3.0, 0.0))
    c1_ground = circuit.anchor("c1_ground", (0.7, -1.55))
    c2_ground = circuit.anchor("c2_ground", (3.0, -1.55))
    load = circuit.anchor("load", (5.1 if harmonic else 4.2, 0.0))
    circuit.port(source, "src", "left")
    circuit.wire(source, input_node)
    circuit.node(input_node)
    circuit.component(
        elm.Capacitor,
        input_node,
        c1_ground,
        label=r"$C_1$",
        label_location="left",
        name="C1",
    )
    circuit.ground(c1_ground)
    circuit.component(elm.Inductor, input_node, output_node, label=r"$L_1$", name="L1")
    circuit.node(output_node)
    circuit.component(
        elm.Capacitor,
        output_node,
        c2_ground,
        label=r"$C_2$",
        label_location="left",
        name="C2",
    )
    circuit.ground(c2_ground)
    circuit.wire(output_node, load)
    circuit.port(load, "load", "right")
    if harmonic:
        harmonic_node = circuit.anchor("harmonic_node", (4.1, 0.0))
        harmonic_mid = circuit.anchor("harmonic_mid", (4.1, -1.3))
        harmonic_ground = circuit.anchor("harmonic_ground", (4.1, -2.6))
        circuit.node(harmonic_node)
        circuit.component(
            elm.Inductor,
            harmonic_node,
            harmonic_mid,
            label=r"$L_h$",
            label_location="right",
            name="Lh",
        )
        circuit.component(
            elm.Capacitor,
            harmonic_mid,
            harmonic_ground,
            label=r"$C_h$",
            label_location="left",
            name="Ch",
        )
        circuit.ground(harmonic_ground)
    circuit.finish(TOPOLOGY_VIEWPORT)


def figure_analysis_boundary(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.45], hspace=0.55, wspace=0.34)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    _panel(ax, "(a)", "Analysis boundary")
    y = 1.3
    ax.plot([0.5, 9.5], [y, y], color=INK, lw=1.25)
    ax.add_patch(mpatches.Circle((0.5, y), 0.07, facecolor=WHITE, edgecolor=INK, lw=1.1))
    ax.text(0.42, y + 0.35, r"$V_{src}, I_{src}$", ha="left")
    ax.annotate("", xy=(1.6, y + 0.13), xytext=(0.9, y + 0.13), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.text(1.22, y + 0.3, r"$I_{src}$", ha="center")
    ax.add_patch(mpatches.Rectangle((2.0, 0.72), 2.15, 1.16, facecolor=WHITE, edgecolor=INK, lw=1.1))
    ax.text(3.075, 1.36, "Candidate\nmatching network", ha="center", va="center")
    ax.annotate(
        "Control state",
        xy=(3.05, 1.9),
        xytext=(3.05, 2.55),
        ha="center",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": MUTED},
        color=MUTED,
    )
    ax.axvline(5.2, ymin=0.16, ymax=0.78, color=BLUE, lw=1.15, ls=(0, (4, 3)))
    ax.text(5.2, 2.32, "declared load\nreference plane", ha="center", va="center", color=BLUE)
    ax.add_patch(mpatches.Rectangle((6.25, 0.72), 2.05, 1.16, facecolor=WHITE, edgecolor=INK, lw=1.1))
    ax.text(7.275, 1.36, "Scenario one-port\n" + r"$Z_{load}(f)$", ha="center", va="center")
    ax.plot([9.5, 9.5], [y, 0.45], color=INK, lw=1.25)
    ax.plot([9.25, 9.75], [0.45, 0.45], color=INK, lw=1.1)
    ax.plot([9.32, 9.68], [0.32, 0.32], color=INK, lw=1.1)
    ax.plot([9.4, 9.6], [0.19, 0.19], color=INK, lw=1.1)
    ax.text(
        0.5,
        0.22,
        rf"$Z_{{in}}=V_{{src}}/I_{{src}}$;  $Z_0={data['reference_impedance_ohm']:g}\,\Omega$"
        " is a calculation reference, not a drawn series resistor",
        ha="left",
        color=MUTED,
    )

    axes = [fig.add_subplot(gs[1, index]) for index in range(3)]
    _panel(axes[0], "(b)", "L-match")
    _draw_l_match(axes[0])
    _panel(axes[1], "(c)", "Pi-match")
    _draw_pi_match(axes[1])
    _panel(axes[2], "(d)", "Pi-match + harmonic branch")
    _draw_pi_match(axes[2], harmonic=True)
    _title(
        fig,
        "RF analysis boundary and matching topologies",
        "Connections used by the public circuit builders; the chamber enters only as a declared electrical one-port.",
    )
    _footer(
        fig,
        "Source: pcd.sim_methods circuit builders. A1-A3 are topology-conformance fixtures, not a topology ranking.",
    )
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.055, right=0.94)
    return fig


def _draw_impedance_point(ax: plt.Axes) -> None:
    circuit = CircuitDiagram(ax)
    positive = circuit.anchor("p", (0.0, 0.0))
    midpoint = circuit.anchor("reactive_start", (1.75, 0.0))
    reactive_end = circuit.anchor("reactive_end", (3.5, 0.0))
    negative = circuit.anchor("n", (4.45, 0.0))
    circuit.port(positive, "p", "left")
    circuit.component(elm.Resistor, positive, midpoint, label=r"$R_{point}$", name="Rpoint")
    circuit.component(elm.RBox, midpoint, reactive_end, label=r"$jX(f_0)$", name="jX")
    circuit.wire(reactive_end, negative)
    circuit.port(negative, "n", "right")
    circuit.finish(LOAD_VIEWPORT)
    ax.text(
        0.5,
        -0.08,
        r"$X>0:\ L=X/\omega_0$    $X<0:\ C=-1/(\omega_0X)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )


def _draw_ccp(ax: plt.Axes) -> None:
    circuit = CircuitDiagram(ax)
    positive = circuit.anchor("p", (0.0, 0.0))
    resistance_end = circuit.anchor("resistance_end", (1.35, 0.0))
    inductance_end = circuit.anchor("inductance_end", (2.75, 0.0))
    sheath_end = circuit.anchor("sheath_end", (4.1, 0.0))
    negative = circuit.anchor("n", (4.45, 0.0))
    circuit.port(positive, "p", "left")
    circuit.component(elm.Resistor, positive, resistance_end, label=r"$R_{eff}$", name="Reff")
    circuit.component(elm.Inductor, resistance_end, inductance_end, label=r"$L_{eff}$", name="Leff")
    circuit.component(
        elm.Capacitor,
        inductance_end,
        sheath_end,
        label=r"$C_{sheath,eq}$",
        name="Csheath",
    )
    circuit.wire(sheath_end, negative)
    circuit.port(negative, "n", "right")
    circuit.finish(LOAD_VIEWPORT)
    ax.text(
        0.5,
        -0.08,
        r"$Z=R_{eff}+j\omega L_{eff}+1/(j\omega C_{sheath,eq})$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )


def _draw_icp(ax: plt.Axes) -> None:
    circuit = CircuitDiagram(ax)
    positive = circuit.anchor("p", (0.0, 0.0))
    input_node = circuit.anchor("input_node", (0.2, 0.0))
    resistance_end = circuit.anchor("resistance_end", (1.45, 0.0))
    inductance_end = circuit.anchor("inductance_end", (2.8, 0.0))
    reflected_end = circuit.anchor("reflected_end", (4.1, 0.0))
    output_node = circuit.anchor("output_node", (4.6, 0.0))
    negative = circuit.anchor("n", (4.8, 0.0))
    parallel_start = circuit.anchor("parallel_start", (0.2, -1.2))
    parallel_end = circuit.anchor("parallel_end", (4.6, -1.2))
    circuit.port(positive, "p", "left")
    circuit.wire(positive, input_node)
    circuit.node(input_node)
    circuit.component(elm.Resistor, input_node, resistance_end, label=r"$R_{coil}$", name="Rcoil")
    circuit.component(elm.Inductor, resistance_end, inductance_end, label=r"$L_{coil}$", name="Lcoil")
    circuit.component(elm.RBox, inductance_end, reflected_end, label=r"$Z_{ref}(\omega)$", name="Zref")
    circuit.wire(reflected_end, output_node)
    circuit.node(output_node)
    circuit.wire(output_node, negative)
    circuit.port(negative, "n", "right")
    circuit.wire(input_node, parallel_start)
    circuit.component(
        elm.Capacitor,
        parallel_start,
        parallel_end,
        label=r"$C_{parallel}$",
        name="Cparallel",
    )
    circuit.wire(parallel_end, output_node)
    circuit.finish(LOAD_VIEWPORT)
    ax.text(
        0.5,
        -0.08,
        r"$Z_{ref}=\omega^2L_{reflected}/(\gamma+j\omega)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )


def figure_load_models() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=PAGE_SIZE)
    _panel(axes[0], "(a)", "Impedance point")
    _draw_impedance_point(axes[0])
    _panel(axes[1], "(b)", "Effective CCP one-port")
    _draw_ccp(axes[1])
    _panel(axes[2], "(c)", "Reduced ICP coil one-port")
    _draw_icp(axes[2])
    _title(
        fig,
        "Electrical load models used by the core benchmarks",
        "Each load is a named-reference-plane one-port; geometry, plasma state and species-power partition remain outside scope.",
    )
    _footer(
        fig,
        "Scope: A4 checks the series CCP terminal form; A5 checks the identifiable ICP reduction.\n"
        "ICP transformer lineage: Lee et al. (2020), DOI 10.1063/1.5133862.",
    )
    fig.subplots_adjust(top=0.7, bottom=0.2, left=0.045, right=0.985, wspace=0.34)
    return fig


def _block(ax: plt.Axes, xy: tuple[float, float], width: float, text: str, edge: str = INK) -> None:
    x, y = xy
    ax.add_patch(mpatches.Rectangle((x, y - 0.34), width, 0.68, facecolor=WHITE, edgecolor=edge, lw=1.05, zorder=3))
    ax.text(x + width / 2, y, text, ha="center", va="center", zorder=4)


def _reference_plane(ax: plt.Axes, x: float, label: str) -> None:
    ax.axvline(x, ymin=0.14, ymax=0.86, color=BLUE, lw=1.05, ls=(0, (4, 3)))
    ax.text(x, 0.9, label, ha="center", va="bottom", color=BLUE, transform=ax.get_xaxis_transform())


def _ground_termination(ax: plt.Axes, x: float) -> None:
    ax.plot([x, x], [0, -0.48], color=INK, lw=1.1, zorder=2)
    ax.plot([x - 0.22, x + 0.22], [-0.48, -0.48], color=INK, lw=1.0, zorder=2)
    ax.plot([x - 0.15, x + 0.15], [-0.58, -0.58], color=INK, lw=1.0, zorder=2)
    ax.plot([x - 0.08, x + 0.08], [-0.68, -0.68], color=INK, lw=1.0, zorder=2)


def figure_reference_circuits(data: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=PAGE_SIZE, sharex=True)
    for ax in axes:
        ax.set_xlim(0, 12)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        _block(ax, (0.8, 0), 2.1, "Pi matcher")

    d1 = data["reference_plane"]["D1"]
    d2 = data["reference_plane"]["D2"]
    d3 = data["reference_plane"]["D3"]
    circuit = data["reference_circuit"]
    _panel(axes[0], "(a)", "D1 — fixture explicit")
    axes[0].plot([0.4, 10.65], [0, 0], color=INK, lw=1.2)
    _reference_plane(axes[0], 3.8, "electrode plane")
    _block(
        axes[0],
        (4.35, 0),
        1.1,
        rf"$R_f={circuit['fixture_resistance_ohm']:g}\,\Omega$",
    )
    _block(
        axes[0],
        (5.8, 0),
        1.6,
        rf"$L_f={circuit['fixture_inductance_H'] * 1e9:.3f}\,nH$",
    )
    _reference_plane(axes[0], 8.0, "plasma plane")
    _block(
        axes[0],
        (8.55, 0),
        2.1,
        rf"$Z_p={circuit['plasma_resistance_ohm']:g}"
        rf"{circuit['plasma_reactance_ohm']:+g}j\,\Omega$",
    )
    axes[0].text(
        11.65,
        0.55,
        f"$Z_{{in}}={d1['scenarios'][0]['input_resistance_ohm']:.3f}"
        f"{d1['scenarios'][0]['input_reactance_ohm']:+.3f}j$ $\\Omega$\n"
        f"$|\\Gamma|={d1['worst_reflection_magnitude']:.4f}$",
        ha="right",
        va="center",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0},
        zorder=5,
    )
    _ground_termination(axes[0], 10.65)

    _panel(axes[1], "(b)", "D2 — fixture embedded once")
    axes[1].plot([0.4, 8.15], [0, 0], color=INK, lw=1.2)
    _reference_plane(axes[1], 3.8, "electrode plane")
    _block(
        axes[1],
        (5.0, 0),
        3.15,
        rf"$Z_e=Z_f+Z_p={circuit['embedded_resistance_ohm']:g}"
        rf"{circuit['embedded_reactance_ohm']:+.3f}j\,\Omega$",
    )
    axes[1].text(
        11.65,
        0.55,
        f"$Z_{{in}}={d2['scenarios'][0]['input_resistance_ohm']:.3f}"
        f"{d2['scenarios'][0]['input_reactance_ohm']:+.3f}j$ $\\Omega$\n"
        f"$|\\Gamma|={d2['worst_reflection_magnitude']:.4f}$",
        ha="right",
        va="center",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0},
        zorder=5,
    )
    _ground_termination(axes[1], 8.15)

    _panel(axes[2], "(c)", "D3 — fixture counted twice")
    axes[2].plot([0.4, 10.0], [0, 0], color=INK, lw=1.2)
    _reference_plane(axes[2], 3.8, "electrode plane")
    _block(axes[2], (4.35, 0), 1.1, r"$R_f$")
    _block(axes[2], (5.8, 0), 1.1, r"$L_f$")
    _block(axes[2], (7.35, 0), 2.65, r"$Z_e$ (already includes $Z_f$)", edge=ORANGE)
    axes[2].annotate(
        "duplicate fixture",
        xy=(8.65, 0.34),
        xytext=(8.65, 0.82),
        ha="center",
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.9},
    )
    axes[2].text(
        11.65,
        0.55,
        f"$Z_{{in}}={d3['scenarios'][0]['input_resistance_ohm']:.3f}"
        f"{d3['scenarios'][0]['input_reactance_ohm']:+.3f}j$ $\\Omega$\n"
        f"$|\\Gamma|={d3['worst_reflection_magnitude']:.4f}$",
        ha="right",
        va="center",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0},
        zorder=5,
    )
    _ground_termination(axes[2], 10.0)

    _title(
        fig,
        "Reference-plane representations used by D1-D3",
        "The same lossy R-L fixture is represented once in D1 and D2; D3 deliberately includes it twice.",
    )
    _footer(
        fig,
        "Synthetic boundary benchmark. It verifies one known series transform, not general S-parameter de-embedding or undocumented-fixture detection.",
    )
    fig.subplots_adjust(top=0.81, bottom=0.07, left=0.06, right=0.99, hspace=0.58)
    return fig


def _status_marker(scenario: dict[str, Any]) -> str:
    violations = set(scenario["violated_constraints"])
    if "min_control_margin" in violations and len(violations) == 1:
        return "D"
    if violations:
        return "^"
    return "o"


def _control_margin_bounds(values: list[float], fraction: float = 0.2) -> tuple[float, float]:
    low, high = min(values), max(values)
    inset = 0.5 * fraction * (high - low)
    return low + inset, high - inset


def figure_control_authority(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.88, 1.22, 1.32], wspace=0.42)
    ax_load = fig.add_subplot(gs[0, 0])
    ax_gamma = fig.add_subplot(gs[0, 1])
    ax_control = fig.add_subplot(gs[0, 2])

    _panel(ax_load, "(a)", "Declared load points")
    by_id = {row["scenario_id"]: row for row in data["load_scenarios"]}
    for scenario_id in SCENARIO_ORDER:
        row = by_id[scenario_id]
        ax_load.scatter(
            row["resistance_ohm"],
            row["reactance_ohm"],
            s=38,
            marker="o",
            facecolor=WHITE,
            edgecolor=INK,
            lw=1.0,
            zorder=3,
        )
        ax_load.annotate(
            SCENARIO_CODES[scenario_id],
            (row["resistance_ohm"], row["reactance_ohm"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.2,
        )
    ax_load.axhline(0, color=GRID, lw=0.7)
    ax_load.set_xlabel(r"load resistance $R$ [$\Omega$]")
    ax_load.set_ylabel(r"load reactance $X$ [$\Omega$]")
    ax_load.grid(True, which="major")
    ax_load.set_xlim(8, 55)
    ax_load.set_ylim(-175, -30)

    _panel(ax_gamma, "(b)", "Selected reflection by scenario")
    y = np.arange(len(SCENARIO_ORDER), dtype=float)
    case_style = {"B1": (MUTED, -0.2), "B2": (BLUE, 0.0), "B3": (ORANGE, 0.2)}
    for case_name, (color, offset) in case_style.items():
        scenarios = _scenario_map(data["control_cases"][case_name])
        for index, scenario_id in enumerate(SCENARIO_ORDER):
            row = scenarios[scenario_id]
            marker = _status_marker(row)
            face = color if marker == "o" else WHITE
            ax_gamma.scatter(
                row["reflection_magnitude"],
                y[index] + offset,
                marker=marker,
                s=38,
                facecolor=face,
                edgecolor=color,
                lw=1.2,
                zorder=3,
            )
    limit = float(data["acceptance"]["max"])
    ax_gamma.axvline(limit, color=INK, lw=1.0, ls=(0, (5, 3)))
    ax_gamma.text(limit + 0.012, -0.48, "10% power limit", rotation=90, va="top", fontsize=6.8, color=MUTED)
    ax_gamma.set_yticks(y, [SCENARIO_CODES[item] for item in SCENARIO_ORDER])
    ax_gamma.invert_yaxis()
    ax_gamma.set_xlim(0, 0.83)
    ax_gamma.set_xlabel(r"reflection magnitude $|\Gamma|$")
    ax_gamma.grid(True, axis="x")
    status_handles = [
        plt.Line2D([], [], marker="o", ls="none", color=INK, markerfacecolor=INK, label="accepted"),
        plt.Line2D([], [], marker="^", ls="none", color=INK, markerfacecolor=WHITE, label="reflection fail"),
        plt.Line2D([], [], marker="D", ls="none", color=INK, markerfacecolor=WHITE, label="margin-only fail"),
    ]
    ax_gamma.legend(handles=status_handles, loc="lower right", frameon=False, handletextpad=0.35)

    _panel(ax_control, "(c)", "Selected tuner coordinates")
    control_styles = {"B2": (BLUE, "D"), "B3": (ORANGE, "o")}
    for case_name, (color, marker) in control_styles.items():
        grid = data["control_grids"][case_name]
        c1_pf = np.asarray(grid["C1"]) * 1e12
        c2_pf = np.asarray(grid["C2"]) * 1e12
        gx, gy = np.meshgrid(c1_pf, c2_pf)
        ax_control.scatter(gx.ravel(), gy.ravel(), s=7, color=GRID, marker=".", zorder=1)
        xlo, xhi = _control_margin_bounds(list(c1_pf))
        ylo, yhi = _control_margin_bounds(list(c2_pf))
        ax_control.add_patch(
            mpatches.Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                fill=False,
                edgecolor=color,
                lw=1.05,
                ls="-" if case_name == "B2" else "--",
            )
        )
        scenarios = _scenario_map(data["control_cases"][case_name])
        for scenario_id in SCENARIO_ORDER:
            row = scenarios[scenario_id]
            c1 = float(row["control"]["C1"]) * 1e12
            c2 = float(row["control"]["C2"]) * 1e12
            accepted = not row["violated_constraints"]
            ax_control.scatter(
                c1, c2, s=39, marker=marker, facecolor=color if accepted else WHITE, edgecolor=color, lw=1.2, zorder=3
            )
            vertical = 5 if case_name == "B2" else -9
            ax_control.annotate(
                SCENARIO_CODES[scenario_id],
                (c1, c2),
                xytext=(3, vertical),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
            )
    ax_control.set_xlim(360, 1040)
    ax_control.set_xticks([400, 600, 800, 1000])
    ax_control.set_ylim(8, 248)
    ax_control.set_xlabel(r"$C_1$ [pF]")
    ax_control.set_ylabel(r"$C_2$ [pF]")
    ax_control.grid(True)

    _title(
        fig,
        "Control-authority benchmark (B1-B3)",
        "One five-point synthetic load envelope; B2/B3 select a discrete C1-C2 state independently for each scenario.",
    )
    _footer(
        fig,
        "Panel (b): B1 grey, B2 blue, B3 orange; fill/shape gives status.\n"
        "Panel (c): blue diamonds = B2, orange circles = B3; rectangles give 20% reserve regions.",
    )
    fig.subplots_adjust(top=0.81, bottom=0.16, left=0.08, right=0.985)
    return fig


def figure_a5_frequency(data: dict[str, Any]) -> plt.Figure:
    case = data["a5"]["result"]
    scenarios = _scenario_map(case)
    order = ["low_frequency", "nominal_frequency", "high_frequency"]
    frequencies = [float(data["a5"]["frequency_MHz"][scenario_id]) for scenario_id in order]
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    _panel(ax, "(a)", "Independent frequency points")
    limit = float(data["acceptance"]["max"])
    ax.axhline(limit, color=INK, lw=1.0, ls=(0, (5, 3)))
    ax.text(19.8, limit + 0.025, "10% reflected-power limit", ha="right", va="bottom", fontsize=7.0, color=MUTED)
    for frequency, scenario_id in zip(frequencies, order, strict=True):
        row = scenarios[scenario_id]
        accepted = row["feasible"]
        marker = "o" if accepted else "X"
        color = BLUE if accepted else ORANGE
        ax.scatter(
            frequency,
            row["reflection_magnitude"],
            s=72,
            marker=marker,
            color=color,
            edgecolor=INK if accepted else color,
            lw=0.9,
            zorder=3,
        )
        z_label = f"{row['input_resistance_ohm']:.3f}{row['input_reactance_ohm']:+.3f}j Ω"
        y_offset = 12 if accepted else -24
        ax.annotate(
            z_label,
            (frequency, row["reflection_magnitude"]),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
        )
    ax.set_xticks(frequencies, ["10", "13.56", "20"])
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel(r"reflection magnitude $|\Gamma|$")
    ax.set_xlim(8.5, 21.5)
    ax.set_ylim(-0.03, 1.09)
    ax.grid(True, axis="y")
    _title(
        fig,
        "A5 effective-ICP frequency conformance",
        "Fixed L-match designed at 13.56 MHz; three synthetic terminal-model conditions are evaluated independently.",
    )
    _footer(
        fig,
        "Markers: real ngspice results. No line is drawn because the benchmark does not claim a continuous qualified bandwidth.",
    )
    fig.subplots_adjust(top=0.79, bottom=0.18, left=0.12, right=0.98)
    return fig


def figure_b5_stress(data: dict[str, Any]) -> plt.Figure:
    rows = {row["scenario_id"]: row for row in data["b5"]["scenarios"]}
    limits = data["b5"]["limits"]
    low, high = rows["low_drive"], rows["high_drive"]
    fig = plt.figure(figsize=PAGE_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.65], wspace=0.5)
    ax_match = fig.add_subplot(gs[0, 0])
    ax_util = fig.add_subplot(gs[0, 1])
    _panel(ax_match, "(a)", "Match quality")
    limit = float(data["acceptance"]["max"])
    ax_match.axhline(limit, color=INK, lw=1.0, ls=(0, (5, 3)))
    for index, (row, color, marker) in enumerate(((low, BLUE, "o"), (high, ORANGE, "s"))):
        ax_match.scatter(index, row["reflection_magnitude"], s=52, marker=marker, color=color, zorder=3)
        ax_match.annotate(
            f"{row['reflection_magnitude']:.5f}",
            (index, row["reflection_magnitude"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
        )
    ax_match.set_xticks([0, 1], ["25 Vpk", "100 Vpk"])
    ax_match.set_ylabel(r"reflection magnitude $|\Gamma|$")
    ax_match.set_ylim(0, 0.35)
    ax_match.grid(True, axis="y")

    _panel(ax_util, "(b)", "Engineering-limit utilization")
    metrics = [
        ("component_L1_current_rms_A", "L1 current"),
        ("component_L1_loss_W", "L1 loss"),
        ("source_current_rms_A", "source current"),
        ("source_apparent_power_VA", "source VA"),
    ]
    y = np.arange(len(metrics), dtype=float)
    ax_util.axvline(1.0, color=INK, lw=1.0, ls=(0, (5, 3)))
    ax_util.text(
        1.0,
        1.02,
        "declared limit",
        transform=ax_util.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    for index, (metric, _label) in enumerate(metrics):
        maximum = float(limits[metric])
        low_ratio = low["metrics"][metric] / maximum
        high_ratio = high["metrics"][metric] / maximum
        ax_util.plot([low_ratio, high_ratio], [index, index], color=GRID, lw=1.1, zorder=1)
        ax_util.scatter(low_ratio, index, s=42, marker="o", color=BLUE, zorder=3)
        ax_util.scatter(high_ratio, index, s=42, marker="s", color=ORANGE, zorder=3)
        ax_util.annotate(
            f"{low_ratio:.2f}x",
            (low_ratio, index),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=BLUE,
        )
        high_offset = 8 if index == len(metrics) - 1 else -11
        ax_util.annotate(
            f"{high_ratio:.2f}x",
            (high_ratio, index),
            xytext=(0, high_offset),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=ORANGE,
        )
    ax_util.set_xscale("log")
    ax_util.set_xlim(0.09, 5.8)
    ax_util.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0], ["0.1", "0.2", "0.5", "1", "2", "5"])
    ax_util.set_yticks(y, [item[1] for item in metrics])
    ax_util.invert_yaxis()
    ax_util.set_ylim(len(metrics) - 0.45, -0.45)
    ax_util.set_xlabel("calculated value / declared limit")
    ax_util.grid(True, axis="x", which="major")
    _title(
        fig,
        "B5 high-drive component and source stress",
        "The linear circuit remains matched at both amplitudes; declared L1 and source-terminal limits reject the high-drive scenario.",
    )
    _footer(
        fig,
        "Blue circle = 25 Vpk; orange square = 100 Vpk. Utilization = calculated value / limit.\n"
        "Effective ESR/DCR loss is not a temperature or lifetime prediction.",
    )
    fig.subplots_adjust(top=0.79, bottom=0.17, left=0.105, right=0.985)
    return fig


def _corner_matrix(ax: plt.Axes, rows: list[dict[str, Any]], c1_factor: float, label: str) -> None:
    subset = [row for row in rows if math.isclose(row["C1_factor"], c1_factor)]
    for row in subset:
        x = 0 if math.isclose(row["C2_factor"], 0.85) else 1
        y = 0 if math.isclose(row["L1_factor"], 0.85) else 1
        color = BLUE_LIGHT if row["feasible"] else ORANGE_LIGHT
        hatch = None if row["feasible"] else "////"
        edge = BLUE if row["feasible"] else ORANGE
        ax.add_patch(
            mpatches.Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, facecolor=color, edgecolor=edge, hatch=hatch, lw=1.35)
        )
        status = "PASS" if row["feasible"] else "FAIL"
        ax.text(x, y + 0.08, f"{row['reflection_magnitude']:.6f}", ha="center", va="center", fontsize=7.4)
        ax.text(x, y - 0.18, status, ha="center", va="center", fontsize=6.8, fontweight="bold")
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(-0.52, 1.52)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1], ["0.85", "1.15"])
    ax.set_yticks([0, 1], ["0.85", "1.15"])
    ax.set_xlabel("C2 factor")
    ax.set_ylabel("L1 factor")
    _panel(ax, label, rf"$C_1$ factor = {c1_factor:.2f}")


def figure_b8_corners(data: dict[str, Any]) -> plt.Figure:
    rows = list(data["b8"])
    fig = plt.figure(figsize=PAGE_SIZE)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 0.92, 1.45], wspace=0.45)
    ax_low = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[0, 2])
    _corner_matrix(ax_low, rows, 0.85, "(a)")
    _corner_matrix(ax_high, rows, 1.15, "(b)")

    _panel(ax_z, "(c)", "Input-impedance plane")
    gamma_limit = float(data["acceptance"]["max"])
    z0 = float(data["reference_impedance_ohm"])
    center = z0 * (1.0 + gamma_limit**2) / (1.0 - gamma_limit**2)
    radius = 2.0 * z0 * gamma_limit / (1.0 - gamma_limit**2)
    ax_z.add_patch(
        mpatches.Circle((center, 0), radius, facecolor=BLUE_LIGHT, edgecolor=INK, lw=1.0, ls=(0, (5, 3)), alpha=0.55)
    )
    ax_z.plot(50, 0, marker="+", ms=8, mew=1.2, color=INK)
    ax_z.annotate(r"$Z_0$", (50, 0), xytext=(4, 4), textcoords="offset points", fontsize=6.4)
    label_positions = {
        "c1_low_l1_low_c2_low": (35, 8),
        "c1_low_l1_low_c2_high": (36, 3),
        "c1_low_l1_high_c2_low": (81, 8),
        "c1_low_l1_high_c2_high": (103, 8),
        "c1_high_l1_low_c2_low": (35, -8),
        "c1_high_l1_low_c2_high": (36, -13),
        "c1_high_l1_high_c2_low": (58, -34),
        "c1_high_l1_high_c2_high": (82, -47),
    }
    for row in rows:
        marker = "o" if row["feasible"] else "X"
        color = BLUE if row["feasible"] else ORANGE
        ax_z.scatter(
            row["input_resistance_ohm"], row["input_reactance_ohm"], marker=marker, s=42, color=color, zorder=3
        )
        code = "".join("H" if row[name] > 1 else "L" for name in ("C1_factor", "L1_factor", "C2_factor"))
        ax_z.annotate(
            code,
            (row["input_resistance_ohm"], row["input_reactance_ohm"]),
            xytext=label_positions[row["scenario_id"]],
            textcoords="data",
            fontsize=6.2,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.45, "shrinkA": 2.0, "shrinkB": 2.0},
        )
    ax_z.axhline(0, color=GRID, lw=0.7)
    ax_z.set_xlabel(r"input resistance [$\Omega$]")
    ax_z.set_ylabel(r"input reactance [$\Omega$]")
    ax_z.set_aspect("equal", adjustable="box")
    ax_z.set_xlim(18, 110)
    ax_z.set_ylim(-52, 42)
    ax_z.set_xticks([20, 40, 60, 80, 100])
    ax_z.set_yticks([-40, -20, 0, 20, 40])
    ax_z.grid(True)
    ax_z.text(50, 34, r"$|\Gamma|\leq\sqrt{0.1}$", ha="center", fontsize=6.9, color=MUTED)
    ax_z.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="none", color=BLUE, label="pass"),
            plt.Line2D([], [], marker="X", ls="none", color=ORANGE, label="fail"),
        ],
        loc="upper right",
        frameon=False,
    )
    _title(
        fig,
        "B8 deterministic component-value corners",
        r"Nominal $C_1/L_1/C_2$ Candidate with the complete 0.85/1.15 factor product; cell values are $|\Gamma|$.",
    )
    _footer(
        fig,
        "Three of eight vertices pass. Point labels give C1/L1/C2 as L = 0.85 or H = 1.15; the count is corner coverage, not manufacturing yield.",
    )
    fig.subplots_adjust(top=0.75, bottom=0.17, left=0.075, right=0.985)
    return fig


def figure_reference_results(data: dict[str, Any]) -> plt.Figure:
    cases = data["reference_plane"]
    gamma_limit = float(data["acceptance"]["max"])
    z0 = float(data["reference_impedance_ohm"])
    center = z0 * (1.0 + gamma_limit**2) / (1.0 - gamma_limit**2)
    radius = 2.0 * z0 * gamma_limit / (1.0 - gamma_limit**2)
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    _panel(ax, "(a)", "Source-plane impedance")
    ax.add_patch(
        mpatches.Circle((center, 0), radius, facecolor=BLUE_LIGHT, edgecolor=INK, lw=1.0, ls=(0, (5, 3)), alpha=0.55)
    )
    ax.plot(50, 0, marker="+", ms=9, mew=1.3, color=INK)
    styles = {"D1": (BLUE, "o", BLUE), "D2": (BLUE, "o", "none"), "D3": (ORANGE, "s", ORANGE)}
    for name in ("D1", "D2", "D3"):
        scenario = cases[name]["scenarios"][0]
        color, marker, face = styles[name]
        size = 80 if name == "D2" else 48
        ax.scatter(
            scenario["input_resistance_ohm"],
            scenario["input_reactance_ohm"],
            s=size,
            marker=marker,
            facecolor=face,
            edgecolor=color,
            lw=1.35,
            zorder=4 if name == "D2" else 3,
            label=name,
        )
    d1 = cases["D1"]["scenarios"][0]
    d3 = cases["D3"]["scenarios"][0]
    ax.annotate(
        "D1 / D2",
        (d1["input_resistance_ohm"], d1["input_reactance_ohm"]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=7.6,
    )
    ax.annotate(
        "D3",
        (d3["input_resistance_ohm"], d3["input_reactance_ohm"]),
        xytext=(7, -2),
        textcoords="offset points",
        fontsize=7.6,
        color=ORANGE,
    )
    ax.text(
        22,
        35,
        f"D1 = D2: |Γ| = {cases['D1']['worst_reflection_magnitude']:.4f}\nD3: |Γ| = {cases['D3']['worst_reflection_magnitude']:.4f}",
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.axhline(0, color=GRID, lw=0.7)
    ax.set_xlabel(r"input resistance [$\Omega$]")
    ax.set_ylabel(r"input reactance [$\Omega$]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(18, 112)
    ax.set_ylim(-55, 48)
    ax.grid(True)
    ax.text(60, 43, r"$|\Gamma|\leq\sqrt{0.1}$", ha="center", fontsize=6.9, color=MUTED)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    _title(
        fig,
        "D1-D3 reference-plane benchmark results",
        "D1/D2 overlap; duplicate fixture D3 fails the 10% reflected-power limit.",
    )
    _footer(
        fig,
        r"Source evaluation plane; the blue circle is the exact $Z_0=50\,\Omega$, $|\Gamma|\leq\sqrt{0.1}$ region.",
    )
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.15, right=0.97)
    return fig


def _phasor_value(payload: dict[str, float]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def _reconstructed_fundamental(payload: dict[str, float], phase_rad: np.ndarray) -> np.ndarray:
    phasor = _phasor_value(payload)
    return np.real(phasor * np.exp(1j * phase_rad))


def _wrapped_phase_difference(first: dict[str, float], second: dict[str, float]) -> float:
    difference = float(second["phase_deg"]) - float(first["phase_deg"])
    return (difference + 180.0) % 360.0 - 180.0


def figure_b5_port_waveforms(data: dict[str, Any]) -> plt.Figure:
    """Show the B5 input/output fundamental without implying transient data."""

    rows = {row["scenario_id"]: row for row in data["b5"]["scenarios"]}
    phase_deg = np.linspace(0.0, 360.0, 361)
    phase_rad = np.deg2rad(phase_deg)
    fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE, sharex=True, sharey="col")
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    for row_index, scenario_id in enumerate(("low_drive", "high_drive")):
        row = rows[scenario_id]
        phasors = row["phasors"]
        drive = float(row["drive_peak_V"])
        voltage_axis, current_axis = axes[row_index]

        source_voltage = phasors["source_voltage_V"]
        load_voltage = phasors["load_voltage_V"]
        voltage_axis.plot(
            phase_deg,
            _reconstructed_fundamental(source_voltage, phase_rad),
            color=INK,
            lw=1.15,
            ls=(0, (5, 3)),
            label="source input",
        )
        voltage_axis.plot(
            phase_deg,
            _reconstructed_fundamental(load_voltage, phase_rad),
            color=BLUE,
            lw=1.45,
            label="electrode terminal",
        )
        voltage_gain = float(load_voltage["peak"]) / float(source_voltage["peak"])
        voltage_phase = _wrapped_phase_difference(source_voltage, load_voltage)
        voltage_axis.text(
            0.98,
            0.93,
            rf"$|V_{{out}}|/|V_{{in}}|={voltage_gain:.3f}$" + "\n" + rf"$\Delta\phi={voltage_phase:.1f}^\circ$",
            transform=voltage_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
        )

        source_current = phasors["source_current_A"]
        load_current = phasors["load_current_A"]
        current_axis.plot(
            phase_deg,
            _reconstructed_fundamental(source_current, phase_rad),
            color=INK,
            lw=1.15,
            ls=(0, (5, 3)),
            label="source delivered",
        )
        current_axis.plot(
            phase_deg,
            _reconstructed_fundamental(load_current, phase_rad),
            color=BLUE,
            lw=1.45,
            label="load-port current",
        )
        current_ratio = float(load_current["peak"]) / float(source_current["peak"])
        current_phase = _wrapped_phase_difference(source_current, load_current)
        current_axis.text(
            0.98,
            0.93,
            rf"$|I_{{load}}|/|I_{{in}}|={current_ratio:.3f}$" + "\n" + rf"$\Delta\phi={current_phase:.1f}^\circ$",
            transform=current_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
        )

        _panel(voltage_axis, panel_labels[row_index][0], f"{drive:g} Vpk — voltage")
        _panel(current_axis, panel_labels[row_index][1], f"{drive:g} Vpk — current")
        for axis in (voltage_axis, current_axis):
            axis.axhline(0.0, color=GRID, lw=0.7)
            axis.set_xlim(0.0, 360.0)
            axis.set_xticks([0, 90, 180, 270, 360])
            axis.grid(True, axis="x")

    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[0, 1].legend(loc="lower right", frameon=False)
    axes[0, 0].set_ylabel("instantaneous voltage [V]")
    axes[1, 0].set_ylabel("instantaneous voltage [V]")
    axes[0, 1].set_ylabel("instantaneous current [A]")
    axes[1, 1].set_ylabel("instantaneous current [A]")
    axes[1, 0].set_xlabel("source-referenced RF phase [deg]")
    axes[1, 1].set_xlabel("source-referenced RF phase [deg]")
    _title(
        fig,
        "B5 source and electrode-terminal steady-state fundamental",
        "Reconstructed from one 13.56 MHz ngspice AC phasor per drive condition; these curves are not transient or harmonic results.",
    )
    _footer(
        fig,
        "Input = ideal-source node voltage and delivered current.\n"
        "Output = declared electrode_terminal voltage and load current; AC magnitudes are peak values.",
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.985, hspace=0.58, wspace=0.30)
    return fig


def _grouped_bars(
    axis: plt.Axes,
    categories: list[str],
    low_values: list[float],
    high_values: list[float],
    ylabel: str,
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    low_bars = axis.bar(
        x - width / 2,
        low_values,
        width,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        lw=1.0,
        label="25 Vpk",
    )
    high_bars = axis.bar(
        x + width / 2,
        high_values,
        width,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        hatch="////",
        lw=1.0,
        label="100 Vpk",
    )
    axis.bar_label(low_bars, fmt="%.2f", padding=2, fontsize=6.4)
    axis.bar_label(high_bars, fmt="%.2f", padding=2, fontsize=6.4)
    axis.set_xticks(x, categories)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, max(high_values) * 1.18)
    axis.grid(True, axis="y")


def figure_b5_signal_and_power(data: dict[str, Any]) -> plt.Figure:
    """Connect B5 drive amplitude, terminal response, and real-power flow."""

    rows = {row["scenario_id"]: row for row in data["b5"]["scenarios"]}
    low, high = rows["low_drive"], rows["high_drive"]
    fig, axes = plt.subplots(1, 3, figsize=PAGE_SIZE)

    _grouped_bars(
        axes[0],
        ["source", "electrode"],
        [low["phasors"]["source_voltage_V"]["peak"], low["phasors"]["load_voltage_V"]["peak"]],
        [high["phasors"]["source_voltage_V"]["peak"], high["phasors"]["load_voltage_V"]["peak"]],
        "peak voltage [V]",
    )
    _panel(axes[0], "(a)", "Voltage amplitude")
    axes[0].legend(loc="upper left", frameon=False)

    _grouped_bars(
        axes[1],
        ["source", "load port"],
        [low["phasors"]["source_current_A"]["peak"], low["phasors"]["load_current_A"]["peak"]],
        [high["phasors"]["source_current_A"]["peak"], high["phasors"]["load_current_A"]["peak"]],
        "peak current [A]",
    )
    _panel(axes[1], "(b)", "Current amplitude")

    x = np.arange(2, dtype=float)
    load_power = [low["metrics"]["load_real_power_W"], high["metrics"]["load_real_power_W"]]
    network_loss = [low["metrics"]["network_loss_W"], high["metrics"]["network_loss_W"]]
    load_bars = axes[2].bar(
        x,
        load_power,
        0.58,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        lw=1.0,
        label="accepted at load plane",
    )
    axes[2].bar(
        x,
        network_loss,
        0.58,
        bottom=load_power,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        hatch="////",
        lw=1.0,
        label="network loss",
    )
    source_power = [low["metrics"]["source_real_power_W"], high["metrics"]["source_real_power_W"]]
    for index, total in enumerate(source_power):
        efficiency = float((low, high)[index]["metrics"]["transfer_efficiency"])
        axes[2].annotate(
            f"{total:.2f} W\nη {100.0 * efficiency:.2f}%",
            (index, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
    axes[2].bar_label(
        load_bars, labels=[f"load {value:.2f}" for value in load_power], label_type="center", fontsize=6.2
    )
    axes[2].set_xticks(x, ["25 Vpk", "100 Vpk"])
    axes[2].set_ylabel("real power [W]")
    axes[2].set_ylim(0.0, max(source_power) * 1.18)
    axes[2].grid(True, axis="y")
    axes[2].legend(loc="upper left", frameon=False, fontsize=6.5)
    _panel(axes[2], "(c)", "Power disposition")

    _title(
        fig,
        "B5 terminal amplitudes and real-power transfer",
        "Two discrete ngspice AC solutions at one fixed linear network/load; bars do not imply a continuous amplitude sweep.",
    )
    _footer(
        fig,
        "A 4x drive produces 4x terminal amplitudes and 16x power/loss while |Γ| and 97.60% efficiency stay constant.\n"
        "The high-drive condition fails declared source/L1 stress limits, not the match criterion.",
    )
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.08, right=0.985, wspace=0.43)
    return fig


_CASE_TOPIC = {
    "A1_topology_l_match": "A1  L-match equation",
    "A2_topology_pi_match": "A2  Pi-match equation",
    "A3_topology_pi_match_harmonic": "A3  Harmonic branch",
    "A4_ccp_lumped_frequency_conformance": "A4  Effective CCP load",
    "A5_icp_transformer_frequency_conformance": "A5  Effective ICP load",
    "B1_fixed_nominal": "B1  Fixed nominal",
    "B2_limited_tuner": "B2  Limited tuner",
    "B3_full_tuner": "B3  Full tuner",
    "B4_independent_frequency_points": "B4  Frequency points",
    "B5_high_drive_stress": "B5  Drive stress",
    "B6_discrete_hardware_search": "B6  Hardware shortlist",
    "B7_role_factorial_search": "B7  Role factorial",
    "B8_component_value_corner_stress": "B8  Value corners",
    "D1_reference_plane_explicit": "D1  Explicit fixture",
    "D2_reference_plane_embedded": "D2  Embedded fixture",
    "D3_reference_plane_double_counted": "D3  Double-counted fixture",
}


def figure_benchmark_reading_guide(data: dict[str, Any]) -> plt.Figure:
    """Separate benchmark reproducibility from engineering feasibility."""

    cases = list(data["benchmark_cases"])
    fig = plt.figure(figsize=PAGE_SIZE)
    flow = fig.add_axes((0.05, 0.60, 0.90, 0.17))
    matrix = fig.add_axes((0.28, 0.08, 0.58, 0.43))

    flow.set_xlim(0.0, 12.0)
    flow.set_ylim(-0.1, 2.5)
    flow.axis("off")
    _panel(flow, "(a)", "What one benchmark case does")
    blocks = [
        (0.2, "Declared\nproblem"),
        (2.8, "Candidate x scenario\nx control"),
        (5.9, "ngspice +\nmetrics"),
        (8.45, "Engineering\nconstraints"),
    ]
    widths = [1.75, 2.25, 1.75, 1.9]
    for (x, label), width in zip(blocks, widths, strict=True):
        flow.add_patch(mpatches.Rectangle((x, 1.0), width, 0.9, facecolor=WHITE, edgecolor=INK, lw=1.0))
        flow.text(x + width / 2, 1.45, label, ha="center", va="center")
    for index, (first, second) in enumerate(itertools.pairwise(blocks)):
        start_x = first[0] + widths[index]
        flow.annotate("", xy=(second[0], 1.45), xytext=(start_x, 1.45), arrowprops={"arrowstyle": "->", "lw": 0.9})
    flow.annotate("", xy=(10.9, 2.05), xytext=(10.35, 1.58), arrowprops={"arrowstyle": "->", "lw": 0.9})
    flow.annotate("", xy=(10.9, 0.65), xytext=(10.35, 1.32), arrowprops={"arrowstyle": "->", "lw": 0.9})
    flow.text(11.0, 2.05, "expected result\nreproduced?", ha="left", va="center", color=BLUE)
    flow.text(11.0, 0.65, "candidate\nfeasible?", ha="left", va="center", color=ORANGE)

    _panel(matrix, "(b)", "Case outcomes")
    y = np.arange(len(cases), dtype=float)
    matrix.set_yticks(y, [_CASE_TOPIC[item["benchmark_id"]] for item in cases])
    matrix.invert_yaxis()
    for index, case in enumerate(cases):
        matrix.scatter(0.0, index, marker="o", s=32, facecolor=BLUE, edgecolor=BLUE, zorder=3)
        feasible = bool(case["feasible"])
        matrix.scatter(
            1.0,
            index,
            marker="o" if feasible else "X",
            s=35,
            facecolor=BLUE if feasible else ORANGE,
            edgecolor=BLUE if feasible else ORANGE,
            zorder=3,
        )
    for boundary in (4.5, 12.5):
        matrix.axhline(boundary, color=GRID, lw=0.8)
    matrix.set_xlim(-1.2, 1.5)
    matrix.set_xticks([0.0, 1.0], ["expectation\nreproduced", "candidate\nfeasible"])
    matrix.xaxis.tick_top()
    matrix.tick_params(axis="y", length=0, labelsize=6.7)
    matrix.tick_params(axis="x", length=0)
    matrix.grid(True, axis="x")
    matrix.spines[["top", "right", "bottom", "left"]].set_visible(False)
    matrix.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="none", color=BLUE, label="yes"),
            plt.Line2D([], [], marker="X", ls="none", color=ORANGE, label="no"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )
    _title(
        fig,
        "How to read the core benchmark results",
        "Benchmark pass = declared expectation reproduced; candidate feasible = every required engineering constraint satisfied.",
    )
    _footer(
        fig,
        "Frozen run: all 16 benchmark expectations reproduced.\n"
        "Engineering outcome: 6 candidate configurations feasible; 10 intentionally or diagnostically infeasible.",
    )
    return fig


def _export(fig: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    svg_metadata = {
        "Title": stem.replace("-", " "),
        "Creator": "PCD benchmark figure generator",
        "Description": "RF circuit benchmark interpretation",
    }
    assert_text_inside_canvas(fig)
    svg_path = output / f"{stem}.svg"
    fig.savefig(svg_path, metadata=svg_metadata)
    # Matplotlib's path writer leaves spaces before many newlines.  They are
    # visually irrelevant but make `git diff --check` noisy for committed
    # publication masters, so normalize only line endings after rendering.
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(output / f"{stem}.png", dpi=300, metadata={"Software": "PCD benchmark figure generator"})
    pdf.savefig(fig)
    plt.close(fig)


def generate(run_root: Path, output: Path, pdf_output: Path) -> dict[str, Any]:
    if not (run_root / "benchmark_result.json").exists():
        raise FileNotFoundError(f"benchmark_result.json not found under {run_root}")
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.mkdir(parents=True, exist_ok=True)
    _configure_style()
    data = build_figure_data(run_root)
    data_path = output / "figure_data.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figures = [
        ("01-analysis-boundary-and-topologies", figure_analysis_boundary(data)),
        ("02-effective-load-models", figure_load_models()),
        ("03-reference-plane-circuits", figure_reference_circuits(data)),
        ("04-control-authority-results", figure_control_authority(data)),
        ("05-a5-frequency-results", figure_a5_frequency(data)),
        ("06-b5-stress-results", figure_b5_stress(data)),
        ("07-b8-corner-results", figure_b8_corners(data)),
        ("08-reference-plane-results", figure_reference_results(data)),
        ("09-b5-port-waveforms", figure_b5_port_waveforms(data)),
        ("10-b5-signal-and-power", figure_b5_signal_and_power(data)),
        ("11-benchmark-reading-guide", figure_benchmark_reading_guide(data)),
    ]
    pdf_path = pdf_output / "benchmark-figure-pack.pdf"
    with PdfPages(
        pdf_path, metadata={"Title": "PCD benchmark figure pack", "Author": "PCD benchmark figure generator"}
    ) as pdf:
        for stem, fig in figures:
            _export(fig, output, stem, pdf)
    return {
        "output": str(output.resolve()),
        "figure_count": len(figures),
        "formats": ["svg", "png", "pdf"],
        "data": str(data_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "source_sha256": data["source"]["benchmark_result_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    args = parser.parse_args(argv)
    result = generate(args.run_root.resolve(), args.output.resolve(), args.pdf_output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
