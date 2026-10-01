"""Generate an independent target for a dual-RF, DC-superposed etch CCP.

The reference calculation is intentionally independent of PCD and ngspice.
It advances capacitor charge and inductor flux with fixed-step RK4, then
solves the linear circuit constraints at every stage.  The time-varying
plasma elements therefore obey explicit constitutive laws

    q_s(t) = C_s(t) v_s(t),       phi_p(t) = L_p(t) i_p(t)

rather than replacing scalar C or L values without accounting for stored
charge or flux.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "cases"

ENVELOPE_PERIOD_S = 2.5e-6
STOP_S = 7.5e-6
TARGET_START_S = 5.0e-6
TARGET_STEP_S = 1.0e-9
REFERENCE_STEP_S = 1.25e-10
Z0_OHM = 50.0

HF_HZ = 60.0e6
LF_HZ = 2.0e6
HF_PEAK_V = 220.0
LF_PEAK_V = 450.0
DC_ON_V = -800.0

ValueFunction = Callable[[float], float]


@dataclass(frozen=True)
class Resistor:
    name: str
    positive: str
    negative: str
    resistance: ValueFunction


@dataclass(frozen=True)
class Capacitor:
    name: str
    positive: str
    negative: str
    capacitance: ValueFunction


@dataclass(frozen=True)
class Inductor:
    name: str
    positive: str
    negative: str
    inductance: ValueFunction
    series_resistance: ValueFunction


def _constant(value: float) -> ValueFunction:
    return lambda _time_s: value


def _periodic_linear(time_s: float, points: tuple[tuple[float, float], ...]) -> float:
    phase = time_s - math.floor(time_s / ENVELOPE_PERIOD_S) * ENVELOPE_PERIOD_S
    times = np.fromiter((point[0] for point in points), dtype=np.float64)
    values = np.fromiter((point[1] for point in points), dtype=np.float64)
    return float(np.interp(phase, times, values))


def envelope(time_s: float) -> float:
    return _periodic_linear(
        time_s,
        (
            (0.0, 0.0),
            (0.25e-6, 0.0),
            (0.55e-6, 1.0),
            (2.0e-6, 1.0),
            (2.4e-6, 0.0),
            (ENVELOPE_PERIOD_S, 0.0),
        ),
    )


def _between(off_value: float, on_value: float) -> ValueFunction:
    return lambda time_s: off_value + (on_value - off_value) * envelope(time_s)


def hf_source(time_s: float) -> float:
    return HF_PEAK_V * math.sin(2.0 * math.pi * HF_HZ * time_s)


def lf_source(time_s: float) -> float:
    return LF_PEAK_V * math.sin(2.0 * math.pi * LF_HZ * time_s)


def dc_source(time_s: float) -> float:
    return _periodic_linear(
        time_s,
        (
            (0.0, 0.0),
            (0.25e-6, 0.0),
            (0.35e-6, DC_ON_V),
            (2.05e-6, DC_ON_V),
            (2.15e-6, 0.0),
            (ENVELOPE_PERIOD_S, 0.0),
        ),
    )


RP = _between(35.0, 15.0)
LP = _between(35.0e-9, 16.0e-9)
CSU = _between(260.0e-12, 520.0e-12)
CSW = _between(360.0e-12, 720.0e-12)

FIXED_NODES: dict[str, ValueFunction] = {
    "0": _constant(0.0),
    "hf_src": hf_source,
    "lf_src": lf_source,
    "dc_src": dc_source,
}

RESISTORS = (
    Resistor("Rgen_hf", "hf_src", "phf", _constant(50.0)),
    Resistor("Rsheath_u", "upper", "bulk_top", _constant(1500.0)),
    Resistor("Rwall", "bulk_bottom", "0", _constant(5000.0)),
    Resistor("Rsheath_w", "bulk_bottom", "wafer", _constant(2000.0)),
    Resistor("Rgen_lf", "lf_src", "plf", _constant(50.0)),
)

CAPACITORS = (
    Capacitor("Cmatch_hf", "phf", "hf_c", _constant(120.0e-12)),
    Capacitor("Cstray_upper", "upper", "0", _constant(90.0e-12)),
    Capacitor("Csheath_u", "upper", "bulk_top", CSU),
    Capacitor("Csheath_w", "bulk_bottom", "wafer", CSW),
    Capacitor("Cblock_lf", "plf", "lf_c", _constant(4.7e-9)),
    Capacitor("Cstray_wafer", "wafer", "0", _constant(140.0e-12)),
)

INDUCTORS = (
    Inductor("Lmatch_feed_hf", "hf_c", "upper", _constant(100.0e-9), _constant(0.6)),
    Inductor("Ldc_choke", "dc_src", "upper", _constant(200.0e-6), _constant(250.0)),
    Inductor("Lplasma", "bulk_top", "bulk_bottom", LP, RP),
    Inductor("Lmatch_feed_lf", "lf_c", "wafer", _constant(1.41e-6), _constant(0.4)),
)


def _unknown_nodes() -> tuple[str, ...]:
    nodes: set[str] = set()
    for branch in (*RESISTORS, *CAPACITORS, *INDUCTORS):
        nodes.update((branch.positive, branch.negative))
    return tuple(sorted(nodes - set(FIXED_NODES)))


UNKNOWN_NODES = _unknown_nodes()
NODE_INDEX = {name: index for index, name in enumerate(UNKNOWN_NODES)}


def _fixed_values(time_s: float) -> dict[str, float]:
    return {name: function(time_s) for name, function in FIXED_NODES.items()}


def _stamp_node_coefficient(row: np.ndarray, node: str, coefficient: float) -> None:
    if node in NODE_INDEX:
        row[NODE_INDEX[node]] += coefficient


def _fixed_contribution(fixed: dict[str, float], positive: str, negative: str) -> float:
    return fixed.get(positive, 0.0) - fixed.get(negative, 0.0)


def _stamp_resistors(
    time_s: float,
    matrix: np.ndarray,
    rhs: np.ndarray,
    fixed: dict[str, float],
) -> None:
    for resistor in RESISTORS:
        conductance = 1.0 / resistor.resistance(time_s)
        for node, sign in ((resistor.positive, 1.0), (resistor.negative, -1.0)):
            if node not in NODE_INDEX:
                continue
            row = NODE_INDEX[node]
            matrix[row, row] += conductance
            other = resistor.negative if sign > 0 else resistor.positive
            if other in NODE_INDEX:
                matrix[row, NODE_INDEX[other]] -= conductance
            else:
                rhs[row] += conductance * fixed[other]


def _stamp_capacitors(
    time_s: float,
    matrix: np.ndarray,
    rhs: np.ndarray,
    fixed: dict[str, float],
    charges: np.ndarray,
    node_count: int,
) -> None:
    for index, capacitor in enumerate(CAPACITORS):
        current_column = node_count + index
        for node, sign in ((capacitor.positive, 1.0), (capacitor.negative, -1.0)):
            if node in NODE_INDEX:
                matrix[NODE_INDEX[node], current_column] += sign
        row = node_count + index
        _stamp_node_coefficient(matrix[row], capacitor.positive, 1.0)
        _stamp_node_coefficient(matrix[row], capacitor.negative, -1.0)
        rhs[row] = charges[index] / capacitor.capacitance(time_s) - _fixed_contribution(
            fixed, capacitor.positive, capacitor.negative
        )


def _stamp_inductors(
    time_s: float,
    matrix: np.ndarray,
    rhs: np.ndarray,
    fixed: dict[str, float],
    fluxes: np.ndarray,
    node_count: int,
    cap_count: int,
) -> dict[str, float]:
    currents: dict[str, float] = {}
    for index, inductor in enumerate(INDUCTORS):
        current = fluxes[index] / inductor.inductance(time_s)
        currents[inductor.name] = current
        for node, sign in ((inductor.positive, 1.0), (inductor.negative, -1.0)):
            if node in NODE_INDEX:
                rhs[NODE_INDEX[node]] -= sign * current
        row = node_count + cap_count + index
        voltage_column = node_count + cap_count + index
        matrix[row, voltage_column] = 1.0
        _stamp_node_coefficient(matrix[row], inductor.positive, -1.0)
        _stamp_node_coefficient(matrix[row], inductor.negative, 1.0)
        rhs[row] = (
            _fixed_contribution(fixed, inductor.positive, inductor.negative)
            - inductor.series_resistance(time_s) * current
        )
    return currents


def solve_algebraic(time_s: float, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    node_count = len(UNKNOWN_NODES)
    cap_count = len(CAPACITORS)
    ind_count = len(INDUCTORS)
    size = node_count + cap_count + ind_count
    matrix = np.zeros((size, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)
    fixed = _fixed_values(time_s)
    charges = state[:cap_count]
    fluxes = state[cap_count:]
    _stamp_resistors(time_s, matrix, rhs, fixed)
    _stamp_capacitors(time_s, matrix, rhs, fixed, charges, node_count)
    inductor_currents = _stamp_inductors(time_s, matrix, rhs, fixed, fluxes, node_count, cap_count)

    solution = np.linalg.solve(matrix, rhs)
    cap_currents = solution[node_count : node_count + cap_count]
    ind_voltages = solution[node_count + cap_count :]
    derivative = np.concatenate((cap_currents, ind_voltages))
    node_voltages = {name: float(solution[index]) for name, index in NODE_INDEX.items()}
    node_voltages.update(fixed)
    algebraic = {
        "nodes": node_voltages,
        "capacitor_currents": {cap.name: float(cap_currents[index]) for index, cap in enumerate(CAPACITORS)},
        "inductor_currents": inductor_currents,
        "inductor_voltages": {ind.name: float(ind_voltages[index]) for index, ind in enumerate(INDUCTORS)},
    }
    return derivative, algebraic


def _rk4_step(time_s: float, state: np.ndarray, step_s: float) -> np.ndarray:
    k1, _ = solve_algebraic(time_s, state)
    k2, _ = solve_algebraic(time_s + 0.5 * step_s, state + 0.5 * step_s * k1)
    k3, _ = solve_algebraic(time_s + 0.5 * step_s, state + 0.5 * step_s * k2)
    k4, _ = solve_algebraic(time_s + step_s, state + step_s * k3)
    return state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _sample_row(time_s: float, state: np.ndarray) -> dict[str, float]:
    _, values = solve_algebraic(time_s, state)
    nodes = values["nodes"]
    inductor_currents = values["inductor_currents"]
    current_into_upper_port = (nodes["hf_src"] - nodes["phf"]) / Z0_OHM
    reflected = 0.5 * (nodes["phf"] - Z0_OHM * current_into_upper_port)
    incident = 0.5 * (nodes["phf"] + Z0_OHM * current_into_upper_port)
    cap_index = {component.name: index for index, component in enumerate(CAPACITORS)}
    ind_index = {component.name: index for index, component in enumerate(INDUCTORS)}
    return {
        "time_s": time_s,
        "upper_hf_source_voltage_V": nodes["hf_src"],
        "lower_lf_source_voltage_V": nodes["lf_src"],
        "upper_dc_source_voltage_V": nodes["dc_src"],
        "plasma_envelope": envelope(time_s),
        "plasma_resistance_ohm": RP(time_s),
        "plasma_inductance_H": LP(time_s),
        "upper_sheath_capacitance_F": CSU(time_s),
        "wafer_sheath_capacitance_F": CSW(time_s),
        "upper_port_voltage_V": nodes["phf"],
        "upper_incident_voltage_V": incident,
        "upper_reflected_voltage_V": reflected,
        "upper_electrode_voltage_V": nodes["upper"],
        "wafer_voltage_V": nodes["wafer"],
        "upper_sheath_voltage_V": nodes["upper"] - nodes["bulk_top"],
        "wafer_sheath_voltage_V": nodes["bulk_bottom"] - nodes["wafer"],
        "bulk_current_A": inductor_currents["Lplasma"],
        "upper_sheath_charge_C": state[cap_index["Csheath_u"]],
        "plasma_flux_Wb": state[len(CAPACITORS) + ind_index["Lplasma"]],
    }


def solve_reference() -> dict[str, np.ndarray]:
    steps = round(STOP_S / REFERENCE_STEP_S)
    sample_stride = round(TARGET_STEP_S / REFERENCE_STEP_S)
    if not math.isclose(sample_stride * REFERENCE_STEP_S, TARGET_STEP_S, rel_tol=0.0, abs_tol=1e-18):
        raise ValueError("target step must be an integer multiple of reference step")
    state = np.zeros(len(CAPACITORS) + len(INDUCTORS), dtype=np.float64)
    rows: list[dict[str, float]] = []
    for step in range(steps + 1):
        time_s = step * REFERENCE_STEP_S
        if time_s >= TARGET_START_S - 0.5 * REFERENCE_STEP_S and step % sample_stride == 0:
            rows.append(_sample_row(time_s, state))
        if step < steps:
            state = _rk4_step(time_s, state, REFERENCE_STEP_S)
    names = tuple(rows[0])
    return {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in names}


def _write_csv(columns: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row in zip(*(columns[name] for name in names), strict=True):
            writer.writerow([f"{float(value):.12g}" for value in row])


def write_targets(data: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
    observables = output_dir / "etch_ccp_dcs_target_observables.csv"
    wafer = output_dir / "etch_ccp_dcs_target_wafer.csv"
    reflection = output_dir / "etch_ccp_dcs_target_reflection.csv"
    _write_csv(data, observables)
    _write_csv({"time_s": data["time_s"], "voltage_V": data["wafer_voltage_V"]}, wafer)
    _write_csv({"time_s": data["time_s"], "voltage_V": data["upper_reflected_voltage_V"]}, reflection)
    return {
        "outputs": {
            "observables": str(observables),
            "wafer_objective": str(wafer),
            "reflection_objective": str(reflection),
        },
        "rows": len(data["time_s"]),
        "time_range_s": [float(data["time_s"][0]), float(data["time_s"][-1])],
        "target_step_s": TARGET_STEP_S,
        "reference_integrator": "charge-flux MNA with fixed-step RK4",
        "reference_step_s": REFERENCE_STEP_S,
        "states": {"capacitor_charges": len(CAPACITORS), "inductor_fluxes": len(INDUCTORS)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = write_targets(solve_reference(), args.output_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
