"""Generate independent target observables for the dual-frequency CCP study.

The calculation integrates the reduced series-loop charge equation without
calling PCD, ngspice, or an optimizer. It then derives two named engineering
observables from the same state:

* ``wafer_voltage_V``: lower electrode/wafer chuck to grounded chamber;
* ``upper_reflected_voltage_V``: reflected voltage wave at the declared
  50-ohm upper reference plane, ``V- = (Vport - Z0 * Iport) / 2``.

The two thin objective CSV files intentionally expose only one canonical
``voltage_V`` target each. The comprehensive CSV keeps all independently
calculated signals for audit and plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CASE = HERE / "cases" / "dual_frequency_ccp_forward.yaml"
DEFAULT_OUTPUT_DIR = HERE / "cases"
TARGET_STEP_S = 1.0e-9
TARGET_START_S = 2.5e-6
REFERENCE_REFINEMENT = 4


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _named(items: list[dict[str, Any]], name: str, location: str) -> dict[str, Any]:
    matches = [item for item in items if str(item.get("name", item.get("ref", ""))) == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} entry in {location}; found {len(matches)}")
    return matches[0]


def _component_value(case: dict[str, Any], reference: str) -> float:
    circuit = _mapping(case["circuit"], "circuit")
    component = _named(list(circuit["components"]), reference, "circuit.components")
    value = component["value"]
    if isinstance(value, str):
        variable = _mapping(circuit["variables"], "circuit.variables")[value]
        value = _mapping(variable, f"circuit.variables.{value}")["default"]
    return float(value)


def _load_capacitance(case: dict[str, Any], reference: str) -> float:
    load = _mapping(case["load"], "load")
    return float(_named(list(load["components"]), reference, "load.components")["value"])


def _read_profile(case_path: Path, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    load = _mapping(case["load"], "load")
    resistor = _named(list(load["components"]), "Rplasma", "load.components")
    config = _mapping(resistor["value"], "Rplasma.value")
    profile_path = (case_path.parent / str(config["profile"])).resolve()
    with profile_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row[str(config["time_column"])]) for row in rows], dtype=float)
    values = np.asarray([float(row[str(config["value_column"])]) for row in rows], dtype=float)
    period = float(config["repeat_period_s"])
    if times.size < 2 or times[0] != 0.0 or not math.isclose(times[-1], period):
        raise ValueError("the repeated Rplasma profile must span one closed period")
    return times, values, period


def _pulse(source: dict[str, Any], time_s: float) -> float:
    v1 = float(source["v1_V"])
    v2 = float(source["v2_V"])
    delay = float(source.get("delay_s", 0.0))
    rise = float(source["rise_s"])
    fall = float(source["fall_s"])
    width = float(source["width_s"])
    period = float(source["period_s"])
    if time_s < delay:
        return v1
    phase = (time_s - delay) % period
    if phase < rise:
        return v1 + (v2 - v1) * phase / rise
    if phase < rise + width:
        return v2
    if phase < rise + width + fall:
        return v2 + (v1 - v2) * (phase - rise - width) / fall
    return v1


def _profile_value(times: np.ndarray, values: np.ndarray, period: float, time_s: float) -> float:
    phase = time_s - math.floor(time_s / period) * period
    return float(np.interp(phase, times, values))


def solve_reference(case_path: Path) -> dict[str, np.ndarray]:
    payload = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    case = _mapping(payload, "case")
    sources = list(case["sources"])
    hf = _named(sources, "Vhf", "sources")
    bias = _named(sources, "Vbias", "sources")
    rhf = _component_value(case, "Rhf")
    rbias = _component_value(case, "Rbias")
    cblock = _component_value(case, "Cblock")
    ctop = _load_capacitance(case, "Csheath_upper")
    cbottom = _load_capacitance(case, "Csheath_wafer")
    profile_time, profile_r, profile_period = _read_profile(case_path, case)
    measurement = _mapping(case.get("measurement", {}), "measurement")
    z0 = float(measurement.get("reference_impedance_ohm", 50.0))
    stop_s = float(_mapping(_mapping(case["solver"], "solver")["tran"], "solver.tran")["stop_s"])
    declared_step_s = float(_mapping(_mapping(case["solver"], "solver")["tran"], "solver.tran")["step_s"])
    integration_step_s = declared_step_s / REFERENCE_REFINEMENT
    steps = round(stop_s / integration_step_s)
    time_s = np.linspace(0.0, stop_s, steps + 1)
    charge = np.zeros_like(time_s)
    inv_ceq = 1.0 / ctop + 1.0 / cbottom + 1.0 / cblock
    hf_omega = 2.0 * math.pi * float(hf["frequency_Hz"])
    hf_amplitude = float(hf["amplitude_V"])
    hf_phase = math.radians(float(hf.get("phase_deg", 0.0)))

    def state(time_value: float, charge_value: float) -> float:
        vhf = hf_amplitude * math.sin(hf_omega * time_value + hf_phase)
        vbias = _pulse(bias, time_value)
        rplasma = _profile_value(profile_time, profile_r, profile_period, time_value)
        return (vhf - vbias - charge_value * inv_ceq) / (rhf + rbias + rplasma)

    for index in range(steps):
        t0 = float(time_s[index])
        q0 = float(charge[index])
        dt = integration_step_s
        k1 = state(t0, q0)
        k2 = state(t0 + 0.5 * dt, q0 + 0.5 * dt * k1)
        k3 = state(t0 + 0.5 * dt, q0 + 0.5 * dt * k2)
        k4 = state(t0 + dt, q0 + dt * k3)
        charge[index + 1] = q0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    vhf = hf_amplitude * np.sin(hf_omega * time_s + hf_phase)
    vbias = np.asarray([_pulse(bias, float(value)) for value in time_s])
    rplasma = np.asarray([_profile_value(profile_time, profile_r, profile_period, float(value)) for value in time_s])
    current = (vhf - vbias - charge * inv_ceq) / (rhf + rbias + rplasma)
    upper_port = vhf - rhf * current
    wafer = vbias + rbias * current + charge / cblock
    plasma = upper_port - wafer
    incident = 0.5 * (upper_port + z0 * current)
    reflected = 0.5 * (upper_port - z0 * current)
    return {
        "time_s": time_s,
        "upper_hf_source_voltage_V": vhf,
        "lower_bias_source_voltage_V": vbias,
        "plasma_resistance_ohm": rplasma,
        "loop_current_A": current,
        "upper_port_voltage_V": upper_port,
        "upper_incident_voltage_V": incident,
        "upper_reflected_voltage_V": reflected,
        "wafer_voltage_V": wafer,
        "plasma_terminal_voltage_V": plasma,
        "charge_C": charge,
    }


def _sample(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source_time = data["time_s"]
    sample_time = np.arange(TARGET_START_S, float(source_time[-1]) + 0.5 * TARGET_STEP_S, TARGET_STEP_S)
    sampled = {"time_s": sample_time}
    for name, values in data.items():
        if name != "time_s":
            sampled[name] = np.asarray(np.interp(sample_time, source_time, values), dtype=np.float64)
    return sampled


def _write_csv(columns: dict[str, np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for values in zip(*(columns[name] for name in names), strict=True):
            writer.writerow([f"{float(value):.12g}" for value in values])


def write_targets(data: dict[str, np.ndarray], output_dir: Path) -> dict[str, Any]:
    sampled = _sample(data)
    observables_path = output_dir / "dual_frequency_ccp_target_observables.csv"
    wafer_path = output_dir / "dual_frequency_ccp_target_wafer.csv"
    reflection_path = output_dir / "dual_frequency_ccp_target_reflection.csv"
    _write_csv(sampled, observables_path)
    _write_csv({"time_s": sampled["time_s"], "voltage_V": sampled["wafer_voltage_V"]}, wafer_path)
    _write_csv(
        {"time_s": sampled["time_s"], "voltage_V": sampled["upper_reflected_voltage_V"]},
        reflection_path,
    )
    return {
        "outputs": {
            "observables": str(observables_path.resolve()),
            "wafer_objective": str(wafer_path.resolve()),
            "reflection_objective": str(reflection_path.resolve()),
        },
        "rows": int(sampled["time_s"].size),
        "time_range_s": [float(sampled["time_s"][0]), float(sampled["time_s"][-1])],
        "target_step_s": TARGET_STEP_S,
        "reference_integrator": "fixed-step RK4",
        "reference_step_s": float(data["time_s"][1] - data["time_s"][0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = write_targets(solve_reference(args.case.resolve()), args.output_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
