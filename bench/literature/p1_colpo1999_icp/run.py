from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pcd.analysis import at_frequency, input_impedance, read_ac
from pcd.artifacts import write_json
from pcd.case import load_case
from pcd.sim_core import simulate_case

HERE = Path(__file__).resolve().parent
L1_H = 8.5e-6
L2_H = 1.5e-6
C1_F = 60e-12
C2_F = 43e-12
FIXTURE_R_OHM = 1e-6
PUBLISHED_RESONANCES_HZ = (7.0e6, 14.5e6, 19.8e6)


def analytic_impedance(frequency_hz: float) -> complex:
    omega = 2.0 * math.pi * frequency_hz
    first = 1j * omega * L1_H / (1.0 - omega**2 * L1_H * C1_F)
    second = 1j * omega * L2_H / (1.0 - omega**2 * L2_H * C2_F)
    return first + second


def analytic_resonances() -> tuple[float, float, float]:
    parallel_1 = 1.0 / (2.0 * math.pi * math.sqrt(L1_H * C1_F))
    series = math.sqrt((L1_H + L2_H) / (L1_H * L2_H * (C1_F + C2_F))) / (2.0 * math.pi)
    parallel_2 = 1.0 / (2.0 * math.pi * math.sqrt(L2_H * C2_F))
    return parallel_1, series, parallel_2


def _estimate_crossing(frequency: list[float], reactance: list[float], expected: float, pole: bool) -> float:
    best: tuple[float, float] | None = None
    for index in range(len(frequency) - 1):
        x0, x1 = reactance[index], reactance[index + 1]
        if x0 == 0.0:
            estimate = frequency[index]
        elif x0 * x1 > 0.0:
            continue
        else:
            y0, y1 = (1.0 / x0, 1.0 / x1) if pole else (x0, x1)
            if y1 == y0:
                continue
            estimate = frequency[index] - y0 * (frequency[index + 1] - frequency[index]) / (y1 - y0)
        distance = abs(estimate - expected)
        if best is None or distance < best[0]:
            best = (distance, estimate)
    if best is None:
        raise RuntimeError(f"no reactance crossing found near {expected:g} Hz")
    return best[1]


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _complex_at_frequency(response: Any, frequency_hz: float) -> complex:
    frequencies = response["frequency_Hz"].astype(float).tolist()
    resistances = response["resistance_ohm"].astype(float).tolist()
    reactances = response["reactance_ohm"].astype(float).tolist()
    if frequency_hz <= frequencies[0]:
        return complex(resistances[0], reactances[0])
    if frequency_hz >= frequencies[-1]:
        return complex(resistances[-1], reactances[-1])
    for index in range(len(frequencies) - 1):
        lo, hi = frequencies[index], frequencies[index + 1]
        if lo <= frequency_hz <= hi:
            fraction = (frequency_hz - lo) / (hi - lo)
            resistance = resistances[index] + fraction * (resistances[index + 1] - resistances[index])
            reactance = reactances[index] + fraction * (reactances[index + 1] - reactances[index])
            return complex(resistance, reactance)
    raise RuntimeError(f"frequency {frequency_hz:g} Hz is outside the response")


def run(run_root: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    resonance_record = simulate_case(
        load_case(HERE / "dummy_resonance.yaml"),
        run_root=run_root,
        run_id="dummy_resonance",
    )
    if resonance_record.status != "ok" or resonance_record.frequency_response_file is None:
        raise RuntimeError(f"dummy resonance simulation failed: {resonance_record.warnings}")

    response = input_impedance(read_ac(resonance_record.run_dir / resonance_record.frequency_response_file))
    frequency = response["frequency_Hz"].astype(float).tolist()
    reactance = response["reactance_ohm"].astype(float).tolist()
    analytic = analytic_resonances()
    spice = (
        _estimate_crossing(frequency, reactance, analytic[0], pole=True),
        _estimate_crossing(frequency, reactance, analytic[1], pole=False),
        _estimate_crossing(frequency, reactance, analytic[2], pole=True),
    )

    probe_checks = []
    for probe_hz in (5.0e6, 10.0e6, 17.0e6, 23.0e6):
        actual = _complex_at_frequency(response, probe_hz)
        expected = analytic_impedance(probe_hz) + FIXTURE_R_OHM
        error = abs(actual - expected) / max(abs(expected), 1e-30)
        probe_checks.append(
            {
                "frequency_Hz": probe_hz,
                "expected_R_ohm": expected.real,
                "expected_X_ohm": expected.imag,
                "spice_R_ohm": actual.real,
                "spice_X_ohm": actual.imag,
                "relative_complex_error": error,
                "passed": error <= 5e-3,
            }
        )

    graphite_record = simulate_case(
        load_case(HERE / "graphite_dummy.yaml"),
        run_root=run_root,
        run_id="graphite_dummy",
    )
    if graphite_record.status != "ok" or graphite_record.frequency_response_file is None:
        raise RuntimeError(f"graphite dummy simulation failed: {graphite_record.warnings}")
    graphite_response = input_impedance(read_ac(graphite_record.run_dir / graphite_record.frequency_response_file))
    graphite_row = at_frequency(graphite_response, 13.56e6)
    graphite_actual = complex(float(graphite_row["resistance_ohm"]), float(graphite_row["reactance_ohm"]))
    graphite_expected = complex(1.26, 57.0)
    graphite_error = abs(graphite_actual - graphite_expected) / abs(graphite_expected)

    resonance_rows = []
    analytic_limits = (0.03, 0.05, 0.03)
    for name, calculated, simulated, published, published_limit in zip(
        ("parallel_1", "series", "parallel_2"),
        analytic,
        spice,
        PUBLISHED_RESONANCES_HZ,
        analytic_limits,
        strict=True,
    ):
        analytic_error = _relative_error(simulated, calculated)
        published_error = _relative_error(calculated, published)
        resonance_rows.append(
            {
                "name": name,
                "published_Hz": published,
                "analytic_from_rounded_components_Hz": calculated,
                "spice_Hz": simulated,
                "spice_vs_analytic_relative_error": analytic_error,
                "analytic_vs_published_relative_error": published_error,
                "spice_passed": analytic_error <= 5e-3,
                "published_rounding_passed": published_error <= published_limit,
            }
        )

    published_known_load = {
        "measured_R_ohm": 1.26,
        "measured_X_ohm": 57.0,
        "boundary_calculation_R_ohm": 1.4,
        "boundary_calculation_X_ohm": 52.0,
        "R_discrepancy_fraction": _relative_error(1.4, 1.26),
        "X_discrepancy_fraction": _relative_error(52.0, 57.0),
    }
    published_known_load["passed"] = (
        published_known_load["R_discrepancy_fraction"] <= 0.12
        and published_known_load["X_discrepancy_fraction"] <= 0.10
    )

    payload: dict[str, Any] = {
        "benchmark_id": "p1_colpo1999_icp",
        "source": "10.1063/1.369268",
        "scope": "dummy fixture topology and published known-load replay; not plasma-on model validation",
        "resonances": resonance_rows,
        "off_resonance_impedance_checks": probe_checks,
        "graphite_dummy_spice": {
            "expected_R_ohm": graphite_expected.real,
            "expected_X_ohm": graphite_expected.imag,
            "actual_R_ohm": graphite_actual.real,
            "actual_X_ohm": graphite_actual.imag,
            "relative_complex_error": graphite_error,
            "passed": graphite_error <= 2e-5,
        },
        "published_graphite_measurement_vs_calculation": published_known_load,
        "passivity": {
            "minimum_spice_resistance_ohm": min(response["resistance_ohm"].astype(float)),
            "passed": min(response["resistance_ohm"].astype(float)) >= -1e-6,
        },
    }
    payload["passed"] = all(row["spice_passed"] and row["published_rounding_passed"] for row in resonance_rows)
    payload["passed"] = payload["passed"] and all(row["passed"] for row in probe_checks)
    payload["passed"] = payload["passed"] and payload["graphite_dummy_spice"]["passed"]
    payload["passed"] = payload["passed"] and published_known_load["passed"]
    payload["passed"] = payload["passed"] and payload["passivity"]["passed"]
    write_json(run_root / "colpo1999_evaluation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Colpo 1999 literature benchmarks.")
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/colpo1999"))
    args = parser.parse_args()
    payload = run(args.run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
