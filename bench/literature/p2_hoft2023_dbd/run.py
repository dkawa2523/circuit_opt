"""Test whether a diffuse-regime R||C predicts the held-out hybrid DBD current."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pcd.api import load_case, simulate_case
from pcd.artifacts import file_sha256, write_json, yaml_dump
from pcd.records import load_waveform

DATA = Path(__file__).with_name("digitized.csv")
DATA_SHA256 = "06acc0c491f4adb5242db77d38697f2c610101ddc7602512de910abaabeb17d9"
TRAIN = "diffuse_12p6_kVpp"
HOLDOUT = "hybrid_14_kVpp"
FREQUENCY_HZ = 10_000.0
PERIOD_S = 1 / FREQUENCY_HZ
ANGULAR_FREQUENCY = 2 * np.pi * FREQUENCY_HZ
OPTICAL_WINDOW_US = (50, 60)  # phase bin VI in Figs. 6 and 8(a)
PULSE_MISS_KEY = "frame_VI_negative_excursion_underestimate_beyond_ink_uA"
PULSE_MISS_THRESHOLD_UA = 40.0


@dataclass(frozen=True)
class Drive:
    sine_V: float
    cosine_V: float

    @classmethod
    def project(cls, time_s: np.ndarray, voltage_V: np.ndarray) -> Drive:
        phase = ANGULAR_FREQUENCY * time_s
        basis = np.column_stack((np.sin(phase), np.cos(phase)))
        sine, cosine = np.linalg.lstsq(basis, voltage_V, rcond=None)[0]
        return cls(float(sine), float(cosine))

    def evaluate(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase = ANGULAR_FREQUENCY * time_s
        sine, cosine = np.sin(phase), np.cos(phase)
        voltage = self.sine_V * sine + self.cosine_V * cosine
        derivative = ANGULAR_FREQUENCY * (self.sine_V * cosine - self.cosine_V * sine)
        return voltage, derivative


def fit_diffuse(train: pd.DataFrame) -> tuple[float, float]:
    time_s = train["time_us"].to_numpy() * 1e-6
    measured_voltage = train["voltage_V"].to_numpy()
    voltage, derivative = Drive.project(time_s, measured_voltage).evaluate(time_s)
    # With current in µA these columns condition the solve in nS and fF.
    basis = np.column_stack((voltage / 1e3, derivative / 1e9))
    conductance_nS, capacitance_fF = np.linalg.lstsq(basis, train["current_uA"], rcond=None)[0]
    if conductance_nS <= 0 or capacitance_fF <= 0:
        raise ValueError("the diffuse fit must admit a passive parallel resistor and capacitor")
    return float(1e9 / conductance_nS), float(capacitance_fF * 1e-15)


def solve_current(
    root: Path, time_s: np.ndarray, drive: Drive, resistance_ohm: float, capacitance_F: float
) -> tuple[np.ndarray, str]:
    amplitude = np.hypot(drive.sine_V, drive.cosine_V)
    phase_deg = np.degrees(np.arctan2(drive.cosine_V, drive.sine_V))
    root.mkdir()
    (root / "load.cir").write_text(
        "DBD terminal comparison with voltage constrained to its measured fundamental\n"
        f"Vdrive terminal 0 SIN(0 {amplitude:.16g} {FREQUENCY_HZ:.16g} 0 0 {phase_deg:.16g})\n"
        f"Rload terminal 0 {resistance_ohm:.16g}\n"
        f"Cload terminal 0 {capacitance_F:.16g}\n.end\n",
        encoding="ascii",
    )
    case_path = root / "load.yaml"
    case_path.write_text(
        yaml_dump(
            {
                "schema": "pcd.rf.v1",
                "case_id": root.name,
                "circuit": {"file": "load.cir", "format": "deck", "output_node": "terminal"},
                "analysis": {"tran": {"step_s": PERIOD_S / 2000, "stop_s": 2 * PERIOD_S}},
                "output": {"schematic": False, "plots": []},
            }
        ),
        encoding="utf-8",
    )
    record = simulate_case(load_case(case_path), run_root=root / "simulation")
    if record.status != "ok" or record.output_result is None or record.output_result["status"] != "complete":
        raise ValueError(f"DBD circuit simulation did not complete under {record.run_dir}")
    trace = load_waveform(record.run_dir)
    if trace["time_s"].iloc[-1] < time_s[-1] + PERIOD_S:
        raise ValueError("DBD circuit trace does not cover the comparison interval")
    # SPICE reports source branch current. The comparison uses current into the load.
    current_uA = -1e6 * np.interp(time_s + PERIOD_S, trace["time_s"], trace["current_A"])
    return current_uA, (record.run_dir / record.output_result["summary"]).relative_to(root.parent).as_posix()


def relative_l2(time_s: np.ndarray, predicted: np.ndarray, observed: np.ndarray) -> float:
    error = np.trapezoid((predicted - observed) ** 2, time_s)
    reference = np.trapezoid(observed**2, time_s)
    return float(np.sqrt(error / reference))


def trace_metrics(evidence: pd.DataFrame, projected_voltage: np.ndarray, predicted_current: np.ndarray) -> dict:
    time_us = evidence["time_us"].to_numpy()
    time_s = time_us * 1e-6
    window = (time_us >= OPTICAL_WINDOW_US[0]) & (time_us < OPTICAL_WINDOW_US[1])
    # Underestimate of a negative excursion, beyond even the upper (least negative) edge of the ink.
    outside_ink = predicted_current - evidence["ink_upper_uA"].to_numpy()
    pulse_index = np.argmax(np.where(window, outside_ink, -np.inf))
    underestimate = max(0.0, float(outside_ink[pulse_index]))
    return {
        "drive_projection_relative_l2": relative_l2(time_s, projected_voltage, evidence["voltage_V"].to_numpy()),
        "current_waveform_relative_l2": relative_l2(time_s, predicted_current, evidence["current_uA"].to_numpy()),
        PULSE_MISS_KEY: underestimate,
        "frame_VI_maximum_underestimate_time_us": float(time_us[pulse_index]) if underestimate > 0 else None,
    }


def compare(root: Path, evidence: pd.DataFrame, resistance: float, capacitance: float) -> tuple[dict, pd.DataFrame]:
    time_s = evidence["time_us"].to_numpy() * 1e-6
    drive = Drive.project(time_s, evidence["voltage_V"].to_numpy())
    projected_voltage, derivative = drive.evaluate(time_s)
    predicted_current, analysis = solve_current(root, time_s, drive, resistance, capacitance)
    analytic_current = 1e6 * (projected_voltage / resistance + capacitance * derivative)
    return {
        **trace_metrics(evidence, projected_voltage, predicted_current),
        "source_plot_columns": len(evidence),
        "circuit_equation_max_error_uA": float(np.max(np.abs(predicted_current - analytic_current))),
        "circuit_analysis": analysis,
    }, evidence.assign(drive_fundamental_V=projected_voltage, predicted_current_uA=predicted_current)


def held_out_drive_sensitivity(
    holdout: pd.DataFrame, resistance: float, capacitance: float
) -> tuple[dict, pd.DataFrame]:
    """Change the held-out voltage projection while keeping the circuit fit fixed."""
    time_s = holdout["time_us"].to_numpy() * 1e-6
    harmonics = ANGULAR_FREQUENCY * np.arange(1, 4)
    phase = time_s[:, None] * harmonics
    sine, cosine = np.sin(phase), np.cos(phase)
    basis = np.column_stack((np.ones(len(time_s)), sine, cosine))
    coefficients = np.linalg.lstsq(basis, holdout["voltage_V"], rcond=None)[0]
    projected_voltage = basis @ coefficients
    derivative = cosine @ (harmonics * coefficients[1:4]) - sine @ (harmonics * coefficients[4:])
    analytic_current = 1e6 * (projected_voltage / resistance + capacitance * derivative)
    return trace_metrics(holdout, projected_voltage, analytic_current), holdout.assign(
        drive_dc_through_h3_V=projected_voltage, analytic_current_dc_through_h3_uA=analytic_current
    )


def render_comparison(destination: Path, comparisons: list[pd.DataFrame]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col", layout="constrained")
    for column, (frame, title) in enumerate(zip(comparisons, ("12.6 kVpp · fit", "14.0 kVpp · held out"), strict=True)):
        time = frame["time_us"]
        voltage, current = axes[:, column]
        voltage.plot(time, frame["voltage_V"] / 1000, label="figure trace", color="0.6")
        voltage.plot(time, frame["drive_fundamental_V"] / 1000, "--", label="10 kHz drive")
        voltage.set(title=title, ylabel="voltage [kV]")
        current.fill_between(time, frame["ink_lower_uA"], frame["ink_upper_uA"], alpha=0.2, color="tab:red")
        current.plot(time, frame["current_uA"], label="figure trace and ink span", color="tab:red")
        current.plot(time, frame["predicted_current_uA"], "--", label="frozen R∥C prediction")
        current.axvspan(*OPTICAL_WINDOW_US, color="0.5", alpha=0.14, label="optical phase bin VI")
        current.set(xlabel="published phase time [µs]", ylabel="terminal current [µA]")
        for axis in (voltage, current):
            axis.grid(alpha=0.2)
            axis.legend()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    root = parser.parse_args().run_root.resolve()
    if file_sha256(DATA) != DATA_SHA256:
        raise ValueError("committed figure transcription differs from its published extraction")
    data = pd.read_csv(DATA)
    train = data.loc[data["condition"] == TRAIN]
    holdout = data.loc[data["condition"] == HOLDOUT]
    resistance, capacitance = fit_diffuse(train)
    root.mkdir(parents=True, exist_ok=False)
    training, training_data = compare(root / TRAIN, train, resistance, capacitance)
    transfer, transfer_data = compare(root / HOLDOUT, holdout, resistance, capacitance)
    sensitivity, transfer_data = held_out_drive_sensitivity(transfer_data, resistance, capacitance)
    checks = {
        "drive_fundamentals_retain_waveform": all(
            row["drive_projection_relative_l2"] < 0.06 for row in (training, transfer)
        ),
        "ngspice_matches_linear_circuit_equation": all(
            row["circuit_equation_max_error_uA"] < 0.1 for row in (training, transfer)
        ),
        "diffuse_fit_agrees_with_current_trace": training["current_waveform_relative_l2"] < 0.1,
        "held_out_hybrid_negative_excursion_is_missed": transfer[PULSE_MISS_KEY] > PULSE_MISS_THRESHOLD_UA,
        "held_out_excursion_miss_persists_with_dc_through_third_harmonic": sensitivity[PULSE_MISS_KEY]
        > PULSE_MISS_THRESHOLD_UA,
    }
    pd.concat([training_data, transfer_data]).to_csv(root / "electrical_comparison.csv", index=False)
    render_comparison(root / "comparison.png", [training_data, transfer_data])
    passed = all(checks.values())
    write_json(
        root / "benchmark_result.json",
        {
            "passed": passed,
            "checks": checks,
            "source": {"article_doi": "10.1140/epjd/s10053-023-00601-z", "figure_transcription_sha256": DATA_SHA256},
            "fit_condition": TRAIN,
            "held_out_condition": HOLDOUT,
            "reference_plane": "applied electrode voltage and grounded-side current of the published DBD cell",
            "resistance_ohm": resistance,
            "capacitance_F": capacitance,
            "optical_phase_bin_VI_us": OPTICAL_WINDOW_US,
            "optical_observations": {
                TRAIN: "diffuse emission morphology in Figure 6",
                HOLDOUT: "diffuse discharge and filament emission overlaid in Figure 8(a), phase bin VI",
                "sampling": "phase-binned ICCD images accumulated independently of the electrical trace",
                "morphology_reference": "Figures 12(a) and 13 distinguish filamentary and diffuse emission",
            },
            "training": training,
            "held_out": transfer,
            "held_out_analytic_drive_sensitivity_dc_through_third_harmonic": sensitivity,
            "figure": "comparison.png",
        },
    )
    print(f"{'PASS' if passed else 'FAIL'} {root / 'benchmark_result.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
