"""Check terminal pulse energy and reject an energy-matched current-waveform model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd.api import load_case, simulate_case
from pcd.artifacts import file_sha256, read_json, write_json, yaml_dump
from pcd.records import load_waveform
from pcd.signals.power import cumulative_energy
from pcd.solver import diagnose_solver
from pcd.validation import validate_case

SOURCE = Path(__file__).with_name("source")
DOI = "10.57745/8LZ7LU"
ELECTRICAL = "Fig3_current-voltage-energy.txt"
OPTICAL = "Fig7_ne_vs_time.txt"
SHA256 = {
    ELECTRICAL: "e25dc7d9aa9cf72a6fb1975251755ef377371c4df49628ab06af00c6bfbe25d5",
    OPTICAL: "20ad2ad6ac7142c09770bfe0f8f121dab645bc36200273adce6e534c6f4815fa",
}


def load_evidence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    for name, digest in SHA256.items():
        if file_sha256(SOURCE / name) != digest:
            raise ValueError(f"published source bytes differ for {name}")
    time_s, voltage_kV, current_A, energy_J = np.loadtxt(SOURCE / ELECTRICAL, skiprows=2, usecols=(0, 1, 2, 3)).T
    if (
        len(time_s) != 1251
        or np.any(np.diff(time_s) <= 0)
        or not np.all(np.isfinite([time_s, voltage_kV, current_A, energy_J]))
    ):
        raise ValueError("the electrical source must contain 1251 finite, ordered observations")
    optical = pd.read_csv(SOURCE / OPTICAL, sep="\t", skiprows=3, header=None, usecols=[0, 3, 8], na_values=["--"])
    optical.columns = ["published_time_ns", "electron_density_cm^-3", "reported_uncertainty_cm^-3"]
    return time_s - time_s[0], 1000 * voltage_kV, current_A, energy_J - energy_J[0], optical


def simulate_resistor(root: Path, time: np.ndarray, voltage: np.ndarray, resistance: float) -> dict[str, Any]:
    """Drive a one-port circuit with the measured terminal voltage on its original grid."""

    points = "\n".join(f"+ {t:.16g} {v:.16g}" for t, v in zip(time, voltage, strict=True))
    (root / "pulse.cir").write_text(
        f"Measured terminal voltage and constant resistance\nVdrive electrode 0 PWL(\n{points}\n+ )\n"
        f"R1 electrode 0 {resistance:.16g}\n.end\n",
        encoding="ascii",
    )
    path = root / "pulse.yaml"
    path.write_text(
        yaml_dump(
            {
                "schema": "pcd.rf.v1",
                "case_id": "pulse_energy_matched_resistor",
                "circuit": {"file": "pulse.cir", "format": "deck", "output_node": "electrode"},
                "analysis": {"tran": {"step_s": float(np.min(np.diff(time))), "stop_s": float(time[-1])}},
                "output": {"schematic": False, "plots": []},
            }
        ),
        encoding="utf-8",
    )
    case = load_case(path)
    validation = validate_case(case)
    if not validation.ok:
        raise ValueError(validation.format_text())
    record = simulate_case(case, run_root=root / "simulation")
    if record.status != "ok" or record.output_result is None or record.output_result["status"] != "complete":
        raise ValueError(f"pulse replay did not produce a complete result under {record.run_dir}")
    summary = read_json(record.run_dir / record.output_result["summary"])
    trace = load_waveform(record.run_dir)
    return {
        "current_A": -np.interp(time, trace["time_s"], trace["current_A"]),
        "source_net_energy_J": summary["analyses"]["transient"]["energy"]["source_net_energy_J"],
        "analysis": (record.run_dir / record.output_result["summary"]).relative_to(root).as_posix(),
    }


def render_comparison(data: pd.DataFrame, optical: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(8, 10), layout="constrained")
    t = data["electrical_time_from_1A_ns"]
    axes[0].plot(t, data["voltage_V"] / 1000, color="tab:blue")
    axes[0].set(ylabel="voltage [kV]", title="Measured anode-terminal voltage (prescribed circuit input)")
    axes[1].plot(t, data["measured_current_A"], label="measured")
    axes[1].plot(t, data["resistor_current_A"], "--", label="energy-matched resistor")
    time_label = "electrical time from first current sample above 1 A [ns]"
    axes[1].set(ylabel="current [A]", xlabel=time_label)
    axes[2].plot(t, 1e3 * data["published_energy_J"], label="published")
    axes[2].plot(t, 1e3 * data["integrated_energy_J"], ":", label="integrated measured $VI$")
    axes[2].plot(t, 1e3 * data["resistor_energy_J"], "--", label="resistor under measured V")
    axes[2].set(ylabel="net energy [mJ]", xlabel=time_label)
    for axis in axes[:3]:
        axis.axvspan(0, 30, color="0.9", zorder=-1)
        axis.grid(alpha=0.25)
    for axis in axes[:2]:
        axis.set_xlim(-8, 65)
    axes[0].text(15, 0.08, "energy-fit window", ha="center", transform=axes[0].get_xaxis_transform())
    axes[1].legend()
    axes[2].legend()
    axes[3].errorbar(
        optical["published_time_ns"],
        optical["electron_density_cm^-3"] / 1e15,
        yerr=optical["reported_uncertainty_cm^-3"] / 1e15,
        fmt=".",
        color="tab:purple",
    )
    axes[3].set(
        ylabel="electron density [10$^{15}$ cm$^{-3}$]",
        xlabel="published optical time [ns]",
        title="Optical observation 2.25 mm from cathode (published time axis)",
    )
    axes[3].grid(alpha=0.25)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="new directory for the pulse benchmark")
    root = parser.parse_args().run_root.resolve()
    solver = diagnose_solver()
    if not solver["batch_runnable"]:
        parser.exit(2, "ngspice is required for the circuit comparison\n")
    time, voltage, current, published, optical = load_evidence()
    root.mkdir(parents=True, exist_ok=False)
    measured = cumulative_energy(time, voltage * current)
    start = int(np.flatnonzero(current > 1.0)[0])
    end = int(np.argmin(np.abs(time - time[start] - 30e-9)))
    fit = slice(start, end + 1)
    resistance = float(cumulative_energy(time[fit], voltage[fit] ** 2)[-1] / (measured[end] - measured[start]))
    solved = simulate_resistor(root, time, voltage, resistance)
    predicted_current = solved["current_A"]
    predicted_energy = cumulative_energy(time, voltage * predicted_current)
    fit_energy_error = abs((predicted_energy[end] - predicted_energy[start]) / (measured[end] - measured[start]) - 1)
    waveform_error = np.sqrt(
        cumulative_energy(time[fit], (predicted_current[fit] - current[fit]) ** 2)[-1]
        / cumulative_energy(time[fit], current[fit] ** 2)[-1]
    )
    published_error = float(np.max(np.abs(measured - published)) / published[-1])
    before, after = voltage[:-1], voltage[1:]
    expected_resistor_energy = float(np.dot(np.diff(time), before**2 + before * after + after**2) / (3 * resistance))
    checks = {
        "measured_vi_reproduces_published_energy_curve": published_error < 0.01,
        "spice_current_matches_resistor_equation": bool(
            np.allclose(predicted_current, voltage / resistance, rtol=1e-4, atol=1e-6)
        ),
        "spice_energy_matches_resistor_equation": abs(solved["source_net_energy_J"] / expected_resistor_energy - 1)
        < 0.01,
        "resistor_matches_selected_window_energy": bool(fit_energy_error < 0.01),
        "resistor_fails_current_waveform": bool(waveform_error > 0.5),
    }
    data = pd.DataFrame(
        {
            "electrical_time_from_1A_ns": (time - time[start]) * 1e9,
            "voltage_V": voltage,
            "measured_current_A": current,
            "resistor_current_A": predicted_current,
            "published_energy_J": published,
            "integrated_energy_J": measured,
            "resistor_energy_J": predicted_energy,
        }
    )
    data.to_csv(root / "electrical_comparison.csv", index=False)
    optical.to_csv(root / "optical_evidence.csv", index=False)
    render_comparison(data, optical, root / "pulse_comparison.png")
    passed = all(checks.values())
    write_json(
        root / "benchmark_result.json",
        {
            "passed": passed,
            "checks": checks,
            "source": {"doi": DOI, "version": 2, "sha256": SHA256},
            "solver": solver,
            "reference_plane": "anode electrical probes in the published setup",
            "electrical_scope": "published shot-averaged terminal waveform, including reflected pulses",
            "optical_scope": "separate electron-density measurement; published optical time axis",
            "fit_window_ns_from_1A": [0.0, float((time[end] - time[start]) * 1e9)],
            "fit_resistance_ohm": resistance,
            "measured_full_net_energy_J": float(measured[-1]),
            "published_full_net_energy_J": float(published[-1]),
            "published_curve_max_error_fraction": published_error,
            "fit_energy_relative_error": float(fit_energy_error),
            "fit_current_waveform_nrmse": float(waveform_error),
            "resistor_full_net_energy_J": solved["source_net_energy_J"],
            "resistor_expected_net_energy_J": expected_resistor_energy,
            "circuit_analysis": solved["analysis"],
            "figure": "pulse_comparison.png",
        },
    )
    print(f"{'PASS' if passed else 'FAIL'} {root / 'benchmark_result.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
