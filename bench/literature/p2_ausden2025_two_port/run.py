"""Fit one acquisition and predict complex transmission on a second acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

import matplotlib
import numpy as np
import pandas as pd

from pcd.api import run_network
from pcd.artifacts import write_json, yaml_dump
from pcd.reporting import analyze_run
from pcd.solver import diagnose_solver

matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE_DOI = "10.5281/zenodo.22658069"
SOURCES = {
    "broadband": ("5dfe5b6872f07a533279c256090a14885151b38b67181662a12c51c7c628c9bf", 871),
    "coaxial": ("10502556b26765905e479b574fe5feacc94f65493b7828ed7dc4924d173b03dc", 191),
}
PORT_IMPEDANCE_OHM = 50.0
FIT_STOP_HZ = 12.5e9
S_PARAMETERS = ("s11", "s21", "s12", "s22")
DIRECTIONS = ("forward", "reverse")
MODELS = ("delay", "resistive")
TRANSFER_COMPLEX_RMSE_MAX = 0.005
TRANSFER_MAGNITUDE_RMSE_DB_MAX = 0.10
REFLECTION_COMPLEX_RMSE_MAX = 0.04


class SweepArtifacts(TypedDict):
    summary: str
    bode: str
    bode_complex_max_error: float


def read_sweep(acquisition: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact archived RI-format Touchstone files and their frequency grid."""

    raw = (HERE / "source" / f"{acquisition}.s2p").read_bytes()
    digest, count = SOURCES[acquisition]
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"published source checksum mismatch for {acquisition}")
    lines = raw.decode("ascii").splitlines()
    options = [line.lower().split() for line in lines if line.startswith("#")]
    if options != [["#", "hz", "s", "ri", "r", "50.0"]]:
        raise ValueError(f"expected frequency in Hz and real/imaginary S at 50 ohms for {acquisition}")
    values = np.loadtxt([line for line in lines if line.strip() and not line.startswith(("#", "!"))])
    grid = 2.5e9 + 0.25e9 * np.arange(count)
    if values.shape != (count, 9) or not np.isfinite(values).all() or not np.array_equal(values[:, 0], grid):
        raise ValueError(f"unexpected frequency grid or invalid complex samples for {acquisition}")
    return grid, values[:, 1::2] + 1j * values[:, 2::2]


def fit_transfer(frequency: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    fit = frequency <= FIT_STOP_HZ
    transmission = measured[fit, 1:3]
    omega_rad_per_ps = 2 * np.pi * frequency[fit] * 1e-12
    phase = np.unwrap(np.angle(transmission), axis=0)
    gain = float(np.mean(np.abs(transmission)))
    delay_ps = -float(np.mean(omega_rad_per_ps @ phase) / (omega_rad_per_ps @ omega_rad_per_ps))
    if not 0 < gain < 1 or not np.isfinite(delay_ps) or delay_ps <= 0:
        raise ValueError("training acquisition does not support the chosen passive attenuation-and-delay model")
    return gain, delay_ps


def write_network_input(model: str, gain: float, delay_ps: float, frequency: np.ndarray, root: Path) -> Path:
    z0 = PORT_IMPEDANCE_OHM
    loss = 1 / gain
    series = z0 * (loss - 1) / (loss + 1)
    shunt = 2 * z0 * loss / (loss**2 - 1)
    pad_end = "far" if model == "delay" else "p2"
    components = [
        f"R1 p1 pad {series:.15g}",
        f"R2 pad 0 {shunt:.15g}",
        f"R3 pad {pad_end} {series:.15g}",
    ]
    if model == "delay":
        components.append(f"Tdelay far 0 p2 0 Z0={z0:g} TD={delay_ps:.15g}p")
    (root / f"{model}.cir").write_text("\n".join(components) + "\n", encoding="utf-8")
    path = root / f"{model}.yaml"
    specification = {
        "schema": "pcd.network.v1",
        "dut": {"file": f"{model}.cir"},
        "ports": [{"name": name, "p": name, "reference_impedance_ohm": z0} for name in ("p1", "p2")],
        "ac": {
            "sweep": "lin",
            "points": len(frequency),
            "start_Hz": float(frequency[0]),
            "stop_Hz": float(frequency[-1]),
        },
    }
    path.write_text(yaml_dump(specification), encoding="utf-8")
    return path


def solve(
    model: str, gain: float, delay_ps: float, frequency: np.ndarray, root: Path
) -> tuple[np.ndarray, dict[str, SweepArtifacts]]:
    path = write_network_input(model, gain, delay_ps, frequency, root)
    network_root = root / model
    result = run_network(path, network_root)
    if result["status"] != "complete":
        raise ValueError(f"incomplete scattering evidence: {network_root}")
    table = pd.read_csv(network_root / result["artifacts"]["scattering"])
    matrices = []
    for receiver, driver in (("p1", "p1"), ("p2", "p1"), ("p1", "p2"), ("p2", "p2")):
        entry = table[(table["outgoing_port"] == receiver) & (table["incident_port"] == driver)]
        if not np.array_equal(entry["frequency_Hz"].to_numpy(), frequency):
            raise ValueError(f"wrong scattering frequency coverage: {network_root}")
        matrices.append(entry["s_re"].to_numpy() + 1j * entry["s_im"].to_numpy())
    runs: dict[str, SweepArtifacts] = {}
    for index, (direction, excitation) in enumerate(zip(DIRECTIONS, result["excitations"], strict=True)):
        run_dir = (network_root / excitation["manifest"]).parents[1]
        analysis = analyze_run(
            run_dir,
            output={
                "schematic": False,
                "plots": [
                    {
                        "type": "bode",
                        "name": "transmission",
                        "numerator": f"port_{2 - index}_v",
                        "denominator": "source_voltage_V",
                    }
                ],
            },
        )
        if analysis["status"] != "complete":
            raise ValueError(f"incomplete Bode export: {run_dir}")
        summary_path = Path(analysis["output_dir"]) / "summary.json"
        plot = analysis["output"]["plots"][0]
        bode_path = summary_path.parent / plot["data"][0]["file"]
        bode = pd.read_csv(bode_path)
        if not np.array_equal(bode["frequency_Hz"].to_numpy(), frequency):
            raise ValueError(f"wrong frequency coverage in exported Bode data: {bode_path}")
        # Equal fixture and wave reference impedances give half the generator voltage as the incident voltage.
        reported_transfer = (
            2 * 10 ** (bode["gain_dB"].to_numpy() / 20) * np.exp(1j * np.deg2rad(bode["phase_deg"].to_numpy()))
        )
        error = float(np.max(np.abs(matrices[1 + index] - reported_transfer)))
        if not np.isfinite(error):
            raise ValueError(f"non-finite Bode evidence: {bode_path}")
        runs[direction] = {
            "summary": summary_path.relative_to(root).as_posix(),
            "bode": bode_path.relative_to(root).as_posix(),
            "bode_complex_max_error": error,
        }
    return np.column_stack(matrices), runs


def error_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    transmission = measured[:, 1:3]
    predicted_transfer = predicted[:, 1:3]
    return {
        "transfer_complex_rmse": float(np.sqrt(np.mean(np.abs(predicted_transfer - transmission) ** 2))),
        "transfer_magnitude_rmse_dB": float(
            np.sqrt(np.mean((20 * np.log10(np.abs(predicted_transfer / transmission))) ** 2))
        ),
        "transfer_phase_rmse_deg": float(
            np.sqrt(np.mean(np.rad2deg(np.angle(predicted_transfer / transmission)) ** 2))
        ),
        "reflection_complex_rmse": float(np.sqrt(np.mean(np.abs(predicted[:, [0, 3]] - measured[:, [0, 3]]) ** 2))),
    }


def write_comparison(root: Path, frequency: np.ndarray, measured: np.ndarray, predicted: dict[str, np.ndarray]) -> None:
    rows = {"frequency_Hz": frequency, "frequency_scope": np.where(frequency <= FIT_STOP_HZ, "overlap", "extension")}
    for label, values in {"measured_coaxial": measured, **predicted}.items():
        for parameter, response in zip(S_PARAMETERS, values.T, strict=True):
            rows[f"{label}.{parameter}.real"] = response.real
            rows[f"{label}.{parameter}.imag"] = response.imag
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
    series = (
        ("Coaxial measurement", measured, "0.2"),
        ("Pad with delay", predicted["delay"], "tab:blue"),
        ("Resistive pad", predicted["resistive"], "tab:orange"),
    )
    for label, values, color in series:
        for port, linestyle in enumerate(("-", "--")):
            axes[0].plot(
                frequency / 1e9,
                20 * np.log10(np.abs(values[:, 1 + port])),
                linestyle,
                color=color,
                label=label if not port else None,
            )
            axes[1].plot(frequency / 1e9, np.rad2deg(np.angle(values[:, 1 + port])), linestyle, color=color)
            axes[2].plot(frequency / 1e9, np.abs(values[:, 3 * port]), linestyle, color=color)
    for ax in axes:
        ax.axvline(FIT_STOP_HZ / 1e9, color="0.55", linewidth=0.9)
        ax.grid(alpha=0.25)
    axes[0].legend()
    axes[0].set_title("Coaxial holdout  •  solid port 1 drive, dashed port 2 drive")
    axes[0].set_ylabel("Transfer (dB)")
    axes[1].set_ylabel("Transfer phase (°)")
    axes[2].set_ylabel("Reflection magnitude")
    axes[2].set_xlabel("Frequency (GHz)  •  vertical line at training-band end")
    figure.savefig(root / "comparison.png", dpi=150)
    plt.close(figure)


def run(root: Path) -> dict[str, Any]:
    solver = diagnose_solver()
    if not solver["batch_runnable"]:
        raise RuntimeError(f"ngspice is unavailable: {solver['notes']}")
    train_frequency, training = read_sweep("broadband")
    gain, delay_ps = fit_transfer(train_frequency, training)
    frequency, measured = read_sweep("coaxial")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    predicted = {}
    runs = {}
    for model in MODELS:
        predicted[model], runs[model] = solve(model, gain, delay_ps, frequency, root)
    masks = {"overlap": frequency <= FIT_STOP_HZ, "extension": frequency > FIT_STOP_HZ}
    errors = {
        model: {region: error_metrics(measured[mask], values[mask]) for region, mask in masks.items()}
        for model, values in predicted.items()
    }
    checks = {
        "public_bode_matches_wave_conversion": all(
            run_info["bode_complex_max_error"] < 1e-10
            for model_runs in runs.values()
            for run_info in model_runs.values()
        ),
        "both_models_meet_transmission_magnitude_limit": all(
            region["transfer_magnitude_rmse_dB"] < TRANSFER_MAGNITUDE_RMSE_DB_MAX
            for model_errors in errors.values()
            for region in model_errors.values()
        ),
        "delay_predicts_complex_transmission_on_both_holdout_bands": all(
            region["transfer_complex_rmse"] < TRANSFER_COMPLEX_RMSE_MAX for region in errors["delay"].values()
        ),
        "magnitude_only_control_fails_complex_transmission_on_extension": (
            errors["resistive"]["extension"]["transfer_complex_rmse"] > TRANSFER_COMPLEX_RMSE_MAX
        ),
        "matched_model_limitation_is_visible_in_reflection": all(
            region["reflection_complex_rmse"] > REFLECTION_COMPLEX_RMSE_MAX for region in errors["delay"].values()
        ),
    }
    write_comparison(root, frequency, measured, predicted)
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": {"doi": SOURCE_DOI, "sha256": {name: source[0] for name, source in SOURCES.items()}},
        "solver": solver,
        "solver_sweeps": len(MODELS) * len(DIRECTIONS),
        "solver_frequency_points": len(MODELS) * len(DIRECTIONS) * len(frequency),
        "fit": {
            "acquisition": "broadband",
            "rows": int(np.count_nonzero(train_frequency <= FIT_STOP_HZ)),
            "gain": gain,
            "delay_ps": delay_ps,
        },
        "holdout": {
            "acquisition": "coaxial",
            "rows": {region: int(np.count_nonzero(mask)) for region, mask in masks.items()},
        },
        "engineering_limits": {
            "transfer_complex_rmse_max": TRANSFER_COMPLEX_RMSE_MAX,
            "transfer_magnitude_rmse_dB_max": TRANSFER_MAGNITUDE_RMSE_DB_MAX,
            "reflection_complex_rmse_max": REFLECTION_COMPLEX_RMSE_MAX,
        },
        "errors": errors,
        "runs": runs,
        "networks": {model: f"{model}/network_result.json" for model in MODELS},
        "comparison": "comparison.csv",
        "figure": "comparison.png",
    }
    write_json(root / "benchmark_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    options = parser.parse_args()
    result = run(options.run_root)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
