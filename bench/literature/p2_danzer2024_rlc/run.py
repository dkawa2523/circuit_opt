"""Compare two passive terminal models on a measured impedance sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd.api import load_case, run_identification
from pcd.artifacts import read_json, write_json, yaml_dump
from pcd.solver import diagnose_solver
from pcd.validation import validate_case

SOURCE = Path(__file__).with_name("rlc_circuit.json")
SOURCE_DOI = "10.5281/zenodo.10794584"
SOURCE_SHA256 = "5ba93aeef5570f8eaf13bf84e0d2b3bb5221f2d208ce365c9088b62873ef5cbd"
FIT_ROWS = (0, 10, 20, 25, 28, 29, 35, 36, 40, 50, 65, 80)
HOLDOUT_ROWS = (30, 31, 32, 33, 34)
ERROR_SCALE_OHM = 100.0


def measured_data() -> pd.DataFrame:
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("the measured dataset does not match the frozen source digest")
    raw = json.loads(source)
    frequency = np.asarray(raw["f"], dtype=float)
    impedance = np.asarray(raw["z_real"], dtype=float) + 1j * np.asarray(raw["z_imag"], dtype=float)
    if (
        len(frequency) != 81
        or not np.all(np.isfinite(frequency) & (frequency > 0))
        or not np.all(np.diff(frequency) < 0)
        or not np.all(np.isfinite(impedance) & (impedance.real > 0))
        or not np.allclose(np.abs(impedance), raw["z_modulus"], rtol=1e-12, atol=0)
        or not np.allclose(np.angle(impedance), raw["z_phase"], rtol=0, atol=1e-12)
    ):
        raise ValueError("unexpected frequency order, impedance, magnitude or phase in the published sweep")
    data = pd.DataFrame(
        {
            "scenario_id": [f"point_{i:02d}" for i in range(len(frequency))],
            "frequency_Hz": frequency,
            "observed_resistance_ohm": impedance.real,
            "observed_reactance_ohm": impedance.imag,
            "partition": "unused",
        }
    )
    data.loc[list(FIT_ROWS), "partition"] = "fit"
    data.loc[list(HOLDOUT_ROWS), "partition"] = "holdout"
    return data


def case_definition(model: str) -> dict[str, Any]:
    series = model == "series_loss"
    return {
        "schema": "pcd.rf.v1",
        "case_id": model,
        "source": {"type": "dc_voltage", "voltage_V": 0, "ac_magnitude_V": 1},
        "circuit": {
            "output_node": "src",
            "components": [
                {"ref": "R1", "n1": "src", "n2": "coil" if series else "0", "value": "R"},
                {"ref": "L1", "n1": "coil" if series else "src", "n2": "0", "value": "L"},
                {"ref": "C1", "n1": "src", "n2": "0", "value": "C"},
            ],
        },
        "parameters": {
            "latent": {
                "R": {"bounds": [0.001, 30] if series else [1, 1000], "default": 6 if series else 200},
                "L": {"bounds": ["3 mH", "7 mH"], "default": "5 mH"},
                "C": {"bounds": ["1.5 uF", "3 uF"], "default": "2.2 uF"},
            }
        },
        "analysis": {"ac": {"frequency_Hz": "frequency_Hz"}},
        "identification": {
            "observations": {"file": "observations.csv"},
            "error_scale_ohm": ERROR_SCALE_OHM,
            "fit_loss_max": 0.02,
            "holdout_loss_max": 0.08,
            "condition_number_max": 100,
        },
        "execution": {"trials": 60},
    }


def run_model(root: Path, data: pd.DataFrame, model: str) -> dict[str, Any]:
    case_path = root / f"{model}.yaml"
    case_path.write_text(yaml_dump(case_definition(model)), encoding="utf-8")
    case = load_case(case_path)
    validation = validate_case(case)
    if not validation.ok:
        raise ValueError(validation.format_text())
    result = run_identification(case, root)
    if not result["estimated_latent"]:
        raise ValueError(f"{model} did not produce complete fit evidence: {result.get('reason')}")

    # Separate circuit algebra checks the fitted topology and sign against
    # the actual ngspice replay; the measured dataset checks prediction error.
    r, inductance, capacitance = (result["estimated_latent"][key] for key in ("R", "L", "C"))
    jw = 2j * np.pi * data["frequency_Hz"].to_numpy(float)
    coil = 1 / (r + jw * inductance) if model == "series_loss" else 1 / r + 1 / (jw * inductance)
    prediction = 1 / (coil + jw * capacitance)
    data[f"{model}_resistance_ohm"] = prediction.real
    data[f"{model}_reactance_ohm"] = prediction.imag
    observed = data["observed_resistance_ohm"] + 1j * data["observed_reactance_ohm"]
    data[f"{model}_error_ohm"] = abs(prediction - observed)
    report: dict[str, Any] = {
        "result": (Path(result["run_root"]) / "identification_result.json").relative_to(root).as_posix(),
        "status": result["status"],
        "parameters_SI": result["estimated_latent"],
        "rank": result["identifiability"]["rank"],
        "condition_number": result["identifiability"]["condition_number"],
    }
    for partition in ("fit", "holdout"):
        replay = pd.read_csv(Path(result["run_root"]) / result["artifacts"][f"{partition}_observations"])
        expected = data.loc[data["partition"] == partition].set_index("scenario_id")
        replay = replay.set_index("scenario_id").reindex(expected.index)
        difference = (replay["predicted.resistance_ohm"] - expected[f"{model}_resistance_ohm"]) + 1j * (
            replay["predicted.reactance_ohm"] - expected[f"{model}_reactance_ohm"]
        )
        error = abs(difference).to_numpy(float)
        if not np.all(np.isfinite(error)) or float(error.max()) > 1e-6:
            raise ValueError(f"{model} {partition}: ngspice replay disagrees with the circuit equation")
        rmse = math.sqrt(float(np.mean(expected[f"{model}_error_ohm"].to_numpy(float) ** 2)))
        reported_rmse = ERROR_SCALE_OHM * result[partition]["value"]
        if not math.isclose(rmse, reported_rmse, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"{model} {partition}: reported RMS error disagrees with the measured-point residuals")
        report[f"{partition}_rmse_ohm"] = rmse
        report[f"{partition}_passed"] = result[partition]["passed"]
        report[f"{partition}_max_spice_equation_difference_ohm"] = float(error.max())
        report[f"{partition}_study"] = read_json(Path(result["run_root"]) / result["artifacts"][f"{partition}_result"])[
            "dataset"
        ]
    return report


def plot_comparison(data: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 8), layout="constrained")
    f = data["frequency_Hz"]
    withheld = f.loc[data["partition"] == "holdout"]
    for axis, quantity in zip(axes, ("resistance", "reactance", "error"), strict=True):
        axis.axvspan(withheld.min(), withheld.max(), color="0.85", label="withheld band")
        axis.plot(f, data[f"series_loss_{quantity}_ohm"], label="series coil loss")
        axis.plot(f, data[f"parallel_loss_{quantity}_ohm"], "--", label="parallel loss")
        if quantity != "error":
            observed = data[f"observed_{quantity}_ohm"]
            for partition, marker, size in (("unused", ".", 12), ("fit", "o", 23), ("holdout", "^", 29)):
                rows = data["partition"] == partition
                axis.scatter(
                    f[rows], observed[rows], s=size, marker=marker, color="0.25", label=f"measured {partition}"
                )
        axis.set(xscale="log", ylabel=f"{quantity} [ohm]")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)
    axes[-1].set(xlabel="frequency [Hz]", ylabel="|predicted Z - measured Z| [ohm]")
    figure.suptitle("Measured parallel RLC — two terminal models, one withheld frequency band")
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="new directory for the benchmark evidence")
    root = parser.parse_args().run_root.resolve()
    solver = diagnose_solver()
    if not solver["batch_runnable"]:
        parser.exit(2, f"ngspice cannot run: {solver}\n")
    data = measured_data()
    root.mkdir(parents=True, exist_ok=False)
    inputs = data.loc[data["partition"] != "unused", ["scenario_id", "frequency_Hz", "partition"]].copy()
    impedance = data["observed_resistance_ohm"] + 1j * data["observed_reactance_ohm"]
    current = (1 / impedance).loc[inputs.index].to_numpy(complex)
    inputs.assign(voltage_re_V=1.0, voltage_im_V=0.0, current_re_A=current.real, current_im_A=current.imag).to_csv(
        root / "observations.csv", index=False
    )
    series = run_model(root, data, "series_loss")
    parallel = run_model(root, data, "parallel_loss")
    passed = (
        series["status"] == "identified"
        and parallel["status"] == "not_identified"
        and not parallel["holdout_passed"]
        and parallel["holdout_rmse_ohm"] > 2 * series["holdout_rmse_ohm"]
    )
    data.to_csv(root / "impedance_comparison.csv", index=False)
    plot_comparison(data, root / "impedance_comparison.png")
    write_json(
        root / "benchmark_result.json",
        {
            "passed": passed,
            "dataset": {
                "doi": SOURCE_DOI,
                "sha256": SOURCE_SHA256,
                "partitions": data["partition"].value_counts().to_dict(),
            },
            "solver": solver,
            "series_loss": series,
            "parallel_loss": parallel,
            "scope": "withheld frequencies from one acquisition; terminal model comparison only",
        },
    )
    print(f"{'PASS' if passed else 'FAIL'} {root / 'benchmark_result.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
