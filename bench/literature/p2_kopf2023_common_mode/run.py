"""Replay calibrated common-mode impedance at the published VNA frequencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcd.analysis import input_impedance  # noqa: E402
from pcd.api import load_case, run_case_study  # noqa: E402
from pcd.records import load_frequency_response  # noqa: E402
from pcd.results import read_best_candidate, selected_evaluation  # noqa: E402
from pcd.validation import validate_case  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCES = {
    "phantom_side": (
        "scenario1_phantomside_30k_3G_V2.s1p",
        "74a0c1445afbbe373aa781a970af63e8e00b9c5a4f0c0cd7f20270fd7425d714",
    ),
    "pulser_side": (
        "SCENARIO1_pulserside_30k_3G.s1p",
        "80c5764492c563f4ff6c13dc06f5ec108f0bd60991084d07c292ec27b50d59f0",
    ),
}
ANCHORS_MHZ = (3.0, 5.0, 8.0, 13.56, 20.0, 27.12)
REFERENCE_OHM = 50.0
ALTERNATE_OHM = 75.0


def read_source(path: Path, expected_sha: str) -> pd.DataFrame:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"source checksum differs from the archived version: {path.name}")
    lines = raw.decode("ascii").splitlines()
    options = [line.lower().split() for line in lines if line.startswith("#")]
    if options != [["#", "hz", "s", "db", "r", "50"]]:
        raise ValueError(f"expected one-port S11 in Hz, dB/degree at 50 ohms: {path.name}")
    data = [line for line in lines if line.strip() and not line.startswith(("!", "#"))]
    hz, db, degrees = np.loadtxt(data, unpack=True)
    if not np.all(np.isfinite([hz, db, degrees])) or not np.all(np.diff(hz) > 0):
        raise ValueError(f"source samples must be finite and in increasing frequency order: {path.name}")
    gamma = 10 ** (db / 20) * np.exp(1j * np.deg2rad(degrees))
    impedance = REFERENCE_OHM * (1 + gamma) / (1 - gamma)
    # Directly transform the measured scattering coefficient at the same plane.
    offset = (REFERENCE_OHM - ALTERNATE_OHM) / (REFERENCE_OHM + ALTERNATE_OHM)
    gamma_75 = (gamma + offset) / (1 + offset * gamma)
    return pd.DataFrame({"frequency_Hz": hz, "impedance": impedance, "gamma_50": gamma, "gamma_75": gamma_75})


def replay_case(
    plane: str, points: list[tuple[float, complex, complex, complex]], root: Path, reference_ohm: float
) -> dict[str, Any]:
    scenarios = [
        {
            "id": f"point_{index + 1}",
            "values": {
                "frequency_Hz": frequency,
                "R_load": impedance.real,
                "X_load": impedance.imag,
            },
        }
        for index, (frequency, impedance, _, _) in enumerate(points)
    ]
    defaults = scenarios[0]["values"]
    case_data = {
        "schema": "case_yaml.v1",
        "case_id": plane,
        "variables": {name: {"role": "operating", "default": value} for name, value in defaults.items()},
        "source": {"type": "sine_voltage", "amplitude_V": 1, "frequency_Hz": "$frequency_Hz"},
        "circuit": {"output_node": "src", "components": []},
        "load": {
            "name": "impedance_point",
            "resistance_ohm": "$R_load",
            "reactance_ohm": "$X_load",
            "model_frequency_Hz": "$frequency_Hz",
            "reference_plane": f"scenario_1_{plane}_common_mode_sma",
            "characterization": {"source": "https://doi.org/10.5281/zenodo.10016478", "method": "VNA S11"},
        },
        "measurement": {"reference_impedance_ohm": reference_ohm},
        "solver": {"ac": {"sweep": "lin", "points": 1, "start_Hz": "$frequency_Hz", "stop_Hz": "$frequency_Hz"}},
        "target": {"objective": "impedance_match"},
        "study": {"scenarios": scenarios},
        "optimizer": {"name": "grid", "seed": 0},
        "run": {"trials": 1},
    }
    case_path = root / f"{plane}_{reference_ohm:g}ohm.yaml"
    case_path.write_text(yaml.safe_dump(case_data, sort_keys=False), encoding="utf-8")
    case = load_case(case_path)
    report = validate_case(case)
    if not report.ok:
        raise ValueError(report.format_text())
    return run_case_study(case, root / "studies" / f"{reference_ohm:g}ohm")


def read_point(
    study_root: Path, scenario: dict[str, Any], frequency: float, reference_ohm: float
) -> tuple[complex, complex, float, float]:
    evaluation = selected_evaluation(scenario)
    raw = evaluation["raw"]
    if raw["status"] != "ok" or evaluation["from_cache"] is not False:
        raise ValueError(f"selected VNA point lacks a fresh successful solve: {study_root}")
    manifest = study_root / raw["artifacts"]["manifest"]
    ac = load_frequency_response(manifest)
    if len(ac) != 1 or not np.isclose(ac.iloc[0]["frequency_Hz"], frequency, rtol=1e-12, atol=0):
        raise ValueError(f"simulation did not evaluate the source frequency: {manifest}")
    row = input_impedance(ac, reference_ohm).iloc[0]
    z = complex(row["resistance_ohm"], row["reactance_ohm"])
    gamma = complex(row["reflection_re"], row["reflection_im"])
    metrics = evaluation["metrics"]
    reported = float(metrics["reflection_magnitude"])
    reported_reference = float(metrics["reference_impedance_ohm"])
    if not np.isfinite([z.real, z.imag, gamma.real, gamma.imag, reported, reported_reference]).all():
        raise ValueError(f"VNA replay contains a non-finite point: {manifest}")
    return z, gamma, reported, reported_reference


def compare(plane: str, source: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    hz = source["frequency_Hz"].to_numpy(float)
    indices = [int(np.argmin(np.abs(hz - target * 1e6))) for target in ANCHORS_MHZ]
    if len(set(indices)) != len(ANCHORS_MHZ):
        raise ValueError(f"source must provide a distinct sample for each replay anchor: {plane}")
    selected = source.iloc[indices]
    points = [
        (float(frequency), complex(impedance), complex(gamma_50), complex(gamma_75))
        for frequency, impedance, gamma_50, gamma_75 in zip(
            selected["frequency_Hz"].to_numpy(),
            selected["impedance"].to_numpy(),
            selected["gamma_50"].to_numpy(),
            selected["gamma_75"].to_numpy(),
            strict=True,
        )
    ]
    if any(impedance.real <= 0 for _, impedance, _, _ in points):
        raise ValueError(f"selected points must support a passive series equivalent: {plane}")
    replays = []
    for reference in (REFERENCE_OHM, ALTERNATE_OHM):
        study = replay_case(plane, points, root, reference)
        study_root = Path(study["run_root"])
        if study["verification"]["status"] != "verified":
            raise ValueError(f"VNA study did not verify its selected points: {study_root}")
        scenarios = read_best_candidate(study_root)["scenarios"]
        by_id = {scenario["scenario"]["scenario_id"]: scenario for scenario in scenarios}
        if len(scenarios) != len(points) or set(by_id) != {f"point_{i + 1}" for i in range(len(points))}:
            raise ValueError(f"VNA study does not cover the expected points: {study_root}")
        replays.append((reference, study, study_root, by_id))
    rows = []
    for index, (frequency, impedance, gamma_50, gamma_75) in enumerate(points, start=1):
        primary, alternate = (
            read_point(study_root, by_id[f"point_{index}"], frequency, reference)
            for reference, _, study_root, by_id in replays
        )
        z, g50, reported_50, reference_50 = primary
        z75, g75, reported_75, reference_75 = alternate
        reflection = {
            "measured_gamma_50": gamma_50,
            "simulated_gamma_50": g50,
            "renormalized_gamma_75": gamma_75,
            "simulated_gamma_75": g75,
        }
        rows.append(
            {
                "plane": plane,
                "frequency_Hz": frequency,
                "measured_R_ohm": impedance.real,
                "measured_X_ohm": impedance.imag,
                "simulated_R_ohm": z.real,
                "simulated_X_ohm": z.imag,
                **{f"{name}_real": value.real for name, value in reflection.items()},
                **{f"{name}_imag": value.imag for name, value in reflection.items()},
                **{f"{name}_magnitude": abs(value) for name, value in reflection.items()},
                "impedance_error_ohm": max(abs(z - impedance), abs(z75 - impedance)),
                "gamma_50_error": abs(g50 - gamma_50),
                "gamma_75_error": abs(g75 - gamma_75),
                "reported_reference_50_ohm": reference_50,
                "reported_reference_75_ohm": reference_75,
                "reported_gamma_50_magnitude": reported_50,
                "reported_gamma_75_magnitude": reported_75,
                "wrong_reference_gamma_error": abs(gamma_75 - gamma_50),
            }
        )
    evidence = pd.DataFrame(rows)
    return evidence, {
        "study_results_by_reference_ohm": {
            f"{reference:g}": (study_root / "study_result.json").relative_to(root).as_posix()
            for reference, _, study_root, _ in replays
        },
        "source_samples": len(source),
        "samples_outside_passive_disk": int(np.count_nonzero(np.abs(source["gamma_50"]) > 1)),
        "solver_evaluations": sum(
            study["n_evaluations"] + study["verification"]["n_evaluations"] for _, study, _, _ in replays
        ),
        "max_impedance_error_ohm": float(evidence["impedance_error_ohm"].max()),
        "max_gamma_50_error": float(evidence["gamma_50_error"].max()),
        "max_gamma_75_error": float(evidence["gamma_75_error"].max()),
        "max_wrong_reference_gamma_error": float(evidence["wrong_reference_gamma_error"].max()),
    }


def plot(sources: dict[str, pd.DataFrame], evidence: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row", layout="constrained")
    for column, (plane, source) in enumerate(sources.items()):
        band = source[source["frequency_Hz"].between(1e6, 30e6)]
        points = evidence[evidence["plane"] == plane]
        top, bottom = axes[:, column]
        hz = band["frequency_Hz"] / 1e6
        for label, suffix, values, color in (
            ("R", "R", np.real(band["impedance"]), "tab:blue"),
            ("X", "X", np.imag(band["impedance"]), "tab:orange"),
        ):
            top.plot(hz, values, color=color, label=f"Measured {label}")
            top.scatter(points["frequency_Hz"] / 1e6, points[f"simulated_{suffix}_ohm"], color=color, s=17)
        for reference, label, color in ((50, "VNA 50 Ω", "tab:blue"), (75, "Same plane, 75 Ω", "tab:green")):
            bottom.plot(hz, np.abs(band[f"gamma_{reference}"]), color=color, label=label)
            bottom.scatter(
                points["frequency_Hz"] / 1e6, points[f"simulated_gamma_{reference}_magnitude"], color=color, s=17
            )
        outside = np.abs(band["gamma_50"]) > 1
        if outside.any():
            bottom.scatter(
                hz[outside],
                np.abs(band.loc[outside, "gamma_50"]),
                marker="x",
                color="tab:red",
                label="Outside passive disk",
                zorder=4,
            )
        top.set(title=plane.replace("_", " ").title(), ylabel="Impedance [Ω]", yscale="symlog")
        bottom.set(xlabel="Frequency [MHz]", ylabel="Reflection magnitude")
        bottom.axhline(1, color="0.6", linestyle=":", linewidth=0.8)
        for axis in (top, bottom):
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=8)
    figure.suptitle("Published common-mode fixtures · markers: ngspice point replay")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    root = parser.parse_args().run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = {plane: read_source(HERE / "source" / file, digest) for plane, (file, digest) in SOURCES.items()}
    results = {plane: compare(plane, data, root) for plane, data in sources.items()}
    evidence = pd.concat([table for table, _ in results.values()], ignore_index=True)
    evidence.to_csv(root / "comparison.csv", index=False)
    plot(sources, evidence, root / "comparison.png")
    checks = {
        "replayed_complex_impedance": bool(evidence["impedance_error_ohm"].max() < 1e-3),
        "replayed_complex_reflection_at_source_reference": bool(evidence["gamma_50_error"].max() < 1e-5),
        "replayed_complex_reflection_at_alternate_reference": bool(evidence["gamma_75_error"].max() < 1e-5),
        "case_reference_reaches_reported_measurement": all(
            bool((evidence[f"reported_reference_{reference:g}_ohm"] == reference).all())
            and bool(
                np.allclose(
                    evidence[f"reported_gamma_{reference:g}_magnitude"],
                    evidence[f"{source}_gamma_{reference:g}_magnitude"],
                    rtol=0,
                    atol=1e-5,
                )
            )
            for reference, source in ((REFERENCE_OHM, "measured"), (ALTERNATE_OHM, "renormalized"))
        ),
        "unchanged_reflection_fails_reference_change": bool(evidence["wrong_reference_gamma_error"].max() > 0.1),
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "source": "https://doi.org/10.5281/zenodo.10016478",
        "reference_impedance_ohm": REFERENCE_OHM,
        "alternate_reference_impedance_ohm": ALTERNATE_OHM,
        "planes": {
            plane: {"file": SOURCES[plane][0], "sha256": SOURCES[plane][1], **result}
            for plane, (_, result) in results.items()
        },
        "comparison": "comparison.csv",
        "figure": "comparison.png",
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (root / "benchmark_result.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
