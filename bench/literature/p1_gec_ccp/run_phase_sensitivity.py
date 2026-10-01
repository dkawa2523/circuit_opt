"""Run a common-mode phase sensitivity on the Hargis 66 Pa CCP window."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_suite import run_case  # noqa: E402
from pcd.artifacts import write_json  # noqa: E402

CENTRAL_PATH = HERE / "derived_impedance_66pa.csv"
EXPECTATIONS = yaml.safe_load((HERE / "phase_sensitivity_expectations.yaml").read_text(encoding="utf-8"))
CASE_PATHS = (
    HERE / "match_full_central.yaml",
    HERE / "match_full_phase_minus6.yaml",
    HERE / "match_full_phase_plus6.yaml",
)
SHIFT_DATASETS = {
    "phase_minus6": (-6.0, HERE / "phase_systematic_minus6.csv"),
    "phase_plus6": (6.0, HERE / "phase_systematic_plus6.csv"),
}
REQUIRED_FREQUENCY_HZ = 13_560_000.0
REQUIRED_REFERENCE_PLANE = "powered_electrode_surface"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _complex(row: dict[str, str]) -> complex:
    return complex(float(row["resistance_ohm"]), float(row["reactance_ohm"]))


def _angle_deg(value: complex) -> float:
    return math.degrees(math.atan2(value.imag, value.real))


def _load_point(row: dict[str, str], *, phase_shift_deg: float) -> dict[str, Any]:
    impedance = _complex(row)
    return {
        "scenario_id": row["scenario_id"],
        "resistance_ohm": impedance.real,
        "reactance_ohm": impedance.imag,
        "magnitude_ohm": abs(impedance),
        "phase_deg": _angle_deg(impedance),
        "phase_shift_deg": phase_shift_deg,
    }


def _validate_datasets() -> tuple[dict[str, Any], dict[str, dict[str, dict[str, str]]]]:
    central_rows = _read_rows(CENTRAL_PATH)
    central = {row["scenario_id"]: row for row in central_rows}
    datasets: dict[str, dict[str, dict[str, str]]] = {"baseline": central}
    details: dict[str, Any] = {}
    all_checks: dict[str, bool] = {
        "central_row_count": len(central_rows) == 8,
        "unique_central_ids": len(central) == len(central_rows),
        "central_frequency": all(float(row["frequency_Hz"]) == REQUIRED_FREQUENCY_HZ for row in central_rows),
        "central_reference_plane": all(row["reference_plane"] == REQUIRED_REFERENCE_PLANE for row in central_rows),
    }

    for dataset_id, (expected_shift, path) in SHIFT_DATASETS.items():
        rows = _read_rows(path)
        shifted = {row["central_scenario_id"]: row for row in rows}
        datasets[dataset_id] = shifted
        magnitude_errors: list[float] = []
        phase_errors: list[float] = []
        vector_errors: list[float] = []
        for central_id, row in shifted.items():
            if central_id not in central:
                continue
            central_z = _complex(central[central_id])
            shifted_z = _complex(row)
            target_phase = _angle_deg(central_z) + expected_shift
            target_z = abs(central_z) * complex(
                math.cos(math.radians(target_phase)),
                math.sin(math.radians(target_phase)),
            )
            magnitude_errors.append(abs(abs(shifted_z) - abs(central_z)) / abs(central_z))
            phase_errors.append(abs((_angle_deg(shifted_z) - _angle_deg(central_z)) - expected_shift))
            vector_errors.append(abs(shifted_z - target_z))

        checks = {
            "row_count": len(rows) == 8,
            "unique_central_ids": len(shifted) == len(rows),
            "exact_central_pairing": set(shifted) == set(central),
            "single_common_phase_shift": all(float(row["phase_shift_deg"]) == expected_shift for row in rows),
            "frequency": all(float(row["frequency_Hz"]) == REQUIRED_FREQUENCY_HZ for row in rows),
            "reference_plane": all(row["reference_plane"] == REQUIRED_REFERENCE_PLANE for row in rows),
            "magnitude_preserved": bool(magnitude_errors) and max(magnitude_errors) <= 1e-10,
            "phase_shift_replayed": bool(phase_errors) and max(phase_errors) <= 1e-9,
            "vector_replayed": bool(vector_errors) and max(vector_errors) <= 1e-8,
            "passive_capacitive": all(_complex(row).real > 0.0 and _complex(row).imag < 0.0 for row in rows),
        }
        all_checks.update({f"{dataset_id}_{name}": passed for name, passed in checks.items()})
        details[dataset_id] = {
            "expected_common_phase_shift_deg": expected_shift,
            "checks": checks,
            "max_magnitude_relative_error": max(magnitude_errors, default=math.inf),
            "max_phase_shift_error_deg": max(phase_errors, default=math.inf),
            "max_vector_replay_error_ohm": max(vector_errors, default=math.inf),
        }

    return {"passed": all(all_checks.values()), "checks": all_checks, "datasets": details}, datasets


def _normalize_failures(case: dict[str, Any], shifted: dict[str, dict[str, str]] | None) -> set[str]:
    if shifted is None:
        return {str(item) for item in case["infeasible_scenarios"]}
    scenario_to_central = {row["scenario_id"]: central_id for central_id, row in shifted.items()}
    return {scenario_to_central[str(item)] for item in case["infeasible_scenarios"]}


def _selected_by_central(
    case: dict[str, Any],
    shifted: dict[str, dict[str, str]] | None,
) -> dict[str, dict[str, Any]]:
    if shifted is None:
        return {str(item["scenario_id"]): item for item in case["scenarios"]}
    scenario_to_central = {row["scenario_id"]: central_id for central_id, row in shifted.items()}
    return {scenario_to_central[str(item["scenario_id"])]: item for item in case["scenarios"]}


def _summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    scenario_count = len(case["scenarios"])
    feasible_scenario_count = round(float(case["feasible_fraction"]) * scenario_count)
    return {
        "benchmark_id": case["benchmark_id"],
        "passed": case["passed"],
        "feasible": case["feasible"],
        "feasible_scenario_count": feasible_scenario_count,
        "scenario_count": scenario_count,
        "feasible_fraction": case["feasible_fraction"],
        "worst_reflection_magnitude": case["worst_reflection_magnitude"],
        "worst_reflected_power_fraction": case["worst_reflected_power_fraction"],
        "control_margin": case.get("control_margin"),
        "infeasible_scenarios": case["infeasible_scenarios"],
        "n_evaluations": case["n_evaluations"],
    }


def run(run_root: Path, solver: str) -> dict[str, Any]:
    integrity, datasets = _validate_datasets()
    design_root = run_root / "design_runs"
    cases = [run_case(path, design_root, solver, EXPECTATIONS) for path in CASE_PATHS]
    case_by_id = {str(case["benchmark_id"]): case for case in cases}
    baseline_case = case_by_id["P1_CCP_phase_baseline"]
    baseline_selected = _selected_by_central(baseline_case, None)
    baseline_failures = _normalize_failures(baseline_case, None)

    comparisons: dict[str, Any] = {}
    for dataset_id, benchmark_id in (
        ("phase_minus6", "P1_CCP_phase_minus6"),
        ("phase_plus6", "P1_CCP_phase_plus6"),
    ):
        case = case_by_id[benchmark_id]
        selected = _selected_by_central(case, datasets[dataset_id])
        failures = _normalize_failures(case, datasets[dataset_id])
        comparisons[dataset_id] = {
            "phase_shift_deg": SHIFT_DATASETS[dataset_id][0],
            "classification_changed_from_baseline": bool(case["feasible"]) != bool(baseline_case["feasible"]),
            "feasible_fraction_delta": float(case["feasible_fraction"]) - float(baseline_case["feasible_fraction"]),
            "worst_reflection_magnitude_delta": float(case["worst_reflection_magnitude"])
            - float(baseline_case["worst_reflection_magnitude"]),
            "lost_feasibility": sorted(failures - baseline_failures),
            "gained_feasibility": sorted(baseline_failures - failures),
            "per_central_scenario": {
                central_id: {
                    "baseline_feasible": baseline_selected[central_id]["feasible"],
                    "shifted_feasible": selected[central_id]["feasible"],
                    "baseline_reflection_magnitude": baseline_selected[central_id]["reflection_magnitude"],
                    "shifted_reflection_magnitude": selected[central_id]["reflection_magnitude"],
                    "shifted_selected_control": selected[central_id]["control"],
                }
                for central_id in sorted(baseline_selected)
            },
        }

    load_points = []
    for central_id in sorted(datasets["baseline"]):
        load_points.append(
            {
                "central_scenario_id": central_id,
                "baseline": _load_point(datasets["baseline"][central_id], phase_shift_deg=0.0),
                "phase_minus6": _load_point(datasets["phase_minus6"][central_id], phase_shift_deg=-6.0),
                "phase_plus6": _load_point(datasets["phase_plus6"][central_id], phase_shift_deg=6.0),
            }
        )

    design_regression_passed = all(bool(case["passed"]) for case in cases)
    passed = bool(integrity["passed"]) and design_regression_passed
    payload: dict[str, Any] = {
        "schema": "literature_phase_sensitivity.v2",
        "benchmark": "p1_gec_ccp_common_mode_phase_sensitivity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "source": {
            "hargis_doi": "https://doi.org/10.1063/1.1144770",
            "sobolewski_doi": "https://doi.org/10.6028/jres.100.026",
            "hargis_scope": "66 Pa central rows from Tables III and IV",
            "sobolewski_finding": (
                "omitting cell and shunt resistive parasitics produced a systematic phase error "
                "typically 5 degrees in the NIST cell"
            ),
            "benchmark_choice": (
                "symmetric +/-6 degree common-mode bracketing is a conservative engineering stress, "
                "not a paper-reported confidence interval"
            ),
        },
        "scope": {
            "drive_frequency_Hz": REQUIRED_FREQUENCY_HZ,
            "pressure_Pa": 66,
            "reference_plane": REQUIRED_REFERENCE_PLANE,
            "resonance_group_labels": "24 and 34 MHz are empty-cell resonance groups, not drive frequencies",
            "power": (
                "not used as an independent golden; Hargis reported power contains up to five harmonics, "
                "and these phase-shifted loads are sensitivity constructions"
            ),
            "acceptance": "10% reflected incident power is a platform criterion, not a paper criterion",
        },
        "sensitivity_construction": {
            "passed": bool(integrity["passed"]),
            "integrity": integrity,
            "load_points": load_points,
            "claim": "two alternative common-mode model-form shifts; not measured scenarios or confidence limits",
        },
        "platform_design_challenge": {
            "regression_passed": design_regression_passed,
            "outcomes": [_summarize_case(case) for case in cases],
            "comparisons_to_baseline": comparisons,
        },
        "n_evaluations": sum(int(case["n_evaluations"]) for case in cases),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "evaluation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/gec_ccp_phase_sensitivity"))
    parser.add_argument("--solver", default="ngspice_cli")
    args = parser.parse_args()
    report = run(args.run_root.resolve(), str(args.solver))
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "n_evaluations": report["n_evaluations"],
                "design_cases": report["platform_design_challenge"]["outcomes"],
                "comparisons_to_baseline": report["platform_design_challenge"]["comparisons_to_baseline"],
                "result": str(args.run_root.resolve() / "evaluation.json"),
            },
            indent=2,
        )
    )
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
