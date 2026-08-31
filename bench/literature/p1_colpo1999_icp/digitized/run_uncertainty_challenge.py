"""Expand and evaluate the Colpo 1999 digitization-only uncertainty window."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.literature.p1_colpo1999_icp.digitized.run import run as run_central  # noqa: E402
from pcd.artifacts import write_json  # noqa: E402
from pcd.case import load_case  # noqa: E402
from pcd.study import run_case_study  # noqa: E402

CORNER_COLUMNS = (
    "scenario_id",
    "frequency_Hz",
    "resistance_ohm",
    "reactance_ohm",
    "weight",
    "parent_scenario_id",
    "argon_pressure_mTorr",
    "reported_rf_power_W",
    "resistance_bound",
    "reactance_bound",
    "resistance_offset_ohm",
    "reactance_offset_ohm",
    "reference_plane",
    "source_doi",
    "evidence_class",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def _one_decimal(value: Decimal) -> str:
    return f"{value:.1f}"


def _derive_corners(
    central_rows: list[dict[str, str]], resistance_offsets: list[float], reactance_offsets: list[float]
) -> list[dict[str, str]]:
    bounds = {-1: "lo", 1: "hi"}
    derived: list[dict[str, str]] = []
    for row in central_rows:
        r_center = Decimal(row["resistance_ohm"])
        x_center = Decimal(row["reactance_ohm"])
        for r_offset_float in resistance_offsets:
            for x_offset_float in reactance_offsets:
                r_offset = Decimal(str(r_offset_float))
                x_offset = Decimal(str(x_offset_float))
                r_sign = 1 if r_offset > 0 else -1
                x_sign = 1 if x_offset > 0 else -1
                r_bound = f"r{bounds[r_sign]}"
                x_bound = f"x{bounds[x_sign]}"
                derived.append(
                    {
                        "scenario_id": f"{row['scenario_id']}__{r_bound}_{x_bound}",
                        "frequency_Hz": row["frequency_Hz"],
                        "resistance_ohm": _one_decimal(r_center + r_offset),
                        "reactance_ohm": _one_decimal(x_center + x_offset),
                        "weight": "1",
                        "parent_scenario_id": row["scenario_id"],
                        "argon_pressure_mTorr": row["argon_pressure_mTorr"],
                        "reported_rf_power_W": row["reported_rf_power_W"],
                        "resistance_bound": r_bound,
                        "reactance_bound": x_bound,
                        "resistance_offset_ohm": _one_decimal(r_offset),
                        "reactance_offset_ohm": _one_decimal(x_offset),
                        "reference_plane": row["reference_plane"],
                        "source_doi": row["source_doi"],
                        "evidence_class": "rectangular_digitization_only_corner",
                    }
                )
    return derived


def _selected_scenarios(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in candidate["scenarios"]:
        evaluation = item["selected"]
        constraints = evaluation.get("constraints", []) or []
        violated = sorted(str(row["name"]) for row in constraints if not bool(row["satisfied"]))
        feasible = evaluation["raw"]["status"] == "ok" and not violated
        gamma = float(evaluation["metrics"]["reflection_magnitude"])
        selected.append(
            {
                "scenario_id": str(item["scenario"]["scenario_id"]),
                "feasible": feasible,
                "reflection_magnitude": gamma,
                "reflected_power_fraction": gamma**2,
                "control": dict(evaluation["request"]["control"]["values"]),
                "violated_constraints": violated,
            }
        )
    return selected


def _decision_stability(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Group epistemic reading corners by their physical parent condition."""

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        parent_id = str(scenario["scenario_id"]).rsplit("__", 1)[0]
        by_parent.setdefault(parent_id, []).append(scenario)
    robust_parents = sorted(
        parent_id for parent_id, rows in by_parent.items() if all(bool(row["feasible"]) for row in rows)
    )
    failed_parents = sorted(
        parent_id for parent_id, rows in by_parent.items() if not any(bool(row["feasible"]) for row in rows)
    )
    mixed_parents = sorted(set(by_parent) - set(robust_parents) - set(failed_parents))
    return {
        "parent_condition_count": len(by_parent),
        "robust_parent_count": len(robust_parents),
        "mixed_parent_count": len(mixed_parents),
        "failed_parent_count": len(failed_parents),
        "robust_parent_conditions": robust_parents,
        "mixed_parent_conditions": mixed_parents,
        "failed_parent_conditions": failed_parents,
        "interpretation": "four digitization corners per physical pressure-power condition",
    }


def _run_design_case(path: Path, run_root: Path, solver: str) -> dict[str, Any]:
    case = load_case(path)
    configured_trials = int((case.data.get("run", {}) or {}).get("trials", 1))
    configured_optimizer = str((case.data.get("optimizer", {}) or {}).get("name", "random"))
    study = run_case_study(
        case,
        n_trials=configured_trials,
        run_root=run_root,
        optimizer_name=configured_optimizer,
        solver_override=solver,
        seed=0,
    )
    candidate_dir = Path(study["run_root"]) / "candidates"
    candidate_path = candidate_dir / f"{study['best']['candidate']['candidate_id']}.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    scenarios = _selected_scenarios(candidate)
    worst = max(float(item["reflection_magnitude"]) for item in scenarios)
    feasible_count = sum(bool(item["feasible"]) for item in scenarios)
    return {
        "case_id": case.case_id,
        "case_path": str(path.resolve()),
        "feasible": feasible_count == len(scenarios) and int(study["n_failed_evaluations"]) == 0,
        "feasible_scenario_count": feasible_count,
        "scenario_count": len(scenarios),
        "feasible_fraction": feasible_count / len(scenarios),
        "worst_reflection_magnitude": worst,
        "worst_reflected_power_fraction": worst**2,
        "control_margin": candidate.get("control_margin"),
        "n_candidates": int(study["n_candidates"]),
        "n_evaluations": int(study["n_evaluations"]),
        "n_failed_evaluations": int(study["n_failed_evaluations"]),
        "infeasible_scenarios": [item["scenario_id"] for item in scenarios if not item["feasible"]],
        "decision_stability": _decision_stability(scenarios),
        "scenarios": scenarios,
    }


def _pi_reflection(load: complex, frequency: float, c1: float, inductance: float, c2: float) -> float:
    omega = 2.0 * math.pi * frequency
    output_parallel = 1.0 / (1.0 / load + 1j * omega * c2)
    series_path = output_parallel + 1j * omega * inductance
    input_impedance = 1.0 / (1.0 / series_path + 1j * omega * c1)
    return abs((input_impedance - 50.0) / (input_impedance + 50.0))


def _render_report(payload: dict[str, Any]) -> str:
    fixed, bounded = payload["platform_design_challenge"]["outcomes"]
    status = "PASS" if payload["benchmark_integrity_passed"] else "FAIL"
    return f"""# Colpo 1999 digitization-corner challenge

Overall: **{status}**

- Central points: {payload["central_point_count"]}
- Explicit rectangular corners: {payload["corner_point_count"]}
- Central one-port replay maximum relative error: {payload["source_fidelity"]["central_replay"]["maximum_spice_replay_relative_error"]:.6g}

| design fixture | robust parent conditions | mixed parent conditions | feasible corners | worst `|Gamma|` |
|---|---:|---:|---:|---:|
| fixed, middle-condition setting | {fixed["decision_stability"]["robust_parent_count"]}/15 | {fixed["decision_stability"]["mixed_parent_count"]}/15 | {fixed["feasible_scenario_count"]}/{fixed["scenario_count"]} | {fixed["worst_reflection_magnitude"]:.4f} |
| bounded, 16 control states | {bounded["decision_stability"]["robust_parent_count"]}/15 | {bounded["decision_stability"]["mixed_parent_count"]}/15 | {bounded["feasible_scenario_count"]}/{bounded["scenario_count"]} | {bounded["worst_reflection_magnitude"]:.4f} |

The +/-6 ohm resistance and +/-7 ohm reactance widths cover digitization and
marker-centre ambiguity only. They are not RFZ60 measurement uncertainty or a
statistical confidence interval. The design fixtures and the 10% reflected-power
criterion belong to the platform benchmark; the paper does not publish matcher
settings for reproduction. The ordinary `impedance_table` path does not expand
uncertainty columns: this benchmark materializes every corner explicitly.
The 60 corners are outcomes within 15 parent pressure-power conditions, not 60
independent operating observations.
"""


def run(run_root: Path, solver: str) -> dict[str, Any]:
    manifest = yaml.safe_load((HERE / "uncertainty_challenge.yaml").read_text(encoding="utf-8"))
    central_rows = _read_csv(HERE / manifest["source"]["central_table"])
    corner_path = HERE / manifest["corner_expansion"]["derived_table"]
    corner_rows = _read_csv(corner_path)
    expansion = manifest["corner_expansion"]
    expected_rows = _derive_corners(
        central_rows,
        [float(value) for value in expansion["resistance_offsets_ohm"]],
        [float(value) for value in expansion["reactance_offsets_ohm"]],
    )
    source_checks: list[dict[str, Any]] = []

    central_ids = [row["scenario_id"] for row in central_rows]
    corner_ids = [row["scenario_id"] for row in corner_rows]
    parent_counts = Counter(row["parent_scenario_id"] for row in corner_rows)
    source_checks.append(
        _check(
            "central_and_corner_grain",
            len(central_rows) == int(expansion["expected_central_points"])
            and len(central_ids) == len(set(central_ids))
            and len(corner_rows) == int(expansion["expected_corner_points"])
            and len(corner_ids) == len(set(corner_ids))
            and set(parent_counts) == set(central_ids)
            and set(parent_counts.values()) == {int(expansion["corners_per_central_point"])},
            central_points=len(central_rows),
            corner_points=len(corner_rows),
        )
    )
    exact_rows = len(corner_rows) == len(expected_rows) and all(
        tuple(actual.get(column, "") for column in CORNER_COLUMNS)
        == tuple(expected[column] for column in CORNER_COLUMNS)
        for actual, expected in zip(corner_rows, expected_rows, strict=True)
    )
    source_checks.append(_check("committed_table_matches_declared_expansion", exact_rows))
    source_checks.append(
        _check(
            "provenance_reference_plane_and_scope_preserved",
            all(
                row["source_doi"] == manifest["source"]["doi"]
                and row["reference_plane"] == manifest["source"]["reference_plane"]
                and float(row["frequency_Hz"]) == float(manifest["source"]["frequency_Hz"])
                and row["evidence_class"] == "rectangular_digitization_only_corner"
                for row in corner_rows
            ),
        )
    )
    source_checks.append(
        _check(
            "all_derived_corners_are_passive_and_inductive",
            all(float(row["resistance_ohm"]) > 0.0 and float(row["reactance_ohm"]) > 0.0 for row in corner_rows),
            R_range_ohm=[
                min(float(row["resistance_ohm"]) for row in corner_rows),
                max(float(row["resistance_ohm"]) for row in corner_rows),
            ],
            X_range_ohm=[
                min(float(row["reactance_ohm"]) for row in corner_rows),
                max(float(row["reactance_ohm"]) for row in corner_rows),
            ],
        )
    )
    boundaries = manifest["qualification"]["boundaries"]
    source_checks.append(
        _check(
            "uncertainty_and_claim_boundaries_are_explicit",
            not bool(boundaries["impedance_table_automatically_propagates_uncertainty_columns"])
            and not bool(boundaries["paper_matcher_reproduction_claimed"])
            and not bool(boundaries["less_than_one_percent_paper_reflection_reproduced"])
            and not bool(boundaries["production_hardware_qualified"]),
        )
    )

    central_replay = run_central(run_root / "central_replay")
    source_checks.append(_check("central_point_spice_replay", bool(central_replay["passed"])))

    fixture = manifest["design_fixture"]
    representative = next(
        row for row in central_rows if row["scenario_id"] == fixture["fixed_network"]["representative_central_scenario"]
    )
    representative_gamma = _pi_reflection(
        complex(float(representative["resistance_ohm"]), float(representative["reactance_ohm"])),
        float(manifest["source"]["frequency_Hz"]),
        float(fixture["fixed_network"]["C1_F"]),
        float(fixture["fixed_inductor_H"]),
        float(fixture["fixed_network"]["C2_F"]),
    )
    design_checks: list[dict[str, Any]] = []
    design_checks.append(
        _check(
            "fixed_fixture_is_near_matched_at_declared_central_condition",
            representative_gamma <= float(fixture["fixed_network"]["representative_reflection_magnitude_max"]),
            scenario_id=representative["scenario_id"],
            analytic_reflection_magnitude=representative_gamma,
        )
    )

    fixed = _run_design_case(HERE / "match_fixed_uncertainty_corners.yaml", run_root / "design", solver)
    bounded = _run_design_case(HERE / "match_bounded_uncertainty_corners.yaml", run_root / "design", solver)
    expected_design = manifest["qualification"]["expected_design_classification"]
    fixed_expected = expected_design["fixed_network"]
    bounded_expected = expected_design["bounded_tuner"]
    design_checks.append(
        _check(
            "fixed_network_negative_control",
            fixed["feasible"] is bool(fixed_expected["feasible"])
            and float(fixed_expected["feasible_fraction"][0])
            <= float(fixed["feasible_fraction"])
            <= float(fixed_expected["feasible_fraction"][1])
            and float(fixed_expected["worst_reflection_magnitude"][0])
            <= float(fixed["worst_reflection_magnitude"])
            <= float(fixed_expected["worst_reflection_magnitude"][1])
            and int(fixed["n_failed_evaluations"]) == 0,
            feasible_fraction=fixed["feasible_fraction"],
            worst_reflection_magnitude=fixed["worst_reflection_magnitude"],
        )
    )
    design_checks.append(
        _check(
            "bounded_tuner_positive_control",
            bounded["feasible"] is bool(bounded_expected["feasible"])
            and float(bounded["feasible_fraction"]) == 1.0
            and float(bounded["worst_reflection_magnitude"])
            <= float(bounded_expected["worst_reflection_magnitude_max"])
            and int(bounded["n_failed_evaluations"]) == 0,
            feasible_fraction=bounded["feasible_fraction"],
            worst_reflection_magnitude=bounded["worst_reflection_magnitude"],
        )
    )

    source_passed = all(check["passed"] for check in source_checks)
    design_regression_passed = all(check["passed"] for check in design_checks)
    passed = source_passed and design_regression_passed
    payload: dict[str, Any] = {
        "schema": "literature_uncertainty_challenge_result.v2",
        "benchmark_id": manifest["benchmark_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "benchmark_integrity_passed": passed,
        "source": manifest["source"],
        "central_point_count": len(central_rows),
        "corner_point_count": len(corner_rows),
        "source_fidelity": {
            "passed": source_passed,
            "corner_generation": expansion,
            "central_replay": central_replay,
            "checks": source_checks,
            "claim": "central transcription replay and exact digitization-corner derivation",
        },
        "platform_design_challenge": {
            "regression_passed": design_regression_passed,
            "design_fixture": fixture,
            "outcomes": [fixed, bounded],
            "checks": design_checks,
            "claim": "decision stability over four reading corners for each of 15 parent conditions",
        },
        "limitations": boundaries,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "evaluation.json", payload)
    (run_root / "REPORT.md").write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/colpo1999_uncertainty_challenge"))
    parser.add_argument("--solver", default="ngspice_cli")
    parser.add_argument("--output", type=Path, help="optional additional JSON result path")
    args = parser.parse_args()
    payload = run(args.run_root.resolve(), str(args.solver))
    if args.output:
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "benchmark_id": payload["benchmark_id"],
                "passed": payload["passed"],
                "central_points": payload["central_point_count"],
                "corner_points": payload["corner_point_count"],
                "central_replay_max_error": payload["source_fidelity"]["central_replay"][
                    "maximum_spice_replay_relative_error"
                ],
                "designs": [
                    {
                        "case_id": item["case_id"],
                        "feasible": item["feasible"],
                        "parent_stability": (
                            f"{item['decision_stability']['robust_parent_count']}/"
                            f"{item['decision_stability']['parent_condition_count']} robust; "
                            f"{item['decision_stability']['mixed_parent_count']} mixed"
                        ),
                        "corner_outcomes": f"{item['feasible_scenario_count']}/{item['scenario_count']}",
                        "worst_reflection_magnitude": item["worst_reflection_magnitude"],
                        "n_evaluations": item["n_evaluations"],
                    }
                    for item in payload["platform_design_challenge"]["outcomes"]
                ],
                "result": str(args.run_root.resolve() / "evaluation.json"),
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
