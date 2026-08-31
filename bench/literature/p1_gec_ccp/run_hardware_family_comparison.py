"""Compare fixed matcher candidates without flattening unlike evidence families."""

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

from pcd.artifacts import write_json  # noqa: E402
from pcd.case import load_case  # noqa: E402
from pcd.study import run_case_study  # noqa: E402

SPEC_PATH = HERE / "hardware_family_spec.yaml"
EXPECTATIONS_PATH = HERE / "hardware_family_expectations.yaml"
REQUIRED_COLUMNS = {"scenario_id", "frequency_Hz", "resistance_ohm", "reactance_ohm"}
REQUIRED_FREQUENCY_HZ = 13_560_000.0
REQUIRED_REFERENCE_PLANE = "powered_electrode_surface"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping: {path}")
    return data


def _read_source_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), set(reader.fieldnames or [])


def _apparatus_group(row: dict[str, str]) -> str:
    explicit = str(row.get("resonance_group_MHz", "")).strip()
    if explicit:
        return str(int(float(explicit)))
    scenario_id = str(row["scenario_id"])
    for group in ("24", "34"):
        if f"_r{group}_" in scenario_id:
            return group
    raise ValueError(f"cannot identify the empty-cell apparatus group: {scenario_id}")


def _materialize_family_case(
    spec: dict[str, Any], family: dict[str, Any], input_dir: Path
) -> tuple[Path, dict[str, Any], dict[str, dict[str, str]]]:
    source_path = HERE / str(family["file"])
    rows, columns = _read_source_rows(source_path)
    source_ids = [str(row["scenario_id"]).strip() for row in rows]
    checks = {
        "source_exists": source_path.is_file(),
        "required_columns": columns >= REQUIRED_COLUMNS,
        "row_count": len(rows) == int(family["expected_rows"]),
        "unique_ids": len(source_ids) == len(set(source_ids)),
        "frequency": all(float(row["frequency_Hz"]) == REQUIRED_FREQUENCY_HZ for row in rows),
        "reference_plane": "reference_plane" in columns
        and all(row["reference_plane"] == REQUIRED_REFERENCE_PLANE for row in rows),
        "passive_capacitive": all(
            float(row["resistance_ohm"]) > 0.0 and float(row["reactance_ohm"]) < 0.0 for row in rows
        ),
    }
    metadata = {
        str(row["scenario_id"]): {
            "source_file": str(family["file"]),
            "apparatus_group_MHz": _apparatus_group(row),
        }
        for row in rows
    }

    input_dir.mkdir(parents=True, exist_ok=True)
    table_path = input_dir / "loads.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "frequency_Hz", "resistance_ohm", "reactance_ohm", "weight"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario_id": row["scenario_id"],
                    "frequency_Hz": row["frequency_Hz"],
                    "resistance_ohm": row["resistance_ohm"],
                    "reactance_ohm": row["reactance_ohm"],
                    "weight": row.get("weight", 1),
                }
            )

    controls = dict(spec["controls"])
    family_id = str(family["id"])
    case = {
        "schema": "pcd.rf.v1",
        "case_id": f"{spec['case_id_prefix']}_{family_id}",
        "description": f"Fixed pi-matcher comparison for the independent {family_id} evidence family.",
        "network": {
            "type": "pi_match",
            "search": {"L1": {"values": list(spec["candidate_L1_H"])}},
            "tuning": {"C1": list(controls["C1_F"]), "C2": list(controls["C2_F"])},
        },
        "load": {
            "type": "impedance_table",
            "file": table_path.name,
            "reference_plane": REQUIRED_REFERENCE_PLANE,
            "evidence": {
                "origin": "literature_load_platform_design_challenge",
                "family_id": family_id,
                "family_kind": str(family["kind"]),
                "source_file": str(family["file"]),
            },
        },
        "acceptance": {"reflected_power_fraction_max": float(spec["acceptance"]["reflected_power_fraction_max"])},
        "execution": {
            "solver": "ngspice_cli",
            "optimizer": "grid",
            "candidate_state_limit": 10,
            "control_state_limit": 32,
        },
    }
    case_path = input_dir / "case.yaml"
    case_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return case_path, {"passed": all(checks.values()), "checks": checks}, metadata


def _selected_is_feasible(selected: dict[str, Any]) -> bool:
    constraints = list(selected.get("constraints") or [])
    return selected["raw"]["status"] == "ok" and all(bool(item["satisfied"]) for item in constraints)


def _at_grid_edge(value: Any, values: list[float]) -> bool:
    number = float(value)
    return math.isclose(number, min(values), rel_tol=1e-12) or math.isclose(number, max(values), rel_tol=1e-12)


def _candidate_summary(
    candidate: dict[str, Any], metadata: dict[str, dict[str, str]], controls: dict[str, Any], limit: float
) -> dict[str, Any]:
    feasible_count = 0
    endpoint_count = 0
    gammas: list[float] = []
    margins: list[float] = []
    by_group: dict[str, dict[str, Any]] = {}
    infeasible: list[str] = []

    for item in candidate["scenarios"]:
        scenario_id = str(item["scenario"]["scenario_id"])
        selected = item["selected"]
        feasible = _selected_is_feasible(selected)
        gamma = float(selected["metrics"]["reflection_magnitude"])
        control = dict(selected["request"]["control"]["values"])
        edge = _at_grid_edge(control["C1"], [float(value) for value in controls["C1_F"]]) or _at_grid_edge(
            control["C2"], [float(value) for value in controls["C2_F"]]
        )
        group_id = metadata[scenario_id]["apparatus_group_MHz"]
        group = by_group.setdefault(
            group_id, {"scenario_count": 0, "feasible_scenarios": 0, "worst_reflection_magnitude": 0.0}
        )
        group["scenario_count"] += 1
        group["feasible_scenarios"] += int(feasible)
        group["worst_reflection_magnitude"] = max(float(group["worst_reflection_magnitude"]), gamma)
        feasible_count += int(feasible)
        endpoint_count += int(edge)
        gammas.append(gamma)
        if item.get("control_margin") is not None:
            margins.append(float(item["control_margin"]))
        if not feasible:
            infeasible.append(scenario_id)

    scenario_count = len(candidate["scenarios"])
    for group in by_group.values():
        group["feasible_fraction"] = int(group["feasible_scenarios"]) / int(group["scenario_count"])
    worst = max(gammas)
    return {
        "L1_H": float(candidate["candidate"]["values"]["L1"]),
        "feasible": feasible_count == scenario_count and float(candidate["success_fraction"]) == 1.0,
        "feasible_scenarios": feasible_count,
        "scenario_count": scenario_count,
        "feasible_fraction": feasible_count / scenario_count,
        "worst_reflection_magnitude": worst,
        "worst_reflected_power_fraction": worst**2,
        "worst_reflection_margin": limit - worst,
        "minimum_control_margin": min(margins) if margins else None,
        "endpoint_scenarios": endpoint_count,
        "n_evaluations": scenario_count * len(controls["C1_F"]) * len(controls["C2_F"]),
        "by_apparatus_group": by_group,
        "infeasible_scenarios": infeasible,
    }


def _within(value: float, bounds: list[Any]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def _check_family(
    study: dict[str, Any], candidates: list[dict[str, Any]], expected: dict[str, Any], control_count: int
) -> dict[str, bool]:
    scenario_count = int(expected["scenario_count"])
    by_l1 = {float(item["L1_H"]): item for item in candidates}
    checks = {
        "scenario_count": all(int(item["scenario_count"]) == scenario_count for item in candidates),
        "candidate_count": int(study["n_candidates"]) == len(expected["candidates"]),
        "evaluation_count": int(study["n_evaluations"]) == scenario_count * len(candidates) * control_count,
        "all_evaluations_succeeded": int(study["n_failed_evaluations"]) == 0,
    }
    for item in expected["candidates"]:
        wanted = float(item["L1_H"])
        actual = next((row for value, row in by_l1.items() if math.isclose(value, wanted, rel_tol=1e-12)), None)
        label = f"L1_{wanted * 1e6:.1f}uH"
        checks[f"{label}_present"] = actual is not None
        if actual is None:
            continue
        checks[f"{label}_coverage"] = int(actual["feasible_scenarios"]) == int(item["feasible_scenarios"])
        checks[f"{label}_worst_reflection"] = _within(
            float(actual["worst_reflection_magnitude"]), list(item["worst_reflection_magnitude"])
        )
        for group_id, count in dict(item["by_apparatus_group"]).items():
            checks[f"{label}_group_{group_id}"] = int(
                actual["by_apparatus_group"][str(group_id)]["feasible_scenarios"]
            ) == int(count)
    return checks


def _family_leaders(candidates: list[dict[str, Any]]) -> list[float]:
    best_count = max(int(item["feasible_scenarios"]) for item in candidates)
    return [float(item["L1_H"]) for item in candidates if int(item["feasible_scenarios"]) == best_count]


def _render_report(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    group_rows: list[str] = []
    for family in payload["families"]:
        for item in family["candidates"]:
            rows.append(
                f"| {family['family_id']} | {1e6 * float(item['L1_H']):.1f} | "
                f"{item['feasible_scenarios']}/{item['scenario_count']} | "
                f"{item['worst_reflection_magnitude']:.4f} | {item['endpoint_scenarios']}/{item['scenario_count']} |"
            )
            for group_id, group in item["by_apparatus_group"].items():
                group_rows.append(
                    f"| {family['family_id']} | {group_id} | {1e6 * float(item['L1_H']):.1f} | "
                    f"{group['feasible_scenarios']}/{group['scenario_count']} | "
                    f"{group['worst_reflection_magnitude']:.4f} |"
                )
    status = "PASS" if payload["benchmark_integrity_passed"] else "FAIL"
    return f"""<!-- generated by run_hardware_family_comparison.py; do not edit -->
# GEC CCP hardware family comparison

Regression status: **{status}**. This checks reproducibility of the declared
platform challenge. It does not qualify a matcher or select a universal L1.

The published operating conditions, reported apparatus spread, and the two
counterfactual phase shifts are evaluated independently. No 80-row coverage
score and no overall candidate ranking are produced because their relative
weights have not been specified.

| evidence family | L1 [uH] | feasible conditions | worst `|Gamma|` | controls at edge |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Empty-cell apparatus groups

The 24/34 MHz labels below identify empty-cell resonance populations; every
load is driven at 13.56 MHz.

| evidence family | apparatus group [MHz] | L1 [uH] | feasible conditions | worst `|Gamma|` |
|---|---:|---:|---:|---:|
{chr(10).join(group_rows)}

## Interpretation

- 1.6 uH has the largest central operating-condition coverage.
- 1.5 uH has the largest reported-spread and -6 degree sensitivity coverage.
- 1.5 and 1.6 uH tie under the +6 degree counterfactual.
- Therefore the evidence supplies a result vector, not a literature-derived
  scalar objective or a production hardware decision.
"""


def run(run_root: Path, solver: str) -> dict[str, Any]:
    spec = _read_yaml(SPEC_PATH)
    expectations = _read_yaml(EXPECTATIONS_PATH)
    controls = dict(spec["controls"])
    control_count = len(controls["C1_F"]) * len(controls["C2_F"])
    limit = math.sqrt(float(spec["acceptance"]["reflected_power_fraction_max"]))
    family_results: list[dict[str, Any]] = []

    for family in spec["families"]:
        family_id = str(family["id"])
        case_path, integrity, metadata = _materialize_family_case(spec, family, run_root / "input" / family_id)
        study = run_case_study(load_case(case_path), run_root=run_root / "studies" / family_id, solver_override=solver)
        candidate_dir = Path(study["run_root"]) / "candidates"
        raw_candidates = [
            json.loads(path.read_text(encoding="utf-8")) for path in sorted(candidate_dir.glob("trial_*.json"))
        ]
        candidates = [_candidate_summary(item, metadata, controls, limit) for item in raw_candidates]
        candidates.sort(key=lambda item: float(item["L1_H"]))
        checks = _check_family(study, candidates, expectations["families"][family_id], control_count)
        family_results.append(
            {
                "family_id": family_id,
                "kind": str(family["kind"]),
                "evidence": str(family["evidence"]),
                "source_file": str(family["file"]),
                "integrity": integrity,
                "regression_checks": checks,
                "regression_passed": bool(integrity["passed"]) and all(checks.values()),
                "coverage_leaders_L1_H": _family_leaders(candidates),
                "n_evaluations": int(study["n_evaluations"]),
                "candidates": candidates,
            }
        )

    total_evaluations = sum(int(family["n_evaluations"]) for family in family_results)
    total_expected = int(expectations["total_evaluation_count"])
    passed = all(bool(family["regression_passed"]) for family in family_results) and total_evaluations == total_expected
    payload: dict[str, Any] = {
        "schema": "literature_hardware_family_comparison.v2",
        "benchmark": "p1_gec_ccp_hardware_family_comparison",
        "benchmark_kind": "platform_design_challenge",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "benchmark_integrity_passed": passed,
        "design": {
            "topology": "pi_match",
            "candidate_parameter": "L1_H",
            "candidate_L1_H": list(spec["candidate_L1_H"]),
            "current_L1_H": float(spec["current_L1_H"]),
            "controls": controls,
        },
        "acceptance": {
            "reflection_magnitude_max": limit,
            "reflected_power_fraction_max": float(spec["acceptance"]["reflected_power_fraction_max"]),
            "origin": str(spec["acceptance"]["origin"]),
        },
        "comparison": {
            "universal_winner_declared": False,
            "combined_coverage_score_produced": False,
            "reason": "operating, apparatus-spread, and model-sensitivity families have no declared relative weights",
        },
        "families": family_results,
        "n_evaluations": total_evaluations,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "evaluation.json", payload)
    (run_root / "REPORT.md").write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/gec_ccp_hardware_families"))
    parser.add_argument("--solver", default="ngspice_cli")
    args = parser.parse_args()
    report = run(args.run_root.resolve(), str(args.solver))
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "benchmark_integrity_passed": report["benchmark_integrity_passed"],
                "n_evaluations": report["n_evaluations"],
                "families": [
                    {
                        "family_id": family["family_id"],
                        "coverage_leaders_L1_H": family["coverage_leaders_L1_H"],
                        "candidates": [
                            {
                                "L1_H": item["L1_H"],
                                "coverage": f"{item['feasible_scenarios']}/{item['scenario_count']}",
                                "worst_reflection_magnitude": item["worst_reflection_magnitude"],
                            }
                            for item in family["candidates"]
                        ],
                    }
                    for family in report["families"]
                ],
                "result": str(args.run_root.resolve() / "evaluation.json"),
            },
            indent=2,
        )
    )
    return 0 if bool(report["benchmark_integrity_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
