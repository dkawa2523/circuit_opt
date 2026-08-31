"""Run deterministic decision benchmarks for PCD's actual design purpose.

Frozen-design cases exercise conditions, bounded controls, constraints,
realized-value corners, and worst-case aggregation. One finite-search case
additionally proves that an explicit hardware shortlist is enumerated
completely and ranked by feasibility.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pcd.artifacts import write_json
from pcd.case import load_case
from pcd.study import run_case_study

ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "cases"
CASE_PATHS = tuple(sorted(CASE_DIR.glob("*.yaml")))
EXPECTATIONS = yaml.safe_load((ROOT / "expectations.yaml").read_text(encoding="utf-8")) or {}


def _within(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def _candidate_is_feasible(candidate: dict[str, Any]) -> bool:
    return float(candidate["feasible_fraction"]) == 1.0 and float(candidate["success_fraction"]) == 1.0


def _parameters_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for name, wanted in expected.items():
        if name not in actual:
            return False
        try:
            if not math.isclose(float(actual[name]), float(wanted), rel_tol=1e-12, abs_tol=0.0):
                return False
        except (TypeError, ValueError):
            if actual[name] != wanted:
                return False
    return True


def _selected_scenarios(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in candidate["scenarios"]:
        evaluation = item["selected"]
        constraints = evaluation.get("constraints", []) or []
        violated = sorted(str(row["name"]) for row in constraints if not bool(row["satisfied"]))
        feasible = evaluation["raw"]["status"] == "ok" and all(bool(row["satisfied"]) for row in constraints)
        gamma = float(evaluation["metrics"]["reflection_magnitude"])
        selected.append(
            {
                "scenario_id": str(item["scenario"]["scenario_id"]),
                "feasible": feasible,
                "reflection_magnitude": gamma,
                "reflected_power_fraction": gamma**2,
                "return_loss_dB": -20.0 * math.log10(max(gamma, 1e-15)),
                "input_resistance_ohm": float(evaluation["metrics"]["resistance_ohm"]),
                "input_reactance_ohm": float(evaluation["metrics"]["reactance_ohm"]),
                "control": dict(evaluation["request"]["control"]["values"]),
                "control_margin": item.get("control_margin"),
                "edge_limited": bool(item.get("edge_limited", False)),
                "violated_constraints": violated,
                "from_cache": bool(evaluation.get("from_cache", False)),
            }
        )
    return selected


def _input_impedance_matches(scenarios: list[dict[str, Any]], expected: dict[str, Any], tolerance_ohm: float) -> bool:
    actual = {str(item["scenario_id"]): item for item in scenarios}
    for scenario_id, pair in expected.items():
        if scenario_id not in actual or not isinstance(pair, list | tuple) or len(pair) != 2:
            return False
        item = actual[scenario_id]
        if not math.isclose(float(item["input_resistance_ohm"]), float(pair[0]), rel_tol=0.0, abs_tol=tolerance_ohm):
            return False
        if not math.isclose(float(item["input_reactance_ohm"]), float(pair[1]), rel_tol=0.0, abs_tol=tolerance_ohm):
            return False
    return True


def _selected_controls_match(scenarios: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    actual = {str(item["scenario_id"]): dict(item["control"]) for item in scenarios}
    return all(
        scenario_id in actual and _parameters_match(actual[scenario_id], values)
        for scenario_id, values in expected.items()
    )


def _metric_ranges_match(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual = {
        str(item["scenario"]["scenario_id"]): dict(item["selected"].get("metrics") or {})
        for item in candidate["scenarios"]
    }
    for scenario_id, ranges in expected.items():
        if scenario_id not in actual:
            return False
        for metric, bounds in dict(ranges).items():
            if metric not in actual[scenario_id] or not _within(float(actual[scenario_id][metric]), list(bounds)):
                return False
    return True


def _optional_expectation_checks(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    study: dict[str, Any],
    expected: dict[str, Any],
    violations: dict[str, list[str]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    expected_violations = {
        str(scenario): sorted(str(name) for name in names)
        for scenario, names in dict(expected.get("violated_constraints") or {}).items()
    }
    if expected_violations:
        checks["violated_constraints"] = violations == expected_violations
    if "control_margin" in expected:
        margin = candidate.get("control_margin")
        checks["control_margin"] = margin is not None and _within(float(margin), list(expected["control_margin"]))
    exact_values = {
        "edge_limited": bool(candidate.get("edge_limited", False)),
        "n_candidates": len(candidates),
        "feasible_candidates": sum(_candidate_is_feasible(item) for item in candidates),
        "n_evaluations": int(study["n_evaluations"]),
    }
    for name, actual in exact_values.items():
        if name in expected:
            checks[name] = actual == expected[name]
    if expected_parameters := dict(expected.get("best_parameters") or {}):
        checks["best_parameters"] = _parameters_match(candidate["candidate"]["values"], expected_parameters)
    if expected_controls := dict(expected.get("selected_controls") or {}):
        checks["selected_controls"] = _selected_controls_match(scenarios, expected_controls)
    if expected_impedance := dict(expected.get("input_impedance_ohm") or {}):
        checks["input_impedance_ohm"] = _input_impedance_matches(
            scenarios,
            expected_impedance,
            float(expected.get("input_impedance_tolerance_ohm", 0.02)),
        )
    if expected_metrics := dict(expected.get("metric_ranges") or {}):
        checks["metric_ranges"] = _metric_ranges_match(candidate, expected_metrics)
    return checks


def run_case(
    path: Path,
    run_root: Path,
    solver: str,
    expectations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case = load_case(path)
    metadata = dict((expectations or EXPECTATIONS)[path.name])
    expected = dict(metadata.get("expected") or {})
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
    candidates = [
        json.loads(candidate_path.read_text(encoding="utf-8"))
        for candidate_path in sorted(candidate_dir.glob("trial_*.json"))
    ]
    best_id = str(study["best"]["candidate"]["candidate_id"])
    candidate = next(item for item in candidates if str(item["candidate"]["candidate_id"]) == best_id)
    scenarios = _selected_scenarios(candidate)
    worst = float(candidate["aggregates"]["reflection_magnitude"])
    feasible_fraction = float(candidate["feasible_fraction"])
    infeasible = sorted(item["scenario_id"] for item in scenarios if not item["feasible"])
    actual_feasible = _candidate_is_feasible(candidate)
    feasible_candidates = sum(_candidate_is_feasible(item) for item in candidates)
    candidate_values = dict(candidate["candidate"]["values"])
    violations = {item["scenario_id"]: item["violated_constraints"] for item in scenarios}
    checks = {
        "feasible_classification": actual_feasible == bool(expected["feasible"]),
        "feasible_fraction": _within(feasible_fraction, list(expected["feasible_fraction"])),
        "worst_reflection_magnitude": _within(worst, list(expected["worst_reflection_magnitude"])),
        "infeasible_scenarios": infeasible == sorted(str(item) for item in expected["infeasible_scenarios"]),
        "all_evaluations_succeeded": int(study["n_failed_evaluations"]) == 0,
    }
    checks.update(_optional_expectation_checks(candidate, candidates, scenarios, study, expected, violations))
    return {
        "benchmark_id": str(metadata["id"]),
        "case_id": case.case_id,
        "case_path": str(path.resolve()),
        "role": str(metadata["role"]),
        "question": str(metadata["question"]),
        "demonstrates": str(metadata["demonstrates"]),
        "does_not_establish": str(metadata["does_not_establish"]),
        "expected": expected,
        "passed": all(checks.values()),
        "checks": checks,
        "feasible": actual_feasible,
        "feasible_fraction": feasible_fraction,
        "success_fraction": float(candidate["success_fraction"]),
        "worst_reflection_magnitude": worst,
        "worst_reflected_power_fraction": worst**2,
        "worst_return_loss_dB": -20.0 * math.log10(max(worst, 1e-15)),
        "control_margin": candidate.get("control_margin"),
        "edge_limited": bool(candidate.get("edge_limited", False)),
        "infeasible_scenarios": infeasible,
        "violated_constraints": violations,
        "best_candidate_id": best_id,
        "best_candidate_values": candidate_values,
        "n_candidates": len(candidates),
        "feasible_candidates": feasible_candidates,
        "n_evaluations": int(study["n_evaluations"]),
        "n_failed_evaluations": int(study["n_failed_evaluations"]),
        "scenarios": scenarios,
    }


def _single_impedance(case: dict[str, Any]) -> complex:
    if len(case["scenarios"]) != 1:
        raise ValueError(f"cross-case invariant requires one scenario: {case['benchmark_id']}")
    scenario = case["scenarios"][0]
    return complex(float(scenario["input_resistance_ohm"]), float(scenario["input_reactance_ohm"]))


def _suite_checks(cases: list[dict[str, Any]]) -> dict[str, bool]:
    """Cross-case physical invariants that one result cannot prove alone."""

    cfg = dict((EXPECTATIONS.get("_suite") or {}).get("reference_plane") or {})
    if not cfg:
        return {}
    by_id = {str(item["benchmark_id"]): item for item in cases}
    equivalent_ids = [str(item) for item in cfg["equivalent_cases"]]
    if len(equivalent_ids) != 2 or any(item not in by_id for item in equivalent_ids):
        return {"reference_plane_cases_present": False}
    first, second = (by_id[item] for item in equivalent_ids)
    double_id = str(cfg["double_counted_case"])
    if double_id not in by_id:
        return {"reference_plane_cases_present": False}
    double = by_id[double_id]
    tolerance = float(cfg["equivalent_impedance_tolerance_ohm"])
    minimum_separation = float(cfg["double_counted_impedance_separation_min_ohm"])
    return {
        "reference_plane_cases_present": True,
        "equivalent_reference_plane_impedance": abs(_single_impedance(first) - _single_impedance(second)) <= tolerance,
        "equivalent_reference_plane_decision": bool(first["feasible"]) == bool(second["feasible"]),
        "double_counted_fixture_changes_impedance": abs(_single_impedance(double) - _single_impedance(first))
        >= minimum_separation,
        "double_counted_fixture_is_rejected": not bool(double["feasible"]),
    }


def _format_control(control: dict[str, Any]) -> str:
    if not control:
        return "fixed"
    return ", ".join(f"{name}={float(value) * 1e12:.0f} pF" for name, value in sorted(control.items()))


def _format_margin(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def render_report(payload: dict[str, Any]) -> str:
    rows = []
    details = []
    for case in payload["cases"]:
        rows.append(
            f"| {'PASS' if case['passed'] else '**FAIL**'} | {case['benchmark_id']} | {case['role']} | "
            f"{case['n_candidates']} | {'yes' if case['feasible'] else 'no'} | "
            f"{case['feasible_fraction']:.0%} | "
            f"{case['worst_reflection_magnitude']:.4f} | {case['worst_reflected_power_fraction']:.2%} | "
            f"{case['worst_return_loss_dB']:.2f} | "
            f"{_format_margin(case['control_margin'])} |"
        )
        scenario_rows = "\n".join(
            f"| {item['scenario_id']} | {'yes' if item['feasible'] else '**no**'} | "
            f"{item['input_resistance_ohm']:.4f} | {item['input_reactance_ohm']:.4f} | "
            f"{item['reflection_magnitude']:.4f} | {item['reflected_power_fraction']:.2%} | "
            f"{item['return_loss_dB']:.2f} | {_format_control(item['control'])} | "
            f"{_format_margin(item['control_margin'])} | "
            f"{', '.join(item['violated_constraints']) or 'none'} |"
            for item in case["scenarios"]
        )
        candidate_evidence = ""
        if case["n_candidates"] > 1:
            values = ", ".join(f"{name}={float(value):.7g}" for name, value in case["best_candidate_values"].items())
            candidate_evidence = (
                f"\n**Search evidence:** {case['feasible_candidates']}/{case['n_candidates']} candidates were feasible; "
                f"best `{case['best_candidate_id']}` has {values}.\n"
            )
        details.append(
            f"""### {case["benchmark_id"]}

**Question:** {case["question"]}
{candidate_evidence}

| scenario | feasible | input R [ohm] | input X [ohm] | reflection magnitude | reflected power | return loss [dB] | selected control | control margin | violated limits |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
{scenario_rows}

**What it demonstrates:** {case["demonstrates"]}

**What it does not establish:** {case["does_not_establish"]}
"""
        )

    invariant_rows = "\n".join(
        f"| {'PASS' if passed else '**FAIL**'} | `{name}` |" for name, passed in payload["suite_checks"].items()
    )
    status = "All benchmark classifications and invariants reproduced" if payload["passed"] else "BENCHMARK FAILURE"
    return f"""<!-- generated by bench/run_suite.py; do not edit by hand -->
# PCD design benchmark report

{status}. Generated at {payload["generated_at"]} with `{payload["solver"]}`.

The acceptance limit is `|Gamma| <= sqrt(0.1) = 0.31623`, equivalent to no
more than 10% reflected incident power.  A candidate is feasible only when a
successful, bounded control setting meets that limit in every scenario.

## Classification summary

| | case | role | candidates | best feasible across window | feasible scenarios | worst reflection magnitude | worst reflected power | worst return loss [dB] | worst control margin |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Per-case evidence

{chr(10).join(details)}

## Cross-case invariants

| | invariant |
|---|---|
{invariant_rows}

## Scope boundary

A1-A3 compare each public network topology with a frozen complex-impedance
oracle. A4 and A5 exercise the effective CCP and ICP ports across three
frequencies. B1-B8
exercise fixed hardware, bounded controls, external scenarios, stress limits,
realized-value corners, and finite candidate enumeration. D1-D3 compare two equivalent lossy
reference-plane representations with a deliberate fixture-double-counting
negative control.
All inputs are synthetic and are not a qualified reactor data set. These cases
do not validate microscopic plasma physics, nonlinear sheath harmonics,
internal temperature, tuner dynamics, or closed-loop stability.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="ngspice_cli")
    parser.add_argument("--run-root", default="runs/benchmark_suite")
    parser.add_argument("--report", default=None, help="Markdown report path; defaults inside run-root")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    cases = sorted(
        (run_case(path, run_root, str(args.solver)) for path in CASE_PATHS),
        key=lambda item: item["benchmark_id"],
    )
    suite_checks = _suite_checks(cases)
    payload = {
        "schema": "design_benchmark_suite.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver": str(args.solver),
        "acceptance": {"metric": "reflection_magnitude", "max": math.sqrt(0.1)},
        "passed": all(item["passed"] for item in cases) and all(suite_checks.values()),
        "suite_checks": suite_checks,
        "n_candidates": sum(item["n_candidates"] for item in cases),
        "n_evaluations": sum(item["n_evaluations"] for item in cases),
        "cases": cases,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "benchmark_result.json", payload)
    report_path = Path(args.report).resolve() if args.report else run_root / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({**payload, "report": str(report_path)}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
