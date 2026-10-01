"""Run the preregistered P3 retrospective ranking benchmark."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pcd.artifacts import file_sha256, write_json
from pcd.case import Case, load_case
from pcd.ml import evaluate_candidate_ranking
from pcd.results import candidate_summary, read_evaluation_table
from pcd.study import run_case_study

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "ranking_protocol.yaml"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = _mapping(payload, "ranking protocol")
    if data.get("schema") != "pcd.ml_ranking_protocol.v1":
        raise ValueError("ranking protocol schema must be 'pcd.ml_ranking_protocol.v1'")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("ranking protocol must declare at least two cases")
    if data.get("claim") != "retrospective_fixed_candidate_pool":
        raise ValueError("ranking protocol supports retrospective fixed pools only")
    model = _mapping(data.get("model"), "model")
    if model.get("name") != "standardized_distance_weighted_knn":
        raise ValueError("ranking protocol supports the fixed three-neighbour model only")
    criterion = _mapping(data.get("success_criterion"), "success_criterion")
    required = (
        criterion.get("require_every_case") is True
        and criterion.get("require_all_solver_evaluations") is True
        and criterion.get("verification") == "full_scenario_x_control_ngspice_rerun"
    )
    if not required:
        raise ValueError("ranking protocol cannot weaken complete evidence or verification requirements")
    return data


def _python_scalar(value: object) -> Any:
    return value.item() if hasattr(value, "item") else value


def _design_values(candidates: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    selected = candidates.loc[candidates["candidate_id"].astype(str).eq(candidate_id)]
    if len(selected) != 1:
        raise ValueError(f"selected candidate {candidate_id!r} is not unique in the pool")
    row = selected.iloc[0]
    return {
        name.removeprefix("design."): _python_scalar(row[name])
        for name in candidates.columns
        if name.startswith("design.")
    }


def _verification_case(source: Case, values: Mapping[str, Any]) -> Case:
    data = deepcopy(source.data)
    data["case_id"] = f"{source.case_id}_ranking_verification"
    study = _mapping(data.get("study") or {}, "study")
    study["candidate_enumeration"] = "exact"
    study["design_variables"] = {name: {"choices": [value]} for name, value in values.items()}
    data["study"] = study
    data["optimizer"] = {"name": "grid", "seed": 0}
    data["run"] = {"trials": 1}
    return Case(path=source.path, data=data)


def _same_design(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for name, wanted in expected.items():
        try:
            if not math.isclose(float(str(actual[name])), float(str(wanted)), rel_tol=1e-12, abs_tol=0.0):
                return False
        except ValueError:
            if actual[name] != wanted:
                return False
    return True


def _verification(
    source: Case,
    candidates: pd.DataFrame,
    candidate_id: str,
    run_root: Path,
    solver: str,
    expected_evaluations: int,
) -> dict[str, Any]:
    values = _design_values(candidates, candidate_id)
    result = run_case_study(
        _verification_case(source, values),
        run_root=run_root,
        solver_override=solver,
    )
    evaluations = read_evaluation_table(result["run_root"])
    cache_hits = evaluations["from_cache"].astype(str).str.lower().eq("true").sum()
    checks = {
        "one_selected_candidate": int(result["n_candidates"]) == 1,
        "complete_scenario_x_control": int(result["n_evaluations"]) == expected_evaluations,
        "all_solver_evaluations_succeeded": int(result["n_failed_evaluations"]) == 0,
        "selected_design_matches": _same_design(result["best"]["candidate"]["values"], values),
        "fully_feasible": (
            float(result["best"]["success_fraction"]) == 1.0 and float(result["best"]["feasible_fraction"]) == 1.0
        ),
        "fresh_solver_evidence": cache_hits == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_id": candidate_id,
        "design": values,
        "ngspice_evaluations": int(result["n_evaluations"]),
        "cache_hits": cache_hits,
        "study_root": str(Path(result["run_root"]).resolve()),
    }


def _declared_objectives(study: Mapping[str, Any]) -> dict[str, str]:
    objectives = _mapping(study.get("study"), "study result study").get("objectives")
    if not isinstance(objectives, list):
        raise ValueError("study result objectives must be a list")
    return {
        f"objective.{_mapping(item, 'study objective')['metric']!s}": str(
            _mapping(item, "study objective").get("direction", "minimize")
        )
        for item in objectives
    }


def _case_result(
    raw_spec: object,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    run_root: Path,
    solver: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = _mapping(raw_spec, "ranking case")
    case_path = (protocol_path.parent / str(spec["path"])).resolve()
    case = load_case(case_path)
    if case.case_id != str(spec["case_id"]):
        raise ValueError(f"protocol case_id does not match {case_path}")
    study = run_case_study(case, run_root=run_root / "pools", solver_override=solver)
    candidates = candidate_summary(study["run_root"])
    expected_candidates = int(spec["expected_candidates"])
    evaluations_per_candidate = int(spec["solver_evaluations_per_candidate"])
    expected_evaluations = expected_candidates * evaluations_per_candidate
    pool_checks = {
        "candidate_count": len(candidates) == expected_candidates,
        "solver_evaluation_count": int(study["n_evaluations"]) == expected_evaluations,
        "all_solver_evaluations_succeeded": int(study["n_failed_evaluations"]) == 0,
        "objective_directions_match": _declared_objectives(study)
        == _mapping(spec["objective_directions"], "objective_directions"),
    }
    criterion = _mapping(protocol["success_criterion"], "success_criterion")
    model = _mapping(protocol["model"], "model")
    ranking = evaluate_candidate_ranking(
        candidates,
        feature_transforms=_mapping(spec["feature_transforms"], "feature_transforms"),
        objective_directions=_mapping(spec["objective_directions"], "objective_directions"),
        initial_count=int(spec["initial_observations"]),
        initial_seed=int(spec["initial_seed"]),
        top_fraction=float(criterion["top_fraction"]),
        neighbors=int(model["neighbors"]),
        feasibility_threshold=float(model["feasibility_threshold"]),
        random_repeats=int(criterion["random_baseline_repeats"]),
        random_seed=int(criterion["random_baseline_seed"]),
        required_savings_fraction=float(criterion["required_savings_fraction"]),
    )
    selected_id = str(_mapping(ranking.summary["outcome"], "ranking outcome")["selected_candidate_id"])
    verification = _verification(
        case,
        candidates,
        selected_id,
        run_root / "verification",
        solver,
        evaluations_per_candidate,
    )
    passed = all(pool_checks.values()) and bool(ranking.summary["case_passed"]) and bool(verification["passed"])
    trace = ranking.trace.copy()
    trace.insert(0, "case_id", case.case_id)
    return (
        {
            "case_id": case.case_id,
            "case_path": str(case_path),
            "case_sha256": file_sha256(case_path),
            "pool_study_root": str(Path(study["run_root"]).resolve()),
            "pool_checks": pool_checks,
            "ranking": ranking.summary,
            "verification": verification,
            "passed": passed,
        },
        trace,
    )


def run_benchmark(protocol_path: Path, run_root: Path, solver: str) -> dict[str, Any]:
    """Execute all frozen pools and publish one aggregate go/no-go result."""

    if run_root.exists():
        raise FileExistsError(f"ranking run root must not already exist; choose a fresh path: {run_root}")
    protocol = _protocol(protocol_path)
    results: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    for raw_spec in protocol["cases"]:
        result, trace = _case_result(raw_spec, protocol_path, protocol, run_root, solver)
        results.append(result)
        traces.append(trace)

    passed = all(bool(result["passed"]) for result in results)
    output = {
        "schema": "pcd.ml_ranking_benchmark.v1",
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": file_sha256(protocol_path),
        "solver": solver,
        "cases": results,
        "passed": passed,
        "sequential_proposal_allowed": passed,
        "decision": (
            "implement_one_bounded_sequential_sizing_loop"
            if passed
            else "keep_ml_offline_and_do_not_offer_candidate_proposals"
        ),
        "limitations": list(protocol.get("limitations") or []),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    pd.concat(traces, ignore_index=True).to_csv(run_root / "ranking_trace.csv", index=False)
    write_json(run_root / "ranking_evaluation.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=Path("runs/p3_ranking"))
    parser.add_argument("--solver", default="ngspice_cli")
    args = parser.parse_args()
    result = run_benchmark(args.protocol.resolve(), args.run_root.resolve(), args.solver)
    for case in result["cases"]:
        outcome = case["ranking"]["outcome"]
        print(
            f"{case['case_id']}: {'PASS' if case['passed'] else 'FAIL'}; "
            f"evaluations={outcome['candidate_evaluations']}, "
            f"random_median={outcome['random_baseline_median_evaluations']}, "
            f"savings={100.0 * outcome['savings_fraction']:.1f}%"
        )
    print(f"P3 ranking gate: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Result: {args.run_root.resolve() / 'ranking_evaluation.json'}")


if __name__ == "__main__":
    main()
