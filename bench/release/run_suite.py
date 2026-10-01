"""Reproduce the five supported PCD v1 user workflows through the public CLI.

The suite qualifies the circuit-analysis path, not a chamber process.  Expected
engineering rejection is a valid result when the declared failed limits are
reproduced.  A separate, preregistered ML result may be attached so release
closure does not confuse circuit readiness with ML proposal readiness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pcd import __version__
from pcd.artifacts import atomic_write_text, write_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    workflow: str
    path: Path
    mode: Literal["simulation_analysis", "study"]
    require_acceptance: bool = False


CASES = (
    CaseSpec(
        "transient_analysis",
        "saved AC/transient analysis",
        ROOT / "examples" / "advanced" / "rf_port_transient.yaml",
        "simulation_analysis",
    ),
    CaseSpec(
        "scenario_control_study",
        "Candidate x Scenario x Control study",
        ROOT / "bench" / "cases" / "role_factorial_search.yaml",
        "study",
        require_acceptance=True,
    ),
    CaseSpec(
        "declared_rejection",
        "Candidate x Scenario x Control study",
        ROOT / "examples" / "rf_impedance_point_study.yaml",
        "study",
    ),
    CaseSpec(
        "frequency_sweep",
        "frequency sweep",
        ROOT / "bench" / "literature" / "p1_colpo1999_icp" / "dummy_resonance.yaml",
        "simulation_analysis",
    ),
    CaseSpec(
        "quasi_static_profile",
        "prescribed time variation",
        ROOT / "examples" / "rf_quasi_static_profile.yaml",
        "study",
    ),
    CaseSpec(
        "time_varying_resistor",
        "prescribed time variation",
        ROOT / "examples" / "advanced" / "time_varying_resistor.yaml",
        "simulation_analysis",
    ),
    CaseSpec(
        "target_waveform_sizing",
        "target-waveform sizing",
        ROOT / "examples" / "advanced" / "generic_rc_filter.yaml",
        "study",
        require_acceptance=True,
    ),
)


def _json_command(label: str, args: list[str], log_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pcd", *args, "--json"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        log_dir / f"{label}.log",
        f"command={json.dumps(command)}\nexit_code={completed.returncode}\n\n"
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {log_dir / f'{label}.log'}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} did not return JSON; see {log_dir / f'{label}.log'}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object JSON result")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _artifact_files(base: Path, artifacts: dict[str, Any], names: tuple[str, ...]) -> bool:
    return all(name in artifacts and (base / str(artifacts[name])).is_file() for name in names)


def _evaluation_rows(study: dict[str, Any], study_root: Path) -> list[dict[str, str]]:
    table = study_root / str(study["artifacts"]["evaluation_table"])
    with table.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cache_hits(rows: list[dict[str, str]]) -> int:
    return sum(row.get("from_cache", "").lower() in {"1", "true", "yes"} for row in rows)


def _conditions(study: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_id"]): row for row in study["best"]["conditions"]}


def _simulation_artifacts(bundle: dict[str, Any], names: tuple[str, ...]) -> bool:
    simulation = bundle["simulation"]
    return _artifact_files(Path(str(simulation["run_dir"])), simulation["artifacts"], names)


def _analysis_artifacts(bundle: dict[str, Any], names: tuple[str, ...]) -> bool:
    analysis = bundle["analysis"]
    return _artifact_files(Path(str(analysis["output_dir"])), analysis["artifacts"], names)


def _check_transient(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    simulation = bundle["simulation"]
    analysis = bundle["analysis"]
    transient = analysis["analyses"]["transient"]
    periodic = transient["periodic"]
    power = float(periodic["load_real_power_W"])
    checks = {
        "solver_succeeded": simulation["status"] == "ok",
        "no_validation_or_runtime_warnings": not simulation["warnings"],
        "canonical_waveform_and_replay_manifest": _simulation_artifacts(bundle, ("waveform", "debug_manifest")),
        "reference_plane_is_explicit": analysis["measurement"]["reference_plane"] == "load_terminal",
        "periodic_measurement_settled": periodic["status"] == "ok" and bool(periodic["periodic_settled"]),
        "load_power_is_25W": 24.9 <= power <= 25.1,
        "standard_figures_exist": _analysis_artifacts(
            bundle, ("transient_response", "operating_point", "harmonic_spectrum")
        ),
    }
    return checks, {"samples": int(transient["samples"]), "load_real_power_W": power}


def _check_frequency_sweep(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    simulation = bundle["simulation"]
    analysis = bundle["analysis"]
    sweep = analysis["analyses"]["ac"]["sweep"]
    resonance = sweep.get("half_power_resonance") or {}
    frequency = float(resonance.get("resonant_frequency_Hz", 0.0))
    bandwidth = float(resonance.get("bandwidth_Hz", 0.0))
    quality = float(resonance.get("loaded_quality_factor", 0.0))
    checks = {
        "solver_succeeded": simulation["status"] == "ok",
        "no_validation_or_runtime_warnings": not simulation["warnings"],
        "canonical_ac_table_and_replay_manifest": _simulation_artifacts(
            bundle, ("frequency_response", "debug_manifest")
        ),
        "reference_plane_is_explicit": analysis["measurement"]["reference_plane"] == "dummy_coil_terminal",
        "dense_frequency_sweep": int(sweep["sample_count"]) >= 8000,
        "resonance_is_bracketed": 13.0e6 <= frequency <= 14.5e6 and bandwidth > 0.0,
        "loaded_q_is_finite": math.isfinite(quality) and 10.0 <= quality <= 20.0,
        "frequency_figures_exist": _analysis_artifacts(bundle, ("frequency_response", "operating_point")),
    }
    return checks, {"resonant_frequency_Hz": frequency, "bandwidth_Hz": bandwidth, "loaded_quality_factor": quality}


def _check_time_varying_resistor(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    simulation = bundle["simulation"]
    analysis = bundle["analysis"]
    transient = analysis["analyses"]["transient"]
    waveform = Path(str(simulation["run_dir"])) / str(simulation["artifacts"]["waveform"])
    with waveform.open(encoding="utf-8", newline="") as handle:
        columns = set(next(csv.reader(handle)))
    checks = {
        "solver_succeeded": simulation["status"] == "ok",
        "no_validation_or_runtime_warnings": not simulation["warnings"],
        "canonical_waveform_and_replay_manifest": _simulation_artifacts(bundle, ("waveform", "debug_manifest")),
        "component_terminal_probes_exist": {
            "component_Rchamber_voltage_V",
            "component_Rchamber_current_A",
        }.issubset(columns),
        "time_profile_was_sampled": int(transient["samples"]) >= 250,
        "nonperiodic_signal_not_misclassified": transient["periodic"]["status"] == "not_applicable",
        "transient_figure_exists": _analysis_artifacts(bundle, ("transient_response",)),
    }
    return checks, {"samples": int(transient["samples"]), "periodic_status": transient["periodic"]["status"]}


def _study_common(bundle: dict[str, Any], required: tuple[str, ...]) -> dict[str, bool]:
    study = bundle["study"]
    root = Path(str(study["run_root"]))
    rows = bundle["evaluation_rows"]
    return {
        "all_solver_evaluations_succeeded": int(study["n_failed_evaluations"]) == 0,
        "fresh_solver_evidence": _cache_hits(rows) == 0,
        "complete_audit_and_selected_evidence": _artifact_files(
            root,
            study["artifacts"],
            ("best_candidate", "history", "evaluation_table", *required),
        ),
        "evaluation_table_is_complete": len(rows) == int(study["n_evaluations"]),
    }


def _check_scenario_control(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    study = bundle["study"]
    conditions = _conditions(study)
    checks = _study_common(bundle, ())
    checks.update(
        {
            "full_factorial_evaluation_count": int(study["n_evaluations"]) == 8,
            "declared_acceptance_met": study["best"]["status"] == "meets_declared_acceptance",
            "condition_coverage_complete": study["best"]["coverage"] == {"conditions": 2, "solved": 2, "accepted": 2},
            "condition_controls_selected": math.isclose(
                float(conditions["control_20pF_required"]["selected_control"]["C1"]), 20e-12
            )
            and math.isclose(float(conditions["control_80pF_required"]["selected_control"]["C1"]), 80e-12),
        }
    )
    return checks, {"evaluations": int(study["n_evaluations"]), "coverage": study["best"]["coverage"]}


def _check_declared_rejection(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    study = bundle["study"]
    conditions = list(_conditions(study).values())
    failed = [
        row
        for row in conditions
        if [item["name"] for item in row["failed_constraints"]] == ["max_reflection_magnitude"]
    ]
    checks = _study_common(bundle, ())
    checks.update(
        {
            "declared_grid_is_complete": int(study["n_evaluations"]) == 81
            and study["best"]["search_completeness"] == "complete",
            "electrical_rejection_is_not_solver_failure": study["best"]["status"]
            == "does_not_meet_declared_acceptance",
            "condition_coverage_is_reported": study["best"]["coverage"]
            == {"conditions": 3, "solved": 3, "accepted": 1},
            "failed_limits_are_named": len(failed) == 2,
        }
    )
    return checks, {"evaluations": int(study["n_evaluations"]), "coverage": study["best"]["coverage"]}


def _check_quasi_static(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    study = bundle["study"]
    root = Path(str(study["run_root"]))
    snapshot = root / str(study["artifacts"]["snapshot_response"])
    with snapshot.open(encoding="utf-8", newline="") as handle:
        snapshot_rows = list(csv.DictReader(handle))
    checks = _study_common(bundle, ("snapshot_response",))
    checks.update(
        {
            "snapshot_mode_is_explicit": study["study"]["metadata"]["analysis_mode"] == "quasi_static_snapshot",
            "all_snapshot_control_states_evaluated": int(study["n_evaluations"]) == 15,
            "one_selected_row_per_snapshot": len(snapshot_rows) == 5,
            "snapshot_coverage_is_reported": study["best"]["coverage"] == {"conditions": 5, "solved": 5, "accepted": 3},
            "rejection_is_not_hidden": study["best"]["status"] == "does_not_meet_declared_acceptance",
        }
    )
    return checks, {"evaluations": int(study["n_evaluations"]), "snapshots": len(snapshot_rows)}


def _check_target_sizing(bundle: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    study = bundle["study"]
    aggregates = study["best"]["aggregates"]
    pareto = study.get("pareto") or {}
    checks = _study_common(bundle, ("pareto_front",))
    checks.update(
        {
            "fixed_budget_and_seed": int(study["n_evaluations"]) == 24 and int(study["execution"]["seed"]) == 3,
            "declared_acceptance_met": study["best"]["status"] == "meets_declared_acceptance",
            "objective_quality_reproduced": float(aggregates["normalized_rmse"]) <= 0.55
            and float(aggregates["peak_abs_voltage_V"]) <= 4.1,
            "pareto_scope_is_honest": pareto.get("scope") == "observed_candidates"
            and int(pareto.get("front_candidates", 0)) > 0,
        }
    )
    return checks, {
        "evaluations": int(study["n_evaluations"]),
        "normalized_rmse": float(aggregates["normalized_rmse"]),
        "peak_abs_voltage_V": float(aggregates["peak_abs_voltage_V"]),
    }


CHECKERS: dict[str, Callable[[dict[str, Any]], tuple[dict[str, bool], dict[str, Any]]]] = {
    "transient_analysis": _check_transient,
    "scenario_control_study": _check_scenario_control,
    "declared_rejection": _check_declared_rejection,
    "frequency_sweep": _check_frequency_sweep,
    "quasi_static_profile": _check_quasi_static,
    "time_varying_resistor": _check_time_varying_resistor,
    "target_waveform_sizing": _check_target_sizing,
}


def _run_case(spec: CaseSpec, run_root: Path, log_dir: Path) -> dict[str, Any]:
    validation = _json_command(
        f"{spec.case_id}_validate",
        ["validate-case", str(spec.path), "--strict"],
        log_dir,
    )
    case_root = run_root / "cases" / spec.case_id
    if spec.mode == "simulation_analysis":
        result = _json_command(
            f"{spec.case_id}_simulate",
            ["sim-run", str(spec.path), "--run-root", str(case_root)],
            log_dir,
        )
        run_dir = Path(str(result["run_dir"]))
        simulation = _read_json(run_dir / "summary.json")
        analysis = _json_command(f"{spec.case_id}_analyze", ["analyze", str(run_dir)], log_dir)
        bundle = {"simulation": simulation, "analysis": analysis}
        evaluations = 1
        cache_hits = 0
        evidence = {
            "simulation": str(run_dir / "summary.json"),
            "analysis": str(Path(analysis["output_dir"]) / "summary.json"),
        }
    else:
        args = ["run", str(spec.path), "--output", str(case_root)]
        if spec.require_acceptance:
            args.append("--require-acceptance")
        result = _json_command(f"{spec.case_id}_study", args, log_dir)
        study_root = Path(str(result["run_root"]))
        study = _read_json(study_root / "study_result.json")
        rows = _evaluation_rows(study, study_root)
        bundle = {"study": study, "evaluation_rows": rows}
        evaluations = int(study["n_evaluations"])
        cache_hits = _cache_hits(rows)
        evidence = {"study": str(study_root / "study_result.json"), "generation": str(study["artifacts"]["generation"])}

    checks, observed = CHECKERS[spec.case_id](bundle)
    checks = {"strict_validation": bool(validation["ok"]), **checks}
    return {
        "case_id": spec.case_id,
        "workflow": spec.workflow,
        "case_path": str(spec.path),
        "mode": spec.mode,
        "passed": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "ngspice_evaluations": evaluations,
        "cache_hits": cache_hits,
        "evidence": evidence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ml_case_integrity(case: dict[str, Any]) -> bool:
    pool_checks = case.get("pool_checks") or {}
    verification = case.get("verification") or {}
    ranking = case.get("ranking") or {}
    savings = (ranking.get("outcome") or {}).get("savings_fraction")
    valid_savings = isinstance(savings, int | float) and not isinstance(savings, bool) and math.isfinite(savings)
    return all(bool(value) for value in pool_checks.values()) and bool(verification.get("passed")) and valid_savings


def _ml_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    ranking = case.get("ranking") or {}
    return {
        "case_id": case.get("case_id"),
        "savings_fraction": (ranking.get("outcome") or {}).get("savings_fraction"),
        "case_passed": case.get("passed"),
        "verification_passed": (case.get("verification") or {}).get("passed"),
    }


def _ml_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_attached", "integrity_passed": False}
    result = _read_json(path.resolve())
    protocol_path = ROOT / "bench" / "ml" / "ranking_protocol.yaml"
    cases = result.get("cases") or []
    integrity = {
        "schema": result.get("schema") == "pcd.ml_ranking_benchmark.v1",
        "protocol_hash": result.get("protocol_sha256") == _sha256(protocol_path),
        "case_evidence": len(cases) == 2 and all(_ml_case_integrity(case) for case in cases),
        "proposal_decision_matches_gate": bool(result.get("sequential_proposal_allowed")) == bool(result.get("passed")),
    }
    return {
        "status": "attached",
        "path": str(path.resolve()),
        "protocol_id": result.get("protocol_id"),
        "protocol_sha256": result.get("protocol_sha256"),
        "integrity_passed": all(integrity.values()),
        "integrity_checks": integrity,
        "ranking_gate_passed": bool(result.get("passed")),
        "sequential_proposal_allowed": bool(result.get("sequential_proposal_allowed")),
        "cases": [_ml_case_summary(case) for case in cases],
    }


def _release_decisions(circuit_passed: bool, ml_attached: bool, ml_ready: bool) -> dict[str, str]:
    if not ml_attached:
        ml_decision = "NOT ASSESSED"
        full_scope_decision = "NOT ASSESSED" if circuit_passed else "NO-GO"
    else:
        ml_decision = "GO" if ml_ready else "NO-GO"
        full_scope_decision = "GO" if circuit_passed and ml_ready else "NO-GO"
    return {
        "circuit_foundation": "GO" if circuit_passed else "NO-GO",
        "ml_candidate_proposal": ml_decision,
        "full_requested_scope": full_scope_decision,
    }


def _release_status(cases: list[dict[str, Any]], ml: dict[str, Any]) -> dict[str, Any]:
    circuit_passed = all(case["passed"] for case in cases)
    ml_attached = ml["status"] == "attached"
    evidence_complete = ml_attached and bool(ml["integrity_passed"])
    ml_ready = evidence_complete and bool(ml["ranking_gate_passed"])
    return {
        "passed": circuit_passed and (not ml_attached or evidence_complete),
        "workflow_acceptance_passed": circuit_passed,
        "ml_evidence_complete": evidence_complete,
        "release_decision": _release_decisions(circuit_passed, ml_attached, ml_ready),
    }


def _release_totals(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "ngspice_evaluations": sum(case["ngspice_evaluations"] for case in cases),
        "cache_hits": sum(case["cache_hits"] for case in cases),
    }


def _format_savings(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.1%}"
    return "n/a"


def _ml_report_note(ml: dict[str, Any]) -> str:
    if ml["status"] != "attached":
        return "No ML ranking evidence was attached, so ML and full-scope readiness were not assessed."
    if ml["ranking_gate_passed"]:
        return "The attached ML gate passed; candidate-proposal readiness is still limited to its declared scope."
    return (
        "The ML failure is retained as evidence. It does not invalidate the circuit foundation, and the "
        "circuit-suite PASS does not authorize an ML proposal command."
    )


def _render_report(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {'PASS' if case['passed'] else '**FAIL**'} | {case['workflow']} | `{case['case_id']}` | "
        f"{case['ngspice_evaluations']} | {case['cache_hits']} |"
        for case in payload["cases"]
    )
    decisions = payload["release_decision"]
    ml = payload["ml_ranking_evidence"]
    ml_rows = (
        "\n".join(
            f"| `{case['case_id']}` | {_format_savings(case['savings_fraction'])} | "
            f"{'PASS' if case['case_passed'] else 'FAIL'} |"
            for case in ml.get("cases", [])
        )
        or "| not attached | - | NOT ASSESSED |"
    )
    ml_note = _ml_report_note(ml)
    return f"""<!-- generated by bench/release/run_suite.py; do not edit -->
# PCD v1 release closure

Generated at {payload["generated_at"]} with `{payload["solver"]["version"]}`.

## User-workflow acceptance

| result | workflow | case | ngspice evaluations | cache hits |
|---|---|---|---:|---:|
{rows}

The suite passed: **{str(payload["passed"]).upper()}**.  PASS means the expected
electrical result and its replay evidence were reproduced.  An intentionally
rejected design passes this suite only when its named limits and condition
coverage are reproduced without a solver failure.

## Release decision

| scope | decision |
|---|---|
| circuit analysis and deterministic optimization foundation | **{decisions["circuit_foundation"]}** |
| ML candidate proposal | **{decisions["ml_candidate_proposal"]}** |
| full originally requested scope | **{decisions["full_requested_scope"]}** |
| chamber/plasma process qualification | **OUT OF SCOPE** |

## Fixed ML ranking evidence

| case | evaluation saving | gate |
|---|---:|---|
{ml_rows}

{ml_note} No result here qualifies plasma chemistry, thermal lifetime, process
yield, or a self-consistent plasma/circuit model.
"""


def run_release_suite(run_root: Path, ml_ranking_result: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"release run root already exists: {run_root}")
    log_dir = run_root / "logs"
    solver = _json_command("solver_diagnose", ["solver-diagnose"], log_dir)
    cases = [_run_case(spec, run_root, log_dir) for spec in CASES]
    ml = _ml_evidence(ml_ranking_result)
    payload = {
        "schema": "pcd.release_closure.v1",
        "platform_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver": solver,
        **_release_status(cases, ml),
        **_release_totals(cases),
        "cases": cases,
        "ml_ranking_evidence": ml,
        "scope_boundary": (
            "Electrical circuit analysis and deterministic sizing only; no chamber process qualification, "
            "plasma chemistry, thermal lifetime, or self-consistent plasma/circuit claim."
        ),
    }
    write_json(run_root / "release_result.json", payload)
    atomic_write_text(run_root / "REPORT.md", _render_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/release_acceptance")
    parser.add_argument("--ml-ranking-result", help="P3 ranking_evaluation.json used for the separate ML decision")
    args = parser.parse_args(argv)
    payload = run_release_suite(
        Path(args.run_root),
        Path(args.ml_ranking_result) if args.ml_ranking_result else None,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
