"""Run literature evidence reproduction separately from platform design challenges."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pcd import __version__

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _payload_passed(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if "benchmark_integrity_passed" in payload:
        return bool(payload["benchmark_integrity_passed"])
    if "passed" in payload:
        return bool(payload["passed"])
    if isinstance(payload.get("summary"), dict) and "passed" in payload["summary"]:
        return bool(payload["summary"]["passed"])
    return str(payload.get("status", "")).upper() == "PASS"


def _execute(label: str, command: list[str], result_path: Path, log_dir: Path) -> dict[str, Any]:
    result_path.unlink(missing_ok=True)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.log").write_text(
        f"exit_code={completed.returncode}\n\nSTDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    payload: dict[str, Any] | None = None
    error: str | None = None
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            error = f"invalid result file: {exc}"
    else:
        error = "result file was not produced"
    return {
        "label": label,
        "command": command,
        "exit_code": completed.returncode,
        "result_path": str(result_path.resolve()),
        "payload": payload,
        "execution_error": error,
        "passed": completed.returncode == 0 and error is None and _payload_passed(payload),
    }


def _relative_complex(actual: dict[str, float], expected: dict[str, float]) -> float:
    left = complex(float(actual["real"]), float(actual["imag"]))
    right = complex(float(expected["real"]), float(expected["imag"]))
    return abs(left - right) / max(abs(right), 1e-300)


def _by_label(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["label"]): record for record in records}


def _source_fidelity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = _by_label(records)
    rows: list[dict[str, Any]] = []

    lee = by_label["lee2021_bias_reference_planes"]
    lee_payload = lee["payload"] or {}
    lee_source = lee_payload.get("source_fidelity", {})
    table = lee_source.get("table_i_closure", {})
    oracle = table.get("independent_oracle_ohm", {})
    rows.append(
        {
            "id": "lee2021_bias_table_i",
            "passed": lee["execution_error"] is None and bool(lee_source.get("passed")),
            "evidence_class": "published numeric table + circuit equation",
            "observed": (
                f"corrected terminal Z={float(oracle.get('resistance', 0.0)):.6g}"
                f"{float(oracle.get('reactance', 0.0)):+.6g}j ohm"
            ),
            "establishes": "Table-I topology arithmetic only",
            "result": lee["result_path"],
        }
    )

    hargis = by_label["hargis1994_all32_source"]
    hargis_payload = hargis["payload"] or {}
    ranges = hargis_payload.get("ranges", {})
    counts = hargis_payload.get("counts", {})
    rows.append(
        {
            "id": "hargis1994_tables_iii_iv",
            "passed": bool(hargis["passed"]),
            "evidence_class": "published V/I/phase tables",
            "observed": (
                f"{counts.get('derived_rows', 0)}/32 central rows; "
                f"8-row 66 Pa view; 32-row reported-spread view; "
                f"R={float(ranges.get('resistance_ohm', {}).get('min', 0.0)):.3f}.."
                f"{float(ranges.get('resistance_ohm', {}).get('max', 0.0)):.3f} ohm"
            ),
            "establishes": "immutable table transcription, derived views, power closure and one-port replay",
            "result": hargis["result_path"],
        }
    )

    fixture_record = by_label["colpo1999_fixture"]
    fixture = fixture_record["payload"] or {}
    resonances = fixture.get("resonances", [])
    graphite_error = float(fixture.get("graphite_dummy_spice", {}).get("relative_complex_error", 0.0))
    rows.append(
        {
            "id": "colpo1999_fixture_and_graphite",
            "passed": bool(fixture_record["passed"]),
            "evidence_class": "published rounded fixture values + known dummy point",
            "observed": (
                "resonances="
                + ", ".join(f"{float(item['spice_Hz']) / 1e6:.3f} MHz" for item in resonances)
                + f"; graphite replay error={graphite_error:.3g}"
            ),
            "establishes": "fixture resonance and global-terminal dummy replay",
            "result": fixture_record["result_path"],
        }
    )

    colpo_record = by_label["colpo1999_digitized_centers"]
    colpo = colpo_record["payload"] or {}
    width = colpo.get("digitization_uncertainty_half_width_ohm", {})
    rows.append(
        {
            "id": "colpo1999_digitized_centers",
            "passed": bool(colpo_record["passed"]),
            "evidence_class": "two-reader plot transcription",
            "observed": (
                f"{colpo.get('point_count', 0)} paired centers; "
                f"digitization widths R +/-{float(width.get('resistance', 0.0)):.0f}, "
                f"X +/-{float(width.get('reactance', 0.0)):.0f} ohm"
            ),
            "establishes": "paired global RFZ60-plane center points; not measurement uncertainty",
            "result": colpo_record["result_path"],
        }
    )

    corner_record = by_label["colpo1999_digitization_stability"]
    corner_payload = corner_record["payload"] or {}
    corner_source = corner_payload.get("source_fidelity", {})
    rows.append(
        {
            "id": "colpo1999_digitization_derivation",
            "passed": corner_record["execution_error"] is None and bool(corner_source.get("passed")),
            "evidence_class": "deterministic derivation from digitized centers",
            "observed": (
                f"{corner_payload.get('central_point_count', 0)} parents -> "
                f"{corner_payload.get('corner_point_count', 0)} reading corners"
            ),
            "establishes": "exact corner derivation; corners are not independent operating observations",
            "result": corner_record["result_path"],
        }
    )
    return rows


def _model_conformance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record = _by_label(records)["lee2020_icp_equations"]
    payload = record["payload"] or {}
    cases = payload.get("cases", [])
    z_errors = [
        _relative_complex(case["ngspice_impedance_ohm"], case["oracle_impedance_ohm"])
        for case in cases
        if case.get("ngspice_impedance_ohm")
    ]
    power_errors = [
        abs(float(case["ngspice_absorbed_power_W"]) - float(case["absorbed_power_eq19_W"]))
        / max(abs(float(case["absorbed_power_eq19_W"])), 1e-300)
        for case in cases
        if case.get("ngspice_absorbed_power_W") is not None
    ]
    return [
        {
            "id": "lee2020_icp_transformer_equations",
            "passed": bool(record["passed"]),
            "evidence_class": "paper equation conformance; synthetic damping regimes",
            "observed": (
                f"{payload.get('summary', {}).get('passed_case_count', 0)}/"
                f"{payload.get('summary', {}).get('case_count', 0)} regimes; "
                f"max Z error={max(z_errors, default=0.0):.3g}; max power error={max(power_errors, default=0.0):.3g}"
            ),
            "establishes": "terminal transformer algebra, not experimental ICP validation",
            "result": record["result_path"],
        }
    ]


def _design_row(
    case: dict[str, Any], *, row_id: str, source: str, scope: str, regression_passed: bool | None = None
) -> dict[str, Any]:
    scenario_count = len(case.get("scenarios", [])) or int(case.get("scenario_count", 0))
    feasible_fraction = float(case["feasible_fraction"])
    feasible_count = int(case.get("feasible_scenario_count", round(feasible_fraction * scenario_count)))
    return {
        "id": row_id,
        "source": source,
        "scope": scope,
        "regression_passed": bool(case.get("passed", True)) if regression_passed is None else regression_passed,
        "design_feasible": bool(case["feasible"]),
        "feasible_count": feasible_count,
        "scenario_count": scenario_count,
        "worst_reflection_magnitude": float(case["worst_reflection_magnitude"]),
        "control_margin": case.get("control_margin", case.get("minimum_control_margin")),
        "n_evaluations": int(case.get("n_evaluations", 0)),
    }


def _design_challenges(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = _by_label(records)
    rows: list[dict[str, Any]] = []

    lee = by_label["lee2021_bias_reference_planes"]
    lee_design = (lee["payload"] or {}).get("platform_design_challenge", {})
    for outcome in lee_design.get("outcomes", []):
        rows.append(
            {
                "id": str(outcome["case_id"]),
                "source": "Lee 2021 bias-path Table I",
                "scope": f"frozen matcher fixture; substituted plane={outcome['reference_plane']}",
                "regression_passed": bool(outcome["passed"]),
                "design_feasible": bool(outcome["feasible"]),
                "feasible_count": int(bool(outcome["feasible"])),
                "scenario_count": 1,
                "worst_reflection_magnitude": float(outcome["spice_reflection_magnitude"]),
                "control_margin": None,
                "n_evaluations": 1,
            }
        )

    gec = by_label["hargis1994_control_authority"]
    for case in (gec["payload"] or {}).get("cases", []):
        benchmark_id = str(case["benchmark_id"])
        scope = "66 Pa reported apparatus spread" if "spread" in benchmark_id.lower() else "66 Pa central conditions"
        rows.append(_design_row(case, row_id=benchmark_id, source="Hargis 1994", scope=scope))

    phase = by_label["hargis_sobolewski_phase_sensitivity"]
    phase_challenge = (phase["payload"] or {}).get("platform_design_challenge", {})
    for case in phase_challenge.get("outcomes", []):
        if str(case["benchmark_id"]) == "P1_CCP_phase_baseline":
            continue
        rows.append(
            _design_row(
                case,
                row_id=str(case["benchmark_id"]),
                source="Hargis 1994 + Sobolewski 1995",
                scope="66 Pa common-mode model sensitivity; not a confidence interval",
            )
        )

    hardware = by_label["hargis_hardware_families"]
    for family in (hardware["payload"] or {}).get("families", []):
        for candidate in family["candidates"]:
            rows.append(
                _design_row(
                    candidate,
                    row_id=f"{family['family_id']}__L1_{1e6 * float(candidate['L1_H']):.1f}uH",
                    source="Hargis 1994 / Sobolewski 1995",
                    scope=f"independent {family['kind']} family; no cross-family weighting or global ranking",
                    regression_passed=bool(family["regression_passed"]),
                )
            )

    colpo = by_label["colpo1999_digitization_stability"]
    challenge = (colpo["payload"] or {}).get("platform_design_challenge", {})
    for outcome in challenge.get("outcomes", []):
        stability = outcome["decision_stability"]
        row = _design_row(
            outcome,
            row_id=str(outcome["case_id"]),
            source="Colpo 1999 digitized centers",
            scope=(
                f"{stability['robust_parent_count']}/{stability['parent_condition_count']} parent conditions "
                f"robust to four reading corners; {stability['mixed_parent_count']} mixed"
            ),
            regression_passed=bool(challenge.get("regression_passed")),
        )
        row["parent_condition_count"] = int(stability["parent_condition_count"])
        row["robust_parent_count"] = int(stability["robust_parent_count"])
        rows.append(row)
    return rows


def _reference_inventory() -> list[dict[str, str]]:
    return [
        {
            "source": "Lee 2021 Figures 4-7",
            "status": "reference_only",
            "reason": "plot-only correlation data are weaker than the exact Table-I electrical closure",
        },
        {
            "source": "Cao 2020 commercial planar ICP",
            "status": "reference_only",
            "reason": "no phase-resolved complex terminal impedance is published",
        },
        {
            "source": "Metze 1986 / Saikia 2018",
            "status": "reference_only",
            "reason": "nonlinear sheath, self-bias and time-domain plasma state are outside the one-port AC responsibility",
        },
        {
            "source": "Howling / Guittienne planar ICP",
            "status": "reference_only",
            "reason": "distributed antenna coupling and mode spectra are outside the current lumped one-port model",
        },
        {
            "source": "Qu 2020 pulsed ICP",
            "status": "reference_only",
            "reason": "time-varying pulsed matching is outside the steady-state AC study engine",
        },
    ]


def _render_report(payload: dict[str, Any]) -> str:
    source_rows = "\n".join(
        f"| {'PASS' if item['passed'] else '**FAIL**'} | {item['id']} | {item['evidence_class']} | "
        f"{item['observed']} | {item['establishes']} |"
        for item in payload["source_fidelity"]
    )
    model_rows = "\n".join(
        f"| {'PASS' if item['passed'] else '**FAIL**'} | {item['id']} | {item['observed']} | {item['establishes']} |"
        for item in payload["model_conformance"]
    )
    design_rows = "\n".join(
        f"| {'PASS' if item['regression_passed'] else '**FAIL**'} | {item['id']} | "
        f"{'yes' if item['design_feasible'] else 'no'} | {item['feasible_count']}/{item['scenario_count']} | "
        f"{item['worst_reflection_magnitude']:.4f} | {item['scope']} |"
        for item in payload["design_challenges"]
    )
    inventory_rows = "\n".join(
        f"| {item['source']} | {item['status']} | {item['reason']} |" for item in payload["reference_inventory"]
    )
    return f"""<!-- generated by bench/literature/run_suite.py -->
# PCD literature benchmark evaluation

Overall benchmark integrity: **{"PASS" if payload["passed"] else "FAIL"}**.
PASS means that source reproduction, equation conformance, and expected
platform decision classifications were reproduced. A design outcome of "no"
is not a benchmark failure.

## Source fidelity

| result | evidence | evidence class | observed | establishes |
|---|---|---|---|---|
{source_rows}

## Model equation conformance

| result | model | observed | establishes |
|---|---|---|---|
{model_rows}

## Platform design challenges

| regression | challenge | design feasible | covered rows | worst `|Gamma|` | scope |
|---|---|---:|---:|---:|---|
{design_rows}

The Hargis central operating conditions, reported apparatus spread, and two
phase-model sensitivities remain separate result families. No 80-row score,
67/80 winner, or hidden family weighting is produced. Colpo's 60 reading
corners are reported as robustness outcomes within 15 parent pressure-power
conditions, not as 60 independent operating scenarios.

## Reference-only literature

| source | status | reason |
|---|---|---|
{inventory_rows}

These entries are intentionally not executable goldens and do not count as
unfinished work.

## Remaining qualification work

- No specific production chamber is qualified because no target apparatus is
  declared. Add apparatus qualification only when a calibrated, held-out
  complex-Z dataset at one named reference plane is available.
- Component voltage, current, ESR/Q and thermal acceptance still require the
  actual matcher ratings and generator-plane drive definition.
- Nonlinear sheath, distributed antenna and pulsed matching models should be
  added only if the platform responsibility is explicitly expanded.

The public case schema is unchanged. Literature-specific provenance and
derived sensitivity construction stay inside `bench/literature`.
"""


def run(run_root: Path, *, include_core: bool) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    python = sys.executable
    specs: list[tuple[str, list[str], Path]] = [
        (
            "lee2021_bias_reference_planes",
            [
                python,
                str(HERE / "p0_lee2021_bias" / "run.py"),
                "--run-root",
                str(run_root / "lee2021_bias_runs"),
                "--output",
                str(run_root / "lee2021_bias.json"),
            ],
            run_root / "lee2021_bias.json",
        ),
        (
            "hargis1994_all32_source",
            [
                python,
                str(HERE / "p1_gec_ccp" / "run_all32_benchmark.py"),
                "--require-ngspice",
                "--output",
                str(run_root / "hargis1994_all32.json"),
            ],
            run_root / "hargis1994_all32.json",
        ),
        (
            "hargis1994_control_authority",
            [
                python,
                str(HERE / "p1_gec_ccp" / "run_design_cases.py"),
                "--run-root",
                str(run_root / "hargis_control_authority"),
            ],
            run_root / "hargis_control_authority" / "design_evaluation.json",
        ),
        (
            "hargis_sobolewski_phase_sensitivity",
            [
                python,
                str(HERE / "p1_gec_ccp" / "run_phase_sensitivity.py"),
                "--run-root",
                str(run_root / "hargis_phase_sensitivity"),
            ],
            run_root / "hargis_phase_sensitivity" / "evaluation.json",
        ),
        (
            "hargis_hardware_families",
            [
                python,
                str(HERE / "p1_gec_ccp" / "run_hardware_family_comparison.py"),
                "--run-root",
                str(run_root / "hargis_hardware_families"),
            ],
            run_root / "hargis_hardware_families" / "evaluation.json",
        ),
        (
            "lee2020_icp_equations",
            [
                python,
                str(HERE / "p1_lee2020_icp" / "run_benchmark.py"),
                "--require-ngspice",
                "--output",
                str(run_root / "lee2020_icp.json"),
            ],
            run_root / "lee2020_icp.json",
        ),
        (
            "colpo1999_fixture",
            [
                python,
                str(HERE / "p1_colpo1999_icp" / "run.py"),
                "--run-root",
                str(run_root / "colpo1999_fixture"),
            ],
            run_root / "colpo1999_fixture" / "colpo1999_evaluation.json",
        ),
        (
            "colpo1999_digitized_centers",
            [
                python,
                str(HERE / "p1_colpo1999_icp" / "digitized" / "run.py"),
                "--run-root",
                str(run_root / "colpo1999_centers"),
                "--output",
                str(run_root / "colpo1999_centers" / "evaluation.json"),
            ],
            run_root / "colpo1999_centers" / "evaluation.json",
        ),
        (
            "colpo1999_digitization_stability",
            [
                python,
                str(HERE / "p1_colpo1999_icp" / "digitized" / "run_uncertainty_challenge.py"),
                "--run-root",
                str(run_root / "colpo1999_digitization_stability"),
            ],
            run_root / "colpo1999_digitization_stability" / "evaluation.json",
        ),
    ]
    if include_core:
        specs.append(
            (
                "core_decision_regression",
                [python, str(ROOT / "bench" / "run_suite.py"), "--run-root", str(run_root / "core_regression")],
                run_root / "core_regression" / "benchmark_result.json",
            )
        )

    records = [_execute(label, command, result, log_dir) for label, command, result in specs]
    source_fidelity = _source_fidelity(records)
    model_conformance = _model_conformance(records)
    design_challenges = _design_challenges(records)
    execution_regressions = [
        {
            "id": record["label"],
            "passed": bool(record["passed"]),
            "exit_code": int(record["exit_code"]),
            "result": record["result_path"],
            "error": record["execution_error"],
        }
        for record in records
    ]
    passed = (
        all(item["passed"] for item in execution_regressions)
        and all(item["passed"] for item in source_fidelity)
        and all(item["passed"] for item in model_conformance)
        and all(item["regression_passed"] for item in design_challenges)
    )
    result: dict[str, Any] = {
        "schema": "pcd.literature_benchmark_suite.v4",
        "platform_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "benchmark_integrity_passed": passed,
        "pass_meaning": "source/equation reproduction and expected design classifications, not apparatus qualification",
        "source_fidelity": source_fidelity,
        "model_conformance": model_conformance,
        "design_challenges": design_challenges,
        "reference_inventory": _reference_inventory(),
        "execution_regressions": execution_regressions,
        "core_regression_included": include_core,
    }
    (run_root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_root / "REPORT.md").write_text(_render_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/suite"))
    parser.add_argument("--skip-core", action="store_true", help="omit the synthetic core regression")
    args = parser.parse_args()
    payload = run(args.run_root.resolve(), include_core=not args.skip_core)
    print(json.dumps(payload, indent=2))
    return 0 if payload["benchmark_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
