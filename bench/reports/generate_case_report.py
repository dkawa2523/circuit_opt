"""Build a concise benchmark report from the two completed suite summaries.

The source suites own electrical checks and literature reproduction. This
module only projects their committed results into one human-readable report
and two flat audit tables; it does not reopen solver artifacts or reinterpret
engineering acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE = ROOT / "runs" / "benchmark_suite" / "benchmark_result.json"
DEFAULT_LITERATURE = ROOT / "runs" / "literature" / "final_evaluation" / "evaluation.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "reports"

SUMMARY_NAME = "benchmark-summary.json"
REPORT_NAME = "benchmark-report.md"
CORE_TABLE_NAME = "core-cases.csv"
LITERATURE_TABLE_NAME = "literature-evidence.csv"

CORE_FIELDS = (
    "benchmark_id",
    "role",
    "benchmark_passed",
    "design_feasible",
    "feasible_conditions",
    "scenario_count",
    "success_fraction",
    "worst_reflection_magnitude",
    "control_margin",
    "n_candidates",
    "n_evaluations",
    "n_failed_evaluations",
    "failed_conditions",
    "failed_constraints",
    "best_candidate_id",
    "best_candidate_values",
    "question",
    "demonstrates",
    "does_not_establish",
    "case_path",
)

LITERATURE_FIELDS = (
    "category",
    "id",
    "reproduction_passed",
    "design_feasible",
    "feasible_conditions",
    "scenario_count",
    "worst_reflection_magnitude",
    "n_evaluations",
    "evidence_class",
    "source",
    "scope",
    "observed",
    "establishes",
    "reason",
    "result",
)

LITERATURE_CATEGORIES = (
    "source_fidelity",
    "model_conformance",
    "design_challenges",
    "reference_inventory",
    "execution_regressions",
)

MODEL_BOUNDARY = (
    "These results establish circuit implementation, equation/data reproduction, and declared electrical "
    "design decisions only. They do not qualify plasma chemistry, chamber process windows, thermal lifetime, "
    "apparatus yield, or self-consistent plasma-circuit dynamics."
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark result must be a JSON object: {path}")
    return payload


def _records(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    values = payload.get(name)
    if not isinstance(values, list):
        raise ValueError(f"benchmark result field {name!r} must be a list")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"benchmark result field {name!r} must contain objects")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _source_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "schema": payload.get("schema"),
        "platform_version": payload.get("platform_version"),
        "generated_at": payload.get("generated_at"),
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _failed_constraints(case: dict[str, Any]) -> str:
    declared = case.get("violated_constraints") or {}
    if not isinstance(declared, dict):
        return ""
    names = {str(name) for values in declared.values() if isinstance(values, list) for name in values}
    return ";".join(sorted(names))


def core_rows(core: dict[str, Any]) -> list[dict[str, Any]]:
    """Project each core benchmark into one decision-oriented row."""

    rows: list[dict[str, Any]] = []
    for case in _records(core, "cases"):
        scenarios = case.get("scenarios") or []
        if not isinstance(scenarios, list):
            raise ValueError("core benchmark scenarios must be a list")
        failed = [str(row.get("scenario_id", "")) for row in scenarios if not bool(row.get("feasible"))]
        rows.append(
            {
                "benchmark_id": case.get("benchmark_id"),
                "role": case.get("role"),
                "benchmark_passed": bool(case.get("passed")),
                "design_feasible": bool(case.get("feasible")),
                "feasible_conditions": sum(bool(row.get("feasible")) for row in scenarios),
                "scenario_count": len(scenarios),
                "success_fraction": case.get("success_fraction"),
                "worst_reflection_magnitude": case.get("worst_reflection_magnitude"),
                "control_margin": case.get("control_margin"),
                "n_candidates": case.get("n_candidates"),
                "n_evaluations": case.get("n_evaluations"),
                "n_failed_evaluations": case.get("n_failed_evaluations"),
                "failed_conditions": ";".join(failed),
                "failed_constraints": _failed_constraints(case),
                "best_candidate_id": case.get("best_candidate_id"),
                "best_candidate_values": _json_cell(case.get("best_candidate_values")),
                "question": case.get("question"),
                "demonstrates": case.get("demonstrates"),
                "does_not_establish": case.get("does_not_establish"),
                "case_path": case.get("case_path"),
            }
        )
    return rows


def _literature_status(category: str, row: dict[str, Any]) -> bool | None:
    if category == "reference_inventory":
        return None
    if "regression_passed" in row:
        return bool(row["regression_passed"])
    if "passed" in row:
        return bool(row["passed"])
    return None


def literature_rows(literature: dict[str, Any]) -> list[dict[str, Any]]:
    """Project literature evidence without merging reproduction and feasibility."""

    rows: list[dict[str, Any]] = []
    for category in LITERATURE_CATEGORIES:
        for row in _records(literature, category):
            rows.append(
                {
                    "category": category,
                    "id": row.get("id") or row.get("label") or row.get("source"),
                    "reproduction_passed": _literature_status(category, row),
                    "design_feasible": row.get("design_feasible"),
                    "feasible_conditions": row.get("feasible_count"),
                    "scenario_count": row.get("scenario_count"),
                    "worst_reflection_magnitude": row.get("worst_reflection_magnitude"),
                    "n_evaluations": row.get("n_evaluations"),
                    "evidence_class": row.get("evidence_class"),
                    "source": row.get("source"),
                    "scope": row.get("scope"),
                    "observed": row.get("observed"),
                    "establishes": row.get("establishes"),
                    "reason": row.get("reason"),
                    "result": row.get("result"),
                }
            )
    return rows


def build_report(
    core: dict[str, Any],
    literature: dict[str, Any],
    *,
    core_path: Path,
    literature_path: Path,
) -> dict[str, Any]:
    """Build one small report model from already interpreted suite outputs."""

    core_table = core_rows(core)
    literature_table = literature_rows(literature)
    source_artifacts = {
        "core": _source_record(core_path, core),
        "literature": _source_record(literature_path, literature),
    }
    snapshot_id = hashlib.sha256(
        (source_artifacts["core"]["sha256"] + source_artifacts["literature"]["sha256"]).encode()
    ).hexdigest()[:16]
    core_failures = [str(row["benchmark_id"]) for row in core_table if not row["benchmark_passed"]]
    literature_failures = [
        str(row["id"])
        for row in literature_table
        if row["reproduction_passed"] is False and row["category"] != "execution_regressions"
    ]
    overall_passed = (
        bool(core.get("passed"))
        and bool(literature.get("passed"))
        and bool(literature.get("benchmark_integrity_passed"))
        and not core_failures
        and not literature_failures
    )
    return {
        "summary": {
            "schema": "benchmark_report.v1",
            "snapshot_id": snapshot_id,
            "overall_passed": overall_passed,
            "core_benchmark_passed": bool(core.get("passed")),
            "literature_benchmark_passed": bool(literature.get("passed")),
            "literature_integrity_passed": bool(literature.get("benchmark_integrity_passed")),
            "solver": core.get("solver"),
            "counts": {
                "core_cases": len(core_table),
                "core_cases_reproduced": sum(bool(row["benchmark_passed"]) for row in core_table),
                "core_designs_feasible": sum(bool(row["design_feasible"]) for row in core_table),
                "core_candidates": core.get("n_candidates"),
                "core_evaluations": core.get("n_evaluations"),
                "literature_source_records": len(_records(literature, "source_fidelity")),
                "literature_model_records": len(_records(literature, "model_conformance")),
                "literature_design_challenges": len(_records(literature, "design_challenges")),
                "reference_only_sources": len(_records(literature, "reference_inventory")),
            },
            "failed_core_benchmarks": core_failures,
            "failed_literature_evidence": literature_failures,
            "interpretation": {
                "benchmark_pass": literature.get("pass_meaning"),
                "engineering_feasibility": "reported per row and never inferred from benchmark reproduction",
                "model_boundary": MODEL_BOUNDARY,
            },
            "source_artifacts": source_artifacts,
            "artifacts": {
                "report": REPORT_NAME,
                "core_cases": CORE_TABLE_NAME,
                "literature_evidence": LITERATURE_TABLE_NAME,
            },
        },
        "core_rows": core_table,
        "literature_rows": literature_table,
    }


def _mark(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return "yes" if bool(value) else "no"


def _number(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}g}"


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    counts = summary["counts"]
    lines = [
        "# Circuit benchmark report",
        "",
        f"Snapshot: `{summary['snapshot_id']}`",
        "",
        "## Outcome",
        "",
        "| check | result |",
        "|---|---:|",
        f"| overall benchmark reproduction | {_mark(summary['overall_passed'])} |",
        f"| core suite | {_mark(summary['core_benchmark_passed'])} |",
        f"| literature suite | {_mark(summary['literature_benchmark_passed'])} |",
        f"| literature integrity | {_mark(summary['literature_integrity_passed'])} |",
        f"| core cases reproduced | {counts['core_cases_reproduced']}/{counts['core_cases']} |",
        f"| core electrical evaluations | {counts['core_evaluations']} |",
        "",
        "Benchmark reproduction and engineering feasibility are separate columns below. An intentionally infeasible",
        "electrical design can be a successful negative-control benchmark.",
        "",
        "## Core circuit cases",
        "",
        "| ID | role | reproduced | design feasible | condition coverage | evaluations | worst Γ magnitude | failed limits |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["core_rows"]:
        lines.append(
            "| {benchmark_id} | {role} | {reproduced} | {feasible} | {covered}/{total} | {evaluations} | "
            "{gamma} | {limits} |".format(
                benchmark_id=row["benchmark_id"],
                role=row["role"],
                reproduced=_mark(row["benchmark_passed"]),
                feasible=_mark(row["design_feasible"]),
                covered=row["feasible_conditions"],
                total=row["scenario_count"],
                evaluations=row["n_evaluations"],
                gamma=_number(row["worst_reflection_magnitude"]),
                limits=row["failed_constraints"] or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Literature-derived evidence",
            "",
            "| category | ID/source | reproduced | design feasible | condition coverage | worst Γ magnitude |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["literature_rows"]:
        if row["category"] == "execution_regressions":
            continue
        coverage = "n/a"
        if row["scenario_count"] not in (None, ""):
            coverage = f"{row['feasible_conditions']}/{row['scenario_count']}"
        lines.append(
            "| {category} | {id} | {reproduced} | {feasible} | {coverage} | {gamma} |".format(
                category=row["category"],
                id=str(row["id"]).replace("|", "/"),
                reproduced=_mark(row["reproduction_passed"]),
                feasible=_mark(row["design_feasible"]),
                coverage=coverage,
                gamma=_number(row["worst_reflection_magnitude"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            MODEL_BOUNDARY,
            "",
            "## Source artifacts",
            "",
            "| source | schema | SHA-256 | path |",
            "|---|---|---|---|",
        ]
    )
    for name, source in summary["source_artifacts"].items():
        lines.append(f"| {name} | {source['schema']} | `{source['sha256']}` | `{source['path']}` |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / SUMMARY_NAME,
        "report": output_dir / REPORT_NAME,
        "core_cases": output_dir / CORE_TABLE_NAME,
        "literature_evidence": output_dir / LITERATURE_TABLE_NAME,
    }
    paths["summary"].write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(_markdown(report), encoding="utf-8")
    _write_csv(paths["core_cases"], CORE_FIELDS, report["core_rows"])
    _write_csv(paths["literature_evidence"], LITERATURE_FIELDS, report["literature_rows"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-result", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--literature-result", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    core = _load(args.core_result)
    literature = _load(args.literature_result)
    report = build_report(core, literature, core_path=args.core_result, literature_path=args.literature_result)
    paths = write_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "overall_passed": report["summary"]["overall_passed"],
                "snapshot_id": report["summary"]["snapshot_id"],
                "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
