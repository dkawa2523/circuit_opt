"""Run matching decisions against the Hargis 1994 66 Pa CCP load window."""

from __future__ import annotations

import argparse
import json
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

EXPECTATIONS = yaml.safe_load((HERE / "design_expectations.yaml").read_text(encoding="utf-8"))
CASE_PATHS = tuple(HERE / name for name in EXPECTATIONS)


def run(run_root: Path, solver: str) -> dict[str, Any]:
    cases = [run_case(path, run_root, solver, EXPECTATIONS) for path in CASE_PATHS]
    payload: dict[str, Any] = {
        "schema": "literature_design_benchmark.v1",
        "benchmark": "p1_gec_ccp_matching_decisions",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "doi": "https://doi.org/10.1063/1.1144770",
            "tables": ["III", "IV"],
            "drive_frequency_Hz": 13_560_000,
            "pressure_Pa": 66,
        },
        "acceptance": {
            "reflected_power_fraction_max": 0.10,
            "origin": "platform engineering criterion, not a paper acceptance limit",
        },
        "passed": all(bool(case["passed"]) for case in cases),
        "n_evaluations": sum(int(case["n_evaluations"]) for case in cases),
        "cases": cases,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "design_evaluation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/gec_ccp_design"))
    parser.add_argument("--solver", default="ngspice_cli")
    args = parser.parse_args()
    report = run(args.run_root.resolve(), str(args.solver))
    summary = {
        "benchmark": report["benchmark"],
        "passed": report["passed"],
        "n_evaluations": report["n_evaluations"],
        "cases": [
            {
                "id": case["benchmark_id"],
                "passed": case["passed"],
                "feasible": case["feasible"],
                "feasible_fraction": case["feasible_fraction"],
                "worst_reflection_magnitude": case["worst_reflection_magnitude"],
                "infeasible_scenarios": case["infeasible_scenarios"],
            }
            for case in report["cases"]
        ],
        "result": str(args.run_root.resolve() / "design_evaluation.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
