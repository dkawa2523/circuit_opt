"""Behaviour of the concise benchmark-report projection."""

from __future__ import annotations

import json

from bench.reports.generate_case_report import build_report, write_report


def test_report_separates_reproduction_from_design_feasibility(tmp_path):
    core = {
        "schema": "design_benchmark_suite.v1",
        "platform_version": "test",
        "generated_at": "2026-01-01T00:00:00Z",
        "solver": "ngspice_cli",
        "passed": True,
        "n_candidates": 1,
        "n_evaluations": 2,
        "cases": [
            {
                "benchmark_id": "negative_control",
                "role": "negative_control",
                "passed": True,
                "feasible": False,
                "success_fraction": 1.0,
                "worst_reflection_magnitude": 0.5,
                "n_candidates": 1,
                "n_evaluations": 2,
                "n_failed_evaluations": 0,
                "violated_constraints": {"outside_window": ["max_reflection_magnitude"]},
                "best_candidate_id": "fixed",
                "best_candidate_values": {"L1": 1e-6},
                "scenarios": [
                    {"scenario_id": "nominal", "feasible": True},
                    {"scenario_id": "outside_window", "feasible": False},
                ],
            }
        ],
    }
    literature = {
        "schema": "pcd.literature_benchmark_suite.v4",
        "platform_version": "test",
        "generated_at": "2026-01-01T00:00:00Z",
        "passed": True,
        "benchmark_integrity_passed": True,
        "pass_meaning": "reproduction, not apparatus qualification",
        "source_fidelity": [{"id": "source", "passed": True}],
        "model_conformance": [],
        "design_challenges": [
            {
                "id": "design",
                "regression_passed": True,
                "design_feasible": False,
                "feasible_count": 1,
                "scenario_count": 2,
            }
        ],
        "reference_inventory": [{"source": "future data", "status": "reference_only"}],
        "execution_regressions": [],
    }
    core_path = tmp_path / "core.json"
    literature_path = tmp_path / "literature.json"
    core_path.write_text(json.dumps(core), encoding="utf-8")
    literature_path.write_text(json.dumps(literature), encoding="utf-8")

    report = build_report(core, literature, core_path=core_path, literature_path=literature_path)
    paths = write_report(report, tmp_path / "report")

    assert report["summary"]["overall_passed"] is True
    assert report["summary"]["counts"]["core_designs_feasible"] == 0
    assert report["core_rows"][0]["benchmark_passed"] is True
    assert report["core_rows"][0]["design_feasible"] is False
    assert report["core_rows"][0]["failed_constraints"] == "max_reflection_magnitude"
    assert all(path.is_file() for path in paths.values())
    markdown = paths["report"].read_text(encoding="utf-8")
    assert "negative_control | negative_control | yes | no | 1/2" in markdown
    assert "plasma chemistry" in markdown
