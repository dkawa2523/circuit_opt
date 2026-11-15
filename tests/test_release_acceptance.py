"""Release closure verifies only the supported circuit workflows."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from bench.release import run_suite


def test_passing_circuit_workflows_authorize_the_circuit_foundation(tmp_path, monkeypatch):
    fake_case = {
        "case_id": "circuit",
        "workflow": "circuit workflow",
        "case_path": "case.yaml",
        "mode": "study",
        "passed": True,
        "checks": {"evidence": True},
        "observed": {},
        "ngspice_evaluations": 1,
        "cache_hits": 0,
        "evidence": {},
    }
    monkeypatch.setattr(run_suite, "CASES", (object(),))
    monkeypatch.setattr(run_suite, "_run_case", lambda *_args: fake_case)
    monkeypatch.setattr(
        run_suite,
        "_json_command",
        lambda *_args: {"batch_runnable": True, "version": "ngspice-test"},
    )

    result = run_suite.run_release_suite(tmp_path / "release")

    assert result["passed"] is True
    assert result["schema"] == "pcd.release_closure.v2"
    assert result["release_decision"] == "GO"
    assert (tmp_path / "release" / "release_result.json").is_file()
    report = (tmp_path / "release" / "REPORT.md").read_text(encoding="utf-8")
    assert "| chamber/plasma process qualification | **OUT OF SCOPE** |" in report


def test_release_root_must_be_new(tmp_path):
    run_root = tmp_path / "existing"
    run_root.mkdir()

    with pytest.raises(FileExistsError, match="release run root already exists"):
        run_suite.run_release_suite(run_root)


def test_cache_hits_parse_boolean_csv_values():
    assert run_suite._cache_hits([{"from_cache": "True"}, {"from_cache": "0"}, {"from_cache": "yes"}]) == 2


@pytest.mark.parametrize("termination", ["declared_grid_completed", "candidate_budget_exhausted"])
def test_rejected_design_requires_complete_grid_and_evaluation_evidence(tmp_path, termination):
    artifacts = {name: name for name in ("best_candidate", "history", "evaluation_table")}
    scenario_ids = ("nominal", "edge_a", "edge_b")
    history = [{"trial": trial, "candidate_id": f"trial_{trial}", "evaluation_count": 3} for trial in range(27)]
    rows = []
    for item in history:
        for scenario in scenario_ids:
            manifest = f"manifest_{item['trial']}_{scenario}.json"
            (tmp_path / manifest).write_text("{}", encoding="utf-8")
            rows.append(
                {
                    "trial": str(item["trial"]),
                    "candidate_id": item["candidate_id"],
                    "scenario_id": scenario,
                    "control": "{}",
                    "selected_control": "True",
                    "from_cache": "False",
                    "status": "ok",
                    "artifact.manifest": manifest,
                }
            )
    (tmp_path / "history").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "evaluation_table").write_text("evidence", encoding="utf-8")
    best = {"candidate_id": "trial_0"}
    replays: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = {"scenario_id": scenario_id}
        manifest = f"replay_{scenario_id}.json"
        (tmp_path / manifest).write_text("{}", encoding="utf-8")
        replay = {
            "request": {"candidate": best, "scenario": scenario, "control": {"values": {}}},
            "raw": {"status": "ok", "artifacts": {"manifest": manifest}},
            "from_cache": False,
        }
        replays.append({"scenario": scenario, "selected_trial": 0, "trials": [replay]})
    candidate = {"candidate": best, "scenarios": replays}
    best_path = tmp_path / "best_candidate"
    best_path.write_text(json.dumps(candidate), encoding="utf-8")
    study = {
        "run_root": str(tmp_path),
        "artifacts": artifacts,
        "n_evaluations": 81,
        "n_failed_evaluations": 0,
        "execution": {"optimizer_outcome": {"termination": termination}},
        "verification": {"status": "verified", "cache_reused": False, "n_evaluations": 3, "n_failed_evaluations": 0},
        "study": {"scenarios": [{"scenario_id": name} for name in scenario_ids]},
        "best": {
            "candidate": best,
            "status": "does_not_meet_declared_acceptance",
            "evaluation_evidence": "complete",
            "coverage": {"conditions": 3, "solved": 3, "accepted": 1},
            "conditions": [
                {"scenario_id": "nominal", "failed_constraints": []},
                {"scenario_id": "edge_a", "failed_constraints": [{"name": "max_reflection_magnitude"}]},
                {"scenario_id": "edge_b", "failed_constraints": [{"name": "max_reflection_magnitude"}]},
            ],
        },
    }
    bundle = {"study": study, "evaluation_rows": rows}

    checks, _observed = run_suite._check_declared_rejection(bundle)
    assert all(checks.values()) is (termination == "declared_grid_completed")
    assert checks["complete_evaluation_evidence"]

    study["best"]["evaluation_evidence"] = "incomplete_evidence"
    checks, _observed = run_suite._check_declared_rejection(bundle)
    assert not checks["complete_evaluation_evidence"]

    duplicate = {**rows[0], "artifact.manifest": rows[-1]["artifact.manifest"]}
    checks, _ = run_suite._check_declared_rejection({**bundle, "evaluation_rows": [*rows[:-1], duplicate]})
    assert not checks["evaluation_table_is_complete"]
    assert checks["complete_audit_and_selected_evidence"]

    replays[0]["trials"][0]["raw"]["artifacts"]["manifest"] = rows[0]["artifact.manifest"]
    best_path.write_text(json.dumps(candidate), encoding="utf-8")
    checks, _ = run_suite._check_declared_rejection(bundle)
    assert not checks["selected_candidate_was_replayed_fresh"]


def test_time_varying_resistor_rejects_constant_waveform_with_the_same_probes(tmp_path):
    time = np.linspace(0, 2.5e-6, 251)
    resistance = np.interp(time, [0, 1e-6, 2e-6], [25, 100, 25])
    waveform = tmp_path / "transient.csv"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"test figure")
    bundle = {
        "simulation": {
            "status": "ok",
            "warnings": [],
            "run_dir": str(tmp_path),
            "artifacts": {"waveform": waveform.name, "debug_manifest": manifest.name},
        },
        "analysis": {
            "output_dir": str(tmp_path),
            "artifacts": {"transient_response": figure.name},
            "analyses": {"transient": {"samples": len(time), "periodic": {"status": "not_applicable"}}},
        },
    }
    for profile, expected in ((resistance, True), (np.full_like(time, 25.0), False)):
        current = 10 / (50 + profile)
        np.savetxt(
            waveform,
            np.column_stack([time, current * profile, current]),
            delimiter=",",
            header="time_s,component_Rchamber_voltage_V,component_Rchamber_current_A",
            comments="",
        )
        checks, _ = run_suite._check_time_varying_resistor(bundle)
        assert checks["terminal_response_matches_the_prescribed_resistance"] is expected
        assert all(checks.values()) is expected
