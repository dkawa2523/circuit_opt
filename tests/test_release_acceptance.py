"""Release closure keeps circuit readiness separate from the failed ML gate."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench.release import run_suite


def _ranking_result() -> dict:
    protocol = run_suite.ROOT / "bench" / "ml" / "ranking_protocol.yaml"
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    case = {
        "pool_checks": {"count": True, "solver": True},
        "ranking": {"outcome": {"savings_fraction": 0.2}},
        "verification": {"passed": True},
        "passed": False,
    }
    return {
        "schema": "pcd.ml_ranking_benchmark.v1",
        "protocol_id": "fixed-test",
        "protocol_sha256": protocol_hash,
        "passed": False,
        "sequential_proposal_allowed": False,
        "cases": [
            {"case_id": "rc", **case},
            {"case_id": "rf", **case},
        ],
    }


def test_failed_ml_gate_is_valid_evidence_but_not_release_authority(tmp_path, monkeypatch):
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_ranking_result()), encoding="utf-8")
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

    result = run_suite.run_release_suite(tmp_path / "release", ranking_path)

    assert result["passed"] is True
    assert result["workflow_acceptance_passed"] is True
    assert result["ml_evidence_complete"] is True
    assert result["release_decision"] == {
        "circuit_foundation": "GO",
        "ml_candidate_proposal": "NO-GO",
        "full_requested_scope": "NO-GO",
    }
    assert (tmp_path / "release" / "release_result.json").is_file()
    assert "The ML failure is retained as evidence" in (tmp_path / "release" / "REPORT.md").read_text(encoding="utf-8")


def test_missing_ml_evidence_does_not_block_circuit_decision(tmp_path, monkeypatch):
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
    assert result["workflow_acceptance_passed"] is True
    assert result["ml_evidence_complete"] is False
    assert result["release_decision"]["circuit_foundation"] == "GO"
    assert result["release_decision"]["ml_candidate_proposal"] == "NOT ASSESSED"
    assert result["release_decision"]["full_requested_scope"] == "NOT ASSESSED"
    assert "No ML ranking evidence was attached" in (tmp_path / "release" / "REPORT.md").read_text(encoding="utf-8")


def test_release_root_must_be_new(tmp_path):
    run_root = tmp_path / "existing"
    run_root.mkdir()

    with pytest.raises(FileExistsError, match="release run root already exists"):
        run_suite.run_release_suite(run_root)


def test_cache_hits_parse_boolean_csv_values():
    assert run_suite._cache_hits([{"from_cache": "True"}, {"from_cache": "0"}, {"from_cache": "yes"}]) == 2
