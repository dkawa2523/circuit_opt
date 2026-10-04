"""Release closure verifies only the supported circuit workflows."""

from __future__ import annotations

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
    assert "Machine learning is not part of this" in report


def test_release_root_must_be_new(tmp_path):
    run_root = tmp_path / "existing"
    run_root.mkdir()

    with pytest.raises(FileExistsError, match="release run root already exists"):
        run_suite.run_release_suite(run_root)


def test_cache_hits_parse_boolean_csv_values():
    assert run_suite._cache_hits([{"from_cache": "True"}, {"from_cache": "0"}, {"from_cache": "yes"}]) == 2
