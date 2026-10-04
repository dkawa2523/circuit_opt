"""Tests for the persisted simulation-artifact read boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pcd.records import artifact_path, frequency_response_path, load_waveform, read_sim_record, waveform_path


def _write_record(tmp_path: Path) -> dict:
    data = tmp_path / "data"
    debug = tmp_path / "debug"
    data.mkdir()
    debug.mkdir()
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1e-9],
            "voltage_V": [0.0, 1.0],
            "current_A": [0.0, 0.1],
        }
    )
    frame.to_csv(data / "transient.csv", index=False)
    (data / "ac.csv").write_text("frequency_Hz,real_V,imag_V\n1e6,1,0\n", encoding="utf-8")
    manifest = {
        "schema": "simulation_record.v2",
        "case_id": "artifact_reader",
        "run_dir": str(tmp_path),
        "status": "ok",
        "artifacts": {
            "waveform": "data/transient.csv",
            "frequency_response": "data/ac.csv",
        },
    }
    (debug / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "schema": "simulation_summary.v1",
        "case_id": "artifact_reader",
        "run_dir": str(tmp_path),
        "status": "ok",
        "artifacts": {
            "waveform": "data/transient.csv",
            "frequency_response": "data/ac.csv",
            "debug_manifest": "debug/manifest.json",
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return manifest


def test_read_sim_record_accepts_manifest_directory_and_mapping(tmp_path):
    manifest = _write_record(tmp_path)

    assert read_sim_record(tmp_path)["case_id"] == "artifact_reader"
    assert read_sim_record(tmp_path / "debug" / "manifest.json")["run_dir"] == str(tmp_path)
    assert read_sim_record(manifest) == manifest


def test_current_summary_and_directory_resolve_the_debug_manifest(tmp_path):
    _write_record(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert read_sim_record(tmp_path)["case_id"] == "artifact_reader"
    assert read_sim_record(tmp_path / "summary.json")["schema"] == "simulation_record.v2"
    assert read_sim_record(summary)["artifacts"]["waveform"] == "data/transient.csv"
    assert waveform_path(tmp_path) == tmp_path / "data" / "transient.csv"


def test_record_paths_are_resolved_from_the_manifest(tmp_path):
    manifest = _write_record(tmp_path)

    assert waveform_path(manifest) == tmp_path / "data" / "transient.csv"
    assert frequency_response_path(manifest) == tmp_path / "data" / "ac.csv"
    assert frequency_response_path({"run_dir": str(tmp_path), "artifacts": {}}) is None


def test_any_declared_artifact_uses_the_same_path_boundary(tmp_path):
    run_dir = tmp_path / "artifacts" / "evaluation"
    run_dir.mkdir(parents=True)
    shared = tmp_path / "generation"
    shared.mkdir()
    case_path = shared / "case.yaml"
    case_path.write_text("schema: case_yaml.v1\n", encoding="utf-8")
    record = {"run_dir": str(run_dir), "artifacts": {"case": "../../generation/case.yaml"}}

    resolved = artifact_path(record, "case")
    assert resolved == run_dir / "../../generation/case.yaml"
    assert resolved is not None
    assert resolved.resolve() == case_path.resolve()


def test_waveform_path_requires_a_declared_artifact(tmp_path):
    record = {"run_dir": str(tmp_path), "artifacts": {}}
    with pytest.raises(ValueError, match="does not declare a waveform"):
        waveform_path(record)


def test_load_waveform_reads_a_record_or_a_direct_csv_path(tmp_path):
    manifest = _write_record(tmp_path)

    from_record = load_waveform(manifest)
    from_csv = load_waveform(tmp_path / "data" / "transient.csv")
    pd.testing.assert_frame_equal(from_record, from_csv)


def test_a_moved_run_resolves_artifacts_from_its_manifest_directory(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _write_record(original)
    moved = tmp_path / "moved"
    original.rename(moved)

    record = read_sim_record(moved)
    assert record["run_dir"] == str(moved.resolve())
    assert waveform_path(record) == moved / "data" / "transient.csv"
