"""Tests for the persisted simulation-artifact read boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pcd.records import frequency_response_path, load_waveform, read_sim_record, waveform_path


def _write_record(tmp_path: Path) -> dict:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1e-9],
            "voltage_V": [0.0, 1.0],
            "current_A": [0.0, 0.1],
        }
    )
    frame.to_csv(tmp_path / "waveform.csv", index=False)
    (tmp_path / "frequency_response.csv").write_text("frequency_Hz,real_V,imag_V\n1e6,1,0\n", encoding="utf-8")
    manifest = {
        "schema": "simulation_record.v2",
        "case_id": "artifact_reader",
        "run_dir": str(tmp_path),
        "status": "ok",
        "artifacts": {
            "waveform": "waveform.csv",
            "frequency_response": "frequency_response.csv",
        },
    }
    (tmp_path / "sim_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_read_sim_record_accepts_manifest_directory_and_mapping(tmp_path):
    manifest = _write_record(tmp_path)

    assert read_sim_record(tmp_path)["case_id"] == "artifact_reader"
    assert read_sim_record(tmp_path / "sim_manifest.json")["run_dir"] == str(tmp_path)
    assert read_sim_record(manifest) == manifest


def test_record_paths_are_resolved_from_the_manifest(tmp_path):
    manifest = _write_record(tmp_path)

    assert waveform_path(manifest) == tmp_path / "waveform.csv"
    assert frequency_response_path(manifest) == tmp_path / "frequency_response.csv"
    assert frequency_response_path({"run_dir": str(tmp_path), "artifacts": {}}) is None


def test_waveform_path_uses_a_safe_default_for_null_legacy_entries(tmp_path):
    record = {"run_dir": str(tmp_path), "artifacts": {"waveform": None}, "waveform_file": None}
    assert waveform_path(record) == tmp_path / "waveform.csv"


def test_load_waveform_reads_a_record_or_a_direct_csv_path(tmp_path):
    manifest = _write_record(tmp_path)

    from_record = load_waveform(manifest)
    from_csv = load_waveform(tmp_path / "waveform.csv")
    pd.testing.assert_frame_equal(from_record, from_csv)
