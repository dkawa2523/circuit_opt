"""pcd.sim_core — running a case and recording the result.

The rule this layer must never break: **simulation writes waveforms, never
metrics.**  The rest is about producing a complete record even when the run
fails, so an optimizer can score it and keep going.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pcd.solver as solver_module
from pcd.case import load_case
from pcd.metrics import measure_record
from pcd.sim_core import prepare_case, simulate_case
from pcd.sim_registry import available as sim_available

EX = Path(__file__).resolve().parents[1] / "examples" / "advanced"


# --- the layer boundary ----------------------------------------------------


def test_simulation_writes_artifacts_but_never_metrics(tmp_path, rc_case):
    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    assert rec.status == "ok"
    assert (rec.run_dir / "waveform.csv").exists()
    assert (rec.run_dir / "netlist.cir").exists()

    manifest = rec.manifest()
    assert manifest["schema"] == "simulation_record.v2"
    assert not (rec.run_dir / "metrics.json").exists()

    metrics = measure_record(rc_case, manifest)
    assert metrics["loss"] >= 0.0
    assert not (rec.run_dir / "metrics.json").exists(), "measurement must not create a second result store"


def test_preparing_a_case_writes_everything_except_the_waveform(tmp_path, rc_case):
    rec = prepare_case(rc_case, run_root=tmp_path)
    assert rec.status == "prepared"
    assert (rec.run_dir / "netlist.cir").exists()
    assert (rec.run_dir / "case.yaml").exists()
    assert json.loads((rec.run_dir / "params.json").read_text(encoding="utf-8"))["R1"] == 1000
    assert "prepared only" in (rec.run_dir / "solver.log").read_text(encoding="utf-8")


def test_registries_expose_the_documented_methods():
    sim = sim_available()
    assert set(sim) == {"circuit", "load", "solver"}
    assert "ngspice_cli" in sim["solver"]
    assert "test_fake" in sim["solver"]
    assert "dummy" not in sim["solver"]


def test_a_plugin_can_add_circuit_and_objective_methods(tmp_path):
    case = load_case(EX / "plugin_case.yaml")
    rec = simulate_case(case, run_root=tmp_path, solver_override="test_fake")
    assert rec.circuit == "custom_series_lc"
    assert measure_record(case, rec.manifest())["objective"] == "peak_voltage"


# --- provenance ------------------------------------------------------------


def test_provenance_identifies_the_case_and_the_solver(tmp_path, rc_case):
    provenance = prepare_case(rc_case, run_root=tmp_path).provenance
    assert len(provenance["case_data_sha256"]) == 64
    assert len(provenance["params_sha256"]) == 64
    assert provenance["case_path"].endswith("generic_rc_filter.yaml")
    assert provenance["platform_version"]
    assert len(provenance["implementation_sha256"]) == 64


def test_changing_a_parameter_changes_the_params_digest(tmp_path, rc_case):
    a = prepare_case(rc_case, params={"R1": 1000.0}, run_root=tmp_path, run_id="a").provenance
    b = prepare_case(rc_case, params={"R1": 2000.0}, run_root=tmp_path, run_id="b").provenance
    assert a["case_data_sha256"] == b["case_data_sha256"]
    assert a["params_sha256"] != b["params_sha256"]


def test_plugin_files_are_recorded_with_their_digests(tmp_path):
    case = load_case(EX / "plugin_case.yaml")
    plugins = prepare_case(case, run_root=tmp_path).provenance["plugins"]
    assert plugins
    assert plugins[0]["exists"] is True
    assert len(plugins[0]["sha256"]) == 64


def test_provenance_records_the_resolved_solver(tmp_path, rc_case, monkeypatch):
    solver_module.solver_version.cache_clear()
    monkeypatch.setattr(solver_module.sys, "platform", "win32")
    monkeypatch.setattr(
        solver_module.shutil,
        "which",
        lambda exe: r"C:\ngspice\ngspice_con.exe" if exe == "ngspice_con.exe" else None,
    )
    monkeypatch.setattr(
        solver_module.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="ngspice-46\n", stderr=""),
    )
    solver = prepare_case(rc_case, run_root=tmp_path, solver_name="ngspice_cli").provenance["solver"]
    assert solver["executable"] == "ngspice_con.exe"
    assert solver["resolved_executable"] == r"C:\ngspice\ngspice_con.exe"
    assert solver["version"] == "ngspice-46"


# --- failure handling ------------------------------------------------------


def test_a_failed_simulation_still_leaves_a_complete_record(tmp_path, rc_case):
    """A solver failure must not lose the observation record."""

    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="does_not_exist")
    assert rec.status == "failed"
    assert rec.error
    assert (rec.run_dir / "waveform.csv").exists()
    assert (rec.run_dir / "params.json").exists()
    assert (rec.run_dir / "case.yaml").exists()
    assert any("simulation exception" in w for w in rec.warnings)


def test_a_failure_during_preparation_is_also_recorded(tmp_path, make_case):
    """An unknown load fails before a netlist exists."""

    case = make_case(
        {
            "case_id": "broken",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"builder": "from_yaml", "components": [{"ref": "R1", "n1": "src", "n2": "out", "value": 50}]},
            "load": {"name": "definitely_unknown"},
            "solver": {"name": "test_fake", "tran": {"step_s": 1e-9, "stop_s": 1e-7}},
        }
    )
    rec = simulate_case(case, run_root=tmp_path / "runs", solver_override="test_fake")
    assert rec.status == "failed"
    assert rec.circuit == "unknown"
    assert (rec.run_dir / "sim_manifest.json").exists()
    assert (rec.run_dir / "waveform.csv").exists()


def test_run_directory_collisions_are_resolved(tmp_path):
    """The directory name embeds a one-second timestamp, so this can collide."""

    from pcd.sim_core import _ensure_unique_dir

    base = tmp_path / "sim_0000"
    assert _ensure_unique_dir(base) == base
    base.mkdir()
    first = _ensure_unique_dir(base)
    assert first.name == "sim_0000_001"
    first.mkdir()
    assert _ensure_unique_dir(base).name == "sim_0000_002"


def test_long_run_id_is_bounded_without_changing_case_identity(tmp_path, rc_case):
    run_id = "evaluation_from_a_very_long_external_case_name_" * 3
    record = prepare_case(rc_case, run_root=tmp_path, run_id=run_id)

    assert len(record.run_dir.name) == 32
    assert record.run_dir.name != run_id
    assert record.case_id == rc_case.case_id


def test_run_directory_digest_is_stable_across_the_security_flag():
    """`usedforsecurity=False` must not rename existing run directories."""

    import hashlib

    payload = repr(sorted({"C1": 1e-9}.items())).encode("utf-8")
    assert (
        hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:8] == hashlib.sha1(payload).hexdigest()[:8]  # noqa: S324
    )


# --- electrical reference plane -------------------------------------------


def test_the_manifest_records_load_ports_and_reference_plane(tmp_path, make_case):
    case = make_case(
        {
            "case_id": "port_metadata",
            "source": {"type": "sine_voltage", "frequency_Hz": 1e6},
            "circuit": {"builder": "from_yaml", "output_node": "electrode", "components": []},
            "load": {
                "name": "resistor",
                "R_ohm": 50,
                "ports": {"p": "electrode", "n": "return"},
                "reference_plane": "chamber_feedthrough",
            },
            "measurement": {"load_current": "auto"},
            "solver": {"name": "test_fake"},
        }
    )
    rec = prepare_case(case, run_root=tmp_path, solver_name="test_fake")
    assert rec.measurement["load_ports"] == {"p": "electrode", "n": "return"}
    assert rec.measurement["load_current"] == "load_current_A"
    assert rec.measurement["reference_plane"] == "chamber_feedthrough"
