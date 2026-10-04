"""pcd.sim_core — running a case and recording the result.

The rule this layer must never break: **simulation writes waveforms, never
metrics.**  The rest is about producing a complete record even when the run
fails, so an optimizer can score it and keep going.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pcd.sim_core as sim_core_module
import pcd.solver as solver_module
from pcd.case import load_case
from pcd.metrics import measure_record, measure_response
from pcd.records import artifact_path
from pcd.sim_core import execute_case, prepare_case, simulate_case
from pcd.sim_registry import available as sim_available
from pcd.sim_registry import register_solver
from pcd.simulation import SimulationResult
from pcd.simulation_input import SolverRunRequest

EX = Path(__file__).resolve().parents[1] / "examples" / "advanced"


# --- the layer boundary ----------------------------------------------------


def test_simulation_writes_artifacts_but_never_metrics(tmp_path, rc_case):
    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="test_fake")
    assert rec.status == "ok"
    assert (rec.run_dir / rec.waveform_file).exists()
    assert (rec.run_dir / rec.netlist_file).exists()
    assert {path.name for path in rec.run_dir.iterdir()} == {"summary.json", "data", "debug"}

    manifest = rec.manifest()
    assert manifest["schema"] == "simulation_record.v2"
    assert rec.summary()["schema"] == "simulation_summary.v1"
    assert not (rec.run_dir / "metrics.json").exists()

    metrics = measure_record(rc_case, manifest)
    assert metrics["loss"] >= 0.0
    assert not (rec.run_dir / "metrics.json").exists(), "measurement must not create a second result store"


def test_in_memory_and_restored_responses_have_identical_metrics(tmp_path, rc_case):
    run = execute_case(rc_case, run_root=tmp_path, solver_override="test_fake")

    direct = measure_response(rc_case, run.response, run.record.params)
    restored = measure_record(rc_case, run.record.manifest())

    assert direct.keys() == restored.keys()
    assert direct["objective"] == restored["objective"]
    for name in ("loss", "normalized_rmse", "rmse_V", "peak_abs_voltage_V"):
        assert direct[name] == pytest.approx(restored[name], rel=1e-14)


def test_structured_and_imported_circuits_return_the_same_response_type(tmp_path, rc_case, make_case):
    imported = make_case(
        {
            "case_id": "imported_response",
            "source": {"type": "voltage_pulse", "name": "Vsrc", "v1_V": 0, "v2_V": 1},
            "circuit": {
                "builder": "from_netlist",
                "netlist_file": "fragment.cir",
                "netlist_mode": "fragment",
                "source_policy": "replace_named",
                "output_node": "out",
            },
            "load": {"name": "none"},
            "measurement": {"voltage_node": "out", "current_source": "Vsrc"},
            "solver": {"tran": {"step_s": 1e-9, "stop_s": 1e-7}},
        },
        name="imported.yaml",
    )
    (imported.base_dir / "fragment.cir").write_text("R1 src out 1k\nC1 out 0 1n\n", encoding="utf-8")

    structured = execute_case(rc_case, run_root=tmp_path / "structured", solver_override="test_fake")
    external = execute_case(imported, run_root=tmp_path / "external", solver_override="test_fake")

    assert isinstance(structured.response, SimulationResult)
    assert isinstance(external.response, SimulationResult)
    assert structured.response.status == external.response.status == "ok"
    assert structured.response.as_frame().columns.equals(external.response.as_frame().columns)


def test_common_simulation_layer_persists_a_custom_solver_ac_result(tmp_path, rc_case, monkeypatch):
    response = pd.DataFrame({"frequency_Hz": [1e6], "voltage_V_re": [1.0], "voltage_V_im": [0.0]})

    def custom_solver(*_args):
        return SimulationResult(
            time_s=np.asarray([], dtype=float),
            voltage_V=np.asarray([], dtype=float),
            frequency_response=response,
        )

    monkeypatch.setattr(sim_core_module, "invoke_solver", custom_solver)
    record = simulate_case(rc_case, run_root=tmp_path, solver_override="custom_ac")

    assert record.frequency_response_file == "data/ac.csv"
    pd.testing.assert_frame_equal(pd.read_csv(record.run_dir / record.frequency_response_file), response)


def test_a_typed_solver_receives_no_case_dictionary(tmp_path, rc_case):
    observed = []

    @register_solver("test_typed_request")
    def typed_solver(request: SolverRunRequest) -> SimulationResult:
        observed.append(request)
        return SimulationResult(
            time_s=np.asarray([0.0]),
            voltage_V=np.asarray([0.0]),
            current_A=np.asarray([0.0]),
        )

    record = simulate_case(rc_case, run_root=tmp_path, solver_override="test_typed_request")

    assert record.status == "ok"
    assert len(observed) == 1
    assert observed[0].simulation.solver.name == "test_typed_request"
    assert observed[0].netlist_path == record.run_dir / record.netlist_file


def test_the_normal_run_resolves_solver_input_once(tmp_path, rc_case, monkeypatch):
    real_resolve = sim_core_module.resolve_simulation_case
    calls = []

    def counted_resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(sim_core_module, "resolve_simulation_case", counted_resolve)
    record = simulate_case(rc_case, run_root=tmp_path, solver_override="test_fake")

    assert record.status == "ok"
    assert len(calls) == 1


def test_preparing_a_case_writes_everything_except_the_waveform(tmp_path, rc_case):
    rec = prepare_case(rc_case, run_root=tmp_path)
    assert rec.status == "prepared"
    assert (rec.run_dir / rec.netlist_file).exists()
    archived_case = artifact_path(rec.manifest(), "case")
    assert archived_case is not None
    assert archived_case.is_file()
    assert rec.manifest()["params"]["R1"] == 1000
    assert not list(rec.run_dir.rglob("params.json")), "parameters belong in the debug manifest only"
    assert "prepared only" in (rec.run_dir / rec.solver_log_file).read_text(encoding="utf-8")
    assert rec.solver == "ngspice_cli"


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
    solver_module.clear_solver_version_cache()
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


def test_a_solver_reported_failure_still_leaves_a_complete_record(tmp_path, rc_case, monkeypatch):
    """An expected solver failure must not lose the observation record."""

    failed = SimulationResult(
        time_s=np.asarray([0.0]),
        voltage_V=np.asarray([np.nan]),
        current_A=np.asarray([np.nan]),
        status="failed",
        log="convergence failed",
        diagnostics={"returncode": 1},
    )
    monkeypatch.setattr(sim_core_module, "invoke_solver", lambda *_args: failed)
    rec = simulate_case(rc_case, run_root=tmp_path, solver_override="test_failed_solver")
    assert rec.status == "failed"
    assert (rec.run_dir / rec.waveform_file).exists()
    assert not list(rec.run_dir.rglob("params.json"))
    archived_case = artifact_path(rec.manifest(), "case")
    assert archived_case is not None
    assert archived_case.is_file()
    assert rec.diagnostics == {"returncode": 1}
    assert "solver status: failed" in rec.warnings


def test_an_unexpected_solver_exception_propagates_and_leaves_prepared_state(tmp_path, rc_case, monkeypatch):
    def broken_solver(*_args):
        raise RuntimeError("solver adapter bug")

    monkeypatch.setattr(sim_core_module, "invoke_solver", broken_solver)
    with pytest.raises(RuntimeError, match="solver adapter bug"):
        simulate_case(rc_case, run_root=tmp_path, solver_override="test_broken_solver")

    summaries = list(tmp_path.rglob("summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text(encoding="utf-8"))["status"] == "prepared"
    assert not list(tmp_path.rglob("data/transient.csv"))


def test_a_failure_during_preparation_propagates_without_a_fake_result(tmp_path, make_case):
    """An unknown load is invalid input, not a simulation observation."""

    case = make_case(
        {
            "case_id": "broken",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"builder": "from_yaml", "components": [{"ref": "R1", "n1": "src", "n2": "out", "value": 50}]},
            "load": {"name": "definitely_unknown"},
            "solver": {"name": "test_fake", "tran": {"step_s": 1e-9, "stop_s": 1e-7}},
        }
    )
    run_root = tmp_path / "runs"
    with pytest.raises(KeyError, match="definitely_unknown"):
        simulate_case(case, run_root=run_root, solver_override="test_fake")
    assert not run_root.exists()


def test_invalid_analysis_input_is_rejected_before_a_run_directory_is_created(tmp_path, make_case):
    case = make_case(
        {
            "case_id": "invalid_analysis",
            "source": {"type": "sine_voltage"},
            "solver": {"timeout_s": 0, "tran": {"step_s": 1e-9, "stop_s": 1e-7}},
        }
    )
    run_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="timeout_s"):
        simulate_case(case, run_root=run_root, solver_override="test_fake")

    assert not run_root.exists()


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
            "source": {"type": "sine_voltage", "name": "Vrf", "frequency_Hz": 1e6},
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
    assert rec.measurement["current_source"] == "Vrf"
    assert rec.measurement["load_ports"] == {"p": "electrode", "n": "return"}
    assert rec.measurement["load_current"] == "load_current_A"
    assert rec.measurement["reference_plane"] == "chamber_feedthrough"
