"""Solver execution plus the separate ngspice output adapter.

A solver never raises for a simulation problem; it returns a failed result
carrying the reason, so an optimizer keeps collecting observations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import pcd.solver as solver_module
from pcd.case import Case
from pcd.core import Candidate, ControlState, EvaluationRequest, RawResult, Scenario
from pcd.evaluation import CaseEvaluationBackend
from pcd.ngspice_io import read_frequency_response, read_transient_output
from pcd.sim_core import execute_case
from pcd.sim_registry import invoke_solver, register_solver
from pcd.simulation import SimulationResult
from pcd.simulation_input import SolverRunRequest, resolve_simulation_case, resolve_solver_settings
from pcd.solver import diagnose_solver, ngspice_cli, solver_identity

EX = Path(__file__).resolve().parents[1] / "examples" / "advanced"

ATOL_TIME = 1e-18  # s; transient steps are ~1e-9
RTOL_SIGNAL = 1e-9


def _ngspice(case: Case, run_dir: Path, params: dict | None = None):
    request = SolverRunRequest(
        netlist_path=run_dir / "netlist.cir",
        run_dir=run_dir,
        simulation=resolve_simulation_case(case, params or {}),
    )
    return ngspice_cli(request)


# --- reading ngspice output ------------------------------------------------


def test_transient_wrdata_decodes_scale_vector_pairs_and_named_probes(tmp_path):
    t = np.array([0.0, 1e-9, 2e-9, 3e-9])
    v = np.array([0.0, 1.0, 0.0, -1.0])
    i = np.array([0.0, 0.1, 0.0, -0.1])
    load = v / 2
    path = tmp_path / "wrdata.txt"
    np.savetxt(path, np.column_stack([t, t, t, v, t, i, t, load]))

    result = read_transient_output(path, ["load_voltage_V"])
    np.testing.assert_allclose(result.time_s, t, rtol=0, atol=ATOL_TIME)
    np.testing.assert_allclose(result.voltage_V, v, rtol=RTOL_SIGNAL, atol=0)
    assert result.current_A is not None
    np.testing.assert_allclose(result.current_A, i, rtol=RTOL_SIGNAL, atol=0)
    np.testing.assert_allclose(result.probes["load_voltage_V"], load, rtol=RTOL_SIGNAL, atol=0)


def test_ac_wrdata_decodes_complex_vectors_on_the_frequency_scale(tmp_path):
    path = tmp_path / "wrdata.csv"
    np.savetxt(path, [[1e6, 1.0, 2.0, 1e6, -0.1, 0.2, 1e6, 0.5, -0.3]])

    response = read_frequency_response(path, ["load_voltage_V"])

    assert response.iloc[0].to_dict() == {
        "frequency_Hz": 1e6,
        "voltage_re": 1.0,
        "voltage_im": 2.0,
        "current_re": -0.1,
        "current_im": 0.2,
        "load_voltage_V_re": 0.5,
        "load_voltage_V_im": -0.3,
    }
    assert all(response[column].dtype == np.float64 for column in response)


# --- solver configuration --------------------------------------------------


def test_the_timeout_uses_a_default_only_when_omitted():
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {}})
    assert resolve_solver_settings(case).timeout_s == 300.0


def test_an_explicit_timeout_is_preserved():
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"timeout_s": 12.5}})
    assert resolve_solver_settings(case).timeout_s == 12.5


@pytest.mark.parametrize("configured", [0, -1, float("inf"), "soon"])
def test_an_invalid_timeout_is_not_silently_replaced(configured):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"timeout_s": configured}})
    with pytest.raises(ValueError, match="timeout_s"):
        resolve_solver_settings(case)


def test_solver_identity_includes_the_resolved_binary_digest(tmp_path, monkeypatch):
    executable = tmp_path / "solver.bin"
    executable.write_bytes(b"solver-build-one")
    monkeypatch.setattr(solver_module.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        solver_module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="test-1\n", stderr=""),
    )

    case = Case(path=tmp_path / "case.yaml", data={"solver": {"executable": "solver"}})
    identity = solver_identity(resolve_solver_settings(case))
    assert identity["version"] == "test-1"
    assert len(identity["executable_sha256"]) == 64
    assert identity["executable_size"] == executable.stat().st_size
    assert identity["executable_mtime_ns"] == executable.stat().st_mtime_ns


# --- running ngspice -------------------------------------------------------


def test_a_missing_binary_is_reported_as_a_diagnostic(tmp_path):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "not_a_real_ngspice"}})
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert "executable not found: not_a_real_ngspice" in result.diagnostics["error"]


def test_a_timeout_is_reported_as_a_diagnostic(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice", "timeout_s": 0.01}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"process output\xff", stderr=None)

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert "timed out after 0.01s" in result.diagnostics["error"]
    assert "process output" in result.log


def test_a_nonzero_exit_is_reported_as_a_failure(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice"}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)
    monkeypatch.setattr(
        solver_module.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="boom", stderr=""),
    )
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert result.diagnostics["returncode"] == 1
    assert "exited with status 1" in result.diagnostics["error"]


def test_malformed_solver_output_is_reported_as_a_parse_failure(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice"}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        (Path(kwargs["cwd"]) / "waveform.csv").write_text("malformed", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert "invalid ngspice output" in result.diagnostics["error"]
    assert "malformed" in result.diagnostics["error"]


def test_a_partial_transient_is_not_reported_as_success(tmp_path, monkeypatch):
    """ngspice can return zero after a time-step abort while leaving a partial CSV."""

    case = Case(
        path=EX / "generic_rc_filter.yaml",
        data={
            "source": {"type": "dc_voltage", "voltage_V": 1.0},
            "solver": {"executable": "ngspice", "tran": {"step_s": 1e-6, "stop_s": 1e-6}},
        },
    )
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        time_s = np.array([0.0, 1e-9, 2e-9])
        np.savetxt(Path(kwargs["cwd"]) / "waveform.csv", np.column_stack([time_s, time_s] * 5))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert "requested stop_s=1e-06 s" in result.diagnostics["error"]
    assert "last=2e-09 s" in result.diagnostics["error"]


def test_incomplete_extension_response_is_an_error_at_solver_dispatch(rc_case, tmp_path):
    @register_solver("test_incomplete_response")
    def incomplete_solver(_request: SolverRunRequest) -> SimulationResult:
        return SimulationResult(np.array([0.0, 1e-9]), np.array([0.0, 1.0]))

    simulation = resolve_simulation_case(rc_case, solver_override="test_incomplete_response")
    request = SolverRunRequest(tmp_path / "netlist.cir", tmp_path, simulation)

    with pytest.raises(ValueError, match="requested stop_s"):
        invoke_solver(simulation.solver.name, request)


def test_unexpected_parser_exception_propagates(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice"}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        (Path(kwargs["cwd"]) / "waveform.csv").write_text("irrelevant", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        solver_module,
        "read_transient_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parser implementation bug")),
    )
    with pytest.raises(RuntimeError, match="parser implementation bug"):
        _ngspice(case, tmp_path)


def test_an_ac_only_run_succeeds_without_a_waveform_file(tmp_path, monkeypatch):
    case = Case(
        path=EX / "generic_rc_filter.yaml",
        data={
            "solver": {
                "executable": "ngspice",
                "ac": {"sweep": "lin", "points": 1, "start_Hz": 1e6, "stop_Hz": 1e6},
            }
        },
    )
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        np.savetxt(
            Path(kwargs["cwd"]) / "ac.csv",
            [[1e6, 1.0, 0.0, 1e6, -0.02, 0.0, 1e6, 0.5, 0.0, 1e6, 0.5, 0.0]],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "ok"
    assert result.time_s.size == 0
    assert result.frequency_response is not None
    assert result.frequency_response.iloc[0]["load_voltage_V_re"] == pytest.approx(0.5)
    assert result.frequency_response["frequency_Hz"].tolist() == [1e6]


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_ngspice_keeps_authored_component_continuations_without_analysis_continuations(make_case, tmp_path):
    (tmp_path / "rc.cir").write_text(
        "RC\nV1 src\n+ 0 AC 1\nR1 src out 1k\nC1 out 0\n+ 1n\n.tran 1n\n* authored step\n+ 1u\n.end\n",
        encoding="utf-8",
    )
    case = make_case(
        {
            "circuit": {"builder": "from_netlist", "netlist_file": "rc.cir", "netlist_mode": "deck"},
            "solver": {"ac": {"frequency_Hz": 1e6}},
        }
    )

    result = execute_case(case, run_root=tmp_path / "runs").response

    assert result.status == "ok", result.log
    assert result.frequency_response is not None
    row = result.frequency_response.iloc[0]
    assert complex(row["voltage_V_re"], row["voltage_V_im"]) == pytest.approx(1 / (1 + 2j * np.pi))


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_ngspice_serializes_signed_and_real_probes_independently(make_case, tmp_path):
    case = make_case(
        {
            "source": {"type": "sine_voltage", "amplitude_V": 1, "frequency_Hz": 1e6},
            "circuit": {"components": [{"ref": "R1", "n1": "src", "n2": "0", "value": 50}], "output_node": "src"},
            "measurement": {
                "probes": {
                    "signed": "-i(Vsrc)",
                    "amplitude": {"vector": "mag(v(src))", "unit": "V"},
                    "ground": "v(0)",
                    "reverse": "v(0,src)",
                }
            },
            "solver": {"tran": {"step_s": 1e-8, "stop_s": 1e-6}, "ac": {"frequency_Hz": 1e6}},
        }
    )

    result = execute_case(case, run_root=tmp_path / "runs").response

    assert result.status == "ok", result.log
    assert result.current_A is not None
    np.testing.assert_allclose(result.probes["signed"], -result.current_A)
    np.testing.assert_allclose(result.probes["amplitude"], abs(result.voltage_V))
    np.testing.assert_allclose(result.probes["ground"], 0)
    np.testing.assert_allclose(result.probes["reverse"], -result.voltage_V)
    assert result.frequency_response is not None
    row = result.frequency_response.iloc[0]
    assert row[["signed_re", "amplitude_re", "ground_re", "reverse_re"]].tolist() == pytest.approx([0.02, 1, 0, -1])
    assert row[["signed_im", "amplitude_im", "ground_im", "reverse_im"]].tolist() == pytest.approx([0, 0, 0, 0])


def test_wrdata_rejects_unexpected_probe_columns(tmp_path):
    path = tmp_path / "wrdata.csv"
    np.savetxt(path, [[1e6, 1.0, 0.0]])
    with pytest.raises(ValueError, match="needs 6 columns"):
        read_frequency_response(path)

    np.savetxt(path, [np.arange(11)])

    with pytest.raises(ValueError, match="expected 9 columns"):
        read_frequency_response(path, ["probe"])
    with pytest.raises(ValueError, match="expected 8 columns"):
        read_transient_output(path, ["probe"])


@pytest.fixture
def windows_ngspice(monkeypatch):
    """Pretend we are on Windows with only ngspice_con.exe installed."""

    monkeypatch.setattr(solver_module.sys, "platform", "win32")
    monkeypatch.setattr(
        solver_module.shutil,
        "which",
        lambda exe: r"C:\ngspice\ngspice_con.exe" if exe == "ngspice_con.exe" else None,
    )
    return monkeypatch


def test_windows_runs_the_console_binary_without_opening_a_window(tmp_path, windows_ngspice):
    observed = {}

    def fake_run(cmd, **kwargs):
        observed.update(cmd=cmd, kwargs=kwargs)
        t = np.array([0.0, 5e-7, 1e-6])
        # time, observed output, source current/voltage, and actual load voltage.
        np.savetxt(Path(kwargs["cwd"]) / "waveform.csv", np.column_stack([t, t] * 5))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    windows_ngspice.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(Case(path=EX / "generic_rc_filter.yaml", data={"source": {}}), tmp_path)
    assert result.status == "ok"
    assert "source_voltage_V" in result.probes
    assert observed["cmd"][0] == r"C:\ngspice\ngspice_con.exe"
    assert "creationflags" in observed["kwargs"]


def test_solver_diagnostics_report_the_environment(windows_ngspice):
    def fake_run(cmd, **kwargs):
        if cmd[-1] == "--version":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="** ngspice-46 : Circuit level simulation program\n", stderr=""
            )
        np.savetxt(Path(kwargs["cwd"]) / "probe.csv", [[1e6, 1.0, 0.0, 1e6, -0.02, 0.0]])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    windows_ngspice.setattr(solver_module.subprocess, "run", fake_run)
    diag = diagnose_solver("ngspice_cli")
    assert diag["schema"] == "solver_diagnostic.v1"
    assert diag["executable"] == "ngspice_con.exe"
    assert diag["version"] == "ngspice-46"
    assert diag["batch_runnable"] is True


def test_solver_diagnostics_reject_a_different_executable(monkeypatch):
    monkeypatch.setattr(solver_module.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        solver_module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="Python 3.12.13\n", stderr=""),
    )

    diag = diagnose_solver(executable="another-tool")

    assert diag["version"] == "Python 3.12.13"
    assert diag["batch_runnable"] is False


def test_solver_identity_tracks_startup_files_and_binds_environment(tmp_path, monkeypatch, rc_case):
    root = tmp_path / "spice"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    config = scripts / "spinit"
    nested = scripts / "nested init"
    leaf = scripts / "leaf init"
    config.write_text(f'source "{nested.as_posix()}"\ncodemodel ../lib/test.cm\n', encoding="utf-8")
    nested.write_text(f'source "{leaf.as_posix()}"\n', encoding="utf-8")
    leaf.write_text("set numdgt=15\n", encoding="utf-8")
    executable = root / "bin" / "solver.exe"
    executable.parent.mkdir()
    executable.touch()
    model = root / "lib" / "test.cm"
    model.parent.mkdir()
    model.write_bytes(b"code model")
    monkeypatch.setenv("SPICE_LIB_DIR", str(root))
    monkeypatch.setattr(solver_module.shutil, "which", lambda _exe: str(executable))
    settings = resolve_solver_settings(rc_case)
    bound = solver_module.bind_solver_settings(settings)
    prior = solver_identity(bound)["startup"]

    assert {str(path) for path in (config, nested, leaf, model)} <= prior["startup_files_sha256"].keys()
    leaf.write_text("set wr_singlescale\n", encoding="utf-8")
    assert solver_identity(settings)["startup"] != prior
    assert solver_identity(bound)["startup"] == prior
    nested.write_text("source $unresolved_path\ncodemodel $model_path\n", encoding="utf-8")
    unresolved = solver_identity(settings)["startup"]["unresolved_sources"]
    assert {f"{nested}:1", f"{nested}:2"} <= set(unresolved)


@pytest.mark.parametrize("mutation_at", ["before", "during", "cache", "executable"])
def test_bound_ngspice_rejects_input_changes(tmp_path, monkeypatch, rc_case, mutation_at):
    config = tmp_path / "spinit"
    config.write_text("setseed 1\n", encoding="utf-8")
    executable = tmp_path / "solver.exe"
    executable.touch()
    monkeypatch.setenv("SPICE_LIB_DIR", str(tmp_path))
    monkeypatch.setattr(solver_module.shutil, "which", lambda _exe: str(executable))
    bound = solver_module.bind_solver_settings(resolve_solver_settings(rc_case))
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        config.write_text("setseed 2\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    if mutation_at == "executable":
        executable.write_bytes(b"replacement executable")
    elif mutation_at != "during":
        config.write_text("setseed 2\n", encoding="utf-8")

    if mutation_at == "cache":
        backend = CaseEvaluationBackend(rc_case, tmp_path, solver_settings=bound)
        request = EvaluationRequest(Candidate("candidate"), Scenario("condition"), ControlState())
        with pytest.raises(ValueError, match="ngspice startup inputs changed"):
            backend.compute(request, RawResult("ok", artifacts={"manifest": "cached.json"}))
    else:
        with pytest.raises(ValueError, match=r"ngspice (startup inputs|executable) changed"):
            solver_module._run_ngspice(bound, tmp_path / "deck.cir", tmp_path / "log", tmp_path, 10.0)

    assert len(calls) == (1 if mutation_at == "during" else 0)
