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
from pcd.ngspice_io import read_transient_output
from pcd.simulation_input import SolverRunRequest, resolve_simulation_case, resolve_solver_settings
from pcd.solver import diagnose_solver, ngspice_cli, solver_identity
from tests.fakes import fake_waveform

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


# --- the waveform contract -------------------------------------------------


def test_the_waveform_frame_has_the_canonical_columns(rc_case):
    frame = fake_waveform(rc_case, {"R1": 1000.0, "C1": 1e-9})
    assert list(frame.columns) == ["time_s", "voltage_V", "current_A"]
    assert all(frame[col].dtype == np.float64 for col in frame.columns)
    assert np.all(np.isfinite(frame.to_numpy()))
    assert np.all(np.diff(frame["time_s"].to_numpy()) > 0)


def test_the_transient_window_matches_the_case_settings(rc_case):
    stop = float(rc_case.data["solver"]["tran"]["stop_s"])
    assert fake_waveform(rc_case, {})["time_s"].iloc[-1] == pytest.approx(stop, rel=1e-12)


@pytest.mark.parametrize("source_type", ["sine_voltage", "dc_voltage", "pulse"])
def test_the_test_fake_handles_every_source_shape(make_case, source_type):
    case = make_case(
        {
            "case_id": "shapes",
            "source": {"type": source_type, "amplitude_V": 2.0, "frequency_Hz": 1e6, "voltage_V": 3.0},
            "solver": {"tran": {"step_s": 1e-9, "stop_s": 1e-7}},
        }
    )
    frame = fake_waveform(case, {})
    assert len(frame) > 10
    assert np.all(np.isfinite(frame["voltage_V"].to_numpy()))


# --- reading ngspice output ------------------------------------------------


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(3, id="time_v_i"),
        pytest.param(4, id="legacy_4col"),
        pytest.param(5, id="legacy_5col"),
        pytest.param(6, id="ngspice46_scale_vector_pairs"),
    ],
)
def test_every_wrdata_layout_decodes_to_the_same_signal(tmp_path, columns):
    """ngspice writes different column layouts by version; all must agree."""

    t = np.array([0.0, 1e-9, 2e-9, 3e-9])
    v = np.array([0.0, 1.0, 0.0, -1.0])
    i = np.array([0.0, 0.1, 0.0, -0.1])
    layouts = {3: [t, v, i], 4: [t, v, t, i], 5: [t, t, v, t, i], 6: [t, t, t, v, t, i]}
    path = tmp_path / "wrdata.txt"
    np.savetxt(path, np.column_stack(layouts[columns]))

    result = read_transient_output(path)
    np.testing.assert_allclose(result.time_s, t, rtol=0, atol=ATOL_TIME)
    np.testing.assert_allclose(result.voltage_V, v, rtol=RTOL_SIGNAL, atol=0)
    assert result.current_A is not None
    np.testing.assert_allclose(result.current_A, i, rtol=RTOL_SIGNAL, atol=0)


def test_a_two_column_output_has_no_current_channel(tmp_path):
    path = tmp_path / "wrdata.txt"
    np.savetxt(path, np.column_stack([np.array([0.0, 1e-9]), np.array([0.0, 1.0])]))
    result = read_transient_output(path)
    assert result.current_A is None
    assert result.as_frame()["current_A"].isna().all()


def test_a_single_row_output_is_still_parsed(tmp_path):
    path = tmp_path / "wrdata.txt"
    np.savetxt(path, np.array([[0.0, 1.0, 0.1]]))
    assert read_transient_output(path).time_s.shape == (1,)


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


def test_ngspice_is_the_default_solver_identity(rc_case):
    assert solver_identity(resolve_solver_settings(rc_case))["name"] == "ngspice_cli"


def test_solver_identity_includes_the_resolved_binary_digest(tmp_path, monkeypatch):
    executable = tmp_path / "solver.bin"
    executable.write_bytes(b"solver-build-one")
    monkeypatch.setattr(solver_module.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(solver_module, "solver_version", lambda _name: "test-1")
    solver_module.clear_solver_version_cache()

    case = Case(path=tmp_path / "case.yaml", data={"solver": {"executable": "solver"}})
    identity = solver_identity(resolve_solver_settings(case))
    assert len(identity["executable_sha256"]) == 64
    assert identity["executable_size"] == executable.stat().st_size
    assert identity["executable_mtime_ns"] == executable.stat().st_mtime_ns


def test_an_unknown_solver_reports_that_it_cannot_be_diagnosed():
    diag = diagnose_solver("some_other_solver")
    assert diag["batch_runnable"] is False
    assert "no built-in diagnostic" in diag["notes"][0]


# --- running ngspice -------------------------------------------------------


def test_a_missing_binary_is_reported_as_a_diagnostic(tmp_path):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "not_a_real_ngspice"}})
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert result.diagnostics["missing_executable"] is True


def test_a_timeout_is_reported_as_a_diagnostic(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice", "timeout_s": 0.01}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output="out", stderr="err")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert result.diagnostics["timed_out"] is True
    assert "timed out" in result.log


def test_byte_output_from_a_timeout_is_decoded(tmp_path, monkeypatch):
    """TimeoutExpired is typed bytes even in text mode; both must work."""

    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice", "timeout_s": 0.01}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"out\xff", stderr=None)

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    assert "out" in _ngspice(case, tmp_path).log


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


def test_malformed_solver_output_is_reported_as_a_parse_failure(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice"}})
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        (Path(kwargs["cwd"]) / "waveform.csv").write_text("malformed", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        solver_module, "read_transient_output", lambda *_args: (_ for _ in ()).throw(ValueError("bad data"))
    )
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert result.diagnostics["parse_error"] == "ValueError: bad data"


def test_a_partial_transient_is_not_reported_as_success(tmp_path, monkeypatch):
    """ngspice can return zero after a time-step abort while leaving a partial CSV."""

    case = Case(
        path=EX / "generic_rc_filter.yaml",
        data={"solver": {"executable": "ngspice", "tran": {"step_s": 1e-9, "stop_s": 1e-6}}},
    )
    monkeypatch.setattr(solver_module.shutil, "which", lambda exe: exe)

    def fake_run(cmd, **kwargs):
        time_s = np.array([0.0, 1e-9, 2e-9])
        np.savetxt(Path(kwargs["cwd"]) / "waveform.csv", np.column_stack([time_s, time_s] * 4))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "failed"
    assert result.diagnostics["incomplete_transient"] is True
    assert result.diagnostics["last_time_s"] == pytest.approx(2e-9)
    assert result.diagnostics["requested_stop_s"] == pytest.approx(1e-6)


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
        lambda *_args: (_ for _ in ()).throw(RuntimeError("parser implementation bug")),
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
            [[1e6, 1.0, 0.0, 1e6, -0.02, 0.0, 1e6, 0.5, 0.0]],
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(case, tmp_path)
    assert result.status == "ok"
    assert result.time_s.size == 0
    assert result.frequency_response is not None
    assert result.frequency_response.iloc[0]["load_voltage_V_re"] == pytest.approx(0.5)
    assert result.frequency_response["frequency_Hz"].tolist() == [1e6]


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
        # wrdata writes a (scale, value) pair per vector: time, v(out), i(Vsrc), v(src).
        np.savetxt(Path(kwargs["cwd"]) / "waveform.csv", np.column_stack([t, t] * 4))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    windows_ngspice.setattr(solver_module.subprocess, "run", fake_run)
    result = _ngspice(Case(path=EX / "generic_rc_filter.yaml", data={}), tmp_path)
    assert result.status == "ok"
    assert "source_voltage_V" in result.probes
    assert observed["cmd"][0] == "ngspice_con.exe"
    assert "creationflags" in observed["kwargs"]


def test_solver_diagnostics_report_the_environment(windows_ngspice):
    solver_module.clear_solver_version_cache()
    windows_ngspice.setattr(
        solver_module.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="******\n** ngspice-46 : Circuit level simulation program\n", stderr=""
        ),
    )
    diag = diagnose_solver("ngspice_cli")
    assert diag["schema"] == "solver_diagnostic.v1"
    assert diag["executable"] == "ngspice_con.exe"
    assert diag["version"] == "ngspice-46"
    assert diag["batch_runnable"] is True
