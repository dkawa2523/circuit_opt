"""Saved-run summaries and standard electrical figures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pcd.probes import ComponentObservation
from pcd.reporting import analyze_run
from pcd.reporting.figures import (
    render_frequency_response,
    render_operating_point,
    render_transient_response,
)
from pcd.reporting.harmonics import render_harmonic_spectrum


def _waveform(n: int = 64) -> pd.DataFrame:
    time_s = np.linspace(0.0, 4e-6, n)
    phase = 2 * np.pi * 1e6 * time_s
    return pd.DataFrame(
        {
            "time_s": time_s,
            "voltage_V": 50.0 * np.sin(phase),
            "current_A": -2.0 * np.sin(phase),
            "source_voltage_V": 100.0 * np.sin(phase),
            "load_current_A": np.sin(phase),
        }
    )


def _ac(resistance: float, reactance: float, frequencies: tuple[float, ...] = (1e6, 1e7, 1e8)) -> pd.DataFrame:
    impedance = complex(resistance, reactance)
    current = -(1.0 / impedance)
    return pd.DataFrame(
        {
            "frequency_Hz": frequencies,
            "voltage_re": [1.0] * len(frequencies),
            "voltage_im": [0.0] * len(frequencies),
            "current_re": [current.real] * len(frequencies),
            "current_im": [current.imag] * len(frequencies),
        }
    )


def _saved_run(tmp_path, name, case_id, case_text, waveform, frequency_response=None):
    """Write the same public/data/debug layout produced by ``sim-run``."""

    run_dir = tmp_path / name
    data_dir = run_dir / "data"
    debug_dir = run_dir / "debug"
    data_dir.mkdir(parents=True)
    debug_dir.mkdir()
    (debug_dir / "case.yaml").write_text(case_text, encoding="utf-8")
    waveform.to_csv(data_dir / "transient.csv", index=False)
    artifacts = {"case": "debug/case.yaml", "waveform": "data/transient.csv"}
    if frequency_response is not None:
        frequency_response.to_csv(data_dir / "ac.csv", index=False)
        artifacts["frequency_response"] = "data/ac.csv"
    manifest = {
        "schema": "simulation_record.v2",
        "case_id": case_id,
        "status": "ok",
        "solver": "ngspice_cli",
        "run_dir": str(run_dir),
        "params": {},
        "artifacts": artifacts,
        "provenance": {"solver": {"version": "ngspice-test"}},
    }
    (debug_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_standard_frequency_and_transient_figures_are_written(tmp_path):
    frequency = render_frequency_response(_ac(50.0, -20.0), tmp_path / "frequency.png", marker_hz=1e7)
    transient = render_transient_response(
        _waveform(),
        tmp_path / "transient.png",
        load_current_column="load_current_A",
    )

    assert frequency.stat().st_size > 0
    assert transient.stat().st_size > 0


def test_empty_response_data_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="frequency response is empty"):
        render_frequency_response(pd.DataFrame(), tmp_path / "frequency.png")
    with pytest.raises(ValueError, match="waveform is empty"):
        render_transient_response(pd.DataFrame(), tmp_path / "transient.png")


def test_operating_point_figure_is_only_created_for_available_metrics(tmp_path):
    component = ComponentObservation("L1", "src", "out", 0.5)
    assert render_operating_point({}, (component,), tmp_path / "empty.png") is None

    out = render_operating_point(
        {
            "source_real_power_W": 100.0,
            "load_real_power_W": 90.0,
            "component_L1_voltage_peak_V": 350.0,
            "component_L1_current_rms_A": 1.2,
        },
        (component,),
        tmp_path / "operating.png",
    )
    assert out is not None
    assert out.stat().st_size > 0


def test_harmonic_figure_compares_voltage_and_current_to_their_fundamentals(tmp_path):
    out = render_harmonic_spectrum(
        {
            "voltage_harmonic_amplitude_V": {"h1": 50.0, "h2": 1.0, "h3": 0.5},
            "current_harmonic_amplitude_A": {"h1": 1.0, "h2": 0.02, "h3": 0.01},
        },
        tmp_path / "harmonics.png",
    )

    assert out is not None
    assert out.stat().st_size > 0


def test_analyze_run_regenerates_summary_and_figures_from_saved_artifacts(tmp_path):
    case_text = (
        "schema: case_yaml.v1\n"
        "case_id: saved_ac\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: '0', amplitude_V: 1, frequency_Hz: 10000000}\n"
        "circuit: {builder: from_yaml, output_node: out, components: [{ref: R1, n1: src, n2: out, value: 50}]}\n"
        "load: {name: resistor, ports: {p: out, n: '0'}, reference_plane: load_terminal, R_ohm: 50}\n"
        "measurement: {voltage_node: out, current_source: Vsrc, reference_impedance_ohm: 50}\n"
        "solver: {name: ngspice_cli, ac: {frequency_Hz: 10000000}}\n"
    )
    ac = _ac(50.0, 0.0, (1e7,)).assign(load_voltage_V_re=1.0, load_voltage_V_im=0.0)
    empty = pd.DataFrame(columns=["time_s", "voltage_V", "current_A"])
    run_dir = _saved_run(tmp_path, "run", "saved_ac", case_text, empty, ac)

    report_dir = tmp_path / "report"
    summary = analyze_run(run_dir, out_dir=report_dir)

    assert summary["schema"] == "run_analysis.v1"
    assert summary["scope"].startswith("electrical measurements only")
    assert summary["analyses"]["ac"]["resistance_ohm"] == pytest.approx(50.0)
    assert summary["analyses"]["ac"]["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert summary["analyses"]["ac"]["return_loss_db"] == pytest.approx(600.0)
    assert set(summary["artifacts"]) == {"summary", "frequency_response", "operating_point"}
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "frequency_response.png").stat().st_size > 0
    assert (report_dir / "operating_point.png").stat().st_size > 0


def test_analyze_run_recomputes_periodic_metrics_from_saved_waveform(tmp_path):
    case_text = (
        "schema: case_yaml.v1\n"
        "case_id: saved_transient\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: '0', amplitude_V: 100, frequency_Hz: 1000000}\n"
        "circuit: {builder: from_yaml, output_node: out, components: [{raw: 'Rnetwork src out 50'}]}\n"
        "load: {name: resistor, ports: {p: out, n: '0'}, reference_plane: load_terminal, R_ohm: 50}\n"
        "measurement: {load_current: auto}\n"
        "solver: {name: ngspice_cli, tran: {step_s: 1.0e-9, stop_s: 4.0e-6}}\n"
    )
    run_dir = _saved_run(tmp_path, "transient_run", "saved_transient", case_text, _waveform(257))

    summary = analyze_run(run_dir)

    periodic = summary["analyses"]["transient"]["periodic"]
    assert periodic["status"] == "ok"
    assert periodic["load_real_power_W"] == pytest.approx(25.0, rel=2e-3)
    assert periodic["periodic_settled"] is True
    assert set(summary["artifacts"]) == {
        "summary",
        "transient_response",
        "operating_point",
        "harmonic_spectrum",
    }
    assert (run_dir / "analysis" / "summary.json").exists()
    assert (run_dir / "analysis" / "harmonic_spectrum.png").stat().st_size > 0


def test_analyze_run_marks_dc_transient_periodic_metrics_not_applicable(tmp_path):
    case_text = (
        "schema: case_yaml.v1\n"
        "case_id: saved_dc_transient\n"
        "source: {type: dc_voltage, name: Vsrc, p: src, n: '0', voltage_V: 10}\n"
        "circuit:\n"
        "  builder: from_yaml\n"
        "  output_node: out\n"
        "  components: [{ref: R1, n1: src, n2: out, value: 50, observe: true}]\n"
        "load: {name: none}\n"
        "measurement: {voltage_node: out, current_source: Vsrc}\n"
        "solver: {name: ngspice_cli, tran: {step_s: 1.0e-7, stop_s: 1.0e-6}}\n"
    )
    time_s = np.linspace(0.0, 1.0e-6, 11)
    waveform = pd.DataFrame(
        {
            "time_s": time_s,
            "voltage_V": np.linspace(0.0, 5.0, len(time_s)),
            "current_A": np.linspace(-0.2, -0.1, len(time_s)),
            "source_voltage_V": np.full(len(time_s), 10.0),
            "component_R1_voltage_V": np.linspace(10.0, 5.0, len(time_s)),
            "component_R1_current_A": np.linspace(0.2, 0.1, len(time_s)),
        }
    )
    run_dir = _saved_run(tmp_path, "dc_transient_run", "saved_dc_transient", case_text, waveform)

    summary = analyze_run(run_dir)

    assert summary["measurement"]["frequency_Hz"] is None
    assert summary["analyses"]["transient"]["periodic"] == {
        "status": "not_applicable",
        "reason": "no periodic fundamental was declared; basic transient statistics remain available",
    }
    assert set(summary["artifacts"]) == {"summary", "transient_response"}


def test_analyze_run_rejects_failed_records(tmp_path):
    debug = tmp_path / "debug"
    debug.mkdir()
    manifest = debug / "manifest.json"
    manifest.write_text(json.dumps({"status": "failed", "run_dir": str(tmp_path)}), encoding="utf-8")
    with pytest.raises(ValueError, match="status 'failed'"):
        analyze_run(manifest)
