"""Saved-run summaries and standard electrical figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.artifacts import write_json
from pcd.case import load_case
from pcd.probes import signal_catalog
from pcd.reporting import analyze_run
from pcd.sim_core import prepare_case, write_sim_record
from pcd.simulation_input import resolve_simulation_case


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


def _saved_run(tmp_path, name, case_text, waveform, frequency_response=None):
    """Publish controlled observations through the current simulation writer."""

    case_path = tmp_path / f"{name}.yaml"
    case_path.write_text(case_text, encoding="utf-8")
    case = load_case(case_path)
    record = prepare_case(case, run_root=tmp_path, run_id=name)
    record.status = "ok"
    waveform = waveform.copy()
    if "load_voltage_V" not in waveform:
        waveform["load_voltage_V"] = waveform["voltage_V"]
    waveform.to_csv(record.run_dir / record.waveform_file, index=False)
    if frequency_response is not None:
        record.frequency_response_file = "data/ac.csv"
        frequency_response = frequency_response.copy()
        for part in ("re", "im"):
            frequency_response[f"voltage_V_{part}"] = frequency_response[f"load_voltage_V_{part}"]
        frequency_response.to_csv(record.run_dir / record.frequency_response_file, index=False)
    simulation = resolve_simulation_case(case)
    write_json(
        record.run_dir / record.signals_file,
        signal_catalog(
            simulation.probes,
            record.measurement,
            transient_columns=tuple(waveform.columns) if not waveform.empty else (),
            ac_columns=tuple(frequency_response.columns) if frequency_response is not None else (),
        ),
    )
    write_sim_record(record)
    return record.run_dir


def test_analyze_run_regenerates_summary_and_figures_from_saved_artifacts(tmp_path, monkeypatch):
    reference = 100.0
    case_text = (
        "schema: case_yaml.v1\n"
        "case_id: saved_ac\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: '0', amplitude_V: 1, frequency_Hz: 10000000}\n"
        "circuit: {builder: from_yaml, output_node: out, components: [{ref: R1, n1: src, n2: out, value: 50}]}\n"
        "load: {name: resistor, ports: {p: out, n: '0'}, reference_plane: load_terminal, R_ohm: 50}\n"
        f"measurement: {{voltage_node: out, current_source: Vsrc, reference_impedance_ohm: {reference}}}\n"
        "solver: {name: ngspice_cli, ac: {frequency_Hz: 10000000}}\n"
    )
    ac = _ac(reference, 0.0, (1e7,)).assign(load_voltage_V_re=1.0, load_voltage_V_im=0.0)
    empty = pd.DataFrame(columns=["time_s", "voltage_V", "current_A"])
    run_dir = _saved_run(tmp_path, "run", case_text, empty, ac)

    report_dir = tmp_path / "report"
    summary = analyze_run(run_dir, out_dir=report_dir)

    assert summary["schema"] == "run_analysis.v1"
    assert summary["scope"].startswith("electrical measurements only")
    assert summary["analyses"]["ac"]["resistance_ohm"] == pytest.approx(reference)
    assert summary["analyses"]["ac"]["forward_power_W"] == pytest.approx(1 / (2 * reference))
    assert summary["analyses"]["ac"]["reflected_power_W"] == pytest.approx(0, abs=1e-15)
    assert summary["analyses"]["ac"]["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert summary["analyses"]["ac"]["return_loss_db"] == pytest.approx(600.0)
    assert set(summary["artifacts"]) >= {"summary", "frequency_response", "operating_point"}
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "frequency_response.png").stat().st_size > 0
    assert (report_dir / "operating_point.png").stat().st_size > 0
    provenance = summary["provenance"]
    assert provenance["source_integrity"] == "verified"
    assert len(provenance["implementation_sha256"]) == 64
    assert provenance["inputs"]["frequency_response"]["sha256"]
    assert provenance["environment"]["libraries"]["numpy"]
    original = (report_dir / "summary.json").read_bytes()
    repeated = analyze_run(run_dir, out_dir=report_dir)
    assert repeated["provenance"]["analysis_id"] == provenance["analysis_id"]
    assert repeated["output_dir"] != summary["output_dir"]
    assert (report_dir / "summary.json").read_bytes() == original
    changed = analyze_run(run_dir, out_dir=report_dir, frequency_hz=1e7)
    assert changed["provenance"]["analysis_id"] != provenance["analysis_id"]
    monkeypatch.chdir(report_dir)
    relative = analyze_run(run_dir, out_dir=".")
    assert relative["output_dir"] == str(tmp_path / "report_003")


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
    run_dir = _saved_run(tmp_path, "transient_run", case_text, _waveform(257))

    summary = analyze_run(run_dir, output={"schematic": False, "formats": ["png", "svg"]})

    periodic = summary["analyses"]["transient"]["periodic"]
    assert periodic["status"] == "ok"
    assert periodic["load_real_power_W"] == pytest.approx(25.0, rel=2e-3)
    assert periodic["periodic_settled"] is True
    energy = summary["analyses"]["transient"]["energy"]
    assert energy["source_net_energy_J"] == pytest.approx(4e-4, rel=2e-3)
    assert energy["load_net_energy_J"] == pytest.approx(1e-4, rel=2e-3)
    power = pd.read_csv(run_dir / "analysis" / summary["artifacts"]["instantaneous_power"])
    assert power["load_net_energy_J"].iloc[-1] == pytest.approx(energy["load_net_energy_J"])
    assert set(summary["artifacts"]) >= {
        "summary",
        "transient_response",
        "operating_point",
        "harmonic_spectrum",
    }
    assert (run_dir / "analysis" / "summary.json").exists()
    report_dir = run_dir / "analysis"
    for figure in ("transient_response", "harmonic_spectrum"):
        for extension in ("png", "svg"):
            assert (report_dir / f"{figure}.{extension}").stat().st_size > 0
    harmonic_table = pd.read_csv(report_dir / summary["artifacts"]["harmonic_metrics"])
    fundamental = harmonic_table[harmonic_table["harmonic"] == 1]
    assert fundamental["signal"].tolist() == ["load_voltage_V", "load_current_A"]
    assert fundamental["amplitude"].tolist() == pytest.approx([50, 1], rel=2e-3)
    assert fundamental["relative_dBc"].tolist() == [0, 0]


def test_saved_sweep_interpolates_phasors_before_impedance_and_reflection(tmp_path):
    case_text = (
        "schema: case_yaml.v1\ncase_id: interpolated\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: 0, frequency_Hz: 2000000}\n"
        "circuit: {builder: from_yaml, output_node: out, components: []}\n"
        "load: {name: none}\nsolver: {ac: {sweep: lin, points: 2, start_Hz: 1000000, stop_Hz: 3000000}}\n"
    )
    ac = _ac(100.0, 0.0, (1e6, 3e6)).assign(
        current_re=[-0.01, -0.03],
        load_voltage_V_re=1.0,
        load_voltage_V_im=0.0,
    )
    empty = pd.DataFrame(columns=["time_s", "voltage_V", "current_A"])
    run_dir = _saved_run(tmp_path, "sweep", case_text, empty, ac)
    summary = analyze_run(run_dir)
    metrics = summary["analyses"]["ac"]
    assert metrics["resistance_ohm"] == pytest.approx(50.0)
    assert metrics["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["frequency_basis"] == "interpolated"
    assert metrics["frequency_bracket_Hz"] == [1e6, 3e6]
    rows = pd.read_csv(Path(summary["output_dir"]) / "operating_metrics.csv")
    assert rows.loc[0, "frequency_lower_Hz"] == 1e6
    assert rows.loc[0, "frequency_upper_Hz"] == 3e6
    ac.iloc[:1].to_csv(run_dir / "data" / "ac.csv", index=False)
    with pytest.raises(ValueError, match="recorded frequency_response"):
        analyze_run(run_dir)


@pytest.mark.parametrize("drive", [0.0, 1.0])
def test_ac_sweep_reports_without_a_periodic_fundamental(tmp_path, drive):
    case_text = (
        "schema: case_yaml.v1\ncase_id: dc_ac_sweep\n"
        f"source: {{type: dc_voltage, name: Vsrc, p: src, n: 0, voltage_V: 1, ac_magnitude_V: {drive}}}\n"
        "circuit: {builder: from_yaml, output_node: out, components: []}\n"
        "load: {name: none}\nsolver: {ac: {sweep: lin, points: 3, start_Hz: 1000000, stop_Hz: 3000000}}\n"
    )
    ac = _ac(50.0, 0.0, (1e6, 2e6, 3e6)).assign(
        voltage_re=drive, current_re=-drive / 50, load_voltage_V_re=drive, load_voltage_V_im=0.0
    )
    empty = pd.DataFrame(columns=["time_s", "voltage_V", "current_A"])
    run_dir = _saved_run(tmp_path, "sweep", case_text, empty, ac)

    summary = analyze_run(run_dir)
    assert summary["status"] == ("complete" if drive else "partial")
    assert summary["measurement"]["ac_frequency_Hz"] is None
    if drive:
        assert summary["analyses"]["ac"]["sweep"]["reflection_sample_count"] == 3
    else:
        assert "no independent AC excitation" in summary["analyses"]["ac"]["source_port_unavailable_reason"]
    assert "frequency_metrics" in summary["artifacts"]
    assert "operating_metrics" not in summary["artifacts"]

    selected = analyze_run(run_dir, frequency_hz=2e6)
    if drive:
        assert selected["analyses"]["ac"]["reflection_magnitude"] == 0.0
    else:
        assert "reflection_magnitude" not in selected["analyses"]["ac"]
    assert "NaN" not in (run_dir / "analysis_001" / "summary.json").read_text(encoding="utf-8")


def test_ac_reporting_preserves_probes_when_the_observed_source_is_not_driven(tmp_path):
    case_text = (
        "schema: case_yaml.v1\ncase_id: separate_ports\n"
        "sources:\n"
        "  - {type: dc_voltage, name: Vdrive, p: src, voltage_V: 0, ac_magnitude_V: 1}\n"
        "  - {type: dc_voltage, name: Vobserve, p: out, voltage_V: 0}\n"
        "circuit: {builder: from_yaml, output_node: src, components: [{ref: R1, n1: src, n2: out, value: 50}]}\n"
        "measurement: {current_source: Vobserve}\n"
        "solver: {ac: {frequency_Hz: 1000000, source: Vdrive}}\n"
    )
    ac = _ac(50.0, 0.0, (1e6,)).assign(voltage_re=0.0, load_voltage_V_re=1.0, load_voltage_V_im=0.0)
    empty = pd.DataFrame(columns=["time_s", "voltage_V", "current_A"])
    run_dir = _saved_run(tmp_path, "other_port", case_text, empty, ac)

    summary = analyze_run(run_dir)
    assert "do not establish the AC source port" in summary["analyses"]["ac"]["source_port_unavailable_reason"]
    derived = pd.read_csv(run_dir / "analysis" / summary["artifacts"]["frequency_metrics"])
    assert "reflection_magnitude" not in derived
    assert derived["load_voltage_V_re"].iloc[0] == 1.0
    assert summary["output"]["plots"][0]["recipe"]["type"] == "ac"


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
    run_dir = _saved_run(tmp_path, "dc_transient_run", case_text, waveform)

    summary = analyze_run(run_dir)

    assert summary["measurement"]["ac_frequency_Hz"] is None
    assert summary["measurement"]["periodic_fundamental_Hz"] is None
    assert summary["analyses"]["transient"]["periodic"] == {
        "status": "not_applicable",
        "reason": "no periodic fundamental was declared; basic transient statistics remain available",
    }
    assert set(summary["artifacts"]) >= {"summary", "transient_response"}


def test_analyze_run_rejects_failed_records(tmp_path, make_case):
    record = prepare_case(make_case({}), run_root=tmp_path / "run")
    record.status = "failed"
    write_sim_record(record)
    with pytest.raises(ValueError, match="status 'failed'"):
        analyze_run(record.run_dir)
