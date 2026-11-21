"""Physical regressions for the network and source-observation boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.api import execute_case, load_case, measure_response, run_case_study, run_network
from pcd.core.models import MeasurementUnavailableError
from pcd.records import read_sim_record
from pcd.reporting import analyze_run
from pcd.validation import validate_case

pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("ngspice")]


def test_network_recovers_scattering_with_unequal_references_and_mismatched_fixtures(tmp_path):
    (tmp_path / "series.cir").write_text("Rbridge p1 p2 40\n", encoding="utf-8")
    specification = tmp_path / "network.yaml"
    specification.write_text(
        "schema: pcd.network.v1\n"
        "dut: {file: series.cir}\n"
        "ports:\n"
        "  - {name: P1, p: p1, reference_impedance_ohm: 50, termination_ohm: 80}\n"
        "  - {name: P2, p: p2, reference_impedance_ohm: 75, termination_ohm: 90}\n"
        "ac: {sweep: lin, points: 3, start_Hz: 1e6, stop_Hz: 2e6}\n",
        encoding="utf-8",
    )
    destination = tmp_path / "result"
    result = run_network(specification, destination)
    assert result["status"] == "complete"
    assert result["solved_points"] == 3
    table = pd.read_csv(destination / result["artifacts"]["scattering"])
    transfer = 2 * np.sqrt(50 * 75)
    expected = np.array([40 + 75 - 50, transfer, transfer, 40 + 50 - 75]) / (40 + 50 + 75)
    np.testing.assert_allclose(table["s_re"].to_numpy().reshape(3, 4), np.tile(expected, (3, 1)), atol=1e-12)
    np.testing.assert_allclose(table["s_im"], 0, atol=1e-12)
    assert "file: dut.cir" in (destination / "network.yaml").read_text(encoding="utf-8")

    invalid = specification.read_text(encoding="utf-8").replace("p: p2", "p: Rbridge")
    specification.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="not established by a top-level DUT connection"):
        run_network(specification, tmp_path / "invalid")

    specification.write_text(invalid.replace("p: Rbridge", "p: p2"), encoding="utf-8")
    (tmp_path / "series.cir").write_text("Rbridge p1 p2 40\n.option cshunt=1p\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attach to fixture nodes"):
        run_network(specification, tmp_path / "shunted")


def test_generated_second_drive_blocks_impedance_but_preserves_ac_observations(tmp_path):
    (tmp_path / "builder.py").write_text(
        "from pcd.api import Circuit, register_simulation\n"
        "@register_simulation('circuit', 'two_actual_ac_drives')\n"
        "def build(case, params):\n"
        "    circuit = Circuit(output_node='src')\n"
        "    circuit.add('Rinput', 'src', '0', 50)\n"
        "    if params['auxiliary'] == 'enabled':\n"
        "        circuit.raw('Iaux src 0 DC 0 AC 0.01')\n"
        "    return circuit\n",
        encoding="utf-8",
    )
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: actual_drives\n"
        "plugins: [builder.py]\n"
        "variables: {auxiliary: {role: design, choices: [enabled, disabled], default: enabled}}\n"
        "run: {trials: 2}\n"
        "optimizer: {name: grid}\n"
        "source: {type: dc_voltage, name: Vdrive, p: src, voltage_V: 0, ac_magnitude_V: 1}\n"
        "circuit: {builder: two_actual_ac_drives, output_node: src}\n"
        "measurement: {current_source: Vdrive}\n"
        "solver: {ac: {sweep: lin, points: 1, start_Hz: 1e6, stop_Hz: 1e6}}\n"
        "target: {objective: impedance_match, constraints: {metric_bounds: {reflected_power_fraction: {max: 0.1}}}}\n",
        encoding="utf-8",
    )
    case = load_case(case_path)
    assert validate_case(case, strict=True).ok
    run = execute_case(case, run_root=tmp_path)
    assert run.record.status == "ok"
    assert run.response.frequency_response is not None
    with pytest.raises(MeasurementUnavailableError, match="multiple AC excitations"):
        measure_response(case, run.response)
    summary = analyze_run(run.record.run_dir, output={"schematic": False, "plots": []})
    scope = summary["measurement"]["source_scope"]
    assert scope["ac_excitations"] == ["Vdrive", "Iaux"]
    assert "multiple AC excitations" in summary["analyses"]["ac"]["source_port_unavailable_reason"]
    study = run_case_study(case, run_root=tmp_path / "search")
    assert study["n_failed_evaluations"] == 1
    assert study["best"]["candidate"]["values"] == {"auxiliary": "disabled"}
    assert study["best"]["status"] == "meets_declared_acceptance"


@pytest.mark.parametrize(
    ("deck", "expected_resistance"),
    [
        ("Idrive 0 jun AC 1\nVmeter src jun 0\nRmain src 0 50\n", 50.0),
        ("Idrive 0 jun AC 1\nVmeter jun src 0\nRmain src 0 50\n", None),
        ("Idrive 0 jun AC 1\nVmeter src jun 0\nRmain src 0 50\nRshunt jun 0 100\n", None),
        ("Idrive 0 jun AC 1\nVmeter src jun 0\nRmain src 0 50\n.option rshunt=100\n", None),
        ("Vdrive src 0 AC 1\nRmain src 0 50\nRbranch src sense 100\nVmeter 0 sense 0\n", None),
    ],
)
def test_imported_meter_requires_series_source_correspondence(tmp_path, deck, expected_resistance):
    (tmp_path / "meter.cir").write_text(deck, encoding="utf-8")
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: series_meter\n"
        "circuit: {builder: from_netlist, netlist_file: meter.cir, output_node: src}\n"
        "measurement: {current_source: Vmeter, source_voltage: v(src)}\n"
        "solver: {ac: {sweep: lin, points: 1, start_Hz: 1e6, stop_Hz: 1e6}}\n"
        "target: {objective: impedance_match}\n",
        encoding="utf-8",
    )
    case = load_case(case_path)
    run = execute_case(case, run_root=tmp_path)
    assert run.record.status == "ok"
    assert run.response.frequency_response is not None
    if expected_resistance is not None:
        assert measure_response(case, run.response)["resistance_ohm"] == pytest.approx(expected_resistance)
    else:
        with pytest.raises(MeasurementUnavailableError, match="do not establish the AC source port"):
            measure_response(case, run.response)


@pytest.mark.parametrize("voltage", ["v(load_port)", "v(src)"])
def test_raw_source_requires_the_same_voltage_and_current_plane(tmp_path, voltage):
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: raw_source\n"
        "source: {raw: 'Vdrive src 0 DC 0 AC 2 SIN(0 100 1Meg)'}\n"
        "circuit:\n"
        "  builder: from_yaml\n"
        "  output_node: load_port\n"
        "  components: [{ref: Rseries, n1: src, n2: load_port, value: 50}]\n"
        "load: {name: resistor, ports: {p: load_port, n: 0}, R_ohm: 50}\n"
        f"measurement: {{current_source: Vdrive, source_voltage: '{voltage}', load_current: auto}}\n"
        "solver: {ac: {frequency_Hz: 1000000}, tran: {step_s: 1e-9, stop_s: 8e-6}}\n"
        "target: {objective: impedance_match, fundamental_Hz: 1000000}\n",
        encoding="utf-8",
    )
    case = load_case(case_path)
    run = execute_case(case, run_root=tmp_path)
    assert run.record.status == "ok"
    if voltage == "v(src)":
        assert measure_response(case, run.response)["reflected_power_fraction"] == pytest.approx(1 / 9)
    else:
        with pytest.raises(MeasurementUnavailableError, match="do not establish the AC source port"):
            measure_response(case, run.response)

    case.data["target"]["objective"] = "rf_load"
    metrics = measure_response(case, run.response)
    assert metrics["load_real_power_W"] == pytest.approx(25, rel=1e-3)
    if voltage == "v(src)":
        assert metrics["network_loss_W"] == pytest.approx(25, rel=1e-3)
    else:
        assert metrics["source_real_power_W"] is None
        assert metrics["network_loss_W"] is None
        assert "independent transient source port" in metrics["network_loss_W_unavailable_reason"]
        report = analyze_run(run.record.run_dir, output={"schematic": False, "plots": []})
        assert "resistance_ohm" not in report["analyses"]["ac"]
        transient = report["analyses"]["transient"]
        assert transient["periodic"]["source_real_power_W"] is None
        assert "source_net_energy_J" not in transient["energy"]
        assert transient["energy"]["load_net_energy_J"] == pytest.approx(2e-4, rel=1e-3)
        signals_path = Path(report["output_dir"]) / report["artifacts"]["signals"]
        signals = json.loads(signals_path.read_text(encoding="utf-8"))["signals"]
        assert signals["source_voltage_V"]["columns"]["transient"] == ["source_voltage_V"]
        assert signals["current_A"]["columns"]["transient"] == ["current_A"]


def test_ac_study_requires_a_solved_operating_frequency(tmp_path):
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: coarse_sweep\n"
        "source: {type: dc_voltage, name: Vdrive, p: src, voltage_V: 0, ac_magnitude_V: 2}\n"
        "circuit:\n"
        "  builder: from_yaml\n"
        "  output_node: src\n"
        "  components:\n"
        "    - {ref: R1, n1: src, n2: 0, value: 50}\n"
        "    - {ref: L1, n1: src, n2: 0, value: 1e-6}\n"
        "    - {ref: C1, n1: src, n2: 0, value: 3.377372788077926e-8}\n"
        "solver: {ac: {sweep: lin, points: 2, start_Hz: 500000, stop_Hz: 1500000}}\n"
        "target: {objective: impedance_match, fundamental_Hz: 1000000}\n",
        encoding="utf-8",
    )
    case = load_case(case_path)
    run = execute_case(case, run_root=tmp_path)
    assert run.record.status == "ok"
    with pytest.raises(MeasurementUnavailableError, match="requires a solved sample at 1e\\+06 Hz"):
        measure_response(case, run.response)
    report = analyze_run(run.record.run_dir, output={"schematic": False, "plots": []})
    assert report["analyses"]["ac"]["frequency_basis"] == "interpolated"
    assert report["analyses"]["ac"]["frequency_bracket_Hz"] == [500000, 1500000]

    case.data["solver"]["ac"] = {"frequency_Hz": 1000000}
    exact = execute_case(case, run_root=tmp_path)
    assert exact.record.status == "ok"
    metrics = measure_response(case, exact.response)
    assert metrics["frequency_basis"] == "simulated"
    assert metrics["reflected_power_fraction"] == pytest.approx(0.6375561855)


def test_unmetered_behavioral_source_keeps_port_power_but_prevents_transient_balance(tmp_path):
    (tmp_path / "network.cir").write_text(
        "V1 in 0 DC 0 AC 2 SIN(0 10 1meg)\nR1 in out 50\nBextra aux 0 V=7*sin(6283185.307179586*time)\nR2 aux out 50\n",
        encoding="utf-8",
    )
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: unmetered_source\n"
        "circuit: {builder: from_netlist, netlist_file: network.cir, output_node: out}\n"
        "measurement: {current_source: V1, source_voltage: v(in), load_current: auto}\n"
        "load: {name: resistor, R_ohm: 50, ports: {p: out, n: 0}}\n"
        "target: {objective: rf_load, fundamental_Hz: 1e6}\n"
        "solver: {ac: {frequency_Hz: 1e6}, tran: {step_s: 1e-9, stop_s: 8e-6}}\n",
        encoding="utf-8",
    )
    case = load_case(case_path)
    run = execute_case(case, run_root=tmp_path)
    assert run.record.status == "ok"
    metrics = measure_response(case, run.response)
    assert metrics["source_real_power_W"] == pytest.approx(13 / 30, rel=1e-3)
    assert metrics["load_real_power_W"] == pytest.approx(289 / 900, rel=1e-3)
    for name in ("network_loss_W", "transfer_efficiency"):
        assert metrics[name] is None
        assert "Bextra" in metrics[f"{name}_unavailable_reason"]

    record = read_sim_record(run.record.run_dir)
    # Reanalysis applies the current balance rule even when the original run
    # was published before the rule detected this source.
    record["diagnostics"]["source_scope"]["transient_unavailable_reason"] = None
    report = analyze_run(record, output={"schematic": False, "plots": []})
    assert report["analyses"]["ac"]["resistance_ohm"] == pytest.approx(75)
    periodic = report["analyses"]["transient"]["periodic"]
    assert periodic["source_real_power_W"] == pytest.approx(metrics["source_real_power_W"])
    assert periodic["load_real_power_W"] == pytest.approx(metrics["load_real_power_W"])
    assert periodic["network_loss_W"] is None


def test_attached_load_power_uses_its_own_current_and_keeps_the_named_probe(make_case, tmp_path):
    case = make_case(
        {
            "case_id": "attached_load_current",
            "source": {"type": "sine_voltage", "name": "Vdrive", "amplitude_V": 2, "frequency_Hz": 1e6},
            "circuit": {
                "output_node": "out",
                "components": [
                    {"raw": "Vmeter src feed DC 0"},
                    {"ref": "Rseries", "n1": "feed", "n2": "out", "value": 50},
                    {"ref": "Rshunt", "n1": "out", "n2": "0", "value": 50},
                ],
            },
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "feed_current", "probes": {"feed_current": "i(Vmeter)"}},
            "solver": {"ac": {"frequency_Hz": 1e6}, "tran": {"step_s": 1e-9, "stop_s": 8e-6}},
            "target": {"objective": "impedance_match"},
        }
    )
    assert validate_case(case, strict=True).ok
    run = execute_case(case, run_root=tmp_path / "runs")
    assert run.record.status == "ok"
    assert run.record.measurement["load_current_binding"] == "inserted_load_ammeter"
    ac = run.response.frequency_response
    assert ac is not None
    assert ac["feed_current_re"].iloc[0] == pytest.approx(2 * ac["load_current_A_re"].iloc[0])
    actual_power = 1 / 225
    assert measure_response(case, run.response)["load_real_power_W"] == pytest.approx(actual_power)

    report = analyze_run(run.record.run_dir, output={"schematic": False, "plots": []})
    assert report["measurement"]["load_current"] == "load_current_A"
    assert report["analyses"]["transient"]["periodic"]["load_real_power_W"] == pytest.approx(actual_power, rel=1e-3)
