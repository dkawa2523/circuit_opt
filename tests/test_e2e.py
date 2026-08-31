"""End-to-end journeys, exercised through the installed console script.

Marked ``e2e`` so they run in ``quality-nightly`` rather than on every edit.
They cover direct simulation, the unified scenario-aware study path, and the
physical ngspice path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FIXTURES = ROOT / "tests" / "fixtures"
RC_CASE = EXAMPLES / "advanced" / "generic_rc_filter.yaml"
ADVANCED_CASE = FIXTURES / "advanced_case.yaml"
FREQUENCY_TABLE_CASE = EXAMPLES / "rf_impedance_frequency_table.yaml"
COMPONENT_STRESS_CASE = EXAMPLES / "rf_component_stress.yaml"
ngspice_available = shutil.which("ngspice_con.exe") or shutil.which("ngspice")


def pcd(*args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI the way a user does, in a separate process."""

    result = subprocess.run(
        [sys.executable, "-m", "pcd.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if expect_success:
        assert result.returncode == 0, f"pcd {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
    return result


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_simulation_only_journey_produces_the_boundary_artifacts(tmp_path):
    """RF/circuit engineer path: case -> netlist -> waveform, never metrics."""

    run_root = tmp_path / "sim_only"
    pcd("sim-run", str(RC_CASE), "--solver", "ngspice_cli", "--run-root", str(run_root))

    manifests = list(run_root.rglob("sim_manifest.json"))
    assert len(manifests) == 1
    run_dir = manifests[0].parent
    assert (run_dir / "waveform.csv").exists()
    assert (run_dir / "netlist.cir").exists()
    assert not (run_dir / "metrics.json").exists(), "simulation must never score"


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_scenario_aware_study_journey(tmp_path):
    run_root = tmp_path / "study"
    result = pcd(
        "run",
        str(ADVANCED_CASE),
        "--optimizer",
        "random",
        "--solver",
        "ngspice_cli",
        "--trials",
        "5",
        "--seed",
        "11",
        "--output",
        str(run_root),
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["n_candidates"] == 5
    assert payload["n_failed_evaluations"] == 0
    history = json.loads((Path(payload["run_root"]) / "study_history.json").read_text(encoding="utf-8"))
    assert len(history) == 5
    summary = tmp_path / "summary.csv"
    pcd("result-summary", payload["run_root"], "--out", str(summary))
    assert len(pd.read_csv(summary)) == 5


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_production_safe_journey_uses_strict_flags(tmp_path):
    """validate-case --strict gates the run; --strict-exit gates the batch."""

    result = pcd("validate-case", str(RC_CASE), "--strict", expect_success=False)
    assert result.returncode == 0

    run_root = tmp_path / "prod"
    pcd("sim-run", str(RC_CASE), "--solver", "ngspice_cli", "--run-root", str(run_root), "--strict-exit")


def test_netlist_visualization_journey(tmp_path):
    netlist = tmp_path / "level1.cir"
    pcd("sim-netlist", str(RC_CASE), "--out", str(netlist))
    image = tmp_path / "schematic.png"
    summary = tmp_path / "schematic.json"
    pcd("visualize-netlist", str(netlist), "--out", str(image), "--summary-json", str(summary))
    assert image.stat().st_size > 1000
    assert json.loads(summary.read_text(encoding="utf-8"))["n_edges"] >= 2


# --- real solver -----------------------------------------------------------


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_real_ngspice_transient_produces_a_physical_waveform(tmp_path):
    """The installed solver produces the physical transient artifact."""

    run_root = tmp_path / "ngspice"
    result = pcd("sim-run", str(RC_CASE), "--solver", "ngspice_cli", "--run-root", str(run_root), "--strict-exit")
    manifest = json.loads(result.stdout)
    assert manifest["status"] == "ok"
    assert manifest["solver"] == "ngspice_cli"
    assert manifest["provenance"]["solver"]["resolved_executable"]

    frame = pd.read_csv(Path(manifest["run_dir"]) / "waveform.csv")
    # The first three columns are the artifact contract; probes follow, and the
    # source voltage is always one of them so power flow is answerable.
    assert list(frame.columns)[:3] == ["time_s", "voltage_V", "current_A"]
    assert "source_voltage_V" in frame.columns
    assert len(frame) > 10
    assert frame["time_s"].is_monotonic_increasing
    assert frame.notna().all().all()

    stop = 2.0e-6  # advanced/generic_rc_filter.yaml solver.tran.stop_s
    assert frame["time_s"].iloc[-1] == pytest.approx(stop, rel=0.05)
    # RC low-pass driven by a 0..5 V pulse: the output must stay inside the rail.
    assert frame["voltage_V"].abs().max() <= 5.0 + 1e-6


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_real_ngspice_closed_loop_completes(tmp_path):
    run_root = tmp_path / "ngspice_loop"
    result = pcd(
        "run",
        str(RC_CASE),
        "--optimizer",
        "random",
        "--solver",
        "ngspice_cli",
        "--trials",
        "3",
        "--seed",
        "3",
        "--output",
        str(run_root),
        "--json",
    )
    payload = json.loads(result.stdout)
    assert payload["n_candidates"] == 3
    assert payload["n_failed_evaluations"] == 0
    assert payload["best"]["aggregates"]["loss"] >= 0.0


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_measured_frequency_points_are_solved_only_at_their_own_frequency(tmp_path):
    result = pcd(
        "run",
        str(FREQUENCY_TABLE_CASE),
        "--solver",
        "ngspice_cli",
        "--trials",
        "1",
        "--output",
        str(tmp_path / "frequency_table"),
        "--json",
    )
    payload = json.loads(result.stdout)
    candidate = json.loads((Path(payload["run_root"]) / "candidates" / "trial_0000.json").read_text(encoding="utf-8"))

    assert payload["n_evaluations"] == 3
    assert payload["n_failed_evaluations"] == 0
    for scenario in candidate["scenarios"]:
        selected = scenario["selected"]
        values = selected["request"]["scenario"]["values"]
        metrics = selected["metrics"]
        assert metrics["match_frequency_Hz"] == pytest.approx(values["rf_frequency_Hz"])
        assert values["load_resistance_ohm"] > 0
        assert "reflection_magnitude" in metrics


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_public_component_stress_and_loss_are_internally_consistent(tmp_path):
    result = pcd(
        "run",
        str(COMPONENT_STRESS_CASE),
        "--solver",
        "ngspice_cli",
        "--trials",
        "1",
        "--output",
        str(tmp_path / "component_stress"),
        "--json",
    )
    payload = json.loads(result.stdout)
    candidate = json.loads((Path(payload["run_root"]) / "candidates" / "trial_0000.json").read_text(encoding="utf-8"))
    metrics = candidate["scenarios"][0]["selected"]["metrics"]

    for ref, resistance in {"C1": 0.1, "L1": 0.5, "C2": 0.1}.items():
        current_rms = metrics[f"component_{ref}_current_rms_A"]
        assert current_rms > 0
        assert metrics[f"component_{ref}_voltage_peak_V"] > 0
        assert metrics[f"component_{ref}_voltage_rms_V"] > 0
        assert metrics[f"component_{ref}_loss_W"] == pytest.approx(current_rms**2 * resistance, rel=2e-5)
    assert metrics["source_current_rms_A"] > 0
    assert metrics["source_apparent_power_VA"] >= abs(metrics["source_real_power_W"])
    assert 0.0 < metrics["transfer_efficiency"] <= 1.0
    assert metrics["component_loss_balance_residual_W"] == pytest.approx(0.0, abs=2e-5)


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_power_accounting_closes_against_a_known_divider(tmp_path):
    """The probe -> generic RF-port metric chain, checked against algebra.

    A 100 V source drives Rs = 50 into Rload = 50, with a shunt capacitor across
    the source.  The shunt is the point: it draws current the electrode never
    sees, so an accounting that used the source current would over-report the
    delivered power.  The exact answers are 50 W leaving the source, 25 W into
    the load, 25 W burned in Rs.
    """

    case = tmp_path / "divider.yaml"
    case.write_text(
        """
schema: case_yaml.v1
case_id: power_divider
source: {type: sine_voltage, name: Vsrc, p: src, n: 0, amplitude_V: 100, frequency_Hz: 1000000}
circuit:
  builder: from_yaml
  output_node: electrode
  components:
    - {raw: "Cshunt src 0 1e-9"}
    - {raw: "Rs src electrode 50"}
    - {raw: "Vam electrode eload DC 0"}
    - {raw: "Rload eload 0 50"}
load: {name: none}
measurement:
  voltage_node: electrode
  current_source: Vsrc
  probes: ["i(Vam)"]
  load_current: "i(Vam)"
solver: {name: ngspice_cli, tran: {step_s: 1.0e-10, stop_s: 2.0e-5}}
""".lstrip(),
        encoding="utf-8",
    )
    result = pcd("sim-run", str(case), "--run-root", str(tmp_path / "runs"), "--strict-exit")
    run_dir = Path(json.loads(result.stdout)["run_dir"])

    frame = pd.read_csv(run_dir / "waveform.csv")
    assert "i(Vam)" in frame.columns, "the declared probe must reach the waveform"

    from pcd.analysis import rf_port_metrics

    metrics = rf_port_metrics(frame, 1.0e6, "i(Vam)")
    assert metrics["source_real_power_W"] == pytest.approx(50.0, rel=1e-3)
    assert metrics["load_real_power_W"] == pytest.approx(25.0, rel=1e-3)
    assert metrics["network_loss_W"] == pytest.approx(25.0, rel=1e-3)
    assert metrics["transfer_efficiency"] == pytest.approx(0.5, rel=1e-3)


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
@pytest.mark.parametrize(
    "case_name",
    ["impedance_point", "ccp_lumped", "icp_transformer", "icp_transformer_high_q"],
)
def test_each_rf_load_matches_its_analytic_impedance_in_ngspice(tmp_path, case_name):
    from pcd.analysis import at_frequency, input_impedance, read_ac
    from pcd.artifacts import yaml_dump
    from pcd.rf_loads import ccp_lumped_impedance, icp_effective_impedance

    f0 = 13.56e6
    configurations = {
        "impedance_point": (
            {"resistance_ohm": 20.0, "reactance_ohm": -80.0, "model_frequency_Hz": f0},
            complex(20.0, -80.0),
        ),
        "ccp_lumped": (
            {"R_eff_ohm": 25.0, "L_eff_H": 2e-7, "C_sheath_eq_F": 1.2e-10},
            ccp_lumped_impedance(f0, 25.0, 2e-7, 1.2e-10),
        ),
        "icp_transformer": (
            {
                "R_coil_ohm": 0.4,
                "L_coil_H": 2e-6,
                "reflected_inductance_H": 2.45e-7,
                "secondary_damping_rate_rad_s": 2 / 3e-7,
                "C_parallel_F": 2e-11,
            },
            icp_effective_impedance(f0, 0.4, 2e-6, 2.45e-7, 2 / 3e-7, 2e-11),
        ),
        "icp_transformer_high_q": (
            {
                "R_coil_ohm": 0.0,
                "L_coil_H": 2e-6,
                "reflected_inductance_H": 2.5e-7,
                "secondary_damping_rate_rad_s": 851999.927653552,
                "C_parallel_F": 0.0,
            },
            icp_effective_impedance(f0, 0.0, 2e-6, 2.5e-7, 851999.927653552, 0.0),
        ),
    }
    load, expected = configurations[case_name]
    model = case_name.removesuffix("_high_q")
    case = tmp_path / f"{case_name}.yaml"
    case.write_text(
        yaml_dump(
            {
                "schema": "case_yaml.v1",
                "case_id": f"e2e_{case_name}",
                "source": {
                    "type": "sine_voltage",
                    "name": "Vsrc",
                    "p": "src",
                    "n": 0,
                    "amplitude_V": 1,
                    "frequency_Hz": f0,
                },
                "circuit": {
                    "builder": "from_yaml",
                    "output_node": "port",
                    "components": [{"raw": "Rfixture src port 1e-9"}],
                },
                "load": {
                    "name": model,
                    "ports": {"p": "port", "n": 0},
                    "reference_plane": "port",
                    "characterization": {"origin": "analytic_e2e"},
                    **load,
                },
                "solver": {
                    "name": "ngspice_cli",
                    "ac": {"sweep": "lin", "points": 3, "start_Hz": f0 / 2, "stop_Hz": 1.5 * f0},
                },
                "target": {"objective": "impedance_match"},
            }
        ),
        encoding="utf-8",
    )
    result = pcd("sim-run", str(case), "--run-root", str(tmp_path / "runs"), "--strict-exit")
    run_dir = Path(json.loads(result.stdout)["run_dir"])
    row = at_frequency(input_impedance(read_ac(run_dir / "ac.csv")), f0)
    assert complex(row["resistance_ohm"], row["reactance_ohm"]) == pytest.approx(expected, rel=2e-5, abs=1e-7)
    assert pd.read_csv(run_dir / "waveform.csv").empty


@pytest.mark.skipif(not ngspice_available, reason="ngspice is not installed on this machine")
def test_a_nearly_reactive_load_is_measured_correctly_end_to_end(tmp_path):
    """The integration guard for the whole measurement chain.

    Three numbers are enough: a series R-L-C whose reactance dwarfs its
    resistance, so the real power is a thousandth of the instantaneous `v*i`
    product.  Every defect the physics validation turned up shows under those
    conditions -- sample-versus-time averaging, an FFT over uneven timesteps, a
    partial cycle -- and all of them are checked against closed forms rather
    than against a model.

    The record deliberately runs 20.34 cycles rather than 20.  A well-formed
    case would hide the very failures this exists to catch: with a whole number
    of cycles the old harmonic code was already within 1%.

    This covers the arithmetic independently of any plasma-state model.
    """

    import numpy as np

    from pcd.analysis import rf_port_metrics

    r, inductance, capacitance = 0.13, 1.76e-10, 8.33e-11
    f0, amplitude = 13.56e6, 600.0
    case = tmp_path / "rlc.yaml"
    case.write_text(
        f"""
schema: case_yaml.v1
case_id: bare_rlc
source: {{type: sine_voltage, name: Vsrc, p: src, n: 0, amplitude_V: {amplitude}, frequency_Hz: {f0:.0f}}}
circuit:
  builder: from_yaml
  output_node: electrode
  components: [{{raw: "Rwire src electrode 1e-9"}}]
load:
  name: from_yaml
  ports: {{p: electrode, n: "0"}}
  components:
    - raw: "Rload p nl {r}"
    - raw: "Lload nl nc {inductance}"
    - raw: "Cload nc n {capacitance}"
    - raw: "Rleak p n 1e15"
measurement: {{voltage_node: electrode, current_source: Vsrc, load_current: auto}}
solver:
  name: ngspice_cli
  tran: {{step_s: {1 / (f0 * 400):.12g}, stop_s: {20.34 / f0:.12g}}}
  ac: {{sweep: dec, points: 60, start_Hz: {f0 / 10:.0f}, stop_Hz: {f0 * 10:.0f}}}
""".lstrip(),
        encoding="utf-8",
    )
    result = pcd("sim-run", str(case), "--run-root", str(tmp_path / "runs"), "--strict-exit")
    run_dir = Path(json.loads(result.stdout)["run_dir"])
    frame = pd.read_csv(run_dir / "waveform.csv")

    omega = 2 * np.pi * f0
    expected_z = complex(r, omega * inductance - 1 / (omega * capacitance))
    assert abs(expected_z.imag) / expected_z.real > 500, "the load must be nearly reactive for this to bite"

    # The frequency domain: impedance straight off the AC sweep.
    from pcd.analysis import at_frequency, input_impedance, read_ac

    row = at_frequency(input_impedance(read_ac(run_dir / "ac.csv")), f0)
    assert row["reactance_ohm"] == pytest.approx(expected_z.imag, rel=2e-3)
    assert row["resistance_ohm"] == pytest.approx(expected_z.real, rel=0.05)

    metrics = rf_port_metrics(frame, f0, "load_current_A")

    # P = I_rms^2 R must hold for the current the run actually carried.  An
    # ideal source switched on at t=0 rings this circuit at its own 1.3 GHz
    # resonance, and with Q above a thousand that ringing is still present, so
    # the steady-state amplitude is the wrong thing to compare against.
    assert metrics["load_real_power_W"] == pytest.approx(metrics["load_i_rms_A"] ** 2 * r, rel=0.01)
    assert metrics["load_v_rms_V"] == pytest.approx(amplitude / np.sqrt(2), rel=0.05)

    steady_rms = amplitude / (abs(expected_z) * np.sqrt(2))
    assert metrics["load_i_rms_A"] == pytest.approx(steady_rms, rel=0.3), "not the right circuit at all"

    # The spectrum: the drive is a pure sine, so there is nothing above h1.
    harmonics = metrics["voltage_harmonic_amplitude_V"]
    assert harmonics["h1"] == pytest.approx(amplitude, rel=0.02)
    assert harmonics["h2"] < 0.02 * amplitude
