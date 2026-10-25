"""Output requests keep their physical meaning and expose reusable numeric data."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.analysis.traces import transfer_trace
from pcd.api import simulate_case
from pcd.case import Case, load_case
from pcd.identification import run_identification
from pcd.reporting import analyze_run
from pcd.study import run_case_study
from pcd.validation import validate_case

ROOT = Path(__file__).resolve().parents[1]
NGSPICE = shutil.which("ngspice_con.exe") or shutil.which("ngspice")


def test_transfer_tables_keep_poles_and_undefined_phase_unavailable():
    data = pd.DataFrame(
        {"frequency_Hz": [1, 2, 3], "a_re": [1, 1, 0], "a_im": [1, 0, 0], "b_re": [1, 0, 1], "b_im": [0, 0, 0]}
    )
    table = transfer_trace(data, ["a_re", "a_im"], ["b_re", "b_im"])
    assert table["gain_dB"].iloc[0] == pytest.approx(10 * np.log10(2))
    assert table["phase_deg"].iloc[0] == pytest.approx(45)
    assert np.isnan(table["magnitude"].iloc[1])
    assert table["magnitude"].iloc[2] == 0
    assert np.isnan(table["phase_deg"].iloc[2])


@pytest.mark.parametrize(
    ("denominator", "numerator", "status"),
    [
        ([1, 1, 1], [1, 1, 1], "complete"),
        ([1, 0, 1], [1, 1, 1], "partial"),
        ([0, 0, 0], [1, 1, 1], "not_available"),
        ([1, 1, 1], [0, 0, 0], "not_available"),
    ],
)
def test_bode_output_status_describes_plottable_data(tmp_path, denominator, numerator, status):
    from pcd.reporting.recipes import render_recipe

    frame = pd.DataFrame(
        {
            "frequency_Hz": [1.0, 2.0, 3.0],
            "a_re": numerator,
            "a_im": [0.0, 0.0, 0.0],
            "b_re": denominator,
            "b_im": [0.0, 0.0, 0.0],
        }
    )
    signals = {name: {"unit": "V", "columns": {"ac": [f"{name}_re", f"{name}_im"]}} for name in ("a", "b")}
    result = render_recipe(
        {"type": "bode", "name": "transfer", "numerator": "a", "denominator": "b"},
        tmp_path,
        ["svg"],
        {"signals": signals},
        pd.DataFrame(),
        frame,
        title="transfer",
        fundamental_hz=None,
    )
    assert result["status"] == status
    table = pd.read_csv(tmp_path / "transfer_data_1.csv")
    assert len(table) == 3
    assert result["coverage"]["transfer"]["finite_values"]["gain_dB"] == int(np.isfinite(table["gain_dB"]).sum())
    if status != "complete":
        assert result["reason"]
        assert status.upper() in (tmp_path / "transfer.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "output",
    [{"formats": ["unknown"]}, {"formats": [{}]}, {"plots": [{"type": "waveform", "signals": ["x"], "tim_s": [0, 1]}]}],
)
def test_invalid_output_settings_fail_before_a_solve_or_run_directory(tmp_path, output):
    original = load_case(ROOT / "examples/circuit_transient.yaml")
    case = Case(original.path, {**original.data, "output": output})
    assert not validate_case(case).ok
    with pytest.raises(ValueError, match=r"formats|unsupported fields"):
        simulate_case(case, run_root=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_public_output_example_exports_the_selected_ac_node_and_matching_bode_data(tmp_path):
    case = load_case(ROOT / "examples/circuit_outputs.yaml")
    assert validate_case(case, strict=True).ok
    record = simulate_case(case, run_root=tmp_path)
    assert record.status == "ok"
    assert record.output_result is not None
    assert record.frequency_response_file is not None
    assert record.output_result["status"] == "complete"
    summary = json.loads((record.run_dir / record.output_result["summary"]).read_text(encoding="utf-8"))
    ac = pd.read_csv(record.run_dir / record.frequency_response_file)
    row = ac.loc[np.isclose(ac["frequency_Hz"], 1e6)].iloc[0]
    capacitor = 1 / (2j * np.pi * 1e6 * 1e-9)
    expected_filter = capacitor / (1470 + capacitor)
    expected_display = (470 + capacitor) / (1470 + capacitor)
    assert complex(row["voltage_V_re"], row["voltage_V_im"]) == pytest.approx(expected_display)
    assert complex(row["filter_V_re"], row["filter_V_im"]) == pytest.approx(expected_filter)
    catalog = json.loads((record.run_dir / record.signals_file).read_text(encoding="utf-8"))
    assert catalog["signals"]["voltage_V"]["terminals"] == {"p": "mid", "n": "0"}
    assert catalog["signals"]["source_voltage_V"]["columns"]["ac"] == ["voltage_re", "voltage_im"]
    analysis = Path(summary["output_dir"])
    plotted = pd.read_csv(analysis / "filter_transfer_data_1.csv")
    selected = plotted.loc[np.isclose(plotted["frequency_Hz"], 1e6)].iloc[0]
    assert selected["gain_dB"] == pytest.approx(20 * np.log10(abs(expected_filter)))
    assert selected["phase_deg"] == pytest.approx(np.degrees(np.angle(expected_filter)))
    frequency_metrics = pd.read_csv(analysis / "frequency_metrics.csv")
    assert not any(str(name).endswith(".1") for name in frequency_metrics)
    measured = frequency_metrics.loc[np.isclose(frequency_metrics["frequency_Hz"], 1e6)].iloc[0]
    assert measured["source_real_power_W"] == pytest.approx(0.5 * 1470 / abs(1470 + capacitor) ** 2)
    for name in ("final_cycles", "filter_transfer", "probe_response", "output_harmonics", "circuit"):
        for extension in ("png", "svg"):
            assert (analysis / f"{name}.{extension}").is_file()


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_named_network_probes_and_normalized_drive_are_explicit(make_case, tmp_path):
    authored = {
        "schema": "pcd.rf.v1",
        "case_id": "reference",
        "frequency_Hz": 1e6,
        "network": {"type": "l_match", "fixed": {"L1": 1e-6, "C1": 1e-10}},
        "load": {"type": "impedance_point", "resistance_ohm": 50, "reactance_ohm": 0, "reference_plane": "terminal"},
        "acceptance": {"reflection_magnitude_max": 0.9},
        "measurement": {"probes": {"electrode_V": "v(electrode)"}},
    }
    case = make_case(authored)
    assert validate_case(case).ok
    record = simulate_case(case, run_root=tmp_path / "runs")
    assert record.frequency_response_file is not None
    frame = pd.read_csv(record.run_dir / record.frequency_response_file)
    assert "electrode_V_re" in frame
    report = analyze_run(record.run_dir, output={"formats": ["svg"]})
    assert report["measurement"]["drive_basis"] == "normalized_1V_peak"
    assert "1 V peak reference excitation" in (Path(report["output_dir"]) / "operating_point.svg").read_text(
        encoding="utf-8"
    )
    titled = analyze_run(
        record.run_dir,
        output={
            "formats": ["svg"],
            "schematic": False,
            "plots": [{"type": "ac", "signals": ["electrode_V"], "title": "Terminal response"}],
        },
    )
    assert "1 V peak reference excitation" in (Path(titled["output_dir"]) / "ac_1.svg").read_text(encoding="utf-8")


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_requested_missing_plot_is_partial_and_can_be_replotted_without_solving(tmp_path):
    original = load_case(ROOT / "examples/circuit_transient.yaml")
    settings = {"schematic": False, "plots": [{"type": "waveform", "signals": ["missing"]}]}
    case = Case(original.path, {**original.data, "output": settings})
    record = simulate_case(case, run_root=tmp_path)
    assert record.status == "ok"
    assert record.output_result is not None
    assert record.output_result["status"] == "partial"
    before = (record.run_dir / record.netlist_file).stat().st_mtime_ns
    report = analyze_run(
        record.run_dir, output={"schematic": False, "plots": [{"type": "waveform", "signals": ["voltage_V"]}]}
    )
    assert report["status"] == "complete"
    assert (record.run_dir / record.netlist_file).stat().st_mtime_ns == before
    from matplotlib import pyplot as plt

    existing = plt.get_fignums()
    unavailable = analyze_run(
        record.run_dir,
        output={"schematic": False, "plots": [{"type": "waveform", "signals": ["voltage_V"], "yscale": "log"}]},
    )
    assert unavailable["status"] == "partial"
    assert "positive values" in unavailable["output"]["plots"][0]["reason"]
    assert plt.get_fignums() == existing


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_unavailable_solver_probe_has_a_public_error(make_case, tmp_path):
    original = load_case(ROOT / "examples/circuit_transient.yaml")
    data = deepcopy(original.data)
    data["measurement"]["probes"] = {"absent_V": "v(no_such_node)"}
    case = make_case(data)
    record = simulate_case(case, run_root=tmp_path / "runs")
    assert record.status == "failed"
    assert record.error
    assert "no such vector" in record.error.lower()


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_study_export_distinguishes_search_and_fresh_verification(tmp_path):
    original = load_case(ROOT / "bench/cases/role_factorial_search.yaml")
    settings = {"schematic": False, "plots": [], "data": {"waveforms": "all"}}
    result = run_case_study(Case(original.path, {**original.data, "output": settings}), tmp_path)
    assert result["output"]["status"] == "complete"
    root = Path(result["run_root"])
    report = json.loads((root / result["output"]["summary"]).read_text(encoding="utf-8"))
    assert report["data"]["evaluations"] == 10
    folder = (root / result["output"]["summary"]).parent
    exported = pd.read_csv(folder / report["data"]["files"]["ac"])
    assert exported.groupby("context.phase").size().to_dict() == {"search": 8, "verification": 2}
    assert len(report["points"]) == 2


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_arbitrary_probe_names_cannot_replace_study_context(make_case, tmp_path):
    import yaml

    raw = yaml.safe_load((ROOT / "examples/circuit_design.yaml").read_text(encoding="utf-8"))
    raw["parameters"]["design"]["C"] = {"choices": [1e-9, 2e-9], "default": 1e-9}
    raw["objective"]["waveform_file"] = str(ROOT / "examples/advanced/target_rc.csv")
    raw["execution"] = {"optimizer": "grid", "trials": 2}
    aliases = ("phase", "candidate_id", "context.phase", "parameter.C")
    raw["measurement"] = {"probes": dict.fromkeys(aliases, "v(out)")}
    raw["output"] = {"schematic": False, "plots": [], "data": {"waveforms": "all"}}
    case = make_case(raw)
    assert validate_case(case).ok
    result = run_case_study(case, tmp_path / "runs")
    assert result["output"]["status"] == "complete"
    folder = (Path(result["run_root"]) / result["output"]["summary"]).parent
    exported = pd.read_csv(folder / "transient_observations.csv")
    assert set(exported["context.phase"]) == {"search", "verification"}
    assert exported["context.candidate_id"].nunique() == 2
    assert exported["parameter.C"].isin([1e-9, 2e-9]).all()
    for name in aliases:
        assert np.allclose(exported[f"signal.{name}"], exported["signal.voltage_V"])
    assert not any(str(name).endswith(".1") for name in exported)


@pytest.mark.e2e
@pytest.mark.skipif(not NGSPICE, reason="ngspice is not installed")
def test_identification_exports_partitioned_observations_residuals_and_local_sensitivity(tmp_path):
    original = load_case(ROOT / "examples/terminal_identification.yaml")
    data = {**original.data, "output": {"schematic": False, "plots": []}}
    result = run_identification(Case(original.path, data, original.source_data), tmp_path)
    assert result["status"] == "identified"
    assert result["output"]["status"] == "complete"
    folder = Path(result["run_root"]) / "output"
    observed = pd.read_csv(folder / "terminal_observations.csv")
    assert observed.groupby("partition").size().to_dict() == {"fit": 3, "holdout": 2}
    assert observed["residual.resistance_ohm"].abs().max() < 1e-6
    assert (folder / "observed_predicted_and_residuals.png").is_file()
    assert (folder / "local_sensitivity.png").is_file()
