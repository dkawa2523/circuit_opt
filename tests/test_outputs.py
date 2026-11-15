"""Output requests keep their physical meaning and expose reusable numeric data."""

from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.analysis.traces import transfer_trace
from pcd.api import simulate_case
from pcd.artifacts import file_sha256
from pcd.case import Case, load_case
from pcd.cli import main
from pcd.identification import run_identification
from pcd.output import output_settings
from pcd.reporting import analyze_run, analyze_saved
from pcd.reporting.study import _comparison
from pcd.results import read_evaluation_table
from pcd.results.store import GENERATION_RESULT_FILE
from pcd.study import run_case_study
from pcd.validation import validate_case

ROOT = Path(__file__).resolve().parents[1]


def test_voltage_comparison_separates_drive_and_observation_planes(tmp_path):
    points = []
    bases = [
        ("electrode", "terminal", "normalized_1V_peak"),
        ("electrode", "terminal", "declared_or_imported"),
        ("feedthrough", "feedthrough_plane", "normalized_1V_peak"),
    ]
    for i, (node, plane, drive) in enumerate(bases):
        folder = tmp_path / f"point_{i}"
        folder.mkdir()
        terminals = {"p": node, "n": "0"}
        (folder / "signals.json").write_text(
            json.dumps(
                {
                    "axes": {"ac": {"column": "frequency_Hz"}},
                    "drive_basis": drive,
                    "load_ports": terminals,
                    "load_reference_plane": plane,
                    "signals": {
                        "voltage_V": {"terminals": terminals, "columns": {"ac": ["load_re", "load_im"]}},
                    },
                }
            ),
            encoding="utf-8",
        )
        (folder / "summary.json").write_text(
            json.dumps(
                {"analyses": {"ac": {}}, "artifacts": {"signals": "signals.json", "frequency_metrics": "ac.csv"}}
            ),
            encoding="utf-8",
        )
        (folder / "ac.csv").write_text("frequency_Hz,load_re,load_im\n1e6,3,4\n", encoding="utf-8")
        points.append(
            {"summary": f"point_{i}/summary.json", "phase": "verification", "scenario_id": str(i), "control": "{}"}
        )

    comparison = _comparison(points, tmp_path, ["svg"])
    assert comparison["unavailable"] == []
    assert [(panel["plane"], panel["drive_basis"]) for panel in comparison["panels"]] == [
        (plane, drive) for _, plane, drive in bases
    ]
    assert all(panel["traces"][0]["finite_samples"] == 1 for panel in comparison["panels"])


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


@pytest.mark.parametrize("names", [("probe", "PROBE"), ("Circuit",)])
def test_output_plot_names_are_unique_on_case_insensitive_filesystems(names):
    plots = [{"type": "waveform", "name": name, "signals": ["voltage_V"]} for name in names]
    with pytest.raises(ValueError, match="unique case-insensitive"):
        output_settings({"plots": plots})


def test_simulation_publishes_numerical_success_when_requested_output_raises(tmp_path, monkeypatch, capsys):
    def fail(*_args):
        raise RuntimeError("render failed")

    monkeypatch.setattr("pcd.reporting.analyze_run", fail)
    casefile = tmp_path / "case.yaml"
    source = (ROOT / "examples/circuit_transient.yaml").read_text(encoding="utf-8")
    casefile.write_text(source + "\noutput: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["sim-run", str(casefile), "--run-root", str(tmp_path / "run"), "--solver", "test_fake", "--json"])
    assert exc.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["output"] == {"status": "partial", "summary": None, "error": "RuntimeError: render failed"}
    root = Path(result["run_dir"])
    assert (root / result["artifacts"]["waveform"]).is_file()
    assert json.loads((root / "summary.json").read_text(encoding="utf-8"))["output"] == result["output"]


def test_study_publishes_verification_when_requested_output_raises(tmp_path, topology_case, monkeypatch):
    def fail(*_args):
        raise OSError("cannot write plots")

    monkeypatch.setattr("pcd.reporting.study.render_study_outputs", fail)
    case = Case(topology_case.path, {**topology_case.data, "output": {}})
    result = run_case_study(case, tmp_path, n_trials=1, solver_override="test_fake")
    root = Path(result["run_root"])
    assert result["verification"]["status"] == "verified"
    assert result["output"] == {"status": "partial", "summary": None, "error": "OSError: cannot write plots"}
    assert "output_summary" not in result["artifacts"]
    assert (root / result["artifacts"]["best_candidate"]).is_file()
    active = (root / "study_result.json").read_bytes()
    assert active == (root / result["artifacts"]["generation"] / GENERATION_RESULT_FILE).read_bytes()


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_public_output_example_exports_the_selected_ac_node_and_matching_bode_data(tmp_path):
    case = load_case(ROOT / "examples/circuit_outputs.yaml")
    assert validate_case(case, strict=True).ok
    record = simulate_case(case, run_root=tmp_path)
    assert record.status == "ok"
    assert record.output_result is not None
    assert record.frequency_response_file is not None
    assert record.output_result["status"] == "complete"
    summary = json.loads((record.run_dir / record.output_result["summary"]).read_text(encoding="utf-8"))
    assert summary["provenance"]["source_integrity"] == "verified"
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

    # A syntactically valid CSV on the right frequency grid can still be the wrong evidence.
    manifest = record.run_dir / "debug" / "manifest.json"
    original = manifest.read_bytes()
    forged = json.loads(original)
    forged["measurement"]["drive_basis"] = "normalized_1V_peak"
    manifest.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs from its public summary"):
        analyze_run(manifest)
    manifest.write_bytes(original)

    ac["current_re"] *= 2
    ac.to_csv(record.run_dir / record.frequency_response_file, index=False)
    with pytest.raises(ValueError, match="recorded frequency_response"):
        analyze_run(record.run_dir)


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_redraw_preserves_the_saved_source_port_and_checks_the_archived_deck(tmp_path):
    record = simulate_case(load_case(ROOT / "examples/circuit_external.yaml"), run_root=tmp_path)
    settings = {"plots": [], "schematic": False}
    summary = analyze_run(record.run_dir, output=settings)
    assert summary["provenance"]["source_integrity"] == "verified"

    archived = record.run_dir / record.case_files["imported_netlist"]
    original = archived.read_text(encoding="utf-8")
    archived.write_text(original.replace("V1 in 0 AC 1", "V1 out 0 AC 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded imported_netlist"):
        analyze_run(record.run_dir, output=settings)

    # Older archives hashed the manifest and observations but omitted the imported deck.
    manifest_path = record.run_dir / "debug" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("input_manifest", "imported_netlist"):
        del manifest["artifact_sha256"][name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = record.run_dir / "summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_summary["debug_manifest_sha256"] = file_sha256(manifest_path)
    summary_path.write_text(json.dumps(saved_summary), encoding="utf-8")

    legacy = analyze_run(record.run_dir, output=settings)
    assert legacy["provenance"]["source_integrity"] == "unverified_dependencies"
    signals = json.loads((Path(legacy["output_dir"]) / "signals.json").read_text(encoding="utf-8"))
    assert signals["signals"]["source_voltage_V"]["terminals"] == {"p": "in", "n": "0"}


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
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
@pytest.mark.usefixtures("ngspice")
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
@pytest.mark.usefixtures("ngspice")
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
@pytest.mark.usefixtures("ngspice")
def test_study_export_distinguishes_search_and_fresh_verification(tmp_path):
    original = load_case(ROOT / "bench/cases/role_factorial_search.yaml")
    scenarios = tmp_path / "ids.csv"
    source = ROOT / "bench/role_factorial_scenarios.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        header, *rows = csv.reader(handle)
    with scenarios.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([identifier, *row[1:]] for identifier, row in zip(("001", "NA"), rows, strict=True))
    data = deepcopy(original.data)
    data["study"]["scenario_table"]["table_file"] = str(scenarios)
    settings = {"schematic": False, "plots": [], "data": {"waveforms": "all"}}
    result = run_case_study(Case(original.path, {**data, "output": settings}), tmp_path)
    assert result["output"]["status"] == "complete"
    root = Path(result["run_root"])
    report = json.loads((root / result["output"]["summary"]).read_text(encoding="utf-8"))
    assert report["data"]["evaluations"] == 10
    folder = (root / result["output"]["summary"]).parent
    exported = pd.read_csv(folder / report["data"]["files"]["ac"])
    assert exported.groupby("context.phase").size().to_dict() == {"search": 8, "verification": 2}
    assert set(read_evaluation_table(root)["scenario_id"]) == {"001", "NA"}
    with (folder / "observations.csv").open(encoding="utf-8", newline="") as handle:
        assert {(row["context.phase"], row["context.scenario_id"]) for row in csv.DictReader(handle)} == {
            (phase, scenario) for phase in ("search", "verification") for scenario in ("001", "NA")
        }
    assert len(report["points"]) == 2
    panels = report["voltage_comparison"]["panels"]
    assert len(panels) == 1
    assert panels[0]["domain"] == "ac"
    assert {trace["scenario_id"] for trace in panels[0]["traces"]} == {"001", "NA"}


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_analyze_study_uses_copied_generation_and_presentation_override(tmp_path, monkeypatch, capsys):
    result = run_case_study(load_case(ROOT / "bench/cases/role_factorial_search.yaml"), tmp_path / "study")
    moved = tmp_path / "copied_study"
    shutil.copytree(result["run_root"], moved)
    generation = moved / result["artifacts"]["generation"]
    (moved / "study_result.json").write_text("{}", encoding="utf-8")
    before = {path: file_sha256(path) for path in moved.rglob("*") if path.is_file()}
    override = tmp_path / "view.yaml"
    override.write_text(
        "circuit: {invalid: true}\noutput: {schematic: false, plots: [], data: {waveforms: all}}\n",
        encoding="utf-8",
    )

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("offline analysis must use saved solver data")

    monkeypatch.setattr("pcd.sim_core.invoke_solver", unexpected_solve)
    destination = tmp_path / "study_view"
    main(["analyze", str(generation), "--out", str(destination), "--case", str(override), "--json"])
    report = json.loads(capsys.readouterr().out)

    assert report["status"] == "complete"
    assert report["engineering_status"] == result["best"]["status"]
    assert report["data"]["evaluations"] == 10
    assert (destination / report["source"]["generation"]).resolve() == generation
    assert report["source"]["sha256"] == file_sha256(generation / GENERATION_RESULT_FILE)
    assert report["source"]["aggregate_integrity"] == "verified"
    hashes = {(destination / path).resolve(): digest for path, digest in report["data"]["source_hashes"].items()}
    for key in ("case", "best_candidate", "history", "evaluation_table"):
        source_path = moved / result["artifacts"][key]
        assert hashes[source_path] == file_sha256(source_path)
    assert (destination / report["artifacts"]["summary"]).is_file()
    assert {path: file_sha256(path) for path in moved.rglob("*") if path.is_file()} == before

    published = generation / GENERATION_RESULT_FILE
    original = published.read_bytes()
    replacement = json.loads(original)
    replacement["best"]["status"] = "forged_acceptance"
    published.write_text(json.dumps(replacement), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its publication"):
        analyze_saved(generation, out_dir=destination)
    published.write_bytes(original)

    history = moved / result["artifacts"]["history"]
    history.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recorded history"):
        analyze_saved(generation, out_dir=destination)


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_arbitrary_probe_names_cannot_replace_study_context(make_case, tmp_path):
    import yaml

    raw = yaml.safe_load((ROOT / "examples/circuit_design.yaml").read_text(encoding="utf-8"))
    raw["parameters"]["design"]["C"] = {"choices": [1e-9, 2e-9], "default": 1e-9}
    raw["parameters"]["control"] = {"setting": {"choices": ["001", "NA"], "default": "001"}}
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
    searched_controls = exported.loc[exported["context.phase"] == "search", "context.control"].unique()
    assert {json.loads(control)["setting"] for control in searched_controls} == {"001", "NA"}
    assert exported["parameter.C"].isin([1e-9, 2e-9]).all()
    for name in aliases:
        assert np.allclose(exported[f"signal.{name}"], exported["signal.voltage_V"])
    assert not any(str(name).endswith(".1") for name in exported)


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_identification_exports_partitioned_observations_residuals_and_local_sensitivity(
    tmp_path, make_case, monkeypatch
):
    import yaml

    raw = yaml.safe_load((ROOT / "examples/terminal_identification.yaml").read_text(encoding="utf-8"))
    inputs = tmp_path / "terminal_observations.csv"
    with (ROOT / "examples/terminal_observations.csv").open(encoding="utf-8", newline="") as handle:
        header, *rows = csv.reader(handle)
    ids = ("001", "NA", "002", "003", "004")
    with inputs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([identifier, *row[1:]] for identifier, row in zip(ids, rows, strict=True))
    raw["identification"]["observations"]["file"] = inputs.name
    raw["output"] = {"schematic": False, "plots": []}
    result = run_identification(make_case(raw), tmp_path)
    assert result["status"] == "identified"
    assert result["output"]["status"] == "complete"
    folder = Path(result["run_root"]) / "output"
    observed = pd.read_csv(folder / "terminal_observations.csv", converters={"scenario_id": str})
    assert set(observed["scenario_id"]) == set(ids)
    assert observed.groupby("partition").size().to_dict() == {"fit": 3, "holdout": 2}
    assert observed["residual.resistance_ohm"].abs().max() < 1e-6
    assert (folder / "observed_predicted_and_residuals.png").is_file()
    assert (folder / "local_sensitivity.png").is_file()

    moved = tmp_path / "copied_identification"
    shutil.copytree(result["run_root"], moved)
    (moved / "fit" / "study_result.json").write_text("{}", encoding="utf-8")

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("offline analysis must use saved solver data")

    monkeypatch.setattr("pcd.sim_core.invoke_solver", unexpected_solve)
    report = analyze_saved(moved / "identification_result.json", out_dir=tmp_path / "identification_view")
    destination = Path(report["output_dir"])
    assert report["status"] == "complete"
    assert report["identification_status"] == "identified"
    assert report["data"]["evaluations"] == 5
    assert (destination / report["phases"]["fit"]["record"]).resolve() == moved / result["artifacts"]["fit_result"]
    assert (destination / report["terminal_observations"]).read_bytes() == (
        folder / "terminal_observations.csv"
    ).read_bytes()
    assert report["source"]["aggregate_integrity"] == "verified"

    observations = moved / result["artifacts"]["fit_observations"]
    original = observations.read_bytes()
    forged = pd.read_csv(observations, converters={"scenario_id": str})
    forged.loc[0, "predicted.resistance_ohm"] = 999999
    forged.to_csv(observations, index=False)
    with pytest.raises(ValueError, match="recorded fit_observations"):
        analyze_saved(moved, out_dir=destination)
    observations.write_bytes(original)

    record = moved / "identification_result.json"
    incomplete = json.loads(record.read_text(encoding="utf-8"))
    del incomplete["artifacts"]["holdout_result"]
    (moved / incomplete["artifacts"]["holdout_observations"]).unlink()
    record.write_text(json.dumps(incomplete), encoding="utf-8")
    partial = analyze_saved(record, out_dir=destination)
    assert partial["status"] == "partial"
    assert partial["identification_status"] == "identified"
    assert partial["data"]["missing_source_artifacts"] == [
        {"phase": "holdout", "artifact": "holdout_result"},
        {"phase": "holdout", "artifact": "holdout_observations"},
    ]
