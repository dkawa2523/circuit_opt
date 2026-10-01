"""Characterization tests for the CLI surface.

Written before ``pcd.cli.main`` was split into per-command handlers so the
refactor is provably behaviour-preserving.  These pin the *contract*: which
artifacts appear, what is printed, and which exit code is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pcd.artifacts import file_sha256
from pcd.cli import _print_sim_summary, main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FIXTURES = ROOT / "tests" / "fixtures"
RC_CASE = str(EXAMPLES / "advanced" / "generic_rc_filter.yaml")
ADVANCED_CASE = str(FIXTURES / "advanced_case.yaml")
GRID_CASE = str(ROOT / "bench" / "cases" / "match_discrete_hardware_search.yaml")


def _stdout_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# --- informational commands ------------------------------------------------


def test_list_reports_simulation_metrics_and_search(capsys):
    main(["list"])
    payload = _stdout_json(capsys)
    assert set(payload["simulation"]) == {"circuit", "load", "solver"}
    assert "waveform_l2" in payload["metrics"]
    assert "rf_load" in payload["metrics"]
    assert "random" in payload["optimizers"]
    assert "grid" in payload["optimizers"]
    assert "differential_evolution" in payload["optimizers"]
    assert "ngspice_cli" in payload["simulation"]["solver"]
    assert "dummy" not in payload["simulation"]["solver"]


def test_solver_diagnose_rejects_an_unknown_solver(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["solver-diagnose", "--solver", "unknown", "--json"])
    assert excinfo.value.code == 1
    payload = _stdout_json(capsys)
    assert payload["schema"] == "solver_diagnostic.v1"
    assert payload["batch_runnable"] is False


def test_solver_diagnose_exits_nonzero_for_a_missing_executable():
    with pytest.raises(SystemExit) as excinfo:
        main(["solver-diagnose", "--solver", "ngspice_cli", "--executable", "definitely-not-installed", "--json"])
    assert excinfo.value.code == 1


# --- validation ------------------------------------------------------------


def test_validate_case_json_reports_ok(capsys):
    main(["validate-case", RC_CASE, "--json"])
    payload = _stdout_json(capsys)
    assert payload["ok"] is True
    assert isinstance(payload["issues"], list)


def test_validate_case_strict_accepts_the_physical_example(capsys):
    main(["validate-case", RC_CASE, "--strict"])
    assert capsys.readouterr().out.strip() == "OK"


# --- simulation-only commands ---------------------------------------------


def test_sim_netlist_writes_a_netlist(tmp_path, capsys):
    out = tmp_path / "netlist.cir"
    main(["sim-netlist", RC_CASE, "--out", str(out)])
    assert capsys.readouterr().out.strip() == str(out)
    text = out.read_text(encoding="utf-8")
    assert ".end" in text
    assert "wrdata waveform.csv" in text


def test_sim_run_produces_a_waveform_and_no_metrics(tmp_path, capsys):
    main(["sim-run", RC_CASE, "--solver", "test_fake", "--run-root", str(tmp_path), "--json"])
    payload = _stdout_json(capsys)
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "ok"
    assert payload["schema"] == "simulation_summary.v1"
    assert (run_dir / payload["artifacts"]["waveform"]).exists()
    assert (run_dir / payload["artifacts"]["debug_manifest"]).exists()
    assert not (run_dir / "metrics.json").exists()


def test_sim_run_defaults_to_a_short_human_summary(tmp_path, capsys):
    main(["sim-run", RC_CASE, "--solver", "test_fake", "--run-root", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Simulation: generic_rc_filter" in output
    assert "Status: ok" in output
    assert "Solver: test_fake" in output
    assert "Results:" in output
    assert "Artifacts:" in output


def test_sim_summary_shows_solver_version_and_error_without_artifact_noise(
    capsys,
) -> None:
    _print_sim_summary(
        {
            "case_id": "failed_case",
            "status": "failed",
            "solver": "ngspice_cli",
            "solver_version": "ngspice-46",
            "run_dir": "runs/failed",
            "error": "convergence failed",
            "artifacts": {},
        }
    )

    output = capsys.readouterr().out
    assert "Solver: ngspice_cli (ngspice-46)" in output
    assert "Error: convergence failed" in output
    assert "Artifacts:" not in output


def test_sim_run_reports_invalid_input_without_fabricating_a_result(tmp_path, capsys):

    case_path = tmp_path / "broken.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: broken_load\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: '0', amplitude_V: 10, frequency_Hz: 1000000.0}\n"
        "circuit: {builder: from_yaml, output_node: out, components: [{ref: R1, n1: src, n2: out, value: 50}]}\n"
        "load: {name: definitely_unknown, ports: {p: out, n: '0'}}\n"
        "measurement: {voltage_node: out, current_source: Vsrc}\n"
        "solver: {name: test_fake, tran: {step_s: 1.0e-9, stop_s: 1.0e-7}}\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    with pytest.raises(SystemExit) as excinfo:
        main(["sim-run", str(case_path), "--solver", "test_fake", "--run-root", str(run_root), "--json"])
    assert excinfo.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
    assert payload["error"]
    assert not list(run_root.rglob("summary.json"))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "sim-run",
                str(case_path),
                "--solver",
                "test_fake",
                "--run-root",
                str(run_root),
                "--allow-failure",
                "--json",
            ]
        )
    assert excinfo.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_sim_run_can_collect_an_explicit_solver_failure(tmp_path, capsys):
    case_path = tmp_path / "unsupported_ac.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: unsupported_ac\n"
        "source: {type: sine_voltage, name: Vsrc, p: src, n: '0', amplitude_V: 1, frequency_Hz: 1000000.0}\n"
        "circuit: {builder: from_yaml, output_node: out, components: [{ref: R1, n1: src, n2: out, value: 50}]}\n"
        "load: {name: resistor, R_ohm: 50, ports: {p: out, n: '0'}}\n"
        "measurement: {voltage_node: out, current_source: Vsrc}\n"
        "solver: {name: test_fake, ac: {sweep: lin, points: 1, start_Hz: 1000000.0, stop_Hz: 1000000.0}}\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs"
    command = ["sim-run", str(case_path), "--solver", "test_fake", "--run-root", str(run_root), "--json"]

    with pytest.raises(SystemExit) as excinfo:
        main(command)
    assert excinfo.value.code == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "failed"
    assert (Path(failed["run_dir"]) / failed["artifacts"]["waveform"]).is_file()

    main([*command, "--allow-failure"])
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_visualize_netlist_writes_image_and_summary(tmp_path, capsys):
    netlist = tmp_path / "n.cir"
    main(["sim-netlist", RC_CASE, "--out", str(netlist)])
    capsys.readouterr()
    image = tmp_path / "schematic.png"
    summary = tmp_path / "schematic.json"
    main(["visualize-netlist", str(netlist), "--out", str(image), "--summary-json", str(summary)])
    assert image.exists()
    assert image.stat().st_size > 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["n_nodes"] >= 2
    assert "component_counts" in payload


def test_analyze_regenerates_summary_and_figures_from_a_run(tmp_path, capsys):
    main(["sim-run", RC_CASE, "--solver", "test_fake", "--run-root", str(tmp_path / "runs"), "--json"])
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]
    report_dir = tmp_path / "analysis"
    main(["analyze", run_dir, "--out", str(report_dir), "--json"])
    payload = _stdout_json(capsys)
    assert payload["schema"] == "run_analysis.v1"
    assert payload["analyses"]["transient"]["samples"] > 1
    assert set(payload["artifacts"]) == {"summary", "transient_response"}
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "transient_response.png").stat().st_size > 0

    main(["analyze", run_dir, "--out", str(tmp_path / "analysis_text")])
    output = capsys.readouterr().out
    assert "Analysis: generic_rc_filter" in output
    assert "Transient:" in output
    assert "periodic=" in output
    assert "Artifacts: summary=summary.json" in output


def test_analyze_prints_the_bracketed_sweep_bandwidth(monkeypatch, capsys):
    monkeypatch.setattr(
        "pcd.reporting.analyze_run",
        lambda *args, **kwargs: {
            "case_id": "resonance",
            "output_dir": "analysis",
            "analyses": {
                "ac": {
                    "frequency_Hz": 10e6,
                    "resistance_ohm": 50.0,
                    "reactance_ohm": 0.0,
                    "reflection_magnitude": 0.0,
                    "sweep": {
                        "sample_count": 401,
                        "sampled_best_match_frequency_Hz": 10e6,
                        "half_power_resonance": {
                            "basis": "load_real_power_W",
                            "resonant_frequency_Hz": 10e6,
                            "bandwidth_Hz": 1e6,
                            "loaded_quality_factor": 10.0,
                        },
                    },
                }
            },
            "artifacts": {"summary": "summary.json"},
        },
    )

    main(["analyze", "saved-run"])

    output = capsys.readouterr().out
    assert "power peak=10 MHz" in output
    assert "-3 dB BW=1 MHz" in output
    assert "loaded Q=10" in output


# --- study -----------------------------------------------------------------


def test_run_is_the_simple_human_readable_study_entry_point(tmp_path, capsys):
    main(["run", RC_CASE, "--solver", "test_fake", "--output", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Electrical context: TRANSIENT, reference plane=load_ports, Z0=50 ohm" in output
    assert "Feasible across all conditions: yes" in output
    assert "Decision: meets_declared_acceptance" in output
    assert "Selected candidate:" in output
    assert "Condition coverage: accepted 1/1, solved 1/1" in output
    assert "Objective: normalized_rmse=" in output
    assert "Objective: peak_abs_voltage_V=" in output
    assert "Pareto front (observed candidates):" in output
    assert "Worst constraint margins (positive=reserve): max_peak_abs_voltage_V=" in output
    assert "Condition nominal: accepted, control={}" in output
    assert "Candidates: 24" in output
    assert "Results:" in output
    assert "Selected evidence:" in output
    assert len(list(tmp_path.rglob("study_result.json"))) == 1


def test_run_can_gate_automation_on_declared_acceptance(monkeypatch, capsys):
    monkeypatch.setattr(
        "pcd.study.run_case_study",
        lambda *args, **kwargs: {
            "best": {"status": "does_not_meet_declared_acceptance"},
            "n_failed_evaluations": 0,
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["run", RC_CASE, "--require-acceptance", "--json"])

    assert excinfo.value.code == 1
    assert _stdout_json(capsys)["best"]["status"] == "does_not_meet_declared_acceptance"


def test_run_reports_a_public_input_typo_without_a_traceback(tmp_path, capsys):
    case = tmp_path / "typo.yaml"
    case.write_text("schema: pcd.rf.v1\ncase_id: typo\nfrequncy_Hz: 1e6\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(case)])
    assert excinfo.value.code == 2
    assert capsys.readouterr().out.startswith("Input error:")


def test_run_refuses_to_sample_only_part_of_a_resolved_grid(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["run", GRID_CASE, "--trials", "2", "--output", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "candidate enumeration is derived from network.search" in capsys.readouterr().out


def test_run_refuses_to_replace_complete_public_enumeration_with_random_sampling(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                GRID_CASE,
                "--optimizer",
                "random",
                "--trials",
                "3",
                "--output",
                str(tmp_path),
            ]
        )
    assert excinfo.value.code == 2
    assert "candidate enumeration is derived from network.search" in capsys.readouterr().out


def test_run_uses_the_unified_pipeline_for_an_advanced_case(tmp_path, capsys):
    main(
        [
            "run",
            ADVANCED_CASE,
            "--optimizer",
            "random",
            "--solver",
            "test_fake",
            "--trials",
            "3",
            "--seed",
            "4",
            "--output",
            str(tmp_path),
            "--json",
        ]
    )
    payload = _stdout_json(capsys)
    assert payload["schema"] == "study_result.v1"
    assert payload["n_candidates"] == 3
    study_root = Path(payload["run_root"])
    assert (study_root / "study_result.json").exists()
    assert (study_root / payload["artifacts"]["history"]).exists()


def test_ml_prepare_exports_explicit_roles_and_a_group_holdout(tmp_path, capsys):
    main(
        [
            "run",
            RC_CASE,
            "--optimizer",
            "random",
            "--solver",
            "test_fake",
            "--trials",
            "4",
            "--seed",
            "7",
            "--output",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    study = _stdout_json(capsys)
    export = tmp_path / "ml"

    main(
        [
            "ml-prepare",
            study["run_root"],
            "--out",
            str(export),
            "--test-fraction",
            "0.25",
            "--seed",
            "11",
            "--json",
        ]
    )
    manifest = _stdout_json(capsys)
    dataset = pd.read_csv(export / "dataset.csv")

    assert manifest["schema"] == "ml_evaluation_dataset.v1"
    assert manifest["split"]["groups"] == {"total": 4, "train": 3, "test": 1}
    assert manifest["source_artifacts"]["evaluation_table"]["sha256"]
    assert manifest["artifacts"]["dataset"]["sha256"]
    assert manifest["feature_transforms"]["design.C1"] == "log10"
    assert manifest["feature_transforms"]["design.R1"] == "log10"
    assert manifest["source_evaluation_cost_s"]["status"] == "available"
    assert (export / "manifest.json").is_file()
    assert "selected_control" not in dataset
    assert dataset.groupby("candidate_id")["split"].nunique().eq(1).all()
    assert set(dataset["split"]) == {"train", "test"}

    evaluation = tmp_path / "ml-evaluation"
    main(["ml-evaluate", str(export), "--out", str(evaluation), "--json"])
    evidence = _stdout_json(capsys)
    assert evidence["schema"] == "ml_surrogate_evaluation.v1"
    assert evidence["source_artifacts"]["dataset"]["sha256"] == manifest["artifacts"]["dataset"]["sha256"]
    assert evidence["evidence_gate"]["bayesian_optimization_ready"] is False
    assert (evaluation / "predictions.csv").is_file()
    assert (evaluation / "evaluation.json").is_file()

    text_evaluation = tmp_path / "ml-evaluation-text"
    main(["ml-evaluate", str(export), "--out", str(text_evaluation)])
    output = capsys.readouterr().out
    assert "Surrogate evaluation:" in output
    assert "RMSE improvement over mean=" in output
    assert "Next evidence step:" in output

    validation_export = tmp_path / "ml-validation"
    validation_export.mkdir()
    validation_dataset = dataset.copy()
    validation_dataset["dataset_id"] = "validation/generation"
    validation_dataset["study_id"] = "validation"
    validation_dataset["design.C1"] *= 1.01
    validation_dataset["design.R1"] *= 1.01
    validation_dataset["design_group_id"] = "validation_" + validation_dataset["design_group_id"].astype(str)
    validation_dataset_path = validation_export / "dataset.csv"
    validation_dataset_path.write_text(validation_dataset.to_csv(index=False), encoding="utf-8")
    validation_manifest = json.loads(json.dumps(manifest))
    validation_manifest["source"]["dataset_id"] = "validation/generation"
    validation_manifest["source"]["study_id"] = "validation"
    validation_manifest["artifacts"]["dataset"]["sha256"] = file_sha256(validation_dataset_path)
    (validation_export / "manifest.json").write_text(
        json.dumps(validation_manifest),
        encoding="utf-8",
    )
    validated_evaluation = tmp_path / "ml-evaluation-validated"
    main(
        [
            "ml-evaluate",
            str(export),
            "--constraint-validation",
            str(validation_export),
            "--out",
            str(validated_evaluation),
            "--json",
        ]
    )
    validated = _stdout_json(capsys)
    assert validated["independent_constraint_validation"]["status"] == "evaluated"
    assert validated["source_artifacts"]["constraint_validation"]["dataset"]["sha256"]
    assert (validated_evaluation / "constraint_validation_predictions.csv").is_file()

    with pytest.raises(SystemExit) as excinfo:
        main(["ml-evaluate", str(export), "--out", str(export / "nested"), "--json"])
    assert excinfo.value.code == 2
    assert "must not overlap" in _stdout_json(capsys)["error"]

    (export / "dataset.csv").write_text((export / "dataset.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["ml-evaluate", str(export), "--out", str(tmp_path / "tampered"), "--json"])
    assert excinfo.value.code == 2
    assert "SHA-256 does not match" in _stdout_json(capsys)["error"]

    broken = tmp_path / "broken-study"
    broken.mkdir()
    (broken / "study_result.json").write_text('{"artifacts": []}', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["ml-prepare", str(broken), "--out", str(tmp_path / "unused"), "--json"])
    assert excinfo.value.code == 2
    assert _stdout_json(capsys)["error"] == "study_result.json must contain an artifacts mapping"


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
