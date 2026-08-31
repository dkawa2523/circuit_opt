"""Characterization tests for the CLI surface.

Written before ``pcd.cli.main`` was split into per-command handlers so the
refactor is provably behaviour-preserving.  These pin the *contract*: which
artifacts appear, what is printed, and which exit code is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcd.cli import main

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
    main(["sim-run", RC_CASE, "--solver", "test_fake", "--run-root", str(tmp_path)])
    payload = _stdout_json(capsys)
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "ok"
    assert payload["schema"] == "simulation_record.v2"
    assert (run_dir / "waveform.csv").exists()
    assert not (run_dir / "metrics.json").exists()


def test_sim_run_strict_exit_fails_when_the_simulation_fails(tmp_path, capsys):
    """A build failure must still write a record, and --strict-exit must exit 1."""

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
        main(["sim-run", str(case_path), "--solver", "test_fake", "--run-root", str(run_root), "--strict-exit"])
    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error"]
    # A failed preparation still leaves a complete record behind.
    assert (Path(payload["run_dir"]) / "waveform.csv").exists()


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


def test_visualize_response_plots_a_run(tmp_path, capsys):
    main(["sim-run", RC_CASE, "--solver", "test_fake", "--run-root", str(tmp_path / "runs")])
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]
    image = tmp_path / "response.png"
    main(["visualize-response", run_dir, "--out", str(image)])
    assert capsys.readouterr().out.strip() == str(image)
    assert image.stat().st_size > 0


# --- study -----------------------------------------------------------------


def test_run_is_the_simple_human_readable_study_entry_point(tmp_path, capsys):
    main(["run", RC_CASE, "--solver", "test_fake", "--output", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Feasible across all conditions: yes" in output
    assert "Decision: meets_declared_acceptance" in output
    assert "Selected candidate:" in output
    assert "Condition coverage: accepted 1/1, solved 1/1" in output
    assert "Objective: loss=" in output
    assert "Condition nominal: accepted, control={}" in output
    assert "Candidates: 1" in output
    assert "Results:" in output
    assert len(list(tmp_path.rglob("study_result.json"))) == 1


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
    assert (study_root / "study_history.json").exists()


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
