from __future__ import annotations

import ast
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pcd.sim_core as sim_core_module
from pcd.common import Case, default_params, load_case, spice_value
from pcd.ml_core import (
    build_learning_table,
    fit_ridge_surrogate,
    interpolate_to_target,
    predict_candidates_with_surrogate,
    propose_candidates,
    score_record,
    score_run_root,
)
from pcd.ml_registry import available as ml_available
from pcd.records import find_sim_records, import_external_waveform, summary_dataframe
from pcd.records import metric_summary_dataframe
from pcd.sim_core import build_circuit, build_load_subckt, ngspice_cli, parse_wrdata, render_ngspice_netlist, simulate_case
from pcd.sim_registry import available as sim_available
from pcd.netlist_viz import netlist_summary, render_netlist_schematic
from pcd.validation import validate_case, validate_waveform
from pcd.workflow import optimize_closed_loop, simulate_candidates

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def test_registries_are_separate():
    sim = sim_available()
    ml = ml_available()
    assert set(sim) == {"circuit", "load", "solver"}
    assert set(ml) == {"objective", "optimizer"}
    assert "ngspice_cli" in sim["solver"]
    assert "waveform_l2" in ml["objective"]
    assert "random" in ml["optimizer"]


def test_spice_value_param_rendering():
    assert spice_value("1k") == "1k"
    assert spice_value("C1") == "{C1}"
    assert spice_value("$C1") == "{C1}"
    assert spice_value("{C1}") == "{C1}"


def test_simulation_does_not_write_metrics_until_ml_scores(tmp_path):
    case = load_case(EX / "generic_rc_filter.yaml")
    rec = simulate_case(case, run_root=tmp_path, solver_override="dummy")
    assert rec.status == "ok"
    assert (rec.run_dir / "sim_manifest.json").exists()
    assert (rec.run_dir / "waveform.csv").exists()
    manifest = rec.manifest()
    assert manifest["schema"] == "simulation_record.v2"
    assert "provenance" in manifest
    assert "case_data_sha256" in manifest["provenance"]
    assert not (rec.run_dir / "metrics.json").exists()
    metrics = score_record(case, rec.manifest())
    assert "loss" in metrics
    assert (rec.run_dir / "metrics.json").exists()


def test_netlist_generation_no_load_is_clean():
    case = load_case(EX / "generic_rc_filter.yaml")
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    load_name, subckt = build_load_subckt(case, params)
    text = render_ngspice_netlist(case, circuit, subckt, params)
    assert load_name == "none"
    assert "Xload" not in text
    assert "R1 src out {R1}" in text
    assert "C1 out 0 {C1}" in text


def test_netlist_visualization_outputs_topology(tmp_path):
    case = load_case(ROOT / "ccp_benchmark_pack" / "ccp_gec_level1_fixed_match.yaml")
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    netlist = tmp_path / "level1.cir"
    netlist.write_text(render_ngspice_netlist(case, circuit, load, params), encoding="utf-8")
    out = tmp_path / "level1.png"
    render_netlist_schematic(netlist, out)
    summary = netlist_summary(netlist)
    assert out.exists() and out.stat().st_size > 0
    assert summary["n_nodes"] >= 5
    assert summary["component_counts"]["inductor"] >= 2
    assert any(edge["ref"] == "Xload:Rbulk" for edge in summary["edges"])


def test_time_varying_plasma_uses_q_expression(tmp_path):
    case = load_case(EX / "rf_plasma_table.yaml")
    rec = simulate_case(case, run_root=tmp_path, solver_override="dummy")
    text = (rec.run_dir / "netlist.cir").read_text()
    assert "load model: plasma_table_rlcq" in text
    assert "Q = '" in text
    assert "pwl(time" in text


def test_ml_propose_is_independent_of_simulation(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    df = propose_candidates(case, n=5, optimizer_name="random", seed=1)
    assert len(df) == 5
    assert "topology_choice" in df.columns
    assert "load_model" in df.columns
    assert list(tmp_path.rglob("sim_manifest.json")) == []


def test_sim_batch_then_score_then_fit_surrogate(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    candidates = propose_candidates(case, n=4, optimizer_name="random", seed=2)
    records = simulate_candidates(case, candidates, tmp_path, solver_override="dummy")
    assert len(records) == 4
    assert len(find_sim_records(tmp_path)) == 4
    assert not any((Path(r["run_dir"]) / "metrics.json").exists() for r in records)
    scored = score_run_root(case, tmp_path)
    assert len(scored) == 4
    assert "loss" in scored.columns
    table = build_learning_table(tmp_path)
    model = fit_ridge_surrogate(table, target_col="loss")
    assert model["schema"] == "ridge_surrogate.v2"
    assert model["n_train"] == 4


def test_closed_loop_workflow_couples_layers_explicitly(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    result = optimize_closed_loop(case, n_trials=3, run_root=tmp_path, optimizer_name="random", solver_override="dummy", seed=5)
    assert result["n_trials"] == 3
    assert result["best"] is not None
    df = summary_dataframe(tmp_path)
    assert len(df) == 3
    assert "loss" in df.columns


def test_plugin_can_add_sim_and_ml_methods(tmp_path):
    case = load_case(EX / "plugin_case.yaml")
    rec = simulate_case(case, run_root=tmp_path, solver_override="dummy")
    metrics = score_record(case, rec.manifest())
    assert rec.circuit == "custom_series_lc"
    assert metrics["objective"] == "peak_voltage"


def test_wrdata_parser_handles_legacy_repeated_time_columns(tmp_path):
    path = tmp_path / "raw_wrdata.csv"
    t = np.array([0.0, 1e-9, 2e-9])
    v = np.array([0.0, 1.0, 0.0])
    i = np.array([0.0, 0.1, 0.0])
    np.savetxt(path, np.column_stack([t, t, v, t, i]))
    result = parse_wrdata(path)
    assert np.allclose(result.time_s, t)
    assert np.allclose(result.voltage_V, v)
    assert np.allclose(result.current_A, i)


def test_wrdata_parser_handles_ngspice46_pair_columns(tmp_path):
    path = tmp_path / "raw_wrdata_ng46.csv"
    t = np.array([0.0, 1e-9, 2e-9])
    v = np.array([0.0, 1.0, 0.0])
    i = np.array([0.0, 0.1, 0.0])
    np.savetxt(path, np.column_stack([t, t, t, v, t, i]))
    result = parse_wrdata(path)
    assert np.allclose(result.time_s, t)
    assert np.allclose(result.voltage_V, v)
    assert np.allclose(result.current_A, i)


def test_import_external_waveform_allows_ml_only_scoring(tmp_path):
    case = load_case(EX / "generic_rc_filter.yaml")
    t = np.linspace(0.0, 2e-6, 200)
    v = 5.0 * np.exp(-t / 4e-7)
    external = tmp_path / "external_waveform.csv"
    pd.DataFrame({"t": t, "v": v}).to_csv(external, index=False)
    rec = import_external_waveform(case, external, tmp_path / "records", params={"R1": 1000, "C1": 1e-9})
    assert rec["solver"] == "external"
    assert (Path(rec["run_dir"]) / "sim_manifest.json").exists()
    assert not (Path(rec["run_dir"]) / "netlist.cir").exists()
    metrics = score_record(case, Path(rec["run_dir"]))
    assert "loss" in metrics
    assert (Path(rec["run_dir"]) / "metrics.json").exists()


def test_sim_batch_accepts_summary_style_param_columns(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    candidates = pd.DataFrame([
        {"param.topology_choice": "l_match", "param.load_model": "electrode_stray", "param.C1": 1e-10, "metric.loss": 999.0},
        {"param.topology_choice": "pi_match", "param.load_model": "plasma_fixed_rlc", "param.C1": 2e-10, "loss": 999.0},
    ])
    records = simulate_candidates(case, candidates, tmp_path, solver_override="dummy")
    assert len(records) == 2
    assert records[0]["circuit"] == "l_match"
    assert records[1]["load"] == "plasma_fixed_rlc"


def test_surrogate_prediction_ranks_candidates(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    candidates = propose_candidates(case, n=5, optimizer_name="random", seed=10)
    simulate_candidates(case, candidates, tmp_path, solver_override="dummy")
    score_run_root(case, tmp_path)
    table = build_learning_table(tmp_path)
    model = fit_ridge_surrogate(table, target_col="loss")
    pred = predict_candidates_with_surrogate(model, candidates)
    assert "predicted_loss" in pred.columns
    assert len(pred) == len(candidates)


def test_validation_reports_case_and_waveform_issues(tmp_path):
    case = Case(
        path=tmp_path / "bad.yaml",
        data={
            "variables": {"x": {"bounds": [-1, 1], "scale": "log", "default": 0}},
            "source": {"type": "dc_voltage", "value": 1},
            "solver": {"name": "dummy", "tran": {"step_s": 2, "stop_s": 1}},
            "target": {"waveform_file": "missing.csv"},
        },
    )
    report = validate_case(case)
    codes = {issue.code for issue in report.issues}
    assert "variable.log_bounds_non_positive" in codes
    assert "solver.dummy" in codes
    assert "target.waveform_not_found" in codes
    assert not report.ok

    waveform = pd.DataFrame({
        "time_s": [1.0, 0.0, 1.0, np.nan, 2.0],
        "voltage_V": [2.0, 0.0, 4.0, 9.0, np.nan],
        "current_A": [0.0, np.inf, 1.0, 2.0, 3.0],
    })
    wf_report = validate_waveform(waveform)
    wf_codes = {issue.code for issue in wf_report.issues}
    assert "waveform.duplicate_time" in wf_codes
    assert "waveform.non_monotonic_time" in wf_codes
    target = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "voltage_V": [0.0, 1.0, 2.0]})
    _t, _vt, v = interpolate_to_target(target, waveform)
    assert np.allclose(v, [0.0, 3.0, 3.0])


def test_failed_records_get_explicit_penalty_and_can_be_excluded(tmp_path):
    case = load_case(EX / "generic_rc_filter.yaml")
    rec = simulate_case(case, run_root=tmp_path, solver_override="does_not_exist")
    assert rec.status == "failed"
    metrics = score_record(case, rec.manifest())
    assert metrics["loss"] == 1e30
    assert metrics["status"] == "failed"
    assert metrics["reason"] == "simulation_failed"
    assert len(build_learning_table(tmp_path, include_failed=True)) == 1
    assert build_learning_table(tmp_path, include_failed=False).empty


def test_closed_loop_continues_with_failed_penalty(tmp_path):
    case = load_case(EX / "topology_choice_pipeline.yaml")
    result = optimize_closed_loop(case, n_trials=2, run_root=tmp_path, solver_override="does_not_exist", seed=7)
    assert result["n_trials"] == 2
    assert result["n_failed_trials"] == 2
    assert result["best"]["metrics"]["status"] == "failed"


def test_ngspice_diagnostics_for_missing_executable_and_timeout(tmp_path, monkeypatch):
    case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "not_a_real_ngspice"}})
    missing = ngspice_cli(tmp_path / "netlist.cir", tmp_path, case, {})
    assert missing.status == "failed"
    assert missing.diagnostics["missing_executable"] is True

    timeout_case = Case(path=EX / "generic_rc_filter.yaml", data={"solver": {"executable": "ngspice", "timeout_s": 0.01}})
    monkeypatch.setattr(sim_core_module.shutil, "which", lambda exe: exe)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output="out", stderr="err")

    monkeypatch.setattr(sim_core_module.subprocess, "run", fake_run)
    timed_out = sim_core_module.ngspice_cli(tmp_path / "netlist.cir", tmp_path, timeout_case, {})
    assert timed_out.status == "failed"
    assert timed_out.diagnostics["timed_out"] is True


def test_ngspice_default_prefers_windows_console_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_core_module.sys, "platform", "win32")
    monkeypatch.setattr(
        sim_core_module.shutil,
        "which",
        lambda exe: r"C:\ngspice\ngspice_con.exe" if exe == "ngspice_con.exe" else None,
    )
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        t = np.array([0.0, 1e-9, 2e-9])
        v = np.array([0.0, 1.0, 0.0])
        i = np.array([0.0, 0.1, 0.0])
        np.savetxt(Path(kwargs["cwd"]) / "waveform.csv", np.column_stack([t, t, t, v, t, i]))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(sim_core_module.subprocess, "run", fake_run)
    result = sim_core_module.ngspice_cli(tmp_path / "netlist.cir", tmp_path, Case(path=EX / "generic_rc_filter.yaml", data={}), {})
    assert result.status == "ok"
    assert observed["cmd"][0] == "ngspice_con.exe"
    assert "creationflags" in observed["kwargs"]


def test_ngspice_provenance_uses_windows_console_default(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_core_module.sys, "platform", "win32")
    monkeypatch.setattr(
        sim_core_module.shutil,
        "which",
        lambda exe: r"C:\ngspice\ngspice_con.exe" if exe == "ngspice_con.exe" else None,
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="ngspice-46\n", stderr="")

    monkeypatch.setattr(sim_core_module.subprocess, "run", fake_run)
    rec = sim_core_module.prepare_case(load_case(EX / "generic_rc_filter.yaml"), run_root=tmp_path, solver_name="ngspice_cli")
    solver = rec.provenance["solver"]
    assert solver["executable"] == "ngspice_con.exe"
    assert solver["resolved_executable"] == r"C:\ngspice\ngspice_con.exe"
    assert solver["version"] == "ngspice-46"


def test_solver_diagnostic_reports_windows_console_default(monkeypatch):
    monkeypatch.setattr(sim_core_module.sys, "platform", "win32")
    monkeypatch.setattr(
        sim_core_module.shutil,
        "which",
        lambda exe: r"C:\ngspice\ngspice_con.exe" if exe == "ngspice_con.exe" else None,
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="ngspice-46\n", stderr="")

    monkeypatch.setattr(sim_core_module.subprocess, "run", fake_run)
    diag = sim_core_module.diagnose_solver("ngspice_cli")
    assert diag["schema"] == "solver_diagnostic.v1"
    assert diag["executable"] == "ngspice_con.exe"
    assert diag["resolved_executable"] == r"C:\ngspice\ngspice_con.exe"
    assert diag["version"] == "ngspice-46"
    assert diag["batch_runnable"] is True


def test_metric_summary_dataframe_splits_feasible_rows():
    df = pd.DataFrame({
        "status": ["ok", "ok", "failed", "ok"],
        "loss": [1.0, 2.0, 100.0, 4.0],
        "metric.constraint_penalty": [0.0, 0.5, 0.0, 3.0],
    })
    row = metric_summary_dataframe(df).iloc[0]
    assert row["count"] == 4
    assert row["failed_count"] == 1
    assert row["feasible_count"] == 2
    assert row["infeasible_count"] == 2
    assert row["loss_median"] == 3.0
    assert row["feasible_median"] == 50.5
    assert row["infeasible_median"] == 3.0


def test_surrogate_schema_warns_and_can_be_strict():
    table = pd.DataFrame({
        "param.choice": ["a", "b", "a"],
        "param.x": [1.0, 2.0, 3.0],
        "loss": [3.0, 2.0, 1.0],
    })
    model = fit_ridge_surrogate(table)
    assert "feature_schema" in model
    pred = predict_candidates_with_surrogate(model, pd.DataFrame({"choice": ["c"], "x": [1.0]}))
    assert "prediction_warning" in pred.columns
    assert "unknown categories" in pred["prediction_warning"].iloc[0]
    with pytest.raises(ValueError):
        predict_candidates_with_surrogate(model, pd.DataFrame({"choice": ["c"], "x": [1.0]}), strict_schema=True)


def test_surrogate_v2_can_exclude_infeasible_and_transform_targets():
    table = pd.DataFrame({
        "param.choice": ["a", "b", "a", "b"],
        "param.x": [1.0, 2.0, 3.0, 4.0],
        "loss": [1.0, 100.0, 2.0, 200.0],
        "constraint_penalty": [0.0, 1.0, 0.0, 2.0],
    })
    model = fit_ridge_surrogate(
        table,
        exclude_infeasible=True,
        target_transform="log1p",
        clip_target_quantile=0.9,
    )
    assert model["schema"] == "ridge_surrogate.v2"
    assert model["n_train"] == 2
    assert model["exclude_infeasible"] is True
    pred = predict_candidates_with_surrogate(model, pd.DataFrame({"choice": ["a"], "x": [2.0]}))
    assert "predicted_loss" in pred.columns
    assert "predicted_loss_model_space" in pred.columns


def test_surrogate_prediction_accepts_v1_models():
    model = {
        "schema": "ridge_surrogate.v1",
        "feature_columns": ["param.x"],
        "feature_schema": {"raw_columns": ["param.x"], "categorical_columns": {}},
        "x_mean": [0.0],
        "x_std": [1.0],
        "weights": [1.0, 2.0],
    }
    pred = predict_candidates_with_surrogate(model, pd.DataFrame({"x": [3.0]}))
    assert pred["predicted_loss"].iloc[0] == 7.0


def test_ccp_analyzer_writes_v2_outputs(tmp_path):
    from ccp_benchmark_pack import analyze_ngspice_benchmark as analyzer

    run_root = tmp_path / "ng"
    dummy_root = tmp_path / "dummy"
    case_dir = run_root / "level3_topology_load_choice"
    dummy_case_dir = dummy_root / "level3_topology_load_choice"
    case_dir.mkdir(parents=True)
    dummy_case_dir.mkdir(parents=True)

    rows = []
    t = np.linspace(0.0, 1e-6, 64)
    for i, penalty in enumerate([0.0, 0.5, 0.0]):
        trial = case_dir / f"trial_{i:04d}"
        trial.mkdir()
        pd.DataFrame({
            "time_s": t,
            "voltage_V": np.sin(2 * np.pi * 13.56e6 * t) * (100 + 10 * i),
            "current_A": np.cos(2 * np.pi * 13.56e6 * t),
        }).to_csv(trial / "waveform.csv", index=False)
        rows.append({
            "case_id": "ccp_gec_level3_topology_and_load_choice",
            "status": "ok",
            "run_dir": str(trial),
            "param.topology_choice": ["l_match", "pi_match", "l_match"][i],
            "param.load_model": ["plasma_state_rlc", "electrode_stray", "plasma_fixed_rlc"][i],
            "param.x": float(i + 1),
            "loss": [1.0, 5.0, 2.0][i],
            "metric.constraint_penalty": penalty,
            "metric.normalized_rmse": 0.1 + i,
            "metric.harmonic_error": 0.2 + i,
            "metric.v_peak_abs_V": 100.0 + i,
            "metric.i_rms_A": 1.0 + i,
        })
    pd.DataFrame(rows).to_csv(case_dir / "summary.csv", index=False)
    pd.DataFrame(rows).assign(loss=[0.8, 0.9, 1.0], **{"metric.constraint_penalty": [0.0, 0.0, 0.0]}).to_csv(dummy_case_dir / "summary.csv", index=False)

    out_dir = tmp_path / "out"
    manifest = analyzer.analyze(argparse.Namespace(
        ngspice_run_root=str(run_root),
        profile_file=str(ROOT / "ccp_benchmark_pack" / "ngspice_benchmark_profiles.json"),
        dummy_root=str(dummy_root),
        out_dir=str(out_dir),
    ))
    assert manifest["schema"] == "ccp_ngspice_benchmark_analysis.v2"
    for name in [
        "comparison_summary.csv",
        "feasibility_summary.csv",
        "category_stats.csv",
        "top_candidates.csv",
        "harmonic_amplitudes.csv",
        "surrogate_diagnostics.csv",
        "analysis.md",
    ]:
        assert (out_dir / name).exists()
    feasibility = pd.read_csv(out_dir / "feasibility_summary.csv")
    assert set(feasibility["bucket"]) == {"all", "feasible", "infeasible"}


def test_validate_case_cli_strict_exits_on_warnings():
    from pcd.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["validate-case", str(EX / "generic_rc_filter.yaml"), "--strict"])
    assert exc.value.code == 1


def test_layer_imports_stay_separated():
    pkg = ROOT / "pcd"
    forbidden_for_sim = {"pcd.ml_core", "pcd.ml_methods", "pcd.ml_registry", "pcd.records", "pcd.workflow"}
    forbidden_for_ml = {"pcd.sim_core", "pcd.sim_methods", "pcd.sim_registry", "pcd.workflow"}

    def imported_modules(path: Path) -> set[str]:
        tree = ast.parse(path.read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(("pcd." + node.module) if node.level else node.module)
        return mods

    for name in ["sim_core.py", "sim_methods.py", "sim_registry.py"]:
        assert not (imported_modules(pkg / name) & forbidden_for_sim), name
    for name in ["ml_core.py", "ml_methods.py", "ml_registry.py", "records.py"]:
        assert not (imported_modules(pkg / name) & forbidden_for_ml), name


def test_registry_conflict_is_reported_in_validation():
    from pcd.sim_registry import register

    @register("solver", "conflict_probe")
    def solver_conflict_a(netlist_path, run_dir, case, params):
        raise AssertionError("not called")

    with pytest.warns(RuntimeWarning, match="simulation method conflict"):
        @register("solver", "conflict_probe")
        def solver_conflict_b(netlist_path, run_dir, case, params):
            raise AssertionError("not called")

    report = validate_case(load_case(EX / "generic_rc_filter.yaml"), strict=True)
    codes = {issue.code for issue in report.issues}
    assert "registry.sim_conflict" in codes
    assert not report.ok
