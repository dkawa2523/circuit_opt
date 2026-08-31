from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pcd.case import Case, load_case
from pcd.core import Candidate, ControlState, EvaluationRequest, RawResult, Scenario
from pcd.results import candidate_summary
from pcd.results import store as store_module
from pcd.study import (
    _feasibility_first_loss,
    _runtime_fingerprint,
    _simulation_fingerprint,
    build_case_runner,
    resolve_study_case,
    run_case_study,
)
from pcd.study_config import CaseControlPolicy, candidate_case, study_spec_from_case


def test_study_config_separates_design_scenario_and_control(topology_case):
    data = copy.deepcopy(topology_case.data)
    data["study"] = {
        "design_variables": ["C1", "L1"],
        "scenarios": [
            {"id": "light", "values": {"Rp": 10.0}, "controls": {"C2": 1e-10}},
            {"id": "heavy", "values": {"Rp": 100.0}, "controls": {"C2": 5e-10}},
        ],
        "controls": {"variables": {"Ch": {"values": [5e-11, 1e-10]}}, "budget": 2},
        "aggregation": "worst",
    }
    case = Case(topology_case.path, data)
    spec = study_spec_from_case(case)
    projected = candidate_case(case)
    assert [scenario.scenario_id for scenario in spec.scenarios] == ["light", "heavy"]
    assert set(projected.data["variables"]) == {"C1", "L1"}


def test_case_study_runs_through_the_generic_pipeline(tmp_path, topology_case):
    result = run_case_study(
        topology_case,
        n_trials=2,
        run_root=tmp_path,
        optimizer_name="random",
        solver_override="dummy",
        seed=3,
    )
    study_root = Path(result["run_root"])
    assert result["schema"] == "study_result.v1"
    assert result["n_candidates"] == 2
    assert result["n_evaluations"] == 2
    assert result["best"]["candidate"]["candidate_id"].startswith("trial_")
    assert result["execution"] == {
        "solver": "dummy",
        "optimizer": "random",
        "trials": 2,
        "seed": 3,
    }
    assert (study_root / "study_result.json").exists()
    assert len(list((study_root / "evaluations").glob("*/result.json"))) == 2
    stored = json.loads((study_root / "study_result.json").read_text(encoding="utf-8"))
    assert stored["study"]["study_id"] == topology_case.case_id
    archived_case = load_case(study_root / "case.yaml")
    assert archived_case.data["run"]["trials"] == 2
    assert archived_case.data["optimizer"]["seed"] == 3
    summary = candidate_summary(study_root)
    assert len(summary) == 2
    assert "objective.loss" in summary
    assert "control_margin" in summary


def test_candidate_summary_matches_evidence_coverage_and_objective_direction(tmp_path):
    study_root = tmp_path / "summary"
    candidates = study_root / "candidates"
    candidates.mkdir(parents=True)

    def save(candidate_id, success, feasible, score, margin):
        payload = {
            "candidate": {"candidate_id": candidate_id, "values": {}},
            "success_fraction": success,
            "feasible_fraction": feasible,
            "total_violation": 0.0,
            "aggregates": {"score": score},
            "control_margin": margin,
        }
        (candidates / f"{candidate_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    save("failed", 0.5, 0.5, 1000.0, 1.0)
    save("incomplete", 1.0, 0.5, 100.0, 1.0)
    save("complete-low", 1.0, 1.0, 1.0, 1.0)
    save("complete-edge", 1.0, 1.0, 2.0, 0.0)
    save("complete-center", 1.0, 1.0, 2.0, 1.0)
    (study_root / "study_result.json").write_text(
        json.dumps({"study": {"objectives": [{"metric": "score", "direction": "maximize", "aggregation": "worst"}]}}),
        encoding="utf-8",
    )

    summary = candidate_summary(study_root)
    assert summary["candidate_id"].tolist() == [
        "complete-center",
        "complete-edge",
        "complete-low",
        "incomplete",
        "failed",
    ]


def test_study_archives_external_data_once_at_the_study_root(tmp_path):
    case_path = Path(__file__).resolve().parents[1] / "bench" / "cases" / "match_fixed_nominal.yaml"
    _spec, runner, store = build_case_runner(load_case(case_path), tmp_path, solver_override="dummy")
    runner.evaluate_candidate(Candidate("fixed"))
    root = store.root

    assert len(list((root / "inputs").iterdir())) == 1
    assert (root / "input_manifest.json").is_file()
    assert not list((root / "artifacts").rglob("inputs"))
    assert not list((root / "artifacts").rglob("input_manifest.json"))
    evaluation_manifest = json.loads(next((root / "artifacts").glob("*/sim_manifest.json")).read_text(encoding="utf-8"))
    shared_manifest = evaluation_manifest["artifacts"]["input_manifest"]
    assert (Path(evaluation_manifest["run_dir"]) / shared_manifest).resolve() == (
        root / "input_manifest.json"
    ).resolve()


def test_deep_windows_workspace_keeps_internal_artifacts_below_legacy_limit(tmp_path, topology_case):
    long_id = "literature_colpo1999_fixed_digitization_corners"
    case = Case(
        topology_case.path,
        {**topology_case.data, "case_id": long_id},
        topology_case.source_data,
        topology_case.resolved_plan,
    )

    # This root reproduces the old 261-character raw-result temp path while
    # remaining representative of a repository nested below a user profile.
    padding = max(0, 124 - len(str(tmp_path)) - 1)
    deep_root = tmp_path / ("d" * padding) if padding else tmp_path
    old_temp_length = len(str(deep_root)) + 1 + len(long_id) + len("\\raw\\" + "0" * 64 + "\\raw_result.json.tmp")
    assert old_temp_length >= 260

    _spec, runner, store = build_case_runner(case, deep_root, solver_override="dummy")
    runner.evaluate_candidate(Candidate("fixed"))

    files = [path for path in store.root.rglob("*") if path.is_file()]
    assert files
    assert max(len(str(path)) for path in files) <= 248
    assert store.root.name != long_id
    assert len(store.root.name) <= 32
    raw_paths = list((store.root / "raw").glob("*/raw_result.json"))
    assert raw_paths
    assert all(len(path.parent.name) == 24 for path in raw_paths)
    raw_payload = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    assert len(raw_payload["cache_key"]) == 64
    assert raw_payload["cache_key"].startswith(raw_paths[0].parent.name)
    assert not list(store.root.rglob(".tmp-*"))
    input_manifest = json.loads((store.root / "input_manifest.json").read_text(encoding="utf-8"))
    assert len(input_manifest["inputs"][0]["sha256"]) == 64
    assert len(Path(input_manifest["inputs"][0]["artifact"]).name) == 32


def test_shortened_raw_cache_collision_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "_CACHE_PATH_KEY_LENGTH", 1)
    store = store_module.FileResultStore(tmp_path, "collision")
    seen: dict[str, EvaluationRequest] = {}
    first = second = None
    for value in range(100):
        request = EvaluationRequest(Candidate(f"c{value}", {"x": value}), Scenario("s"), ControlState())
        prefix = store.raw_key(request)[0]
        if prefix in seen:
            first, second = seen[prefix], request
            break
        seen[prefix] = request
    assert first is not None
    assert second is not None

    store.save_raw(first, RawResult("ok", {"score": 1.0}))
    with pytest.raises(ValueError, match="shortened raw cache path collision"):
        store.save_raw(second, RawResult("ok", {"score": 2.0}))


def test_explicit_objective_and_control_axes_are_translated(topology_case):
    data = copy.deepcopy(topology_case.data)
    data["study"] = {
        "scenarios": [
            {
                "scenario_id": "corner",
                "values": {"Rp": 25.0},
                "controls": {"bias": 2.0},
                "weight": 3.0,
            }
        ],
        "objectives": [{"metric": "loss", "direction": "maximize", "aggregation": "mean"}],
        "controls": {
            "defaults": {"bias": 0.0},
            "variables": {"tune": {"values": [1.0, 2.0]}},
            "by_scenario": {"corner": {"bias": 1.0}},
            "budget": 2,
        },
    }
    case = Case(topology_case.path, data)
    spec = study_spec_from_case(case)
    assert spec.scenarios[0].weight == 3.0
    assert spec.objectives[0].direction == "maximize"

    policy = CaseControlPolicy(case)
    controls = policy.controls(spec, Candidate("design"), spec.scenarios[0])
    assert [dict(control.values) for control in controls] == [
        {"bias": 2.0, "tune": 1.0},
        {"bias": 2.0, "tune": 2.0},
    ]


def test_scenario_table_is_loaded_and_excluded_from_design_variables(tmp_path, topology_case):
    table = tmp_path / "loads.csv"
    table.write_text("scenario_id,R,X,weight\ncorner_a,20,-90,2\ncorner_b,40,-30,1\n", encoding="utf-8")
    data = copy.deepcopy(topology_case.data)
    data["study"] = {
        "scenario_table": {
            "table_file": str(table),
            "values": {"Rload": "R", "Xload": "X"},
        }
    }
    case = Case(topology_case.path, data)
    spec = study_spec_from_case(case)
    assert [scenario.scenario_id for scenario in spec.scenarios] == ["corner_a", "corner_b"]
    assert spec.scenarios[0].values == {"Rload": 20.0, "Xload": -90.0}
    assert spec.scenarios[0].weight == 2.0
    assert {"Rload", "Xload"}.isdisjoint(candidate_case(case).data["variables"])


@pytest.mark.parametrize(
    ("table_text", "table_cfg", "message"),
    [
        ("scenario_id,R\na,20\n", {}, "table_file is required"),
        ("scenario_id,R\na,20\n", {"table_file": "TABLE"}, "values must map"),
        (
            "scenario_id,R\na,20\n",
            {"table_file": "TABLE", "values": {"Xload": "X"}},
            "missing columns",
        ),
        (
            "scenario_id,R\n,20\n",
            {"table_file": "TABLE", "values": {"Rload": "R"}},
            "empty scenario_id",
        ),
        (
            "scenario_id,R\na,\n",
            {"table_file": "TABLE", "values": {"Rload": "R"}},
            "scenario value is empty",
        ),
        (
            "scenario_id,R\na,nan\n",
            {"table_file": "TABLE", "values": {"Rload": "R"}},
            "must be finite",
        ),
        (
            "scenario_id,R\n",
            {"table_file": "TABLE", "values": {"Rload": "R"}},
            "scenario table is empty",
        ),
    ],
)
def test_invalid_scenario_tables_fail_at_the_data_boundary(tmp_path, topology_case, table_text, table_cfg, message):
    table = tmp_path / "invalid.csv"
    table.write_text(table_text, encoding="utf-8")
    cfg = {key: (str(table) if value == "TABLE" else value) for key, value in table_cfg.items()}
    data = copy.deepcopy(topology_case.data)
    data["study"] = {"scenario_table": cfg}
    with pytest.raises(ValueError, match=message):
        study_spec_from_case(Case(topology_case.path, data))


def test_inline_and_table_scenarios_cannot_both_own_the_same_axis(tmp_path, topology_case):
    table = tmp_path / "loads.csv"
    table.write_text("scenario_id,R\na,20\n", encoding="utf-8")
    data = copy.deepcopy(topology_case.data)
    data["study"] = {
        "scenarios": [{"id": "inline"}],
        "scenario_table": {"table_file": str(table), "values": {"Rload": "R"}},
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        study_spec_from_case(Case(topology_case.path, data))


@pytest.mark.parametrize(
    ("study", "message"),
    [
        (3, "study must be a mapping"),
        ({"scenarios": []}, "scenarios must be a non-empty list"),
        ({"fidelities": []}, "no longer supported"),
        ({"objectives": []}, "objectives must be a non-empty list"),
    ],
)
def test_malformed_study_axes_fail_at_translation(topology_case, study, message):
    data = copy.deepcopy(topology_case.data)
    data["study"] = study
    with pytest.raises(ValueError, match=message):
        study_spec_from_case(Case(topology_case.path, data))


def test_design_projection_accepts_inline_specs_and_rejects_ambiguous_forms(topology_case):
    data = copy.deepcopy(topology_case.data)
    data["study"] = {"design_variables": {"custom": {"bounds": [1.0, 2.0], "default": 1.5}}}
    projected = candidate_case(Case(topology_case.path, data))
    assert projected.data["variables"] == {"custom": {"bounds": [1.0, 2.0], "default": 1.5}}

    data["study"] = {"design_variables": ["not_declared"]}
    with pytest.raises(ValueError, match="not declared"):
        candidate_case(Case(topology_case.path, data))
    data["study"] = {"design_variables": "C1"}
    with pytest.raises(ValueError, match="list or mapping"):
        candidate_case(Case(topology_case.path, data))


def test_runtime_fingerprint_hashes_present_plugins_and_marks_missing_ones(tmp_path, topology_case):
    plugin = tmp_path / "plugin.py"
    plugin.write_text("VALUE = 1\n", encoding="utf-8")
    missing = tmp_path / "missing.py"
    data = copy.deepcopy(topology_case.data)
    data["plugins"] = [str(plugin), str(missing)]
    fingerprint = _runtime_fingerprint(Case(topology_case.path, data), "dummy")
    assert fingerprint["plugins"][str(plugin.resolve())]
    assert fingerprint["plugins"][str(missing.resolve())] is None
    assert fingerprint["implementation_sha256"]
    assert fingerprint["solver"]["name"] == "dummy"


def test_simulation_fingerprint_excludes_study_interpretation_but_keeps_physics(topology_case):
    baseline = copy.deepcopy(topology_case.data)
    first = _simulation_fingerprint(Case(topology_case.path, baseline), "dummy")

    interpretation = copy.deepcopy(baseline)
    interpretation.setdefault("target", {})["objective"] = "rf_load"
    interpretation["target"]["constraints"] = {"metric_bounds": {"load_real_power_W": {"min": 10.0}}}
    interpretation["study"] = {
        "objectives": [{"metric": "load_real_power_W", "direction": "maximize"}],
        "aggregation": "mean",
    }
    interpretation["optimizer"] = {"name": "random", "seed": 99}
    assert _simulation_fingerprint(Case(topology_case.path, interpretation), "dummy") == first
    assert _runtime_fingerprint(Case(topology_case.path, interpretation), "dummy") != _runtime_fingerprint(
        Case(topology_case.path, baseline), "dummy"
    )

    changed_circuit = copy.deepcopy(baseline)
    changed_circuit.setdefault("circuit", {})["output_node"] = "different_node"
    assert _simulation_fingerprint(Case(topology_case.path, changed_circuit), "dummy") != first

    changed_frequency = copy.deepcopy(baseline)
    changed_frequency.setdefault("target", {})["fundamental_Hz"] = 27.12e6
    assert _simulation_fingerprint(Case(topology_case.path, changed_frequency), "dummy") != first


def test_fingerprints_track_external_physics_and_metric_files_separately(tmp_path, topology_case):
    netlist = tmp_path / "circuit.cir"
    target = tmp_path / "target.csv"
    netlist.write_text("R1 in out 50\n", encoding="utf-8")
    target.write_text("time_s,voltage_V\n0,0\n", encoding="utf-8")
    data = copy.deepcopy(topology_case.data)
    data.setdefault("circuit", {})["netlist_file"] = str(netlist)
    data.setdefault("target", {})["waveform_file"] = str(target)
    case = Case(topology_case.path, data)
    raw_before = _simulation_fingerprint(case, "dummy")
    evaluation_before = _runtime_fingerprint(case, "dummy")

    target.write_text("time_s,voltage_V\n0,1\n", encoding="utf-8")
    assert _simulation_fingerprint(case, "dummy") == raw_before
    assert _runtime_fingerprint(case, "dummy") != evaluation_before

    netlist.write_text("R1 in out 75\n", encoding="utf-8")
    assert _simulation_fingerprint(case, "dummy") != raw_before


def test_study_requires_a_positive_trial_budget(tmp_path, topology_case):
    with pytest.raises(ValueError, match="positive"):
        run_case_study(topology_case, n_trials=0, run_root=tmp_path)


def test_effective_execution_settings_replace_the_archived_public_plan():
    path = Path(__file__).resolve().parents[1] / "bench" / "cases" / "match_discrete_hardware_search.yaml"
    case = load_case(path)

    effective = resolve_study_case(
        case,
        n_trials=2,
        optimizer_name="random",
        solver_override="dummy",
        seed=9,
    )

    assert effective.authored_data == case.authored_data
    assert effective.resolved_plan is not None
    assert effective.resolved_plan["execution"] == {
        "solver": "dummy",
        "optimizer": "random",
        "trials": 2,
        "seed": 9,
    }
    assert effective.resolved_plan["case"] == effective.data
    assert case.resolved_plan is not None
    assert case.resolved_plan["execution"]["optimizer"] == "grid"


def test_grid_optimizer_cannot_be_partially_run_through_the_python_api(tmp_path):
    path = Path(__file__).resolve().parents[1] / "bench" / "cases" / "match_discrete_hardware_search.yaml"
    with pytest.raises(ValueError, match="requires exactly 3 trials"):
        run_case_study(load_case(path), n_trials=2, run_root=tmp_path)
    assert not list(tmp_path.rglob("study_result.json"))


def test_optimizer_signal_uses_coverage_while_final_rank_keeps_the_full_order():
    feasible_bad_objective = _feasibility_first_loss((0.0, 0.0, 0.0, 0.0, 1e30))
    higher_coverage = _feasibility_first_loss((1.0, 0.0, 0.25, 100.0, -1e30))
    lower_coverage = _feasibility_first_loss((1.0, 0.0, 0.75, 0.0, -1e30))
    same_coverage_lower_violation = _feasibility_first_loss((1.0, 0.0, 0.25, 0.0, -1e30))

    assert feasible_bad_objective < higher_coverage < lower_coverage
    assert higher_coverage == same_coverage_lower_violation
