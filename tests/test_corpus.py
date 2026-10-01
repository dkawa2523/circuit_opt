from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pcd.artifacts import write_json
from pcd.cli import main
from pcd.corpus import _plain_scalar, export_ac_graph_corpus
from pcd.ml.corpus import _assign_condition_split, _assign_design_split


def _write_study(root: Path) -> Path:
    generation = root / "generations" / "g_test"
    generation.mkdir(parents=True)
    case = {
        "schema": "case_yaml.v1",
        "case_id": "corpus_case",
        "variables": {
            "L1": {"default": 1e-6},
            "C1": {"default": 1e-9},
            "Rload": {"default": 50.0},
        },
        "source": {
            "type": "sine_voltage",
            "name": "Vsrc",
            "p": "src",
            "n": "0",
            "amplitude_V": 10.0,
            "frequency_Hz": 13.56e6,
        },
        "circuit": {
            "builder": "from_yaml",
            "topology_family": "l_match",
            "output_node": "out",
            "components": [
                {
                    "ref": "L1",
                    "n1": "src",
                    "n2": "out",
                    "value": "L1",
                    "series_resistance_ohm": 0.2,
                    "observe": True,
                },
                {"ref": "C1", "n1": "out", "n2": "0", "value": "C1", "observe": True},
            ],
        },
        "load": {"name": "resistor", "R_ohm": "Rload", "ports": {"p": "out", "n": "0"}},
        "measurement": {"current_source": "Vsrc", "reference_impedance_ohm": 50.0},
        "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": 13.56e6}},
    }
    case_path = generation / "case.yaml"
    case_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")

    identity = {
        "table_schema": "evaluation_table.v2",
        "dataset_id": "corpus_study/g_test",
        "study_id": "corpus_study",
        "case_schema": "case_yaml.v1",
        "resolved_case_schema": "case_yaml.v1",
        "runtime_fingerprint_sha256": "runtime",
        "solver_fingerprint_sha256": "solver",
    }
    rows = []
    for trial, inductance in enumerate((1e-6, 2e-6)):
        artifact = root / "artifacts" / f"e{trial}"
        data = artifact / "data"
        debug = artifact / "debug"
        data.mkdir(parents=True)
        debug.mkdir()
        response = pd.DataFrame(
            {
                "frequency_Hz": [13.56e6],
                "voltage_re": [10.0],
                "voltage_im": [0.0],
                "current_re": [-0.1 - trial * 0.01],
                "current_im": [0.02],
                "load_voltage_V_re": [8.0],
                "load_voltage_V_im": [-1.0],
                "load_current_A_re": [0.16],
                "load_current_A_im": [-0.02],
                "component_L1_voltage_V_re": [2.0],
                "component_L1_voltage_V_im": [1.0],
                "component_L1_current_A_re": [0.1],
                "component_L1_current_A_im": [-0.02],
                "component_C1_voltage_V_re": [8.0],
                "component_C1_voltage_V_im": [-1.0],
                "component_C1_current_A_re": [0.01],
                "component_C1_current_A_im": [0.03],
            }
        )
        response.to_csv(data / "ac.csv", index=False)
        write_json(
            debug / "manifest.json",
            {"schema": "simulation_record.v1", "artifacts": {"frequency_response": "data/ac.csv"}},
        )
        rows.append(
            {
                **identity,
                "trial": trial,
                "candidate_id": f"trial_{trial:04d}",
                "scenario_id": "nominal",
                "status": "ok",
                "raw_cache_key": f"raw-{trial}",
                "artifact.manifest": f"artifacts/e{trial}/debug/manifest.json",
                "design.L1": inductance,
                "design.C1": 1e-9,
                "scenario.Rload": 50.0,
            }
        )
    evaluation_path = generation / "evaluations.csv"
    pd.DataFrame(rows).to_csv(evaluation_path, index=False)
    write_json(
        root / "study_result.json",
        {
            "schema": "study_result.v1",
            "dataset": identity,
            "n_evaluations": 2,
            "artifacts": {
                "case": "generations/g_test/case.yaml",
                "evaluation_table": "generations/g_test/evaluations.csv",
            },
        },
    )
    return root


def _generation(root: Path) -> Path:
    return root / "generations" / "g_test"


def _read_result(root: Path) -> dict:
    return json.loads((root / "study_result.json").read_text(encoding="utf-8"))


def _read_case(root: Path) -> dict:
    return yaml.safe_load((_generation(root) / "case.yaml").read_text(encoding="utf-8"))


def _write_case(root: Path, case: dict) -> None:
    (_generation(root) / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")


def _read_evaluations(root: Path) -> pd.DataFrame:
    return pd.read_csv(_generation(root) / "evaluations.csv")


def _write_evaluations(root: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(_generation(root) / "evaluations.csv", index=False)


def _response_path(root: Path, trial: int = 0) -> Path:
    return root / "artifacts" / f"e{trial}" / "data" / "ac.csv"


def _manifest_path(root: Path, trial: int = 0) -> Path:
    return root / "artifacts" / f"e{trial}" / "debug" / "manifest.json"


def test_export_ac_graph_corpus_preserves_physics_and_group_holdouts(tmp_path: Path):
    study = _write_study(tmp_path / "study")
    output = tmp_path / "corpus"

    manifest = export_ac_graph_corpus([study], output, test_fraction=0.5, seed=7)

    assert manifest["schema"] == "ac_graph_corpus.v1"
    assert manifest["counts"]["graphs"] == 2
    assert manifest["counts"]["samples"] == 2
    assert manifest["counts"]["component_response_rows"] == 4
    assert manifest["counts"]["component_target_rows"] == 4
    assert manifest["splits"]["unseen_design_within_known_topologies"]["status"] == "ready"
    assert manifest["splits"]["unseen_external_conditions"]["status"] == "unavailable"

    samples = pd.read_csv(output / "samples.csv")
    assert set(samples["design_split"]) == {"train", "test"}
    assert set(samples["condition_split"]) == {"train"}
    assert set(samples["target.source_current_into_source_re_A"]) == {-0.1, -0.11}
    assert set(samples["context.load.R_ohm"]) == {50.0}

    graphs = [json.loads(line) for line in (output / "graphs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({graph["topology_fingerprint"] for graph in graphs}) == 1
    assert all([component["reference"] for component in graph["components"]] == ["C1", "L1"] for graph in graphs)
    losses = {
        component["series_resistance_ohm"]
        for graph in graphs
        for component in graph["components"]
        if component["reference"] == "L1"
    }
    assert losses == {0.2}
    assert "Vobserve" not in (output / "graphs.jsonl").read_text(encoding="utf-8")

    component_responses = pd.read_csv(output / "component_responses.csv")
    assert component_responses["target_available"].all()
    assert set(component_responses["component_reference"]) == {"L1", "C1"}


def test_export_rejects_invalid_requests_and_source_overlap(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        export_ac_graph_corpus([], tmp_path / "out")
    for fraction in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="test_fraction"):
            export_ac_graph_corpus([tmp_path / "unused"], tmp_path / "out", test_fraction=fraction)
    with pytest.raises(ValueError, match="completed study result"):
        export_ac_graph_corpus([tmp_path / "missing"], tmp_path / "out")

    study = _write_study(tmp_path / "study")
    with pytest.raises(ValueError, match="outside every immutable"):
        export_ac_graph_corpus([study], study / "derived")


def test_export_validates_committed_study_identity(tmp_path: Path):
    cases: list[tuple[str, str]] = []

    root = _write_study(tmp_path / "bad-result-schema")
    result = _read_result(root)
    result["schema"] = "old"
    write_json(root / "study_result.json", result)
    cases.append(("bad-result-schema", "study result must use schema"))

    root = _write_study(tmp_path / "missing-evaluations")
    result = _read_result(root)
    result["artifacts"]["evaluation_table"] = "missing.csv"
    write_json(root / "study_result.json", result)
    cases.append(("missing-evaluations", "does not contain its evaluation table"))

    root = _write_study(tmp_path / "missing-case")
    result = _read_result(root)
    result["artifacts"]["case"] = "missing.yaml"
    write_json(root / "study_result.json", result)
    cases.append(("missing-case", "does not contain its executable case"))

    root = _write_study(tmp_path / "missing-dataset")
    result = _read_result(root)
    result.pop("dataset")
    write_json(root / "study_result.json", result)
    cases.append(("missing-dataset", "does not declare dataset identity"))

    root = _write_study(tmp_path / "empty-table")
    pd.DataFrame().to_csv(_generation(root) / "evaluations.csv", index=False)
    cases.append(("empty-table", "No columns to parse from file"))

    root = _write_study(tmp_path / "missing-column")
    frame = _read_evaluations(root).drop(columns=["artifact.manifest"])
    _write_evaluations(root, frame)
    cases.append(("missing-column", "missing required columns"))

    root = _write_study(tmp_path / "old-table")
    result = _read_result(root)
    result["dataset"]["table_schema"] = "evaluation_table.v1"
    write_json(root / "study_result.json", result)
    cases.append(("old-table", "requires evaluation_table.v2"))

    root = _write_study(tmp_path / "missing-identity")
    result = _read_result(root)
    result["dataset"].pop("solver_fingerprint_sha256")
    write_json(root / "study_result.json", result)
    cases.append(("missing-identity", "identity is missing"))

    root = _write_study(tmp_path / "mismatched-identity")
    frame = _read_evaluations(root)
    frame["dataset_id"] = "different"
    _write_evaluations(root, frame)
    cases.append(("mismatched-identity", "does not match committed study metadata"))

    root = _write_study(tmp_path / "missing-identity-value")
    frame = _read_evaluations(root)
    frame.loc[0, "dataset_id"] = None
    _write_evaluations(root, frame)
    cases.append(("missing-identity-value", "contains missing values"))

    root = _write_study(tmp_path / "wrong-row-count")
    result = _read_result(root)
    result["n_evaluations"] = 3
    write_json(root / "study_result.json", result)
    cases.append(("wrong-row-count", "row count does not match"))

    for name, message in cases:
        with pytest.raises((ValueError, pd.errors.EmptyDataError), match=message):
            export_ac_graph_corpus([tmp_path / name], tmp_path / f"out-{name}")


def test_export_reports_evaluations_that_cannot_supply_ac_graph_samples(tmp_path: Path):
    failed = _write_study(tmp_path / "failed")
    frame = _read_evaluations(failed)
    frame["status"] = "failed"
    _write_evaluations(failed, frame)
    with pytest.raises(ValueError, match="solver_not_ok=2"):
        export_ac_graph_corpus([failed], tmp_path / "failed-out")

    raw = _write_study(tmp_path / "raw")
    case = _read_case(raw)
    case["circuit"]["components"] = [{"raw": "B1 src out V=1"}]
    _write_case(raw, case)
    with pytest.raises(ValueError, match="graph_or_context_unavailable=2"):
        export_ac_graph_corpus([raw], tmp_path / "raw-out")

    transient_only = _write_study(tmp_path / "transient-only")
    for trial in range(2):
        write_json(_manifest_path(transient_only, trial), {"schema": "simulation_record.v1", "artifacts": {}})
    with pytest.raises(ValueError, match="no_frequency_response=2"):
        export_ac_graph_corpus([transient_only], tmp_path / "transient-out")


def test_export_checks_manifest_and_frequency_response_integrity(tmp_path: Path):
    roots: list[tuple[Path, str, type[Exception]]] = []

    root = _write_study(tmp_path / "empty-manifest-cell")
    frame = _read_evaluations(root)
    frame["artifact.manifest"] = None
    _write_evaluations(root, frame)
    roots.append((root, "does not declare artifact.manifest", ValueError))

    root = _write_study(tmp_path / "escaping-manifest")
    frame = _read_evaluations(root)
    frame["artifact.manifest"] = "../outside.json"
    _write_evaluations(root, frame)
    roots.append((root, "must remain inside", ValueError))

    root = _write_study(tmp_path / "missing-manifest")
    frame = _read_evaluations(root)
    frame["artifact.manifest"] = "artifacts/missing/debug/manifest.json"
    _write_evaluations(root, frame)
    roots.append((root, "simulation manifest is missing", FileNotFoundError))

    root = _write_study(tmp_path / "missing-response")
    write_json(
        _manifest_path(root),
        {"schema": "simulation_record.v1", "artifacts": {"frequency_response": "data/missing.csv"}},
    )
    roots.append((root, "frequency response is missing", FileNotFoundError))

    root = _write_study(tmp_path / "missing-column-response")
    response = pd.read_csv(_response_path(root)).drop(columns=["load_voltage_V_im"])
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "missing required columns", ValueError))

    root = _write_study(tmp_path / "empty-response")
    response = pd.read_csv(_response_path(root)).iloc[0:0]
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "frequency response is empty", ValueError))

    root = _write_study(tmp_path / "incomplete-load-current")
    response = pd.read_csv(_response_path(root)).drop(columns=["load_current_A_im"])
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "incomplete load-current", ValueError))

    root = _write_study(tmp_path / "missing-cache-key")
    frame = _read_evaluations(root)
    frame["raw_cache_key"] = None
    _write_evaluations(root, frame)
    roots.append((root, "does not declare raw_cache_key", ValueError))

    root = _write_study(tmp_path / "incomplete-component")
    response = pd.read_csv(_response_path(root)).drop(columns=["component_L1_current_A_im"])
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "incomplete complex response", ValueError))

    root = _write_study(tmp_path / "nonnumeric-component")
    response = pd.read_csv(_response_path(root))
    response["component_L1_current_A_re"] = response["component_L1_current_A_re"].astype(object)
    response.loc[0, "component_L1_current_A_re"] = "bad"
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "must be numeric", ValueError))

    root = _write_study(tmp_path / "nonfinite-component")
    response = pd.read_csv(_response_path(root))
    response.loc[0, "component_L1_current_A_re"] = float("inf")
    response.to_csv(_response_path(root), index=False)
    roots.append((root, "must be finite", ValueError))

    for root, message, exception in roots:
        with pytest.raises(exception, match=message):
            export_ac_graph_corpus([root], tmp_path / f"out-{root.name}")


def test_export_allows_partial_optional_response_coverage(tmp_path: Path):
    study = _write_study(tmp_path / "partial")
    for trial in range(2):
        response = pd.read_csv(_response_path(study, trial))
        optional = [
            name for name in response.columns if name.startswith("component_") or name.startswith("load_current_A_")
        ]
        response.drop(columns=optional).to_csv(_response_path(study, trial), index=False)

    manifest = export_ac_graph_corpus([study], tmp_path / "partial-out")

    assert manifest["counts"]["component_target_rows"] == 0
    assert manifest["counts"]["load_current_target_rows"] == 0
    assert not pd.read_csv(tmp_path / "partial-out" / "component_responses.csv")["target_available"].any()


def test_context_fallbacks_remain_explicit_and_raw_sources_are_skipped(tmp_path: Path):
    study = _write_study(tmp_path / "context")
    case = _read_case(study)
    case["circuit"].pop("topology_family")
    case["source"].pop("name")
    case["measurement"]["current_source"] = "not-present"
    case["load"]["extra"] = [1, "Rload"]
    case["load"]["characterization"] = {"ignored": "path.csv"}
    _write_case(study, case)

    manifest = export_ac_graph_corpus([study], tmp_path / "context-out")
    sample = pd.read_csv(tmp_path / "context-out" / "samples.csv").iloc[0]
    graph = json.loads((tmp_path / "context-out" / "graphs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert graph["topology_family"] == "corpus_case:from_yaml"
    assert sample["context.load.extra.0"] == 1
    assert sample["context.load.extra.1"] == 50
    assert all("characterization" not in name for name in manifest["roles"]["context_features"])

    no_source = _write_study(tmp_path / "no-source")
    case = _read_case(no_source)
    case.pop("source")
    _write_case(no_source, case)
    assert export_ac_graph_corpus([no_source], tmp_path / "no-source-out")["counts"]["samples"] == 2

    raw_source = _write_study(tmp_path / "raw-source")
    case = _read_case(raw_source)
    case["source"] = {"raw": "Vsrc src 0 AC 1"}
    _write_case(raw_source, case)
    with pytest.raises(ValueError, match="graph_or_context_unavailable=2"):
        export_ac_graph_corpus([raw_source], tmp_path / "raw-source-out")

    bad_measurement = _write_study(tmp_path / "bad-measurement")
    case = _read_case(bad_measurement)
    case["measurement"] = ["bad"]
    _write_case(bad_measurement, case)
    with pytest.raises(ValueError, match="graph_or_context_unavailable=2"):
        export_ac_graph_corpus([bad_measurement], tmp_path / "bad-measurement-out")

    empty_family = _write_study(tmp_path / "empty-family")
    case = _read_case(empty_family)
    case["circuit"]["topology_family"] = ""
    _write_case(empty_family, case)
    with pytest.raises(ValueError, match="graph_or_context_unavailable=2"):
        export_ac_graph_corpus([empty_family], tmp_path / "empty-family-out")


def test_role_and_component_identifiers_remain_unambiguous(tmp_path: Path):
    duplicate_role = _write_study(tmp_path / "duplicate-role")
    frame = _read_evaluations(duplicate_role)
    frame["scenario.L1"] = 1e-6
    _write_evaluations(duplicate_role, frame)
    with pytest.raises(ValueError, match="belongs to both design and scenario"):
        export_ac_graph_corpus([duplicate_role], tmp_path / "duplicate-role-out")

    collision = _write_study(tmp_path / "component-collision")
    case = _read_case(collision)
    case["circuit"]["components"] = [
        {"ref": "R-1", "n1": "src", "n2": "out", "value": 10},
        {"ref": "R_1", "n1": "out", "n2": "0", "value": 20},
    ]
    _write_case(collision, case)
    with pytest.raises(ValueError, match="collide after probe-name normalization"):
        export_ac_graph_corpus([collision], tmp_path / "collision-out")


def test_split_protocols_refuse_topology_confounding_and_accept_shared_conditions():
    insufficient_design = pd.DataFrame(
        {
            "topology_family": ["a", "b"],
            "design_group_id": ["da", "db"],
            "condition_group_id": ["c1", "c2"],
            "frequency_Hz": [1.0, 1.0],
        }
    )
    design = _assign_design_split(insufficient_design, 0.5, 0)
    assert design["status"] == "unavailable"
    assert design["affected_topologies"] == ["a", "b"]

    confounded = insufficient_design.drop(columns=["design_split"]).copy()
    condition = _assign_condition_split(confounded, 0.5, 0)
    assert condition["status"] == "unavailable"
    assert condition["reason"] == "condition_split_is_confounded_with_topology_family"

    shared = pd.DataFrame(
        {
            "topology_family": ["a", "a", "b", "b"],
            "design_group_id": ["a1", "a2", "b1", "b2"],
            "condition_group_id": ["c1", "c2", "c1", "c2"],
            "frequency_Hz": [1.0, 2.0, 1.0, 2.0],
        }
    )
    condition = _assign_condition_split(shared, 0.5, 0)
    assert condition["status"] == "ready"
    assert set(shared["condition_split"]) == {"train", "test"}


def test_ml_corpus_cli_reports_machine_and_human_outputs(tmp_path: Path, capsys):
    study = _write_study(tmp_path / "study")
    frame = _read_evaluations(study)
    frame.loc[0, "status"] = "failed"
    _write_evaluations(study, frame)

    main(["ml-corpus", str(study), "--out", str(tmp_path / "json-out"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["skipped_evaluations"] == 1

    main(["ml-corpus", str(study), "--out", str(tmp_path / "text-out")])
    output = capsys.readouterr().out
    assert "AC graph corpus:" in output
    assert "Skipped evaluations: 1 (solver_not_ok=1)" in output


def test_plain_scalar_normalizes_numpy_and_missing_values():
    assert _plain_scalar(np.int64(3)) == 3
    assert _plain_scalar(None) is None
    assert _plain_scalar(pd.NA) is None
    assert _plain_scalar(float("nan")) is None
    assert _plain_scalar(Path("value")) == "value"
