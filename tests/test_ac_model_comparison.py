from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.artifacts import atomic_write_text, file_sha256, write_json
from pcd.circuit_graph import CircuitGraph, GraphComponent, GraphPort, GraphTerminal
from pcd.cli import main
from pcd.corpus_evaluation import evaluate_ac_graph_corpus
from pcd.ml.ac_evaluation import evaluate_ac_models
from pcd.ml.graph_encoding import encode_graph_batch, flat_graph_features
from pcd.ml.neural import fit_mlp, fit_relational_gnn


def _component(reference: str, kind: str, p: str, n: str, value: float) -> GraphComponent:
    return GraphComponent(reference, kind, (GraphTerminal("p", p), GraphTerminal("n", n)), value)


def _graph(family: str, design: int) -> tuple[str, dict]:
    factor = 1.0 + 0.2 * design
    if family == "l_match":
        components = (
            _component("L1", "inductor", "src", "out", 0.6e-6 * factor),
            _component("C1", "capacitor", "out", "0", 100e-12 * factor),
        )
    elif family == "pi_match":
        components = (
            _component("C1", "capacitor", "src", "0", 80e-12 * factor),
            _component("L1", "inductor", "src", "out", 1.0e-6 * factor),
            _component("C2", "capacitor", "out", "0", 30e-12 * factor),
        )
    else:
        components = (
            _component("C1", "capacitor", "src", "0", 80e-12 * factor),
            _component("L1", "inductor", "src", "out", 1.0e-6 * factor),
            _component("C2", "capacitor", "out", "0", 30e-12 * factor),
            _component("Lh", "inductor", "out", "mid", 0.3e-6 * factor),
            _component("Ch", "capacitor", "mid", "0", 60e-12 * factor),
        )
    graph = CircuitGraph(
        family,
        components,
        (GraphPort("source", "src"), GraphPort("load", "out"), GraphPort("ground", "0")),
    )
    graph_id = f"graph_{graph.instance_fingerprint}"
    return graph_id, {"graph_id": graph_id, **graph.to_dict()}


def _comparison_fixture() -> tuple[dict[str, dict], pd.DataFrame, dict]:
    graphs: dict[str, dict] = {}
    rows: list[dict] = []
    family_offsets = {"l_match": -0.7, "pi_match": 0.4, "pi_match_harmonic": 1.2}
    for family, offset in family_offsets.items():
        for design in range(4):
            graph_id, graph = _graph(family, design)
            graphs[graph_id] = graph
            for condition, resistance in enumerate((15.0, 30.0, 60.0)):
                response = offset + 0.35 * design + np.log(resistance)
                rows.append(
                    {
                        "sample_schema": "ac_graph_sample.v1",
                        "sample_id": f"{family}:{design}:{condition}",
                        "graph_id": graph_id,
                        "topology_family": family,
                        "candidate_id": f"d{design}",
                        "scenario_id": f"c{condition}",
                        "physical_group_id": f"physical:{family}:{design}:{condition}",
                        "design_group_id": f"design:{family}:{design}",
                        "condition_group_id": f"condition:{condition}",
                        "design_split": "test" if design == 3 else "train",
                        "condition_split": "test" if condition == 2 else "train",
                        "frequency_Hz": 13.56e6,
                        "context.load.resistance_ohm": resistance,
                        "context.load.name": "impedance_point",
                        "target.port_re": response,
                        "target.port_im": 0.5 * response - 0.2 * design,
                    }
                )
    manifest = {
        "schema": "ac_graph_corpus.v1",
        "roles": {
            "context_features": ["frequency_Hz", "context.load.resistance_ohm", "context.load.name"],
            "port_complex_targets": ["target.port_re", "target.port_im"],
        },
        "splits": {
            "unseen_design_within_known_topologies": {"status": "ready"},
            "unseen_external_conditions": {"status": "ready"},
            "unseen_topology_family": {"status": "ready", "groups": list(family_offsets)},
        },
    }
    return graphs, pd.DataFrame(rows), manifest


def test_graph_encodings_preserve_relations_and_supply_simple_baselines():
    graph_id, graph = _graph("pi_match", 1)
    graphs = {graph_id: graph}

    batch = encode_graph_batch(graphs, [graph_id, graph_id])
    flat = flat_graph_features(graphs, [graph_id])

    assert batch.node_features.shape[0] == 2
    assert batch.adjacency.shape[1] == 4
    assert batch.component_mask[0].sum() == 3
    assert batch.source_mask[0].sum() == batch.load_mask[0].sum() == batch.ground_mask[0].sum() == 1
    assert np.allclose(batch.adjacency.sum(axis=3)[batch.adjacency.sum(axis=3) > 0], 1.0)
    assert flat.matrix.shape == (1, len(flat.names))
    assert flat.matrix[0, flat.names.index("graph.capacitor_count")] == 2
    assert batch.take(np.array([True, False])).node_features.shape[0] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda graphs, graph_id: graphs.pop(graph_id), "unknown graph_id"),
        (lambda graphs, graph_id: graphs[graph_id].update(schema="wrong"), "invalid circuit graph"),
        (lambda graphs, graph_id: graphs[graph_id].update(ports=[]), "source, load, and ground"),
        (
            lambda graphs, graph_id: graphs[graph_id]["components"][0].update(terminals=[]),
            "two terminals",
        ),
    ],
)
def test_graph_encoding_rejects_incomplete_contracts(change, message):
    graph_id, graph = _graph("l_match", 0)
    graphs = {graph_id: graph}
    change(graphs, graph_id)

    with pytest.raises(ValueError, match=message):
        encode_graph_batch(graphs, [graph_id])


def test_same_split_comparison_evaluates_every_model_and_protocol():
    graphs, samples, manifest = _comparison_fixture()

    comparison = evaluate_ac_models(graphs, samples, manifest, seed=11)

    assert len(comparison.predictions) == 57
    assert comparison.summary["decision"]["status"] == "topology_holdout_evaluated"
    for protocol in comparison.summary["protocols"].values():
        assert protocol["status"] == "evaluated"
        assert protocol["winner"] in {"constant", "ridge", "mlp", "relational_gnn"}
        assert all(np.isfinite(list(protocol["mean_fold_normalized_rmse"].values())))
    assert comparison.summary["data"]["context_excluded_as_non_numeric_or_incomplete"] == ["context.load.name"]
    assert {"constant.target.port_re", "relational_gnn.target.port_im"} <= set(comparison.predictions)


def test_small_or_unavailable_splits_remain_explicitly_unevaluable():
    graphs, samples, manifest = _comparison_fixture()
    small = samples.iloc[:2].copy()
    small["sample_id"] = ["a", "b"]
    small["topology_family"] = ["l_match", "pi_match"]
    manifest["splits"]["unseen_design_within_known_topologies"] = {
        "status": "unavailable",
        "reason": "not_enough_designs",
    }
    manifest["splits"]["unseen_external_conditions"] = {
        "status": "unavailable",
        "reason": "not_enough_conditions",
    }
    manifest["splits"]["unseen_topology_family"]["groups"] = ["l_match", "pi_match"]

    comparison = evaluate_ac_models(graphs, small, manifest)

    assert comparison.predictions.empty
    assert comparison.summary["protocols"]["unseen_design_within_known_topologies"]["reason"] == "not_enough_designs"
    assert comparison.summary["protocols"]["unseen_topology_family"]["reason"] == "no_evaluable_folds"
    assert comparison.summary["decision"]["advance_to_model_persistence"] is False


def test_neural_reference_models_validate_shapes_and_training_controls():
    features = np.arange(12, dtype=float).reshape(6, 2)
    targets = features[:, :1] / 10.0
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_mlp(features[:, 0], targets, features, seed=0)
    with pytest.raises(ValueError, match="epochs"):
        fit_mlp(features, targets, features, seed=0, epochs=0)

    graph_id, graph = _graph("l_match", 0)
    batch = encode_graph_batch({graph_id: graph}, [graph_id] * len(features))
    with pytest.raises(ValueError, match="align"):
        fit_relational_gnn(batch, features[:-1], targets, batch, features, seed=0)


def _write_corpus(root: Path) -> None:
    graphs, samples, manifest = _comparison_fixture()
    samples = samples.iloc[:2].copy()
    samples["sample_id"] = ["sample-a", "sample-b"]
    samples["topology_family"] = ["l_match", "pi_match"]
    samples["graph_id"] = [
        next(key for key, value in graphs.items() if value["topology_family"] == family)
        for family in samples["topology_family"]
    ]
    graph_records = {graph_id: graphs[graph_id] for graph_id in samples["graph_id"]}
    graph_path = root / "graphs.jsonl"
    sample_path = root / "samples.csv"
    atomic_write_text(graph_path, "".join(json.dumps(value) + "\n" for value in graph_records.values()))
    atomic_write_text(sample_path, samples.to_csv(index=False))
    manifest["artifacts"] = {
        "graphs": {"path": "graphs.jsonl", "sha256": file_sha256(graph_path)},
        "samples": {"path": "samples.csv", "sha256": file_sha256(sample_path)},
    }
    manifest["splits"]["unseen_design_within_known_topologies"] = {"status": "unavailable", "reason": "small"}
    manifest["splits"]["unseen_external_conditions"] = {"status": "unavailable", "reason": "small"}
    manifest["splits"]["unseen_topology_family"]["groups"] = ["l_match", "pi_match"]
    write_json(root / "manifest.json", manifest)


def test_corpus_evaluation_adapter_verifies_artifacts_and_cli(tmp_path: Path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus(corpus)

    result = evaluate_ac_graph_corpus(corpus, tmp_path / "evaluation", seed=4)
    assert result["schema"] == "ac_model_comparison.v1"
    assert result["decision"]["status"] == "insufficient_topology_evidence"
    assert (tmp_path / "evaluation" / "predictions.csv").is_file()

    main(["ml-corpus-evaluate", str(corpus), "--out", str(tmp_path / "cli-evaluation")])
    assert "unseen_topology_family: unavailable" in capsys.readouterr().out

    with pytest.raises(ValueError, match="outside"):
        evaluate_ac_graph_corpus(corpus, corpus / "nested")

    (corpus / "samples.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_ac_graph_corpus(corpus, tmp_path / "corrupt-evaluation")
