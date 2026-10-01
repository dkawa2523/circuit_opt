"""File adapter for the topology-aware AC model comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import atomic_write_text, file_sha256, read_json, utc_now, write_json
from .ml.ac_evaluation import AcModelComparison, evaluate_ac_models

EVALUATION_SCHEMA = "ac_model_comparison.v1"


def evaluate_ac_graph_corpus(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare fixed models using only one committed AC corpus."""

    root = Path(corpus_root).resolve()
    output = Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("model-comparison output must remain outside the source corpus")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "ac_graph_corpus.v1":
        raise ValueError(f"AC corpus manifest must use schema 'ac_graph_corpus.v1': {manifest_path}")

    graph_path = _verified_artifact(root, manifest, "graphs")
    sample_path = _verified_artifact(root, manifest, "samples")
    graphs = _load_graphs(graph_path)
    samples = pd.read_csv(sample_path)
    comparison = evaluate_ac_models(graphs, samples, manifest, seed=seed)
    return _write_comparison(output, root, manifest_path, graph_path, sample_path, comparison)


def _verified_artifact(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("AC corpus manifest must declare artifacts")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"AC corpus manifest is missing artifact {name!r}")
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected_hash, str):
        raise ValueError(f"AC corpus artifact {name!r} must declare path and sha256")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"AC corpus artifact {name!r} escapes its corpus root")
    if not path.is_file():
        raise FileNotFoundError(f"AC corpus artifact is missing: {path}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"AC corpus artifact hash mismatch: {name}")
    return path


def _load_graphs(path: Path) -> dict[str, Mapping[str, Any]]:
    graphs: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping) or not isinstance(value.get("graph_id"), str):
            raise ValueError(f"invalid graph record at {path}:{line_number}")
        graph_id = str(value["graph_id"])
        if graph_id in graphs:
            raise ValueError(f"duplicate graph_id {graph_id!r} in {path}")
        graphs[graph_id] = value
    if not graphs:
        raise ValueError(f"AC corpus graph artifact is empty: {path}")
    return graphs


def _write_comparison(
    output: Path,
    source_root: Path,
    source_manifest: Path,
    graph_path: Path,
    sample_path: Path,
    comparison: AcModelComparison,
) -> dict[str, Any]:
    predictions_path = output / "predictions.csv"
    evaluation_path = output / "evaluation.json"
    atomic_write_text(predictions_path, comparison.predictions.to_csv(index=False, float_format="%.17g"))
    summary = {
        "schema": EVALUATION_SCHEMA,
        "created_at": utc_now(),
        "purpose": "same_split_ac_response_model_comparison",
        "source": {
            "corpus_root": str(source_root),
            "manifest": {"path": str(source_manifest), "sha256": file_sha256(source_manifest)},
            "graphs_sha256": file_sha256(graph_path),
            "samples_sha256": file_sha256(sample_path),
        },
        **dict(comparison.summary),
        "artifacts": {
            "predictions": {"path": "predictions.csv", "sha256": file_sha256(predictions_path)},
            "evaluation": "evaluation.json",
        },
    }
    write_json(evaluation_path, summary)
    return summary
