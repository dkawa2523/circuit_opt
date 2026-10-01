"""Export completed AC studies as a graph-linked, replayable ML corpus.

This is an application adapter: it reads immutable study artifacts, rebuilds
their structured physical circuits without running a solver, and writes a
relational corpus.  Model code stays in :mod:`pcd.ml`; simulation remains in
the lower-level netlist and record modules.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_text, file_sha256, read_json, utc_now, write_json
from .case import Case, default_params, load_case
from .circuit_graph import CircuitGraph
from .ml import AcResponseRecord, PreparedAcCorpus, prepare_ac_corpus
from .netlist import build_circuit, circuit_config, circuit_to_graph, load_config
from .records import frequency_response_path, load_frequency_response
from .results import study_artifact_path
from .simulation_input import resolve_source_specs
from .spice import resolve_value

CORPUS_SCHEMA = "ac_graph_corpus.v1"

_IDENTITY_COLUMNS = (
    "table_schema",
    "dataset_id",
    "study_id",
    "case_schema",
    "resolved_case_schema",
    "runtime_fingerprint_sha256",
    "solver_fingerprint_sha256",
)
_ROW_COLUMNS = (
    "trial",
    "candidate_id",
    "scenario_id",
    "status",
    "raw_cache_key",
    "artifact.manifest",
)
_CONTEXT_EXCLUDED_KEYS = frozenset({"variables", "characterization", "evidence"})


@dataclass(frozen=True)
class _StudySource:
    root: Path
    result_path: Path
    evaluation_path: Path
    case_path: Path
    result: Mapping[str, Any]
    evaluations: pd.DataFrame
    case: Case
    dataset_id: str
    study_id: str

    @property
    def generation_root(self) -> Path:
        return self.evaluation_path.parent


def export_ac_graph_corpus(
    study_roots: Sequence[str | Path],
    output_dir: str | Path,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Export one or more committed studies without invoking ngspice."""

    if not study_roots:
        raise ValueError("at least one completed study root is required")
    if not math.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be finite and between 0 and 1")

    sources = [_load_study_source(Path(root)) for root in study_roots]
    output = Path(output_dir).resolve()
    _validate_output_location(output, sources)

    records: list[AcResponseRecord] = []
    skipped: Counter[str] = Counter()
    skipped_examples: dict[str, list[str]] = {}
    for source in sources:
        _collect_source(source, records, skipped, skipped_examples)

    if not records:
        reasons = ", ".join(f"{name}={count}" for name, count in sorted(skipped.items())) or "none"
        raise ValueError(f"no graph-ready successful AC samples were found; skipped: {reasons}")

    prepared = prepare_ac_corpus(records, test_fraction=test_fraction, seed=seed)

    graph_path = output / "graphs.jsonl"
    sample_path = output / "samples.csv"
    component_path = output / "component_responses.csv"
    manifest_path = output / "manifest.json"
    graph_text = "".join(
        json.dumps(prepared.graphs[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for key in sorted(prepared.graphs)
    )
    atomic_write_text(graph_path, graph_text)
    atomic_write_text(sample_path, prepared.samples.to_csv(index=False, float_format="%.17g"))
    atomic_write_text(
        component_path,
        prepared.component_responses.to_csv(index=False, float_format="%.17g"),
    )

    manifest = _corpus_manifest(
        sources=sources,
        prepared=prepared,
        skipped=skipped,
        skipped_examples=skipped_examples,
        graph_path=graph_path,
        sample_path=sample_path,
        component_path=component_path,
    )
    write_json(manifest_path, manifest)
    return manifest


def _load_study_source(root: Path) -> _StudySource:
    root = root.resolve()
    result_path = root / "study_result.json"
    if not result_path.is_file():
        raise ValueError(f"completed study result not found: {result_path}")
    result = read_json(result_path)
    if not isinstance(result, Mapping) or result.get("schema") != "study_result.v1":
        raise ValueError(f"study result must use schema 'study_result.v1': {result_path}")
    evaluation_path = study_artifact_path(root, "evaluation_table")
    case_path = study_artifact_path(root, "case")
    if evaluation_path is None or not evaluation_path.is_file():
        raise ValueError(f"committed study does not contain its evaluation table: {root}")
    if case_path is None or not case_path.is_file():
        raise ValueError(f"committed study does not contain its executable case: {root}")
    frame = pd.read_csv(evaluation_path)
    dataset = result.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError(f"study result does not declare dataset identity: {result_path}")
    _validate_evaluation_identity(frame, result, dataset, evaluation_path)
    return _StudySource(
        root=root,
        result_path=result_path,
        evaluation_path=evaluation_path.resolve(),
        case_path=case_path.resolve(),
        result=result,
        evaluations=frame,
        case=load_case(case_path),
        dataset_id=str(dataset["dataset_id"]),
        study_id=str(dataset["study_id"]),
    )


def _validate_evaluation_identity(
    frame: pd.DataFrame,
    result: Mapping[str, Any],
    dataset: Mapping[str, Any],
    path: Path,
) -> None:
    if frame.empty:
        raise ValueError(f"evaluation table is empty: {path}")
    _require_columns(frame, (*_IDENTITY_COLUMNS, *_ROW_COLUMNS), path)
    if dataset.get("table_schema") != "evaluation_table.v2":
        raise ValueError("AC graph corpus requires evaluation_table.v2")
    for name in _IDENTITY_COLUMNS:
        if name not in dataset:
            raise ValueError(f"committed dataset identity is missing {name!r}")
        _validate_identity_column(frame, name, dataset[name])
    if len(frame) != int(result.get("n_evaluations", -1)):
        raise ValueError("evaluation row count does not match committed study metadata")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"evaluation table is missing required columns {missing}: {path}")


def _validate_identity_column(frame: pd.DataFrame, name: str, expected: object) -> None:
    if frame[name].isna().any():
        raise ValueError(f"evaluation table identity column {name!r} contains missing values")
    values = {str(value) for value in frame[name].unique()}
    if values != {str(expected)}:
        raise ValueError(f"evaluation table {name!r} does not match committed study metadata")


def _validate_output_location(output: Path, sources: Sequence[_StudySource]) -> None:
    for source in sources:
        if output == source.root or source.root in output.parents:
            raise ValueError("corpus output must remain outside every immutable source study")


def _collect_source(
    source: _StudySource,
    records: list[AcResponseRecord],
    skipped: Counter[str],
    skipped_examples: dict[str, list[str]],
) -> None:
    for _, row in source.evaluations.iterrows():
        if str(row["status"]) != "ok":
            skipped["solver_not_ok"] += 1
            continue
        role_values = _role_values(row, tuple(source.evaluations.columns))
        params = _merged_params(source.case, role_values)
        try:
            graph, external_context = _graph_and_context(source.case, params)
        except (KeyError, TypeError, ValueError) as exc:
            _record_skip(skipped, skipped_examples, "graph_or_context_unavailable", source, row, exc)
            continue

        manifest_path = _evaluation_manifest_path(source, row["artifact.manifest"])
        response_path = frequency_response_path(manifest_path)
        if response_path is None:
            skipped["no_frequency_response"] += 1
            continue
        if not response_path.is_file():
            raise FileNotFoundError(f"declared frequency response is missing: {response_path}")
        response = load_frequency_response(manifest_path)
        raw_cache_key = _plain_scalar(row["raw_cache_key"])
        records.append(
            AcResponseRecord(
                graph=graph,
                response=response,
                response_label=str(response_path),
                source_dataset_id=source.dataset_id,
                source_study_id=source.study_id,
                trial=_plain_scalar(row["trial"]),
                candidate_id=str(row["candidate_id"]),
                scenario_id=str(row["scenario_id"]),
                raw_cache_key="" if raw_cache_key is None else str(raw_cache_key),
                role_values=role_values,
                external_context=external_context,
            )
        )


def _record_skip(
    skipped: Counter[str],
    examples: dict[str, list[str]],
    reason: str,
    source: _StudySource,
    row: pd.Series,
    error: Exception,
) -> None:
    skipped[reason] += 1
    bucket = examples.setdefault(reason, [])
    if len(bucket) < 3:
        bucket.append(f"{source.dataset_id}:{row.get('candidate_id', '?')}:{row.get('scenario_id', '?')}: {error}")


def _role_values(row: pd.Series, columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    roles = {name: {} for name in ("design", "scenario", "control")}
    for role in roles:
        prefix = role + "."
        for column in sorted(name for name in columns if name.startswith(prefix)):
            value = _plain_scalar(row[column])
            if value is not None:
                roles[role][column[len(prefix) :]] = value
    return roles


def _merged_params(case: Case, role_values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    params = default_params(case)
    owners: dict[str, str] = {}
    for role in ("design", "scenario", "control"):
        for name, value in role_values[role].items():
            if name in owners:
                raise ValueError(f"input {name!r} belongs to both {owners[name]} and {role}")
            params[name] = value
            owners[name] = role
    return params


def _graph_and_context(case: Case, params: dict[str, Any]) -> tuple[CircuitGraph, dict[str, Any]]:
    circuit_name, circuit = build_circuit(case, params)
    config = circuit_config(case)
    declared_family = config.get("topology_family")
    if declared_family is None:
        family = circuit_name if circuit_name != "from_yaml" else f"{case.case_id}:from_yaml"
    else:
        family = str(resolve_value(declared_family, params)).strip()
    if not family:
        raise ValueError("circuit.topology_family must resolve to non-empty text")
    graph = circuit_to_graph(circuit, family, source_node=_source_node(case, params))
    return graph, _external_context(case, params)


def _source_node(case: Case, params: Mapping[str, Any]) -> str:
    sources = resolve_source_specs(case)
    if not sources:
        return "src"
    measurement = case.data.get("measurement") or {}
    requested = str(measurement.get("current_source", "")) if isinstance(measurement, Mapping) else ""
    for index, source in enumerate(sources):
        if "raw" in source:
            continue
        name = str(source.get("name", "Vsrc" if index == 0 else f"Vsrc{index}"))
        if not requested or name == requested:
            return str(resolve_value(source.get("p", "src"), dict(params)))
    return "src"


def _external_context(case: Case, params: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for index, source in enumerate(resolve_source_specs(case)):
        if "raw" in source:
            raise ValueError("raw source statements do not expose reusable ML context")
        _flatten_context(context, f"source.{index}", source, params)
    _flatten_context(context, "load", load_config(case), params)
    measurement = case.data.get("measurement") or {}
    if not isinstance(measurement, Mapping):
        raise TypeError("measurement must be a mapping")
    _flatten_context(context, "measurement", measurement, params)
    return context


def _flatten_context(
    output: dict[str, Any],
    prefix: str,
    value: Any,
    params: Mapping[str, Any],
) -> None:
    resolved = resolve_value(value, dict(params))
    if isinstance(resolved, Mapping):
        for raw_name in sorted(resolved, key=str):
            name = str(raw_name)
            if name in _CONTEXT_EXCLUDED_KEYS:
                continue
            _flatten_context(output, f"{prefix}.{name}", resolved[raw_name], params)
        return
    if isinstance(resolved, (list, tuple)):
        for index, item in enumerate(resolved):
            _flatten_context(output, f"{prefix}.{index}", item, params)
        return
    scalar = _plain_scalar(resolved)
    if scalar is not None:
        output[f"context.{prefix}"] = scalar


def _evaluation_manifest_path(source: _StudySource, raw: object) -> Path:
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("successful evaluation does not declare artifact.manifest")
    declared = Path(text.replace("\\", "/"))
    path = declared if declared.is_absolute() else source.root / declared
    path = path.resolve()
    if path != source.root and source.root not in path.parents:
        raise ValueError("evaluation manifest must remain inside its committed study root")
    if not path.is_file():
        raise FileNotFoundError(f"declared simulation manifest is missing: {path}")
    return path


def _corpus_manifest(
    *,
    sources: Sequence[_StudySource],
    prepared: PreparedAcCorpus,
    skipped: Mapping[str, int],
    skipped_examples: Mapping[str, list[str]],
    graph_path: Path,
    sample_path: Path,
    component_path: Path,
) -> dict[str, Any]:
    core = prepared.manifest
    return {
        "schema": CORPUS_SCHEMA,
        "created_at": utc_now(),
        "purpose": "physical_ac_response_learning_for_structured_rf_circuits",
        "grain": core["grain"],
        "sources": [
            {
                "dataset_id": source.dataset_id,
                "study_id": source.study_id,
                "root": str(source.root),
                "artifacts": {
                    "study_result": {"path": str(source.result_path), "sha256": file_sha256(source.result_path)},
                    "evaluation_table": {
                        "path": str(source.evaluation_path),
                        "sha256": file_sha256(source.evaluation_path),
                    },
                    "case": {"path": str(source.case_path), "sha256": file_sha256(source.case_path)},
                },
                "evaluations": len(source.evaluations),
            }
            for source in sources
        ],
        "counts": {
            "source_studies": len(sources),
            "source_evaluations": sum(len(source.evaluations) for source in sources),
            **dict(core["counts"]),
            "skipped_evaluations": int(sum(skipped.values())),
            "skipped_by_reason": dict(sorted(skipped.items())),
        },
        "skipped_examples": dict(skipped_examples),
        "roles": core["roles"],
        "splits": core["splits"],
        "semantics": core["semantics"],
        "limitations": core["limitations"],
        "artifacts": {
            "graphs": {"path": "graphs.jsonl", "sha256": file_sha256(graph_path)},
            "samples": {"path": "samples.csv", "sha256": file_sha256(sample_path)},
            "component_responses": {
                "path": "component_responses.csv",
                "sha256": file_sha256(component_path),
            },
            "manifest": "manifest.json",
        },
    }


def _plain_scalar(value: object) -> Any | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
