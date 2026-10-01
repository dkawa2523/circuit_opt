"""Pure preparation of graph-linked AC response data for model comparison."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from pcd.circuit_graph import CircuitGraph
from pcd.probes import component_id

from .dataset import assign_group_holdout

SAMPLE_SCHEMA = "ac_graph_sample.v1"
COMPONENT_RESPONSE_SCHEMA = "ac_component_response.v1"

PORT_TARGETS = {
    "voltage_re": "target.source_voltage_re_V",
    "voltage_im": "target.source_voltage_im_V",
    "current_re": "target.source_current_into_source_re_A",
    "current_im": "target.source_current_into_source_im_A",
    "load_voltage_V_re": "target.load_voltage_re_V",
    "load_voltage_V_im": "target.load_voltage_im_V",
    "load_current_A_re": "target.load_current_re_A",
    "load_current_A_im": "target.load_current_im_A",
}
_REQUIRED_AC_COLUMNS = (
    "frequency_Hz",
    "voltage_re",
    "voltage_im",
    "current_re",
    "current_im",
    "load_voltage_V_re",
    "load_voltage_V_im",
)


@dataclass(frozen=True)
class AcResponseRecord:
    """One saved evaluation plus the physical graph and AC response it produced."""

    graph: CircuitGraph
    response: pd.DataFrame
    response_label: str
    source_dataset_id: str
    source_study_id: str
    trial: Any
    candidate_id: str
    scenario_id: str
    raw_cache_key: str
    role_values: Mapping[str, Mapping[str, Any]]
    external_context: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedAcCorpus:
    """Persistence-free graph records, relational target tables, and roles."""

    graphs: Mapping[str, Mapping[str, Any]]
    samples: pd.DataFrame
    component_responses: pd.DataFrame
    manifest: Mapping[str, Any]


def prepare_ac_corpus(
    records: Sequence[AcResponseRecord],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> PreparedAcCorpus:
    """Build one deterministic corpus without discovering or writing files."""

    if not records:
        raise ValueError("at least one AC response record is required")
    if not math.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be finite and between 0 and 1")

    graphs: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for record in records:
        _validate_ac_response(record.response, record.response_label)
        graph_id = f"graph_{record.graph.instance_fingerprint}"
        graph_record = {"graph_id": graph_id, **record.graph.to_dict()}
        previous = graphs.setdefault(graph_id, graph_record)
        if previous != graph_record:  # pragma: no cover - SHA-256 collision/invariant guard
            raise ValueError(f"graph identity collision: {graph_id}")
        _append_response_rows(record, graph_id, samples, components)

    sample_frame = _sample_frame(samples)
    design_split = _assign_design_split(sample_frame, test_fraction, seed)
    condition_split = _assign_condition_split(sample_frame, test_fraction, seed)
    component_frame = _component_frame(components)
    manifest = _corpus_roles(graphs, sample_frame, component_frame, design_split, condition_split, seed, test_fraction)
    return PreparedAcCorpus(graphs, sample_frame, component_frame, manifest)


def _validate_ac_response(frame: pd.DataFrame, label: str) -> None:
    missing = [name for name in _REQUIRED_AC_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"frequency response is missing required columns {missing}: {label}")
    if frame.empty:
        raise ValueError(f"frequency response is empty: {label}")
    optional_pair = ("load_current_A_re", "load_current_A_im")
    if any(name in frame for name in optional_pair) and not all(name in frame for name in optional_pair):
        raise ValueError(f"frequency response has an incomplete load-current complex pair: {label}")


def _append_response_rows(
    record: AcResponseRecord,
    graph_id: str,
    samples: list[dict[str, Any]],
    components: list[dict[str, Any]],
) -> None:
    design_group = _group_id(
        "design",
        {
            "topology_fingerprint": record.graph.topology_fingerprint,
            "design": record.role_values["design"],
        },
    )
    condition_group = _group_id(
        "condition",
        {"external": record.external_context, "scenario": record.role_values["scenario"]},
    )
    raw_key = record.raw_cache_key.strip()
    if not raw_key:
        raise ValueError("successful evaluation does not declare raw_cache_key")
    physical_group = _group_id("physical", raw_key)
    shared_context = dict(record.external_context)
    for role in ("design", "scenario", "control"):
        shared_context.update({f"context.{role}.{name}": value for name, value in record.role_values[role].items()})

    response_columns = tuple(record.response.columns)
    for point_index, (_row_index, point) in enumerate(record.response.reset_index(drop=True).iterrows()):
        frequency = _finite_value(point["frequency_Hz"], "frequency_Hz")
        sample_id = _group_id(
            "sample",
            [record.source_dataset_id, raw_key, point_index, frequency],
        )
        sample: dict[str, Any] = {
            "sample_schema": SAMPLE_SCHEMA,
            "sample_id": sample_id,
            "graph_id": graph_id,
            "topology_family": record.graph.topology_family,
            "topology_fingerprint": record.graph.topology_fingerprint,
            "source_dataset_id": record.source_dataset_id,
            "source_study_id": record.source_study_id,
            "trial": record.trial,
            "candidate_id": record.candidate_id,
            "scenario_id": record.scenario_id,
            "ac_point_index": point_index,
            "physical_group_id": physical_group,
            "design_group_id": design_group,
            "condition_group_id": condition_group,
            "frequency_Hz": frequency,
            **shared_context,
        }
        for source_column, target_column in PORT_TARGETS.items():
            sample[target_column] = (
                _finite_value(point[source_column], source_column) if source_column in record.response else None
            )
        samples.append(sample)
        components.extend(_component_response_rows(sample_id, graph_id, record.graph, point, response_columns))


def _component_response_rows(
    sample_id: str,
    graph_id: str,
    graph: CircuitGraph,
    point: pd.Series,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    normalized: set[str] = set()
    for component in sorted(graph.components, key=lambda item: item.reference):
        metric_id = component_id(component.reference)
        if metric_id in normalized:
            raise ValueError("component references collide after probe-name normalization")
        normalized.add(metric_id)
        source_columns = (
            f"component_{metric_id}_voltage_V_re",
            f"component_{metric_id}_voltage_V_im",
            f"component_{metric_id}_current_A_re",
            f"component_{metric_id}_current_A_im",
        )
        present = [name in columns for name in source_columns]
        if any(present) and not all(present):
            raise ValueError(f"component {component.reference!r} has an incomplete complex response")
        available = all(present)
        terminals = {terminal.name: terminal.net for terminal in component.terminals}
        targets = [_finite_value(point[name], name) if available else None for name in source_columns]
        output.append(
            {
                "response_schema": COMPONENT_RESPONSE_SCHEMA,
                "sample_id": sample_id,
                "graph_id": graph_id,
                "component_reference": component.reference,
                "component_kind": component.kind,
                "positive_net": terminals.get("p"),
                "negative_net": terminals.get("n"),
                "target_available": available,
                "target.voltage_re_V": targets[0],
                "target.voltage_im_V": targets[1],
                "target.current_re_A": targets[2],
                "target.current_im_A": targets[3],
            }
        )
    return output


def _sample_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    fixed = [
        "sample_schema",
        "sample_id",
        "graph_id",
        "topology_family",
        "topology_fingerprint",
        "source_dataset_id",
        "source_study_id",
        "trial",
        "candidate_id",
        "scenario_id",
        "ac_point_index",
        "physical_group_id",
        "design_group_id",
        "condition_group_id",
        "frequency_Hz",
    ]
    names = list(frame.columns)
    context = sorted(name for name in names if name.startswith("context."))
    targets = [name for name in PORT_TARGETS.values() if name in names]
    return frame.loc[:, [*fixed, *context, *targets]].copy()


def _component_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "response_schema",
        "sample_id",
        "graph_id",
        "component_reference",
        "component_kind",
        "positive_net",
        "negative_net",
        "target_available",
        "target.voltage_re_V",
        "target.voltage_im_V",
        "target.current_re_A",
        "target.current_im_A",
    ]
    return pd.DataFrame(rows, columns=columns)


def _assign_design_split(frame: pd.DataFrame, test_fraction: float, seed: int) -> dict[str, Any]:
    by_topology = frame.groupby("topology_family", sort=True)["design_group_id"].nunique()
    insufficient = sorted(cast(str, name) for name, count in by_topology.items() if count < 2)
    if insufficient:
        return _unavailable_split(
            frame,
            "design_group_id",
            "design_split",
            "at_least_two_design_groups_per_topology_required",
            insufficient,
        )

    split = pd.Series("train", index=frame.index, dtype="object")
    train_groups = 0
    test_groups = 0
    for topology in sorted(frame["topology_family"].unique()):
        selected = frame["topology_family"].eq(topology)
        assigned, counts = assign_group_holdout(frame.loc[selected, "design_group_id"], test_fraction, seed)
        split.loc[selected] = assigned
        train_groups += counts["train"]
        test_groups += counts["test"]
    _insert_split(frame, "design_split", split)
    return _ready_split(split, train_groups, test_groups)


def _assign_condition_split(frame: pd.DataFrame, test_fraction: float, seed: int) -> dict[str, Any]:
    groups = _integer(frame["condition_group_id"].nunique())
    if groups < 2:
        return _unavailable_split(
            frame,
            "condition_group_id",
            "condition_split",
            "at_least_two_external_condition_groups_required",
        )
    split, counts = assign_group_holdout(frame["condition_group_id"], test_fraction, seed)
    missing_coverage = _topologies_missing_from_a_split(frame, split)
    if missing_coverage:
        return _unavailable_split(
            frame,
            "condition_group_id",
            "condition_split",
            "condition_split_is_confounded_with_topology_family",
            missing_coverage,
        )
    _insert_split(frame, "condition_split", split)
    return _ready_split(split, counts["train"], counts["test"])


def _unavailable_split(
    frame: pd.DataFrame,
    group_column: str,
    split_column: str,
    reason: str,
    affected_topologies: Sequence[str] = (),
) -> dict[str, Any]:
    groups = _integer(frame[group_column].nunique())
    _insert_split(frame, split_column, pd.Series("train", index=frame.index, dtype="object"))
    return {
        "status": "unavailable",
        "reason": reason,
        "affected_topologies": list(affected_topologies),
        "groups": {"total": groups, "train": groups, "test": 0},
        "rows": {"total": len(frame), "train": len(frame), "test": 0},
    }


def _topologies_missing_from_a_split(frame: pd.DataFrame, split: pd.Series) -> list[str]:
    missing: list[str] = []
    for topology in sorted(frame["topology_family"].unique()):
        selected = frame["topology_family"].eq(topology)
        if set(split.loc[selected]) != {"train", "test"}:
            missing.append(topology)
    return missing


def _insert_split(frame: pd.DataFrame, column: str, split: pd.Series) -> None:
    location = list(frame.columns).index("frequency_Hz")
    frame.insert(location, column, split)


def _ready_split(split: pd.Series, train_groups: int, test_groups: int) -> dict[str, Any]:
    return {
        "status": "ready",
        "groups": {"total": train_groups + test_groups, "train": train_groups, "test": test_groups},
        "rows": {
            "total": len(split),
            "train": _integer(split.eq("train").sum()),
            "test": _integer(split.eq("test").sum()),
        },
    }


def _corpus_roles(
    graphs: Mapping[str, Mapping[str, Any]],
    samples: pd.DataFrame,
    components: pd.DataFrame,
    design_split: Mapping[str, Any],
    condition_split: Mapping[str, Any],
    seed: int,
    test_fraction: float,
) -> dict[str, Any]:
    sample_columns = list(samples.columns)
    context_columns = sorted(
        name for name in sample_columns if name.startswith("context.") and not name.startswith("context.design.")
    )
    design_columns = sorted(name for name in sample_columns if name.startswith("context.design."))
    target_columns = [name for name in sample_columns if name.startswith("target.")]
    topology_families = sorted(samples["topology_family"].unique())
    available_component_rows = _integer(components["target_available"].eq(True).sum())
    load_current_rows = (
        samples["target.load_current_re_A"].notna() & samples["target.load_current_im_A"].notna()
    ).sum()
    load_current_rows = _integer(load_current_rows)
    return {
        "grain": {
            "samples": "successful_evaluation_x_ac_frequency_point",
            "component_responses": "sample_x_logical_circuit_component",
            "graphs": "unique_resolved_physical_circuit_instance",
        },
        "counts": {
            "graphs": len(graphs),
            "samples": len(samples),
            "component_response_rows": len(components),
            "component_target_rows": available_component_rows,
            "load_current_target_rows": load_current_rows,
        },
        "roles": {
            "graph_input": "graphs.jsonl:circuit_graph.v1",
            "context_features": ["frequency_Hz", *context_columns],
            "design_trace": design_columns,
            "port_complex_targets": target_columns,
            "component_complex_targets": [
                "target.voltage_re_V",
                "target.voltage_im_V",
                "target.current_re_A",
                "target.current_im_A",
            ],
            "trace": [
                "sample_id",
                "graph_id",
                "source_dataset_id",
                "source_study_id",
                "trial",
                "candidate_id",
                "scenario_id",
                "ac_point_index",
                "physical_group_id",
            ],
            "grouping": ["design_group_id", "condition_group_id", "topology_family"],
        },
        "splits": {
            "seed": seed,
            "requested_test_fraction": test_fraction,
            "unseen_design_within_known_topologies": {
                "strategy": "deterministic_group_holdout",
                "column": "design_split",
                **dict(design_split),
            },
            "unseen_external_conditions": {
                "strategy": "deterministic_group_holdout",
                "column": "condition_split",
                **dict(condition_split),
            },
            "unseen_topology_family": {
                "strategy": "leave_one_topology_family_out",
                "group_column": "topology_family",
                "status": "ready" if len(topology_families) >= 2 else "unavailable",
                "groups": topology_families,
            },
        },
        "semantics": {
            "source_current": "ngspice current through the source from its positive to negative terminal",
            "component_current": "current through the observation meter from component p to n",
            "component_voltage": "complex voltage V(p)-V(n) across the logical component including declared series loss",
            "topology": "logical components only; zero-volt meters and solver-internal loss nodes are excluded",
        },
        "limitations": [
            "Only saved AC frequency responses are exported; transient waveform learning remains a later milestone.",
            "Raw SPICE and time-profile circuit components are skipped until they declare graph semantics.",
            "Component targets are blank unless the source case observed that logical component.",
            "A topology fingerprint is deterministic for authored wiring but is not a general graph-isomorphism proof.",
            "This corpus does not train a model or replace ngspice validation.",
        ],
    }


def _group_id(prefix: str, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer(value: Any) -> int:
    return int(value)


def _finite_value(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AC response column {name!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"AC response column {name!r} must be finite")
    return number
