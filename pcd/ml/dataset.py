"""Prepare one committed study table for within-study surrogate evaluation.

This module is deliberately model- and persistence-free.  It receives the
committed study metadata and its evaluation table, validates their shared
identity and grain, assigns every column one role, and creates a deterministic
holdout split by fixed-design values.  Loading files and writing exports remain
application responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_DATASET_IDENTITY_COLUMNS = (
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
    "scenario_weight",
    "selected_control",
    "status",
    "feasible",
    "total_violation",
)
_TRACE_COLUMNS = ("dataset_id", "study_id", "trial", "candidate_id", "scenario_id")
_OUTCOME_COLUMNS = ("status", "solver_ok", "feasible", "total_violation")


@dataclass(frozen=True)
class PreparedDataset:
    """A role-limited table and the manifest needed to interpret it."""

    frame: pd.DataFrame
    manifest: dict[str, Any]


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"evaluation table is missing required columns: {', '.join(missing)}")


def _single_identity(frame: pd.DataFrame, column: str, expected: object) -> str:
    if frame[column].isna().any():
        raise ValueError(f"evaluation table identity column {column!r} contains missing values")
    values = {str(value) for value in frame[column].unique()}
    if len(values) != 1:
        raise ValueError(f"evaluation table identity column {column!r} must have one value")
    actual = next(iter(values))
    if actual != str(expected):
        raise ValueError(f"evaluation table {column!r} does not match committed study metadata")
    return actual


def _canonical_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _design_signature(row: pd.Series, columns: list[str]) -> str:
    values = [[name, _canonical_scalar(row[name])] for name in columns]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=False, default=str)


def _design_group_id(signature: str) -> str:
    return "design_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _objective_names(study_result: Mapping[str, Any]) -> list[str]:
    study = _as_mapping(study_result.get("study"), "study_result.study")
    declared = study.get("objectives")
    if not isinstance(declared, list) or not declared:
        raise ValueError("committed study must declare at least one objective")
    names: list[str] = []
    for index, value in enumerate(declared):
        objective = _as_mapping(value, f"study_result.study.objectives[{index}]")
        name = str(objective.get("metric", "")).strip()
        if not name:
            raise ValueError(f"study objective {index} does not declare a metric")
        if name in names:
            raise ValueError(f"study objective metric {name!r} is duplicated")
        names.append(name)
    return names


def _validate_counts(frame: pd.DataFrame, study_result: Mapping[str, Any]) -> None:
    try:
        expected_rows = int(study_result["n_evaluations"])
        expected_candidates = int(study_result["n_candidates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("committed study is missing valid candidate/evaluation counts") from exc
    if len(frame) != expected_rows:
        raise ValueError(f"evaluation row count {len(frame)} does not match committed count {expected_rows}")
    candidates = frame["candidate_id"].nunique(dropna=False)
    if candidates != expected_candidates:
        raise ValueError(f"candidate count {candidates} does not match committed count {expected_candidates}")


def _validate_row_grain(frame: pd.DataFrame, control_columns: list[str]) -> list[str]:
    grain = ["dataset_id", "candidate_id", "scenario_id", *control_columns]
    duplicated = frame.duplicated(grain, keep=False)
    if bool(duplicated.any()):
        raise ValueError("evaluation table contains duplicate Candidate x Scenario x Control rows")

    candidate_trials = frame.groupby("candidate_id", dropna=False)["trial"].nunique(dropna=False)
    trial_candidates = frame.groupby("trial", dropna=False)["candidate_id"].nunique(dropna=False)
    if bool((candidate_trials != 1).any()) or bool((trial_candidates != 1).any()):
        raise ValueError("trial and candidate_id must identify each other one-to-one")
    return grain


def _numeric_objectives(
    frame: pd.DataFrame,
    objective_targets: list[str],
    solver_ok: pd.Series,
) -> pd.Series:
    eligible = solver_ok.copy()
    for column in objective_targets:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if bool(invalid.any()):
            raise ValueError(f"objective column {column!r} contains non-numeric values")
        finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=frame.index)
        if bool((solver_ok & ~finite).any()):
            raise ValueError(f"successful evaluations require a finite value in {column!r}")
        frame[column] = numeric
        eligible &= finite
    return eligible


def assign_group_holdout(group_ids: pd.Series, test_fraction: float, seed: int) -> tuple[pd.Series, dict[str, int]]:
    """Assign whole groups to a deterministic train/test holdout."""

    if not math.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be finite and between 0 and 1")
    groups = sorted(str(value) for value in group_ids.unique())
    if len(groups) < 2:
        raise ValueError("surrogate holdout needs at least two distinct fixed-design groups")
    n_test = min(len(groups) - 1, max(1, math.ceil(len(groups) * test_fraction)))
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )
    test_groups = set(ranked[:n_test])
    split = group_ids.map(lambda value: "test" if str(value) in test_groups else "train")
    return split, {"total": len(groups), "train": len(groups) - n_test, "test": n_test}


def _column_roles(frame: pd.DataFrame, objective_targets: list[str]) -> dict[str, list[str]]:
    def populated(prefix: str) -> list[str]:
        return sorted(name for name in frame.columns if name.startswith(prefix) and frame[name].notna().any())

    design = populated("design.")
    scenario = populated("scenario.")
    control = populated("control.")
    metrics = populated("metric.")
    constraints = populated("constraint.")
    objective_set = set(objective_targets)
    return {
        "design_features": design,
        "scenario_features": scenario,
        "control_features": control,
        "features": [*design, *scenario, *control],
        "objective_targets": objective_targets,
        "auxiliary_response_targets": [name for name in metrics if name not in objective_set],
        "constraint_classification_targets": [name for name in constraints if name.endswith(".satisfied")],
        "constraint_response_targets": [
            name for name in constraints if name.endswith((".value", ".violation", ".margin"))
        ],
        "constraint_context": [name for name in constraints if name.endswith(".limit")],
    }


def _validated_structure(
    frame: pd.DataFrame,
    study_result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[str], dict[str, list[str]], list[str]]:
    if study_result.get("schema") != "study_result.v1":
        raise ValueError("study_result schema must be 'study_result.v1'")
    if frame.empty:
        raise ValueError("evaluation table is empty")

    dataset = _as_mapping(study_result.get("dataset"), "study_result.dataset")
    _require_columns(frame, [*_DATASET_IDENTITY_COLUMNS, *_ROW_COLUMNS])
    for column in _DATASET_IDENTITY_COLUMNS:
        if column not in dataset:
            raise ValueError(f"committed dataset identity is missing {column!r}")
        _single_identity(frame, column, dataset[column])
    if dataset["table_schema"] != "evaluation_table.v2":
        raise ValueError("evaluation table schema must be 'evaluation_table.v2'")
    if frame[["trial", "candidate_id", "scenario_id", "status"]].isna().any().any():
        raise ValueError("evaluation row identifiers and status cannot be missing")
    _validate_counts(frame, study_result)

    objective_targets = [f"metric.{name}" for name in _objective_names(study_result)]
    _require_columns(frame, objective_targets)
    roles = _column_roles(frame, objective_targets)
    design_columns = roles["design_features"]
    if not design_columns:
        raise ValueError("within-study surrogate data requires at least one design.* feature")
    if frame[design_columns].isna().any().any():
        raise ValueError("fixed-design features cannot be missing")
    grain = _validate_row_grain(frame, roles["control_features"])
    return dataset, objective_targets, roles, grain


def _assign_design_groups(frame: pd.DataFrame, design_columns: list[str]) -> None:
    signatures = frame.apply(lambda row: _design_signature(row, design_columns), axis=1)
    candidate_signature_count = signatures.groupby(frame["candidate_id"], dropna=False).nunique(dropna=False)
    if bool((candidate_signature_count != 1).any()):
        raise ValueError("one candidate_id maps to multiple fixed-design values")
    frame.insert(len(_TRACE_COLUMNS), "design_group_id", signatures.map(_design_group_id))


def _prepared_columns(roles: Mapping[str, list[str]]) -> list[str]:
    metric_columns = [*roles["objective_targets"], *roles["auxiliary_response_targets"]]
    constraint_columns = [
        *roles["constraint_classification_targets"],
        *roles["constraint_response_targets"],
        *roles["constraint_context"],
    ]
    columns = [
        *_TRACE_COLUMNS,
        "design_group_id",
        "split",
        *roles["features"],
        "scenario_weight",
        *_OUTCOME_COLUMNS,
        "objective_regression_eligible",
        *metric_columns,
        *constraint_columns,
    ]
    return list(dict.fromkeys(columns))


def _excluded_roles(evaluations: pd.DataFrame, included_columns: list[str]) -> dict[str, list[str]]:
    included = {name for name in included_columns if name in evaluations.columns}
    buckets: dict[str, list[str]] = {
        "post_evaluation_selection_outcome": [],
        "execution_and_artifact_metadata": [],
        "constant_source_identity": [],
        "all_missing_source_columns": [],
        "other_unassigned_source_columns": [],
    }
    for name in sorted(name for name in evaluations.columns if name not in included):
        if name == "selected_control":
            reason = "post_evaluation_selection_outcome"
        elif name in {"duration_s", "from_cache", "raw_cache_key", "error"} or name.startswith("artifact."):
            reason = "execution_and_artifact_metadata"
        elif name in _DATASET_IDENTITY_COLUMNS:
            reason = "constant_source_identity"
        elif evaluations[name].isna().all():
            reason = "all_missing_source_columns"
        else:
            reason = "other_unassigned_source_columns"
        buckets[reason].append(name)
    return buckets


def _boolean_counts(values: pd.Series) -> dict[str, int]:
    missing = values.isna()
    true_values = values.eq(True) & ~missing
    false_values = values.eq(False) & ~missing
    return {
        "true": len(values.loc[true_values]),
        "false": len(values.loc[false_values]),
        "missing": len(values.loc[missing]),
        "other": len(values) - len(values.loc[true_values | false_values | missing]),
    }


def _classification_distribution(
    prepared: pd.DataFrame,
    constraint_labels: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    columns = ["solver_ok", "feasible", *constraint_labels]
    return {
        name: {
            "all": _boolean_counts(prepared[name]),
            "train": _boolean_counts(prepared.loc[prepared["split"].eq("train"), name]),
            "test": _boolean_counts(prepared.loc[prepared["split"].eq("test"), name]),
        }
        for name in columns
    }


def _source_evaluation_cost(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize solver cost without exposing it as a model feature."""

    if "duration_s" not in frame:
        return {"status": "unavailable", "reason": "duration_s_not_recorded"}
    if "from_cache" not in frame:
        return {"status": "unavailable", "reason": "cache_status_not_recorded"}
    cache_status = frame["from_cache"].astype(str).str.strip().str.lower()
    valid_cache = cache_status.isin({"true", "false"})
    if not bool(valid_cache.all()):
        return {"status": "unavailable", "reason": "from_cache_contains_invalid_values"}
    uncached = cache_status.eq("false")
    if not bool(uncached.any()):
        return {
            "status": "unavailable",
            "reason": "no_uncached_evaluations",
            "rows": len(frame),
            "cached_rows": len(frame),
        }
    duration = pd.to_numeric(frame.loc[uncached, "duration_s"], errors="coerce")
    valid = duration.notna() & np.isfinite(duration) & duration.ge(0.0)
    if not bool(valid.all()):
        return {
            "status": "unavailable",
            "reason": "uncached_duration_s_contains_missing_or_invalid_values",
            "valid_rows": len(duration.loc[valid]),
            "uncached_rows": len(duration),
        }
    values = duration.to_numpy(dtype=float)
    return {
        "status": "available",
        "basis": "uncached_evaluations",
        "rows": len(frame),
        "uncached_rows": len(values),
        "cached_rows": len(frame) - len(values),
        "total": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _feature_transforms(
    roles: Mapping[str, list[str]],
    declared: Mapping[str, str] | None,
) -> dict[str, str]:
    features = roles["features"]
    supplied = dict(declared or {})
    unknown = sorted(set(supplied) - set(features))
    if unknown:
        raise ValueError(f"feature transforms name unknown features: {', '.join(unknown)}")
    invalid = sorted(name for name, kind in supplied.items() if kind not in {"linear", "log10"})
    if invalid:
        raise ValueError(f"feature transforms must be 'linear' or 'log10': {', '.join(invalid)}")
    return {name: supplied.get(name, "linear") for name in features}


def _dataset_manifest(
    *,
    dataset: Mapping[str, Any],
    source_frame: pd.DataFrame,
    prepared: pd.DataFrame,
    roles: Mapping[str, list[str]],
    grain: list[str],
    group_counts: Mapping[str, int],
    test_fraction: float,
    seed: int,
    solver_ok: pd.Series,
    objective_eligible: pd.Series,
    excluded: Mapping[str, list[str]],
    feature_transforms: Mapping[str, str],
) -> dict[str, Any]:
    train_rows = len(prepared.loc[prepared["split"].eq("train")])
    test_rows = len(prepared.loc[prepared["split"].eq("test")])
    return {
        "schema": "ml_evaluation_dataset.v1",
        "purpose": "within_study_fixed_design_response_surrogate",
        "source": {
            **{name: str(dataset[name]) for name in _DATASET_IDENTITY_COLUMNS},
            "rows": len(prepared),
            "candidates": len(source_frame["candidate_id"].unique()),
        },
        "grain": {
            "unit": "candidate_x_scenario_x_control_evaluation",
            "source_columns": grain,
        },
        "roles": {
            **roles,
            "trace": list(_TRACE_COLUMNS),
            "aggregation_context": ["scenario_weight"],
            "outcome_targets": list(_OUTCOME_COLUMNS),
            "eligibility": ["objective_regression_eligible"],
            "split": ["design_group_id", "split"],
            "excluded": dict(excluded),
        },
        "split": {
            "strategy": "deterministic_group_holdout_by_fixed_design_values",
            "claim": "unseen_fixed_designs_within_the_same_committed_study",
            "group_source_columns": roles["design_features"],
            "seed": seed,
            "requested_test_fraction": test_fraction,
            "groups": dict(group_counts),
            "rows": {"total": len(prepared), "train": train_rows, "test": test_rows},
        },
        "eligibility": {
            "solver_ok_rows": len(solver_ok.loc[solver_ok]),
            "solver_failed_rows": len(solver_ok.loc[~solver_ok]),
            "objective_regression_rows": len(objective_eligible.loc[objective_eligible]),
            "objective_regression_excluded_rows": len(objective_eligible.loc[~objective_eligible]),
            "classification_label_distribution": _classification_distribution(
                prepared,
                roles["constraint_classification_targets"],
            ),
        },
        "feature_transforms": dict(feature_transforms),
        "feature_missing_values": {name: len(prepared.loc[prepared[name].isna()]) for name in roles["features"]},
        "source_evaluation_cost_s": _source_evaluation_cost(source_frame),
        "limitations": [
            "The split measures unseen fixed-design performance only within this committed case.",
            "It does not establish generalization to another circuit case or independent measurement series.",
            "selected_control is excluded because it is known only after evaluating all control states.",
            "Any surrogate-proposed design still requires full Scenario x Control ngspice evaluation.",
        ],
    }


def prepare_evaluation_dataset(
    evaluations: pd.DataFrame,
    study_result: Mapping[str, Any],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
    feature_transforms: Mapping[str, str] | None = None,
) -> PreparedDataset:
    """Validate and split one committed evaluation table.

    The holdout claim is intentionally narrow: unseen fixed-design values in
    the same committed case.  All scenario/control rows for identical design
    values stay together.  This function does not claim generalization to a
    different circuit case or an independent measurement series.
    """

    if not isinstance(study_result, Mapping):
        raise ValueError("study_result must be a mapping")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    frame = evaluations.copy()
    dataset, objective_targets, roles, grain = _validated_structure(frame, study_result)
    design_columns = roles["design_features"]
    transforms = _feature_transforms(roles, feature_transforms)
    _assign_design_groups(frame, design_columns)

    solver_ok = frame["status"].astype(str).eq("ok")
    objective_eligible = _numeric_objectives(frame, objective_targets, solver_ok)
    split, group_counts = assign_group_holdout(frame["design_group_id"], test_fraction, seed)
    frame.insert(len(_TRACE_COLUMNS) + 1, "split", split)
    frame["solver_ok"] = solver_ok
    frame["objective_regression_eligible"] = objective_eligible

    output_columns = _prepared_columns(roles)
    prepared = frame.loc[:, output_columns].copy()
    manifest = _dataset_manifest(
        dataset=dataset,
        source_frame=evaluations,
        prepared=prepared,
        roles=roles,
        grain=grain,
        group_counts=group_counts,
        test_fraction=test_fraction,
        seed=seed,
        solver_ok=solver_ok,
        objective_eligible=objective_eligible,
        excluded=_excluded_roles(evaluations, output_columns),
        feature_transforms=transforms,
    )
    return PreparedDataset(prepared.reset_index(drop=True), manifest)
