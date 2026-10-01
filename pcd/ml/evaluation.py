"""Evaluate one small, transparent surrogate on a prepared study holdout.

The comparison is intentionally narrow.  A constant training-set baseline and
distance-weighted three-neighbour model see the same declared numeric features
and the same fixed-design holdout.  This module neither proposes candidates nor
executes a circuit solver.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pcd.ml.neighbors import fit_feature_space, neighbor_predictions

_NEIGHBORS = 3
_TRACE_COLUMNS = ("dataset_id", "study_id", "trial", "candidate_id", "scenario_id", "design_group_id")


@dataclass(frozen=True)
class SurrogateEvaluation:
    """Holdout predictions and their compact evidence summary."""

    predictions: pd.DataFrame
    summary: dict[str, Any]
    constraint_validation_predictions: pd.DataFrame | None = None


@dataclass(frozen=True)
class _EvaluationPlan:
    features: list[str]
    objectives: list[str]
    classification_targets: list[str]
    engineering_targets: list[str]


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _role_names(roles: Mapping[str, Any], name: str) -> list[str]:
    value = roles.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"manifest.roles.{name} must be a list of column names")
    return list(value)


def _require_columns(frame: pd.DataFrame, names: list[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"prepared dataset is missing required columns: {', '.join(missing)}")


def _boolean_values(values: pd.Series, name: str) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for position, value in enumerate(values):
        if pd.isna(value):
            continue
        boolean = isinstance(value, (bool, np.bool_))
        binary_number = isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {0.0, 1.0}
        if boolean or binary_number:
            result[position] = float(value)
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            result[position] = float(value.strip().lower() == "true")
        else:
            raise ValueError(f"classification target {name!r} contains a non-boolean value")
    return result


def _split_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split = frame["split"].astype(str)
    unexpected = sorted(set(split) - {"train", "test"})
    if unexpected:
        raise ValueError(f"prepared dataset contains unknown split labels: {', '.join(unexpected)}")
    train = split.eq("train").to_numpy()
    test = split.eq("test").to_numpy()
    if not train.any() or not test.any():
        raise ValueError("prepared dataset must contain both train and test rows")
    grouped = frame.groupby("design_group_id", dropna=False)["split"].nunique(dropna=False)
    if bool((grouped != 1).any()):
        raise ValueError("one fixed-design group crosses the train/test boundary")
    return train, test


def _declared_transforms(manifest: Mapping[str, Any], features: list[str]) -> dict[str, str]:
    raw = _as_mapping(manifest.get("feature_transforms"), "manifest.feature_transforms")
    if set(raw) != set(features):
        raise ValueError("manifest.feature_transforms must name every and only declared feature")
    transforms = {name: str(raw[name]) for name in features}
    invalid = sorted(name for name, kind in transforms.items() if kind not in {"linear", "log10"})
    if invalid:
        raise ValueError(f"unsupported feature transforms: {', '.join(invalid)}")
    return transforms


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residual = predicted - actual
    squared = residual**2
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rows": len(actual),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(squared))),
        "r2": None if denominator == 0.0 else 1.0 - float(squared.sum()) / denominator,
    }


def _regression_target(
    frame: pd.DataFrame,
    features: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    eligible: np.ndarray,
    name: str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    train_rows = train & eligible
    test_rows = test & eligible
    if not train_rows.any() or not test_rows.any():
        return {
            "status": "insufficient_eligible_rows",
            "train_rows": int(train_rows.sum()),
            "test_rows": int(test_rows.sum()),
        }
    if not np.isfinite(numeric[train_rows | test_rows]).all():
        raise ValueError(f"eligible objective target {name!r} must be finite and numeric")
    actual = numeric[test_rows]
    baseline = np.full(len(actual), numeric[train_rows].mean())
    surrogate, neighbors = neighbor_predictions(
        features,
        train_rows,
        test_rows,
        numeric[train_rows],
        neighbors=_NEIGHBORS,
    )
    baseline_metrics = _regression_metrics(actual, baseline)
    surrogate_metrics = _regression_metrics(actual, surrogate)
    baseline_rmse = float(baseline_metrics["rmse"])
    surrogate_rmse = float(surrogate_metrics["rmse"])
    improvement = None if baseline_rmse == 0.0 else (baseline_rmse - surrogate_rmse) / baseline_rmse
    test_index = frame.index[test_rows]
    predictions.loc[test_index, f"actual.{name}"] = actual
    predictions.loc[test_index, f"baseline.{name}"] = baseline
    predictions.loc[test_index, f"surrogate.{name}"] = surrogate
    return {
        "status": "evaluated",
        "train_rows": int(train_rows.sum()),
        "test_rows": int(test_rows.sum()),
        "neighbors_used": neighbors,
        "baseline": baseline_metrics,
        "surrogate": surrogate_metrics,
        "relative_rmse_improvement": improvement,
        "surrogate_beats_baseline": surrogate_rmse < baseline_rmse,
    }


def _class_counts(values: np.ndarray) -> dict[str, int]:
    return {"false": int((values == 0.0).sum()), "true": int((values == 1.0).sum())}


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    true_positive = int((actual_bool & predicted_bool).sum())
    true_negative = int((~actual_bool & ~predicted_bool).sum())
    false_positive = int((~actual_bool & predicted_bool).sum())
    false_negative = int((actual_bool & ~predicted_bool).sum())
    metrics: dict[str, Any] = {
        "rows": len(actual),
        "accuracy": float((actual_bool == predicted_bool).mean()),
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "balanced_accuracy": None,
    }
    if len(np.unique(actual_bool)) == 2:
        true_recall = true_positive / (true_positive + false_negative)
        false_recall = true_negative / (true_negative + false_positive)
        metrics["balanced_accuracy"] = (true_recall + false_recall) / 2.0
    return metrics


def _classification_status(train_values: np.ndarray, test_values: np.ndarray) -> str:
    if not len(train_values) or not len(test_values):
        return "insufficient_labeled_rows"
    if len(np.unique(train_values)) < 2:
        return "insufficient_training_class_diversity"
    if len(np.unique(test_values)) < 2:
        return "insufficient_test_class_diversity"
    return "evaluable"


def _classification_target(
    frame: pd.DataFrame,
    features: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    name: str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    values = _boolean_values(frame[name], name)
    labeled = np.isfinite(values)
    train_rows = train & labeled
    test_rows = test & labeled
    train_values = values[train_rows]
    test_values = values[test_rows]
    status = _classification_status(train_values, test_values)
    result: dict[str, Any] = {
        "status": status,
        "train_labels": _class_counts(train_values),
        "test_labels": _class_counts(test_values),
    }
    if not len(train_values) or not len(test_values):
        return result
    positive_rate = float(train_values.mean())
    baseline = np.full(len(test_values), positive_rate >= 0.5)
    probability, neighbors = neighbor_predictions(
        features,
        train_rows,
        test_rows,
        train_values,
        neighbors=_NEIGHBORS,
    )
    surrogate = probability >= 0.5
    baseline_metrics = _classification_metrics(test_values, baseline)
    surrogate_metrics = _classification_metrics(test_values, surrogate)
    result.update(
        {
            "neighbors_used": neighbors,
            "training_positive_rate": positive_rate,
            "baseline": baseline_metrics,
            "surrogate": surrogate_metrics,
            "surrogate_beats_baseline": (
                status == "evaluable"
                and float(surrogate_metrics["balanced_accuracy"]) > float(baseline_metrics["balanced_accuracy"])
            ),
        }
    )
    test_index = frame.index[test_rows]
    predictions.loc[test_index, f"actual.{name}"] = test_values.astype(bool)
    predictions.loc[test_index, f"baseline.{name}"] = baseline
    predictions.loc[test_index, f"surrogate.{name}"] = surrogate
    predictions.loc[test_index, f"surrogate_probability.{name}"] = probability
    return result


def _runtime_comparison(manifest: Mapping[str, Any], elapsed: float) -> dict[str, Any]:
    source = manifest.get("source_evaluation_cost_s")
    if not isinstance(source, Mapping) or source.get("status") != "available":
        return {"source_median_evaluation_s": None, "model_run_below_source_median": None}
    median = float(source["median"])
    return {
        "source_median_evaluation_s": median,
        "model_run_below_source_median": elapsed < median,
        "model_to_source_median_ratio": None if median == 0.0 else elapsed / median,
    }


def _constraint_classification_evidence(
    classifications: Mapping[str, Mapping[str, Any]] | None,
    engineering_targets: list[str],
) -> tuple[bool, bool]:
    if classifications is None or not engineering_targets:
        return False, False
    evaluable = all(classifications[name].get("status") == "evaluable" for name in engineering_targets)
    beats_baseline = evaluable and all(
        classifications[name].get("surrogate_beats_baseline") is True for name in engineering_targets
    )
    return evaluable, beats_baseline


def _evidence_recommendation(
    *,
    objectives_evaluable: bool,
    objectives_beat_baseline: bool,
    constraint_evaluable: bool,
    constraint_source: str | None,
    runtime_supported: bool,
) -> str:
    if not objectives_evaluable:
        return "collect_eligible_objective_holdout_rows"
    if not objectives_beat_baseline:
        return "do_not_advance_surrogate_model"
    if constraint_source is None and constraint_evaluable:
        return "do_not_advance_constraint_classifier"
    if constraint_source is None:
        return "collect_preregistered_constraint_boundary_evidence"
    if not runtime_supported:
        return "measure_or_reduce_model_overhead"
    return "run_retrospective_candidate_ranking_evaluation"


def _evidence_gate(
    regressions: Mapping[str, Mapping[str, Any]],
    classifications: Mapping[str, Mapping[str, Any]],
    engineering_targets: list[str],
    runtime: Mapping[str, Any],
    independent_classifications: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    objective_evaluated = all(result.get("status") == "evaluated" for result in regressions.values())
    objective_better = objective_evaluated and all(
        result.get("surrogate_beats_baseline") is True for result in regressions.values()
    )
    classification_evaluable = all(result.get("status") == "evaluable" for result in classifications.values())
    holdout_engineering_evaluable, holdout_engineering_better = _constraint_classification_evidence(
        classifications,
        engineering_targets,
    )
    independent_engineering_evaluable, independent_engineering_better = _constraint_classification_evidence(
        independent_classifications,
        engineering_targets,
    )
    if holdout_engineering_better:
        engineering_evidence = "fixed_holdout"
    elif independent_engineering_better:
        engineering_evidence = "independent_constraint_validation"
    else:
        engineering_evidence = None
    recommendation = _evidence_recommendation(
        objectives_evaluable=objective_evaluated,
        objectives_beat_baseline=objective_better,
        constraint_evaluable=holdout_engineering_evaluable or independent_engineering_evaluable,
        constraint_source=engineering_evidence,
        runtime_supported=runtime.get("model_run_below_source_median") is True,
    )
    return {
        "all_objectives_evaluated": objective_evaluated,
        "all_objectives_beat_constant_baseline": objective_better,
        "all_classification_targets_evaluable": classification_evaluable,
        "engineering_constraint_targets": engineering_targets,
        "all_engineering_constraint_targets_evaluable": holdout_engineering_evaluable,
        "all_engineering_constraint_targets_beat_majority_baseline": holdout_engineering_better,
        "independent_engineering_constraint_targets_evaluable": independent_engineering_evaluable,
        "independent_engineering_constraint_targets_beat_majority_baseline": independent_engineering_better,
        "engineering_constraint_evidence_source": engineering_evidence,
        "model_run_below_one_median_source_evaluation": runtime.get("model_run_below_source_median"),
        "ngspice_evaluation_savings": "not_measured",
        "bayesian_optimization_ready": False,
        "recommendation": recommendation,
    }


def _evaluation_plan(manifest: Mapping[str, Any]) -> _EvaluationPlan:
    roles = _as_mapping(manifest.get("roles"), "manifest.roles")
    features = _role_names(roles, "features")
    objectives = _role_names(roles, "objective_targets")
    if not features or not objectives:
        raise ValueError("manifest must declare at least one feature and objective target")
    constraint_targets = _role_names(roles, "constraint_classification_targets")
    classification_targets = list(dict.fromkeys(["solver_ok", "feasible", *constraint_targets]))
    engineering_constraints = [name for name in constraint_targets if name != "constraint.evaluation_success.satisfied"]
    engineering_targets = ["feasible", *engineering_constraints] if engineering_constraints else []
    return _EvaluationPlan(features, objectives, classification_targets, engineering_targets)


def _validated_evaluation_input(
    dataset: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, Mapping[str, Any], _EvaluationPlan]:
    if manifest.get("schema") != "ml_evaluation_dataset.v1":
        raise ValueError("manifest schema must be 'ml_evaluation_dataset.v1'")
    if dataset.empty:
        raise ValueError("prepared dataset is empty")
    plan = _evaluation_plan(manifest)
    frame = dataset.copy().reset_index(drop=True)
    required = [
        *_TRACE_COLUMNS,
        "split",
        "objective_regression_eligible",
        *plan.features,
        *plan.objectives,
        *plan.classification_targets,
    ]
    _require_columns(frame, required)
    source = _as_mapping(manifest.get("source"), "manifest.source")
    if int(source.get("rows", -1)) != len(frame):
        raise ValueError("prepared dataset row count does not match its manifest")
    return frame, source, plan


def _constraint_validation(
    source_frame: pd.DataFrame,
    source: Mapping[str, Any],
    source_plan: _EvaluationPlan,
    source_manifest: Mapping[str, Any],
    validation_dataset: pd.DataFrame,
    validation_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation_frame, validation_source, validation_plan = _validated_evaluation_input(
        validation_dataset,
        validation_manifest,
    )
    if validation_plan != source_plan:
        raise ValueError("constraint validation must declare the same feature and target roles as the source dataset")
    source_transforms = _declared_transforms(source_manifest, source_plan.features)
    validation_transforms = _declared_transforms(validation_manifest, validation_plan.features)
    if validation_transforms != source_transforms:
        raise ValueError("constraint validation must use the same declared feature transforms as the source dataset")
    if validation_source.get("dataset_id") == source.get("dataset_id"):
        raise ValueError("constraint validation must come from a different committed dataset")
    overlap = set(source_frame["design_group_id"].astype(str)) & set(validation_frame["design_group_id"].astype(str))
    if overlap:
        raise ValueError("constraint validation contains fixed-design values used by the source dataset")

    source_train, _source_test = _split_masks(source_frame)
    combined = pd.concat([source_frame, validation_frame], ignore_index=True)
    validation_rows = np.arange(len(combined)) >= len(source_frame)
    training_rows = np.concatenate([source_train, np.zeros(len(validation_frame), dtype=bool)])
    features = fit_feature_space(combined, source_plan.features, source_transforms, training_rows)
    output_columns = [*_TRACE_COLUMNS, *source_plan.features]
    predictions = combined.loc[validation_rows, output_columns].copy()
    classifications = {
        name: _classification_target(
            combined,
            features.matrix,
            training_rows,
            validation_rows,
            name,
            predictions,
        )
        for name in source_plan.classification_targets
    }
    for result in classifications.values():
        result["validation_labels"] = result.pop("test_labels")
    summary = {
        "status": "evaluated",
        "claim": "separate_committed_dataset_with_nonoverlapping_fixed_designs",
        "training_dataset": {name: source.get(name) for name in ("dataset_id", "study_id", "table_schema")},
        "validation_dataset": {
            name: validation_source.get(name) for name in ("dataset_id", "study_id", "table_schema")
        },
        "training_scope": "source_training_split_only",
        "training_rows": int(source_train.sum()),
        "validation_rows": len(validation_frame),
        "fixed_design_overlap": 0,
        "classification": classifications,
        "limitations": [
            "The validation dataset is not used for scaling, fitting, threshold selection, or hyperparameter selection.",
            "A separate simulated design set tests interpolation in this circuit; it is not measurement validation or cross-circuit evidence.",
        ],
    }
    return predictions.reset_index(drop=True), summary


def evaluate_surrogate_dataset(
    dataset: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    constraint_validation_dataset: pd.DataFrame | None = None,
    constraint_validation_manifest: Mapping[str, Any] | None = None,
) -> SurrogateEvaluation:
    """Compare a fixed three-neighbour surrogate with constant baselines."""

    started = time.perf_counter()
    frame, source, plan = _validated_evaluation_input(dataset, manifest)
    train, test = _split_masks(frame)
    transforms = _declared_transforms(manifest, plan.features)
    feature_space = fit_feature_space(frame, plan.features, transforms, train)
    fitted = time.perf_counter()

    output_columns = [*_TRACE_COLUMNS, *plan.features]
    predictions = frame.loc[test, output_columns].copy()
    eligible = _boolean_values(frame["objective_regression_eligible"], "objective_regression_eligible")
    if not np.isfinite(eligible).all():
        raise ValueError("objective_regression_eligible cannot contain missing values")
    eligible_bool = eligible.astype(bool)
    regressions = {
        name: _regression_target(
            frame,
            feature_space.matrix,
            train,
            test,
            eligible_bool,
            name,
            predictions,
        )
        for name in plan.objectives
    }
    classifications = {
        name: _classification_target(
            frame,
            feature_space.matrix,
            train,
            test,
            name,
            predictions,
        )
        for name in plan.classification_targets
    }
    holdout_elapsed = time.perf_counter() - started
    validation_predictions: pd.DataFrame | None = None
    independent_validation: dict[str, Any] = {"status": "not_provided"}
    if (constraint_validation_dataset is None) != (constraint_validation_manifest is None):
        raise ValueError("constraint validation dataset and manifest must be provided together")
    validation_started = time.perf_counter()
    if constraint_validation_dataset is not None and constraint_validation_manifest is not None:
        validation_predictions, independent_validation = _constraint_validation(
            frame,
            source,
            plan,
            manifest,
            constraint_validation_dataset,
            constraint_validation_manifest,
        )
    validation_elapsed = time.perf_counter() - validation_started
    elapsed = time.perf_counter() - started
    runtime = {
        "feature_fit": fitted - started,
        "prediction_and_metrics": holdout_elapsed - (fitted - started),
        "independent_constraint_validation": validation_elapsed,
        "total": elapsed,
        **_runtime_comparison(manifest, holdout_elapsed),
    }
    independent_classifications = independent_validation.get("classification")
    if not isinstance(independent_classifications, Mapping):
        independent_classifications = None
    summary = {
        "schema": "ml_surrogate_evaluation.v1",
        "dataset": {name: source.get(name) for name in ("dataset_id", "study_id", "table_schema")},
        "claim": "fixed_holdout_on_unseen_designs_within_one_committed_study",
        "model": {
            "name": "standardized_distance_weighted_knn",
            "neighbors": _NEIGHBORS,
            "feature_transforms": transforms,
            "active_features": feature_space.names,
            "dropped_constant_features": feature_space.dropped_constants,
            "training_center": feature_space.center,
            "training_scale": feature_space.scale,
        },
        "baselines": {"regression": "training_mean", "classification": "training_majority_class"},
        "regression": regressions,
        "classification": classifications,
        "independent_constraint_validation": independent_validation,
        "runtime_s": runtime,
        "evidence_gate": _evidence_gate(
            regressions,
            classifications,
            plan.engineering_targets,
            runtime,
            independent_classifications,
        ),
        "limitations": [
            "Metrics describe one fixed split inside one committed study, not another circuit or measurement campaign.",
            "One-class train or test labels are reported as insufficient evidence, regardless of accuracy.",
            "No uncertainty estimate, candidate proposal, or ngspice-evaluation saving is measured here.",
            "Every future proposed design must still receive complete Scenario x Control solver evaluation.",
        ],
    }
    return SurrogateEvaluation(
        predictions=predictions.reset_index(drop=True),
        summary=summary,
        constraint_validation_predictions=validation_predictions,
    )
