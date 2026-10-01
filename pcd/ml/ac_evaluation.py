"""Leakage-aware comparison of simple and graph AC response models.

Every model sees the same saved samples, targets, and predeclared holdout.
This module owns evaluation only: it does not read files, execute ngspice,
select circuit candidates, or persist a deployable model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .graph_encoding import GraphBatch, encode_graph_batch, flat_graph_features
from .neighbors import fit_feature_space
from .neural import NeuralFit, fit_mlp, fit_relational_gnn

MODEL_NAMES = ("constant", "ridge", "mlp", "relational_gnn")
_MIN_TRAIN_ROWS = 8
_MIN_TEST_ROWS = 2
_RIDGE_ALPHA = 1e-3


@dataclass(frozen=True)
class AcModelComparison:
    """Test-row predictions and model-comparison evidence."""

    predictions: pd.DataFrame
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _Fold:
    protocol: str
    name: str
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class _Features:
    frame: pd.DataFrame
    names: list[str]
    context_names: set[str]
    excluded_context: list[str]
    graphs: GraphBatch


def evaluate_ac_models(
    graphs: Mapping[str, Mapping[str, Any]],
    samples: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    seed: int = 0,
) -> AcModelComparison:
    """Compare constant, ridge, MLP, and relational GNN on declared splits."""

    _validate_samples(samples)
    targets = _role_columns(manifest, "port_complex_targets")
    contexts = [name for name in _role_columns(manifest, "context_features") if not name.startswith("context.design.")]
    _require_columns(samples, ["graph_id", "topology_family", "physical_group_id", *targets, *contexts])
    graph_ids = samples["graph_id"].astype(str).tolist()
    features = _prepare_features(graphs, samples, graph_ids, contexts)
    protocols = _protocol_folds(samples, manifest)

    prediction_tables: list[pd.DataFrame] = []
    protocol_results: dict[str, Any] = {}
    for protocol, value in protocols.items():
        if isinstance(value, str):
            protocol_results[protocol] = {"status": "unavailable", "reason": value, "folds": []}
            continue
        fold_results: list[dict[str, Any]] = []
        for fold in value:
            result, predictions = _evaluate_fold(samples, targets, features, fold, seed)
            fold_results.append(result)
            if not predictions.empty:
                prediction_tables.append(predictions)
        protocol_results[protocol] = _aggregate_protocol(fold_results)

    predictions = pd.concat(prediction_tables, ignore_index=True, sort=False) if prediction_tables else pd.DataFrame()
    summary = {
        "models": {
            "constant": "training-target mean",
            "ridge": "closed-form ridge on flat graph descriptors plus numeric context",
            "mlp": "fixed two-hidden-layer MLP on the same tabular features",
            "relational_gnn": "two component/net message-passing steps with separate p/n directions",
        },
        "protocols": protocol_results,
        "decision": _comparison_decision(protocol_results),
        "data": {
            "samples": len(samples),
            "graphs": len(graphs),
            "targets_declared": targets,
            "context_excluded_as_non_numeric_or_incomplete": features.excluded_context,
        },
        "training": {
            "seed": seed,
            "minimum_train_rows": _MIN_TRAIN_ROWS,
            "minimum_test_rows": _MIN_TEST_ROWS,
            "selection": "fixed_configuration_no_test_set_tuning",
        },
        "limitations": [
            "This comparison uses simulated AC phasors and does not establish independent chamber validation.",
            "Unavailable corpus splits remain unavailable; rows are never reshuffled to manufacture evidence.",
            "No evaluated model proposes component values or bypasses final ngspice evaluation.",
        ],
    }
    return AcModelComparison(predictions, summary)


def _prepare_features(
    graphs: Mapping[str, Mapping[str, Any]],
    samples: pd.DataFrame,
    graph_ids: list[str],
    contexts: list[str],
) -> _Features:
    flat = flat_graph_features(graphs, graph_ids)
    frame = pd.DataFrame(flat.matrix, columns=flat.names, index=samples.index)
    numeric_context: list[str] = []
    excluded: list[str] = []
    for name in contexts:
        numeric = pd.to_numeric(samples[name], errors="coerce")
        if numeric.notna().all():
            frame[name] = numeric.to_numpy(dtype=float)
            numeric_context.append(name)
        else:
            excluded.append(name)
    return _Features(
        frame,
        [*flat.names, *numeric_context],
        set(numeric_context),
        excluded,
        encode_graph_batch(graphs, graph_ids),
    )


def _protocol_folds(samples: pd.DataFrame, manifest: Mapping[str, Any]) -> dict[str, list[_Fold] | str]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("corpus manifest must declare splits")
    return {
        "unseen_design_within_known_topologies": _declared_fold(
            samples,
            splits,
            "unseen_design_within_known_topologies",
            "design_split",
            "design_group_id",
        ),
        "unseen_external_conditions": _declared_fold(
            samples,
            splits,
            "unseen_external_conditions",
            "condition_split",
            "condition_group_id",
        ),
        "unseen_topology_family": _topology_folds(samples, splits),
    }


def _declared_fold(
    samples: pd.DataFrame,
    splits: Mapping[str, Any],
    name: str,
    column: str,
    group_column: str,
) -> list[_Fold] | str:
    declaration = splits.get(name)
    if not isinstance(declaration, Mapping):
        raise ValueError(f"corpus manifest is missing split {name!r}")
    if declaration.get("status") != "ready":
        return str(declaration.get("reason", "corpus_split_unavailable"))
    _require_columns(samples, [column, group_column])
    labels = samples[column].astype(str)
    if set(labels) != {"train", "test"}:
        raise ValueError(f"ready split {column!r} must contain train and test rows")
    if bool((samples.groupby(group_column, dropna=False)[column].nunique(dropna=False) != 1).any()):
        raise ValueError(f"group {group_column!r} crosses {column!r}")
    return [_Fold(name, "holdout", labels.eq("train").to_numpy(), labels.eq("test").to_numpy())]


def _topology_folds(samples: pd.DataFrame, splits: Mapping[str, Any]) -> list[_Fold] | str:
    declaration = splits.get("unseen_topology_family")
    if not isinstance(declaration, Mapping):
        raise ValueError("corpus manifest is missing topology split")
    if declaration.get("status") != "ready":
        return "fewer_than_two_topology_families"
    families = sorted(samples["topology_family"].astype(str).unique())
    declared = declaration.get("groups")
    if not isinstance(declared, list) or families != sorted(str(item) for item in declared):
        raise ValueError("topology split groups do not match samples")
    return [
        _Fold(
            "unseen_topology_family",
            f"holdout:{family}",
            samples["topology_family"].astype(str).ne(family).to_numpy(),
            samples["topology_family"].astype(str).eq(family).to_numpy(),
        )
        for family in families
    ]


def _evaluate_fold(
    samples: pd.DataFrame,
    targets: list[str],
    features: _Features,
    fold: _Fold,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_rows = int(fold.train.sum())
    test_rows = int(fold.test.sum())
    base = {"name": fold.name, "train_rows": train_rows, "test_rows": test_rows}
    if train_rows < _MIN_TRAIN_ROWS or test_rows < _MIN_TEST_ROWS:
        return ({**base, "status": "insufficient_rows"}, pd.DataFrame())
    _validate_physical_groups(samples, fold)
    target_matrix, active_targets, target_center, target_scale = _targets_for_fold(samples, targets, fold)
    if not active_targets:
        return ({**base, "status": "no_variable_complete_targets"}, pd.DataFrame())

    transforms = dict.fromkeys(features.names, "linear")
    space = fit_feature_space(features.frame, features.names, transforms, fold.train)
    context_indexes = [index for index, name in enumerate(space.names) if name in features.context_names]
    context = space.matrix[:, context_indexes]
    train_targets = (target_matrix[fold.train] - target_center) / target_scale
    test_targets = target_matrix[fold.test]
    fold_seed = _fold_seed(seed, fold.protocol, fold.name)

    constant = np.broadcast_to(target_center, test_targets.shape).copy()
    ridge_standard = _ridge_predict(space.matrix[fold.train], train_targets, space.matrix[fold.test])
    mlp = fit_mlp(space.matrix[fold.train], train_targets, space.matrix[fold.test], seed=fold_seed)
    gnn = fit_relational_gnn(
        features.graphs.take(fold.train),
        context[fold.train],
        train_targets,
        features.graphs.take(fold.test),
        context[fold.test],
        seed=fold_seed,
    )
    model_predictions = {
        "constant": constant,
        "ridge": ridge_standard * target_scale + target_center,
        "mlp": mlp.predictions * target_scale + target_center,
        "relational_gnn": gnn.predictions * target_scale + target_center,
    }
    model_metrics = {
        name: _model_metrics(test_targets, predicted, target_scale, active_targets)
        for name, predicted in model_predictions.items()
    }
    winner = min(MODEL_NAMES, key=lambda name: model_metrics[name]["mean_normalized_rmse"])
    result = {
        **base,
        "status": "evaluated",
        "targets": active_targets,
        "features": space.names,
        "dropped_constant_features": space.dropped_constants,
        "models": model_metrics,
        "winner": winner,
        "training_evidence": {
            "mlp": _training_evidence(mlp, train_rows),
            "relational_gnn": _training_evidence(gnn, train_rows),
        },
    }
    return result, _prediction_rows(samples, fold, active_targets, test_targets, model_predictions)


def _targets_for_fold(
    samples: pd.DataFrame,
    targets: list[str],
    fold: _Fold,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    centers: list[float] = []
    scales: list[float] = []
    selected = fold.train | fold.test
    for name in targets:
        numeric = pd.to_numeric(samples[name], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values[selected]).all():
            continue
        center = float(values[fold.train].mean())
        scale = float(values[fold.train].std())
        threshold = max(1e-12, abs(center) * 1e-12)
        if scale <= threshold:
            continue
        columns.append(values)
        names.append(name)
        centers.append(center)
        scales.append(scale)
    matrix = np.column_stack(columns) if columns else np.empty((len(samples), 0))
    return matrix, names, np.asarray(centers), np.asarray(scales)


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    test = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(design.shape[1]) * _RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return test @ coefficients


def _model_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_scale: np.ndarray,
    targets: list[str],
) -> dict[str, Any]:
    residual = predicted - actual
    per_target: dict[str, Any] = {}
    normalized_rmse: list[float] = []
    for index, name in enumerate(targets):
        rmse = float(np.sqrt(np.mean(residual[:, index] ** 2)))
        nrmse = rmse / float(target_scale[index])
        normalized_rmse.append(nrmse)
        per_target[name] = {
            "mae": float(np.mean(np.abs(residual[:, index]))),
            "rmse": rmse,
            "normalized_rmse": nrmse,
        }
    return {"mean_normalized_rmse": float(np.mean(normalized_rmse)), "targets": per_target}


def _prediction_rows(
    samples: pd.DataFrame,
    fold: _Fold,
    targets: list[str],
    actual: np.ndarray,
    model_predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    trace_names = [
        name
        for name in ("sample_id", "graph_id", "topology_family", "candidate_id", "scenario_id", "frequency_Hz")
        if name in samples
    ]
    output = samples.loc[fold.test, trace_names].reset_index(drop=True).copy()
    output.insert(0, "fold", fold.name)
    output.insert(0, "protocol", fold.protocol)
    for index, target in enumerate(targets):
        output[f"actual.{target}"] = actual[:, index]
        for model, predicted in model_predictions.items():
            output[f"{model}.{target}"] = predicted[:, index]
    return output


def _aggregate_protocol(folds: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [fold for fold in folds if fold["status"] == "evaluated"]
    if not evaluated:
        return {"status": "unavailable", "reason": "no_evaluable_folds", "folds": folds}
    aggregate = {
        model: float(np.mean([fold["models"][model]["mean_normalized_rmse"] for fold in evaluated]))
        for model in MODEL_NAMES
    }
    winner = min(MODEL_NAMES, key=aggregate.__getitem__)
    return {
        "status": "evaluated" if len(evaluated) == len(folds) else "partial",
        "evaluated_folds": len(evaluated),
        "total_folds": len(folds),
        "mean_fold_normalized_rmse": aggregate,
        "winner": winner,
        "folds": folds,
    }


def _comparison_decision(protocols: Mapping[str, Any]) -> dict[str, Any]:
    topology = protocols["unseen_topology_family"]
    if topology.get("status") != "evaluated":
        return {
            "status": "insufficient_topology_evidence",
            "advance_to_model_persistence": False,
        }
    scores = topology["mean_fold_normalized_rmse"]
    winner = topology["winner"]
    gnn_score = float(scores["relational_gnn"])
    best_non_gnn = min(float(scores[name]) for name in MODEL_NAMES if name != "relational_gnn")
    return {
        "status": "topology_holdout_evaluated",
        "preferred_model": winner,
        "gnn_beats_every_non_gnn_model": gnn_score < best_non_gnn,
        "best_non_gnn_normalized_rmse": best_non_gnn,
        "gnn_normalized_rmse": gnn_score,
        "advance_to_model_persistence": winner != "constant" and float(scores[winner]) < float(scores["constant"]),
        "next_evidence_needed": (
            "more_independent_topology_families_and_training_samples"
            if winner == "constant"
            else "repeat_on_an_independent_locked_corpus"
        ),
    }


def _validate_samples(samples: pd.DataFrame) -> None:
    if samples.empty:
        raise ValueError("AC corpus samples are empty")
    _require_columns(samples, ["sample_schema", "sample_id", "design_group_id", "condition_group_id"])
    if set(samples["sample_schema"].astype(str)) != {"ac_graph_sample.v1"}:
        raise ValueError("samples must use schema ac_graph_sample.v1")
    if samples["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id values must be unique")


def _validate_physical_groups(samples: pd.DataFrame, fold: _Fold) -> None:
    train = set(samples.loc[fold.train, "physical_group_id"].astype(str))
    test = set(samples.loc[fold.test, "physical_group_id"].astype(str))
    if train & test:
        raise ValueError(f"physical evaluation group crosses fold {fold.name!r}")


def _role_columns(manifest: Mapping[str, Any], name: str) -> list[str]:
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("corpus manifest must declare roles")
    value = roles.get(name)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"corpus manifest roles.{name} must be a non-empty column list")
    return list(value)


def _require_columns(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"AC corpus is missing required columns: {', '.join(missing)}")


def _fold_seed(seed: int, protocol: str, fold: str) -> int:
    payload = f"{seed}:{protocol}:{fold}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _training_evidence(fit: NeuralFit, train_rows: int) -> dict[str, Any]:
    return {
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "parameter_count": fit.parameter_count,
        "parameters_per_train_row": fit.parameter_count / train_rows,
    }
