"""Small numeric building blocks shared by the fixed surrogate evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpace:
    """Features transformed and standardized from one declared training set."""

    matrix: np.ndarray
    names: list[str]
    dropped_constants: list[str]
    center: dict[str, float]
    scale: dict[str, float]


def fit_feature_space(
    frame: pd.DataFrame,
    features: list[str],
    transforms: Mapping[str, str],
    train_rows: np.ndarray,
) -> FeatureSpace:
    """Fit declared transforms and scaling using only ``train_rows``."""

    columns: list[np.ndarray] = []
    for name in features:
        numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"feature {name!r} must contain only finite numeric values")
        if transforms[name] == "log10":
            if bool((numeric <= 0.0).any()):
                raise ValueError(f"log10 feature {name!r} must be positive")
            numeric = np.log10(numeric)
        columns.append(numeric)

    matrix = np.column_stack(columns)
    means = matrix[train_rows].mean(axis=0)
    scales = matrix[train_rows].std(axis=0)
    active = scales > 0.0
    if not active.any():
        raise ValueError("all model features are constant in the training split")
    active_names = [name for name, keep in zip(features, active, strict=True) if keep]
    dropped = [name for name, keep in zip(features, active, strict=True) if not keep]
    standardized = (matrix[:, active] - means[active]) / scales[active]
    return FeatureSpace(
        standardized,
        active_names,
        dropped,
        dict(zip(active_names, means[active].tolist(), strict=True)),
        dict(zip(active_names, scales[active].tolist(), strict=True)),
    )


def neighbor_predictions(
    features: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    train_values: np.ndarray,
    *,
    neighbors: int,
) -> tuple[np.ndarray, int]:
    """Predict numeric targets with stable inverse-distance neighbours."""

    train_matrix = features[train_rows]
    test_matrix = features[test_rows]
    used = min(neighbors, len(train_matrix))
    distances = np.linalg.norm(test_matrix[:, np.newaxis, :] - train_matrix[np.newaxis, :, :], axis=2)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :used]
    selected = np.take_along_axis(distances, nearest, axis=1)
    exact = selected <= np.finfo(float).eps
    weights = np.zeros_like(selected)
    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        weights[exact_rows] = exact[exact_rows] / exact[exact_rows].sum(axis=1, keepdims=True)
    if (~exact_rows).any():
        inverse = 1.0 / selected[~exact_rows]
        weights[~exact_rows] = inverse / inverse.sum(axis=1, keepdims=True)
    return np.sum(train_values[nearest] * weights, axis=1), used
