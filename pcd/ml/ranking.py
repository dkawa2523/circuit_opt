"""Retrospective candidate ordering on one fixed, fully evaluated design pool.

The evaluator replays which candidate would have been inspected next.  It does
not execute a solver, propose values outside the supplied pool, or establish an
online optimization capability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pcd.ml.neighbors import fit_feature_space, neighbor_predictions

_OUTCOME_COLUMNS = ("success_fraction", "feasible_fraction", "total_violation")


@dataclass(frozen=True)
class RankingEvaluation:
    """A compact decision summary plus the replayed evaluation order."""

    trace: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class _RankingPlan:
    features: dict[str, str]
    objectives: dict[str, str]
    initial_count: int
    initial_seed: int
    top_fraction: float
    neighbors: int
    feasibility_threshold: float
    random_repeats: int
    random_seed: int
    required_savings_fraction: float


def _features(feature_transforms: Mapping[str, str]) -> dict[str, str]:
    features = dict(feature_transforms)
    if not features or any(
        not name.startswith("design.") or kind not in {"linear", "log10"} for name, kind in features.items()
    ):
        raise ValueError("ranking features must be design.* columns with linear or log10 transforms")
    return features


def _objectives(objective_directions: Mapping[str, str]) -> dict[str, str]:
    objectives = dict(objective_directions)
    if not objectives or any(
        not name.startswith("objective.") or direction not in {"minimize", "maximize"}
        for name, direction in objectives.items()
    ):
        raise ValueError("ranking objectives must be objective.* columns with minimize or maximize directions")
    return objectives


def _settings(
    initial_count: int,
    initial_seed: int,
    top_fraction: float,
    neighbors: int,
    feasibility_threshold: float,
    random_repeats: int,
    random_seed: int,
    required_savings_fraction: float,
) -> tuple[float, float, float]:
    if min(initial_count, neighbors, random_repeats) < 1 or min(initial_seed, random_seed) < 0:
        raise ValueError("ranking counts must be positive and seeds must be non-negative")
    fractions = (top_fraction, feasibility_threshold, required_savings_fraction)
    if not all(math.isfinite(value) for value in fractions) or not (
        0.0 < fractions[0] <= 1.0 and 0.0 < fractions[1] <= 1.0 and 0.0 <= fractions[2] <= 1.0
    ):
        raise ValueError("ranking fractions are outside their supported 0-to-1 ranges")
    return fractions


def _plan(
    *,
    feature_transforms: Mapping[str, str],
    objective_directions: Mapping[str, str],
    initial_count: int,
    initial_seed: int,
    top_fraction: float,
    neighbors: int,
    feasibility_threshold: float,
    random_repeats: int,
    random_seed: int,
    required_savings_fraction: float,
) -> _RankingPlan:
    features = _features(feature_transforms)
    objectives = _objectives(objective_directions)
    fractions = _settings(
        initial_count,
        initial_seed,
        top_fraction,
        neighbors,
        feasibility_threshold,
        random_repeats,
        random_seed,
        required_savings_fraction,
    )
    return _RankingPlan(
        features,
        objectives,
        initial_count,
        initial_seed,
        fractions[0],
        neighbors,
        fractions[1],
        random_repeats,
        random_seed,
        fractions[2],
    )


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    original = frame[name]
    numeric = pd.to_numeric(original, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"ranking column {name!r} must contain only finite numeric values")
    return numeric


def _validated_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["candidate_id"].isna().any():
        raise ValueError("candidate_id cannot be missing")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if bool(frame["candidate_id"].str.strip().eq("").any()) or bool(frame["candidate_id"].duplicated().any()):
        raise ValueError("candidate_id must contain unique non-empty values")
    return frame.sort_values("candidate_id").reset_index(drop=True)


def _validated_outcomes(frame: pd.DataFrame) -> None:
    for name in ("success_fraction", "feasible_fraction"):
        if bool(((frame[name] < 0.0) | (frame[name] > 1.0)).any()):
            raise ValueError(f"{name} must be between 0 and 1")
    if bool((frame["total_violation"] < 0.0).any()):
        raise ValueError("total_violation must be non-negative")


def _validated_margin(frame: pd.DataFrame) -> bool:
    margin = pd.to_numeric(frame["control_margin"], errors="coerce")
    has_margin = bool(margin.notna().any())
    if has_margin and bool(margin.isna().any()):
        raise ValueError("control_margin must be finite for every candidate or absent for every candidate")
    if has_margin and bool((~np.isfinite(margin) | (margin < 0.0) | (margin > 1.0)).any()):
        raise ValueError("control_margin must be between 0 and 1")
    frame["control_margin"] = margin
    return has_margin


def _validated_frame(candidates: pd.DataFrame, plan: _RankingPlan) -> tuple[pd.DataFrame, bool]:
    if candidates.empty:
        raise ValueError("candidate ranking pool is empty")
    required = ["candidate_id", *_OUTCOME_COLUMNS, *plan.features, *plan.objectives, "control_margin"]
    if missing := [name for name in required if name not in candidates]:
        raise ValueError(f"candidate ranking pool is missing columns: {', '.join(missing)}")
    frame = _validated_identifiers(candidates.loc[:, required].copy())
    if plan.initial_count >= len(frame):
        raise ValueError("initial_count must leave at least one candidate for ranking")

    for name in [*_OUTCOME_COLUMNS, *plan.features, *plan.objectives]:
        frame[name] = _numeric_column(frame, name)
    _validated_outcomes(frame)
    return frame, _validated_margin(frame)


def _number(frame: pd.DataFrame, index: int, name: str) -> float:
    return float(str(frame.at[index, name]))


def _actual_feasible_order(frame: pd.DataFrame, plan: _RankingPlan, has_margin: bool) -> list[int]:
    complete = frame["success_fraction"].eq(1.0) & frame["feasible_fraction"].eq(1.0)
    complete &= frame["total_violation"].eq(0.0)
    eligible = list(frame.index[complete])

    def key(index: int) -> tuple[Any, ...]:
        objectives = tuple(
            _number(frame, index, name) if direction == "minimize" else -_number(frame, index, name)
            for name, direction in plan.objectives.items()
        )
        margin = (-_number(frame, index, "control_margin"),) if has_margin else ()
        return (*objectives, *margin, str(frame.at[index, "candidate_id"]))

    return sorted(eligible, key=key)


def _target_predictions(
    frame: pd.DataFrame,
    plan: _RankingPlan,
    observed: np.ndarray,
    remaining: np.ndarray,
    has_margin: bool,
) -> pd.DataFrame:
    space = fit_feature_space(frame, list(plan.features), plan.features, observed)
    targets = [*_OUTCOME_COLUMNS, *plan.objectives]
    if has_margin:
        targets.append("control_margin")
    predictions = pd.DataFrame(index=frame.index[remaining])
    for name in targets:
        values = frame[name].to_numpy(dtype=float)
        predicted, _used = neighbor_predictions(
            space.matrix,
            observed,
            remaining,
            values[observed],
            neighbors=plan.neighbors,
        )
        predictions[name] = predicted
    return predictions


def _next_candidate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    plan: _RankingPlan,
    has_margin: bool,
) -> int:
    def key(index: int) -> tuple[Any, ...]:
        complete = (
            _number(predictions, index, "success_fraction") >= plan.feasibility_threshold
            and _number(predictions, index, "feasible_fraction") >= plan.feasibility_threshold
        )
        objectives = tuple(
            _number(predictions, index, name) if direction == "minimize" else -_number(predictions, index, name)
            for name, direction in plan.objectives.items()
        )
        margin = (-_number(predictions, index, "control_margin"),) if has_margin else ()
        return (
            0 if complete else 1,
            1.0 - _number(predictions, index, "success_fraction"),
            1.0 - _number(predictions, index, "feasible_fraction"),
            _number(predictions, index, "total_violation"),
            *objectives,
            *margin,
            str(frame.loc[index, "candidate_id"]),
        )

    return min((int(index) for index in predictions.index), key=key)


def _trace_row(
    frame: pd.DataFrame,
    index: int,
    evaluation_number: int,
    phase: str,
    feasible_ranks: Mapping[int, int],
    targets: set[int],
    plan: _RankingPlan,
    predicted: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluation_number": evaluation_number,
        "phase": phase,
        "candidate_id": str(frame.at[index, "candidate_id"]),
        "fully_feasible": index in feasible_ranks,
        "feasible_rank": feasible_ranks.get(index),
        "target_top_fraction": index in targets,
    }
    result.update({name: _number(frame, index, name) for name in plan.features})
    for name in [*_OUTCOME_COLUMNS, *plan.objectives]:
        result[f"actual.{name}"] = _number(frame, index, name)
        result[f"predicted.{name}"] = None if predicted is None else predicted[name]
    return result


def _random_baseline_counts(
    candidate_count: int,
    initial: list[int],
    targets: set[int],
    *,
    repeats: int,
    seed: int,
) -> list[int]:
    if targets.intersection(initial):
        return [len(initial)] * repeats
    remaining = [index for index in range(candidate_count) if index not in set(initial)]
    rng = np.random.default_rng(seed)
    counts: list[int] = []
    for _repeat in range(repeats):
        order = rng.permutation(remaining).tolist()
        first = next(position for position, index in enumerate(order, start=1) if index in targets)
        counts.append(len(initial) + first)
    return counts


def _replay(
    frame: pd.DataFrame,
    plan: _RankingPlan,
    has_margin: bool,
    feasible_order: list[int],
) -> tuple[pd.DataFrame, list[int], set[int]]:
    target_count = max(1, math.ceil(len(feasible_order) * plan.top_fraction))
    targets = set(feasible_order[:target_count])
    feasible_ranks = {index: rank for rank, index in enumerate(feasible_order, start=1)}
    initial = np.random.default_rng(plan.initial_seed).permutation(len(frame))[: plan.initial_count].tolist()
    observed = list(initial)
    trace = [
        _trace_row(frame, index, number, "initial", feasible_ranks, targets, plan)
        for number, index in enumerate(initial, start=1)
    ]
    while not targets.intersection(observed):
        observed_mask = frame.index.isin(observed)
        remaining_mask = ~observed_mask
        predictions = _target_predictions(frame, plan, observed_mask, remaining_mask, has_margin)
        selected = _next_candidate(frame, predictions, plan, has_margin)
        observed.append(selected)
        trace.append(
            _trace_row(
                frame,
                selected,
                len(observed),
                "surrogate_ranked",
                feasible_ranks,
                targets,
                plan,
                {name: _number(predictions, selected, name) for name in predictions.columns},
            )
        )
    return pd.DataFrame(trace), observed, targets


def evaluate_candidate_ranking(
    candidates: pd.DataFrame,
    *,
    feature_transforms: Mapping[str, str],
    objective_directions: Mapping[str, str],
    initial_count: int,
    initial_seed: int,
    top_fraction: float = 0.1,
    neighbors: int = 3,
    feasibility_threshold: float = 0.5,
    random_repeats: int = 101,
    random_seed: int = 0,
    required_savings_fraction: float = 0.3,
) -> RankingEvaluation:
    """Replay fixed KNN ordering against fixed random-order baselines."""

    plan = _plan(
        feature_transforms=feature_transforms,
        objective_directions=objective_directions,
        initial_count=initial_count,
        initial_seed=initial_seed,
        top_fraction=top_fraction,
        neighbors=neighbors,
        feasibility_threshold=feasibility_threshold,
        random_repeats=random_repeats,
        random_seed=random_seed,
        required_savings_fraction=required_savings_fraction,
    )
    frame, has_margin = _validated_frame(candidates, plan)
    feasible_order = _actual_feasible_order(frame, plan, has_margin)
    if not feasible_order:
        return RankingEvaluation(
            pd.DataFrame(),
            {
                "schema": "ml_candidate_ranking.v1",
                "claim": "retrospective_fixed_candidate_pool",
                "status": "no_fully_feasible_candidates",
                "case_passed": False,
                "pool": {"candidates": len(frame), "fully_feasible_candidates": 0},
            },
        )

    trace, observed, targets = _replay(frame, plan, has_margin, feasible_order)
    initial = trace.loc[trace["phase"].eq("initial"), "candidate_id"].tolist()
    initial_indices = [int(frame.index[frame["candidate_id"].eq(candidate_id)][0]) for candidate_id in initial]
    baseline_counts = _random_baseline_counts(
        len(frame),
        initial_indices,
        targets,
        repeats=plan.random_repeats,
        seed=plan.random_seed,
    )
    baseline_median = float(np.median(baseline_counts))
    model_count = len(observed)
    savings = (baseline_median - model_count) / baseline_median
    selected_targets = targets.intersection(observed)
    selected = min(selected_targets, key=lambda index: feasible_order.index(index))
    complete_evidence = bool(frame["success_fraction"].eq(1.0).all())
    passed = complete_evidence and savings >= plan.required_savings_fraction
    summary = {
        "schema": "ml_candidate_ranking.v1",
        "claim": "retrospective_fixed_candidate_pool",
        "status": "evaluated",
        "pool": {
            "candidates": len(frame),
            "fully_feasible_candidates": len(feasible_order),
            "target_top_fraction": plan.top_fraction,
            "target_candidates": len(targets),
        },
        "protocol": {
            "initial_observations": plan.initial_count,
            "initial_seed": plan.initial_seed,
            "random_baseline_repeats": plan.random_repeats,
            "random_baseline_seed": plan.random_seed,
            "required_savings_fraction": plan.required_savings_fraction,
        },
        "model": {
            "name": "standardized_distance_weighted_knn",
            "neighbors": plan.neighbors,
            "feature_transforms": plan.features,
            "objective_directions": plan.objectives,
            "feasibility_threshold": plan.feasibility_threshold,
            "control_margin_used": has_margin,
        },
        "outcome": {
            "selected_candidate_id": str(frame.loc[selected, "candidate_id"]),
            "selected_feasible_rank": feasible_order.index(selected) + 1,
            "candidate_evaluations": model_count,
            "random_baseline_median_evaluations": baseline_median,
            "random_baseline_min_evaluations": min(baseline_counts),
            "random_baseline_max_evaluations": max(baseline_counts),
            "savings_fraction": savings,
        },
        "complete_solver_evidence": complete_evidence,
        "case_passed": passed,
        "limitations": [
            "This replays a finite pool whose outcomes were already computed; it is not an online optimizer.",
            "The model ranks only declared numeric component values in this circuit and does not generate topology.",
            "A selected candidate still requires a separate full Scenario x Control solver verification.",
        ],
    }
    return RankingEvaluation(trace.reset_index(drop=True), summary)
