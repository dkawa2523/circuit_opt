from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pcd.case import load_case
from pcd.ml import evaluate_candidate_ranking
from pcd.validation import validate_case

ML_BENCH = Path(__file__).resolve().parents[1] / "bench" / "ml"


def _candidate_pool(count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"candidate_{index:02d}" for index in range(count)],
            "design.x": [float(index + 1) for index in range(count)],
            "success_fraction": [1.0] * count,
            "feasible_fraction": [1.0] * count,
            "total_violation": [0.0] * count,
            "control_margin": [None] * count,
            "objective.loss": [float(index) for index in range(count)],
        }
    )


def _evaluate(
    frame: pd.DataFrame,
    *,
    objective_direction: str = "minimize",
    initial_count: int = 5,
    initial_seed: int = 0,
    top_fraction: float = 0.1,
):
    return evaluate_candidate_ranking(
        frame,
        feature_transforms={"design.x": "log10"},
        objective_directions={"objective.loss": objective_direction},
        initial_count=initial_count,
        initial_seed=initial_seed,
        top_fraction=top_fraction,
        random_repeats=101,
        random_seed=17,
        required_savings_fraction=0.3,
    )


def test_candidate_ranking_replays_fixed_pool_and_beats_fixed_random_baseline():
    first = _evaluate(_candidate_pool())
    second = _evaluate(_candidate_pool().sample(frac=1.0, random_state=8))

    assert first.summary["claim"] == "retrospective_fixed_candidate_pool"
    assert first.summary["pool"]["target_candidates"] == 4
    assert first.summary["outcome"]["candidate_evaluations"] == 6
    assert first.summary["outcome"]["random_baseline_median_evaluations"] == 11.0
    assert first.summary["outcome"]["savings_fraction"] > 0.3
    assert first.summary["case_passed"] is True
    assert first.summary["model"]["control_margin_used"] is False
    assert first.trace["candidate_id"].tolist() == second.trace["candidate_id"].tolist()
    assert first.trace.iloc[-1]["target_top_fraction"]


def test_candidate_ranking_keeps_incomplete_evidence_and_no_feasible_pool_explicit():
    incomplete = _candidate_pool()
    incomplete.loc[10, "success_fraction"] = 0.0
    result = _evaluate(incomplete)
    assert result.summary["complete_solver_evidence"] is False
    assert result.summary["case_passed"] is False

    infeasible = _candidate_pool()
    infeasible["feasible_fraction"] = 0.0
    infeasible["total_violation"] = 1.0
    result = _evaluate(infeasible)
    assert result.summary["status"] == "no_fully_feasible_candidates"
    assert result.summary["case_passed"] is False
    assert result.trace.empty


def test_candidate_ranking_rejects_outcome_features_and_ambiguous_pool_values():
    pool = _candidate_pool()
    with pytest.raises(ValueError, match=r"design\.\*"):
        evaluate_candidate_ranking(
            pool,
            feature_transforms={"objective.loss": "linear"},
            objective_directions={"objective.loss": "minimize"},
            initial_count=5,
            initial_seed=0,
        )

    duplicate = pd.concat([pool, pool.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique non-empty"):
        _evaluate(duplicate)

    mixed_margin = pool.copy()
    mixed_margin.loc[0, "control_margin"] = 0.5
    with pytest.raises(ValueError, match="every candidate or absent"):
        _evaluate(mixed_margin)


def test_candidate_ranking_rejects_invalid_protocol_settings():
    pool = _candidate_pool()
    with pytest.raises(ValueError, match="objectives must be"):
        _evaluate(pool, objective_direction="ascending")
    with pytest.raises(ValueError, match="counts must be positive"):
        _evaluate(pool, initial_count=0)
    with pytest.raises(ValueError, match="fractions are outside"):
        _evaluate(pool, top_fraction=0.0)


def test_candidate_ranking_rejects_incomplete_or_invalid_pool_columns():
    with pytest.raises(ValueError, match="pool is empty"):
        _evaluate(_candidate_pool().iloc[0:0])
    with pytest.raises(ValueError, match="missing columns"):
        _evaluate(_candidate_pool().drop(columns="total_violation"))

    pool = _candidate_pool()
    pool.loc[0, "candidate_id"] = None
    with pytest.raises(ValueError, match="cannot be missing"):
        _evaluate(pool)
    pool = _candidate_pool()
    pool.loc[0, "objective.loss"] = float("nan")
    with pytest.raises(ValueError, match="finite numeric"):
        _evaluate(pool)


def test_candidate_ranking_rejects_invalid_outcomes_and_uses_numeric_margin():
    pool = _candidate_pool()
    pool.loc[0, "feasible_fraction"] = 1.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        _evaluate(pool)
    pool = _candidate_pool()
    pool.loc[0, "total_violation"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        _evaluate(pool)
    pool = _candidate_pool()
    pool["control_margin"] = 0.5
    pool.loc[0, "control_margin"] = 1.1
    with pytest.raises(ValueError, match="control_margin must be between"):
        _evaluate(pool)
    with pytest.raises(ValueError, match="leave at least one"):
        _evaluate(_candidate_pool(5))

    pool = _candidate_pool()
    pool["control_margin"] = [index / len(pool) for index in range(len(pool))]
    result = _evaluate(pool, objective_direction="maximize", initial_seed=1)
    assert result.summary["model"]["control_margin_used"] is True
    assert result.trace.iloc[-1]["target_top_fraction"]


def test_p3_protocol_fixes_two_strict_81_candidate_pools_before_execution():
    protocol = yaml.safe_load((ML_BENCH / "ranking_protocol.yaml").read_text(encoding="utf-8"))

    assert protocol["schema"] == "pcd.ml_ranking_protocol.v1"
    assert protocol["success_criterion"]["top_fraction"] == 0.10
    assert protocol["success_criterion"]["required_savings_fraction"] == 0.30
    assert protocol["success_criterion"]["random_baseline_repeats"] == 101
    assert len(protocol["cases"]) == 2
    for spec in protocol["cases"]:
        case = load_case(ML_BENCH / spec["path"])
        assert validate_case(case, strict=True).ok
        assert case.case_id == spec["case_id"]
        assert int(case.data["run"]["trials"]) == spec["expected_candidates"] == 81
        assert spec["initial_observations"] == 6
        assert spec["solver_evaluations_per_candidate"] == 1
