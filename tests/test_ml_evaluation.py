from __future__ import annotations

import copy

import pandas as pd
import pytest

from pcd.ml import evaluate_surrogate_dataset


def _prepared_dataset() -> tuple[pd.DataFrame, dict]:
    rows = []
    for trial, value, split in (
        (0, 0.0, "train"),
        (1, 2.0, "train"),
        (2, 4.0, "train"),
        (3, 6.0, "train"),
        (4, 8.0, "train"),
        (5, 1.0, "test"),
        (6, 7.0, "test"),
    ):
        accepted = value >= 4.0
        rows.append(
            {
                "dataset_id": "study/generation",
                "study_id": "study",
                "trial": trial,
                "candidate_id": f"candidate_{trial}",
                "scenario_id": "nominal",
                "design_group_id": f"design_{trial}",
                "split": split,
                "design.x": value,
                "solver_ok": True,
                "feasible": accepted,
                "objective_regression_eligible": True,
                "metric.loss": value,
                "constraint.accepted.satisfied": accepted,
            }
        )
    frame = pd.DataFrame(rows)
    manifest = {
        "schema": "ml_evaluation_dataset.v1",
        "source": {
            "table_schema": "evaluation_table.v2",
            "dataset_id": "study/generation",
            "study_id": "study",
            "rows": len(frame),
        },
        "roles": {
            "features": ["design.x"],
            "objective_targets": ["metric.loss"],
            "constraint_classification_targets": ["constraint.accepted.satisfied"],
        },
        "feature_transforms": {"design.x": "linear"},
        "source_evaluation_cost_s": {"status": "available", "median": 0.5},
    }
    return frame, manifest


def _constraint_validation_dataset(manifest: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for trial, value in enumerate((0.5, 3.5, 4.5, 7.5)):
        accepted = value >= 4.0
        rows.append(
            {
                "dataset_id": "validation/generation",
                "study_id": "validation",
                "trial": trial,
                "candidate_id": f"validation_{trial}",
                "scenario_id": "nominal",
                "design_group_id": f"validation_design_{trial}",
                "split": "train" if trial < 3 else "test",
                "design.x": value,
                "solver_ok": True,
                "feasible": accepted,
                "objective_regression_eligible": True,
                "metric.loss": value,
                "constraint.accepted.satisfied": accepted,
            }
        )
    validation_manifest = copy.deepcopy(manifest)
    validation_manifest["source"].update(
        {
            "dataset_id": "validation/generation",
            "study_id": "validation",
            "rows": len(rows),
        }
    )
    return pd.DataFrame(rows), validation_manifest


def test_surrogate_evaluation_compares_fixed_baselines_and_reports_class_evidence():
    frame, manifest = _prepared_dataset()

    evaluated = evaluate_surrogate_dataset(frame, manifest)

    regression = evaluated.summary["regression"]["metric.loss"]
    assert regression["status"] == "evaluated"
    assert regression["surrogate_beats_baseline"] is True
    assert regression["surrogate"]["rmse"] < regression["baseline"]["rmse"]
    assert evaluated.summary["classification"]["feasible"]["status"] == "evaluable"
    assert evaluated.summary["classification"]["constraint.accepted.satisfied"]["status"] == "evaluable"
    assert evaluated.summary["classification"]["solver_ok"]["status"] == "insufficient_training_class_diversity"
    assert evaluated.summary["evidence_gate"]["all_objectives_beat_constant_baseline"] is True
    assert evaluated.summary["evidence_gate"]["all_classification_targets_evaluable"] is False
    assert evaluated.summary["evidence_gate"]["all_engineering_constraint_targets_evaluable"] is True
    assert evaluated.summary["evidence_gate"]["bayesian_optimization_ready"] is False
    assert evaluated.summary["evidence_gate"]["recommendation"] == "run_retrospective_candidate_ranking_evaluation"
    assert len(evaluated.predictions) == 2
    assert "surrogate.metric.loss" in evaluated.predictions
    assert "surrogate_probability.feasible" in evaluated.predictions


def test_surrogate_evaluation_rejects_invalid_features_and_split_leakage():
    frame, manifest = _prepared_dataset()
    non_numeric = frame.copy()
    non_numeric["design.x"] = non_numeric["design.x"].astype(object)
    non_numeric.loc[0, "design.x"] = "invalid"
    with pytest.raises(ValueError, match="finite numeric values"):
        evaluate_surrogate_dataset(non_numeric, manifest)

    crossed = frame.copy()
    crossed.loc[crossed.index[-1], "design_group_id"] = "design_0"
    with pytest.raises(ValueError, match="crosses the train/test boundary"):
        evaluate_surrogate_dataset(crossed, manifest)

    missing_transform = copy.deepcopy(manifest)
    missing_transform["feature_transforms"] = {}
    with pytest.raises(ValueError, match="every and only declared feature"):
        evaluate_surrogate_dataset(frame, missing_transform)


def test_surrogate_evaluation_reports_one_class_test_without_claiming_balanced_accuracy():
    frame, manifest = _prepared_dataset()
    frame.loc[frame["split"].eq("test"), "feasible"] = True

    evaluated = evaluate_surrogate_dataset(frame, manifest)

    result = evaluated.summary["classification"]["feasible"]
    assert result["status"] == "insufficient_test_class_diversity"
    assert result["surrogate"]["balanced_accuracy"] is None


def test_independent_constraint_validation_can_supply_missing_boundary_evidence():
    frame, manifest = _prepared_dataset()
    frame.loc[frame["split"].eq("test"), "feasible"] = True
    frame.loc[frame["split"].eq("test"), "constraint.accepted.satisfied"] = True
    validation, validation_manifest = _constraint_validation_dataset(manifest)

    evaluated = evaluate_surrogate_dataset(
        frame,
        manifest,
        constraint_validation_dataset=validation,
        constraint_validation_manifest=validation_manifest,
    )

    independent = evaluated.summary["independent_constraint_validation"]
    result = independent["classification"]["feasible"]
    assert independent["claim"] == "separate_committed_dataset_with_nonoverlapping_fixed_designs"
    assert independent["training_scope"] == "source_training_split_only"
    assert result["status"] == "evaluable"
    assert result["surrogate"]["balanced_accuracy"] > result["baseline"]["balanced_accuracy"]
    assert evaluated.summary["evidence_gate"]["engineering_constraint_evidence_source"] == (
        "independent_constraint_validation"
    )
    assert evaluated.summary["evidence_gate"]["recommendation"] == ("run_retrospective_candidate_ranking_evaluation")
    assert evaluated.constraint_validation_predictions is not None
    assert len(evaluated.constraint_validation_predictions) == len(validation)


def test_independent_constraint_validation_rejects_non_independent_or_partial_inputs():
    frame, manifest = _prepared_dataset()
    validation, validation_manifest = _constraint_validation_dataset(manifest)

    with pytest.raises(ValueError, match="must be provided together"):
        evaluate_surrogate_dataset(frame, manifest, constraint_validation_dataset=validation)

    different_roles = copy.deepcopy(validation_manifest)
    different_roles["roles"]["constraint_classification_targets"] = []
    with pytest.raises(ValueError, match="same feature and target roles"):
        evaluate_surrogate_dataset(
            frame,
            manifest,
            constraint_validation_dataset=validation,
            constraint_validation_manifest=different_roles,
        )

    different_transform = copy.deepcopy(validation_manifest)
    different_transform["feature_transforms"]["design.x"] = "log10"
    with pytest.raises(ValueError, match="same declared feature transforms"):
        evaluate_surrogate_dataset(
            frame,
            manifest,
            constraint_validation_dataset=validation,
            constraint_validation_manifest=different_transform,
        )

    same_identity = copy.deepcopy(validation_manifest)
    same_identity["source"]["dataset_id"] = manifest["source"]["dataset_id"]
    with pytest.raises(ValueError, match="different committed dataset"):
        evaluate_surrogate_dataset(
            frame,
            manifest,
            constraint_validation_dataset=validation,
            constraint_validation_manifest=same_identity,
        )

    overlapping = validation.copy()
    overlapping.loc[0, "design_group_id"] = frame.loc[0, "design_group_id"]
    with pytest.raises(ValueError, match="contains fixed-design values"):
        evaluate_surrogate_dataset(
            frame,
            manifest,
            constraint_validation_dataset=overlapping,
            constraint_validation_manifest=validation_manifest,
        )


def test_surrogate_evaluation_rejects_broken_prepared_contracts():
    frame, manifest = _prepared_dataset()

    wrong_schema = copy.deepcopy(manifest)
    wrong_schema["schema"] = "ml_evaluation_dataset.v0"
    with pytest.raises(ValueError, match="manifest schema"):
        evaluate_surrogate_dataset(frame, wrong_schema)
    with pytest.raises(ValueError, match="prepared dataset is empty"):
        evaluate_surrogate_dataset(frame.iloc[0:0], manifest)

    missing_roles = copy.deepcopy(manifest)
    missing_roles["roles"] = None
    with pytest.raises(ValueError, match=r"manifest\.roles must be a mapping"):
        evaluate_surrogate_dataset(frame, missing_roles)
    malformed_roles = copy.deepcopy(manifest)
    malformed_roles["roles"]["features"] = "design.x"
    with pytest.raises(ValueError, match="must be a list of column names"):
        evaluate_surrogate_dataset(frame, malformed_roles)
    empty_features = copy.deepcopy(manifest)
    empty_features["roles"]["features"] = []
    with pytest.raises(ValueError, match="at least one feature and objective"):
        evaluate_surrogate_dataset(frame, empty_features)

    with pytest.raises(ValueError, match="missing required columns: solver_ok"):
        evaluate_surrogate_dataset(frame.drop(columns="solver_ok"), manifest)
    wrong_rows = copy.deepcopy(manifest)
    wrong_rows["source"]["rows"] += 1
    with pytest.raises(ValueError, match="row count does not match"):
        evaluate_surrogate_dataset(frame, wrong_rows)

    unknown_split = frame.copy()
    unknown_split.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="unknown split labels"):
        evaluate_surrogate_dataset(unknown_split, manifest)
    with pytest.raises(ValueError, match="both train and test rows"):
        evaluate_surrogate_dataset(frame.assign(split="train"), manifest)

    invalid_transform = copy.deepcopy(manifest)
    invalid_transform["feature_transforms"]["design.x"] = "log"
    with pytest.raises(ValueError, match="unsupported feature transforms"):
        evaluate_surrogate_dataset(frame, invalid_transform)
    nonpositive_log = copy.deepcopy(manifest)
    nonpositive_log["feature_transforms"]["design.x"] = "log10"
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_surrogate_dataset(frame, nonpositive_log)
    constant = frame.copy()
    constant["design.x"] = 1.0
    with pytest.raises(ValueError, match="all model features are constant"):
        evaluate_surrogate_dataset(constant, manifest)


def test_surrogate_evaluation_keeps_insufficient_rows_and_invalid_labels_explicit():
    frame, manifest = _prepared_dataset()
    missing_eligibility = frame.copy()
    missing_eligibility["objective_regression_eligible"] = missing_eligibility["objective_regression_eligible"].astype(
        object
    )
    missing_eligibility.loc[0, "objective_regression_eligible"] = None
    with pytest.raises(ValueError, match="cannot contain missing values"):
        evaluate_surrogate_dataset(missing_eligibility, manifest)

    invalid_label = frame.copy()
    invalid_label["feasible"] = invalid_label["feasible"].astype(object)
    invalid_label.loc[0, "feasible"] = "unknown"
    with pytest.raises(ValueError, match="contains a non-boolean value"):
        evaluate_surrogate_dataset(invalid_label, manifest)

    string_labels = frame.copy()
    for name in ("solver_ok", "feasible", "constraint.accepted.satisfied"):
        string_labels[name] = string_labels[name].map({True: "true", False: "false"})
    assert (
        evaluate_surrogate_dataset(string_labels, manifest).summary["classification"]["feasible"]["status"]
        == "evaluable"
    )

    invalid_objective = frame.copy()
    invalid_objective.loc[0, "metric.loss"] = float("inf")
    with pytest.raises(ValueError, match="eligible objective target"):
        evaluate_surrogate_dataset(invalid_objective, manifest)

    no_test_objectives = frame.copy()
    no_test_objectives.loc[no_test_objectives["split"].eq("test"), "objective_regression_eligible"] = False
    no_test_objectives["feasible"] = no_test_objectives["feasible"].astype(object)
    no_test_objectives.loc[no_test_objectives["split"].eq("test"), "feasible"] = None
    result = evaluate_surrogate_dataset(no_test_objectives, manifest).summary
    assert result["regression"]["metric.loss"]["status"] == "insufficient_eligible_rows"
    assert result["classification"]["feasible"]["status"] == "insufficient_labeled_rows"
    assert result["evidence_gate"]["recommendation"] == "collect_eligible_objective_holdout_rows"


def test_surrogate_evidence_gate_requires_gain_and_a_source_cost_comparison():
    frame, manifest = _prepared_dataset()
    baseline_wins = frame.copy()
    baseline_wins.loc[baseline_wins["split"].eq("test"), "metric.loss"] = 4.0
    result = evaluate_surrogate_dataset(baseline_wins, manifest).summary
    assert result["regression"]["metric.loss"]["relative_rmse_improvement"] is None
    assert result["evidence_gate"]["recommendation"] == "do_not_advance_surrogate_model"

    no_cost = copy.deepcopy(manifest)
    del no_cost["source_evaluation_cost_s"]
    result = evaluate_surrogate_dataset(frame, no_cost).summary
    assert result["runtime_s"]["model_run_below_source_median"] is None
    assert result["evidence_gate"]["recommendation"] == "measure_or_reduce_model_overhead"
