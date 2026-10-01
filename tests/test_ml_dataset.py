from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pcd.ml import prepare_evaluation_dataset


def _evaluation_source() -> tuple[pd.DataFrame, dict]:
    identity = {
        "table_schema": "evaluation_table.v2",
        "dataset_id": "study/g_generation",
        "study_id": "study",
        "case_schema": "case_yaml.v1",
        "resolved_case_schema": "case_yaml.v1",
        "runtime_fingerprint_sha256": "runtime",
        "solver_fingerprint_sha256": "solver",
    }
    rows = []
    designs = (10.0, 10.0, 20.0, 30.0)
    for trial, design in enumerate(designs):
        for scenario_id, load in (("light", 25.0), ("heavy", 100.0)):
            for tune in (1.0, 2.0):
                failed = trial == 3 and scenario_id == "heavy" and tune == 2.0
                rows.append(
                    {
                        **identity,
                        "trial": trial,
                        "candidate_id": f"candidate_{trial}",
                        "scenario_id": scenario_id,
                        "scenario_weight": 0.5,
                        "selected_control": tune == 1.0,
                        "status": "failed" if failed else "ok",
                        "feasible": not failed,
                        "total_violation": 1.0 if failed else 0.0,
                        "duration_s": 0.1,
                        "from_cache": False,
                        "raw_cache_key": f"cache-{trial}-{scenario_id}-{tune}",
                        "error": "solver failed" if failed else None,
                        "design.R_ohm": design,
                        "scenario.load_ohm": load,
                        "control.tune": tune,
                        "metric.loss": np.nan if failed else design / load + tune,
                        "metric.peak_V": np.nan if failed else load / design,
                        "constraint.max_peak.satisfied": np.nan if failed else load / design <= 10.0,
                        "constraint.max_peak.violation": np.nan if failed else max(load / design / 10.0 - 1.0, 0.0),
                        "constraint.max_peak.value": np.nan if failed else load / design,
                        "constraint.max_peak.limit": 10.0,
                        "constraint.max_peak.margin": np.nan if failed else 10.0 - load / design,
                        "constraint.evaluation_success.satisfied": not failed,
                        "constraint.evaluation_success.violation": 1.0 if failed else 0.0,
                        "constraint.evaluation_success.value": np.nan,
                        "constraint.evaluation_success.limit": np.nan,
                        "constraint.evaluation_success.margin": np.nan,
                        "artifact.waveform": f"artifact-{trial}.csv",
                    }
                )
    frame = pd.DataFrame(rows)
    result = {
        "schema": "study_result.v1",
        "dataset": identity,
        "study": {
            "study_id": "study",
            "objectives": [{"metric": "loss", "direction": "minimize", "aggregation": "worst"}],
        },
        "n_candidates": len(designs),
        "n_evaluations": len(frame),
    }
    return frame, result


def test_ml_dataset_assigns_roles_and_keeps_fixed_designs_in_one_split():
    source, result = _evaluation_source()

    prepared = prepare_evaluation_dataset(source, result, test_fraction=0.25, seed=17)
    frame = prepared.frame
    manifest = prepared.manifest

    assert len(frame) == len(source)
    assert manifest["schema"] == "ml_evaluation_dataset.v1"
    assert manifest["grain"]["unit"] == "candidate_x_scenario_x_control_evaluation"
    assert manifest["roles"]["features"] == ["design.R_ohm", "scenario.load_ohm", "control.tune"]
    assert manifest["roles"]["objective_targets"] == ["metric.loss"]
    assert manifest["roles"]["auxiliary_response_targets"] == ["metric.peak_V"]
    assert "constraint.max_peak.margin" in manifest["roles"]["constraint_response_targets"]
    assert manifest["roles"]["excluded"]["post_evaluation_selection_outcome"] == ["selected_control"]
    assert "selected_control" not in frame
    assert "artifact.waveform" not in frame
    assert frame.groupby("candidate_id")["split"].nunique().eq(1).all()
    assert frame.groupby("design.R_ohm")["split"].nunique().eq(1).all()
    assert frame.loc[frame["candidate_id"].isin(["candidate_0", "candidate_1"]), "design_group_id"].nunique() == 1
    assert manifest["split"]["groups"] == {"total": 3, "train": 2, "test": 1}
    assert manifest["eligibility"]["solver_failed_rows"] == 1
    assert manifest["eligibility"]["objective_regression_rows"] == len(source) - 1
    assert manifest["eligibility"]["classification_label_distribution"]["solver_ok"]["all"] == {
        "true": len(source) - 1,
        "false": 1,
        "missing": 0,
        "other": 0,
    }
    assert manifest["feature_transforms"] == {
        "design.R_ohm": "linear",
        "scenario.load_ohm": "linear",
        "control.tune": "linear",
    }
    assert manifest["source_evaluation_cost_s"]["median"] == pytest.approx(0.1)
    assert not frame.loc[frame["status"].eq("failed"), "objective_regression_eligible"].item()


def test_ml_dataset_split_is_stable_when_source_rows_are_reordered():
    source, result = _evaluation_source()
    first = prepare_evaluation_dataset(source, result, test_fraction=0.25, seed=9).frame
    shuffled = source.sample(frac=1.0, random_state=4).reset_index(drop=True)
    second = prepare_evaluation_dataset(shuffled, result, test_fraction=0.25, seed=9).frame

    first_groups = first.groupby("candidate_id")[["design_group_id", "split"]].first().sort_index()
    second_groups = second.groupby("candidate_id")[["design_group_id", "split"]].first().sort_index()
    pd.testing.assert_frame_equal(first_groups, second_groups)


def test_ml_dataset_rejects_broken_grain_identity_and_unsplittable_designs():
    source, result = _evaluation_source()

    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    duplicate_result = copy.deepcopy(result)
    duplicate_result["n_evaluations"] = len(duplicated)
    with pytest.raises(ValueError, match="duplicate Candidate x Scenario x Control"):
        prepare_evaluation_dataset(duplicated, duplicate_result)

    mismatched = copy.deepcopy(result)
    mismatched["dataset"]["dataset_id"] = "another/generation"
    with pytest.raises(ValueError, match="does not match committed study metadata"):
        prepare_evaluation_dataset(source, mismatched)

    one_design = source.copy()
    one_design["design.R_ohm"] = 10.0
    with pytest.raises(ValueError, match="at least two distinct fixed-design groups"):
        prepare_evaluation_dataset(one_design, result)

    missing_target = source.drop(columns="metric.loss")
    with pytest.raises(ValueError, match=r"missing required columns: metric\.loss"):
        prepare_evaluation_dataset(missing_target, result)


def test_ml_dataset_rejects_inconsistent_committed_metadata():
    source, result = _evaluation_source()

    with pytest.raises(ValueError, match="study_result must be a mapping"):
        prepare_evaluation_dataset(source, cast(Any, []))

    wrong_schema = copy.deepcopy(result)
    wrong_schema["schema"] = "study_result.v0"
    with pytest.raises(ValueError, match="study_result schema"):
        prepare_evaluation_dataset(source, wrong_schema)

    with pytest.raises(ValueError, match="evaluation table is empty"):
        prepare_evaluation_dataset(source.iloc[0:0], result)

    missing_dataset = copy.deepcopy(result)
    missing_dataset["dataset"] = None
    with pytest.raises(ValueError, match=r"study_result\.dataset must be a mapping"):
        prepare_evaluation_dataset(source, missing_dataset)

    missing_identity = copy.deepcopy(result)
    del missing_identity["dataset"]["solver_fingerprint_sha256"]
    with pytest.raises(ValueError, match="committed dataset identity is missing"):
        prepare_evaluation_dataset(source, missing_identity)

    null_identity = source.copy()
    null_identity.loc[0, "dataset_id"] = None
    with pytest.raises(ValueError, match="identity column 'dataset_id' contains missing"):
        prepare_evaluation_dataset(null_identity, result)

    mixed_identity = source.copy()
    mixed_identity.loc[0, "dataset_id"] = "another/generation"
    with pytest.raises(ValueError, match="identity column 'dataset_id' must have one value"):
        prepare_evaluation_dataset(mixed_identity, result)

    old_table = source.copy()
    old_table["table_schema"] = "evaluation_table.v1"
    old_result = copy.deepcopy(result)
    old_result["dataset"]["table_schema"] = "evaluation_table.v1"
    with pytest.raises(ValueError, match="evaluation table schema"):
        prepare_evaluation_dataset(old_table, old_result)

    missing_counts = copy.deepcopy(result)
    del missing_counts["n_evaluations"]
    with pytest.raises(ValueError, match="missing valid candidate/evaluation counts"):
        prepare_evaluation_dataset(source, missing_counts)

    wrong_rows = copy.deepcopy(result)
    wrong_rows["n_evaluations"] = len(source) - 1
    with pytest.raises(ValueError, match="evaluation row count"):
        prepare_evaluation_dataset(source, wrong_rows)

    wrong_candidates = copy.deepcopy(result)
    wrong_candidates["n_candidates"] = result["n_candidates"] + 1
    with pytest.raises(ValueError, match="candidate count"):
        prepare_evaluation_dataset(source, wrong_candidates)

    no_objectives = copy.deepcopy(result)
    no_objectives["study"]["objectives"] = []
    with pytest.raises(ValueError, match="at least one objective"):
        prepare_evaluation_dataset(source, no_objectives)

    invalid_objective = copy.deepcopy(result)
    invalid_objective["study"]["objectives"] = [None]
    with pytest.raises(ValueError, match=r"objectives\[0\] must be a mapping"):
        prepare_evaluation_dataset(source, invalid_objective)

    unnamed_objective = copy.deepcopy(result)
    unnamed_objective["study"]["objectives"] = [{}]
    with pytest.raises(ValueError, match="does not declare a metric"):
        prepare_evaluation_dataset(source, unnamed_objective)

    duplicate_objective = copy.deepcopy(result)
    duplicate_objective["study"]["objectives"].append(copy.deepcopy(duplicate_objective["study"]["objectives"][0]))
    with pytest.raises(ValueError, match="objective metric 'loss' is duplicated"):
        prepare_evaluation_dataset(source, duplicate_objective)


def test_ml_dataset_rejects_rows_that_cannot_support_the_holdout_claim():
    source, result = _evaluation_source()

    missing_row_id = source.copy()
    missing_row_id.loc[0, "scenario_id"] = None
    with pytest.raises(ValueError, match="row identifiers and status cannot be missing"):
        prepare_evaluation_dataset(missing_row_id, result)

    no_design = source.drop(columns="design.R_ohm")
    with pytest.raises(ValueError, match=r"requires at least one design\.\* feature"):
        prepare_evaluation_dataset(no_design, result)

    missing_design = source.copy()
    missing_design.loc[0, "design.R_ohm"] = np.nan
    with pytest.raises(ValueError, match="fixed-design features cannot be missing"):
        prepare_evaluation_dataset(missing_design, result)

    moving_candidate = source.copy()
    moving_candidate.loc[0, "design.R_ohm"] = 11.0
    with pytest.raises(ValueError, match="candidate_id maps to multiple fixed-design values"):
        prepare_evaluation_dataset(moving_candidate, result)

    reused_trial = source.copy()
    reused_trial.loc[reused_trial["candidate_id"].eq("candidate_1"), "trial"] = 0
    with pytest.raises(ValueError, match="trial and candidate_id must identify each other"):
        prepare_evaluation_dataset(reused_trial, result)

    text_target = source.copy()
    text_target["metric.loss"] = text_target["metric.loss"].astype(object)
    text_target.loc[0, "metric.loss"] = "not-a-number"
    with pytest.raises(ValueError, match=r"objective column 'metric\.loss' contains non-numeric"):
        prepare_evaluation_dataset(text_target, result)

    infinite_target = source.copy()
    infinite_target.loc[0, "metric.loss"] = np.inf
    with pytest.raises(ValueError, match="successful evaluations require a finite value"):
        prepare_evaluation_dataset(infinite_target, result)

    with pytest.raises(ValueError, match="test_fraction must be finite and between 0 and 1"):
        prepare_evaluation_dataset(source, result, test_fraction=0.0)
    with pytest.raises(ValueError, match="test_fraction must be finite and between 0 and 1"):
        prepare_evaluation_dataset(source, result, test_fraction=np.inf)
    with pytest.raises(ValueError, match="seed must be an integer"):
        prepare_evaluation_dataset(source, result, seed=True)

    extra_output = source.assign(undocumented_output=1.0)
    prepared = prepare_evaluation_dataset(extra_output, result)
    assert prepared.manifest["roles"]["excluded"]["other_unassigned_source_columns"] == ["undocumented_output"]

    with pytest.raises(ValueError, match="unknown features"):
        prepare_evaluation_dataset(source, result, feature_transforms={"design.unknown": "log10"})
    with pytest.raises(ValueError, match="must be 'linear' or 'log10'"):
        prepare_evaluation_dataset(source, result, feature_transforms={"design.R_ohm": "log"})


def test_ml_dataset_does_not_misreport_cached_or_invalid_evaluation_cost():
    source, result = _evaluation_source()
    cases = (
        (source.drop(columns="duration_s"), "duration_s_not_recorded"),
        (source.drop(columns="from_cache"), "cache_status_not_recorded"),
        (source.assign(from_cache="unknown"), "from_cache_contains_invalid_values"),
        (source.assign(from_cache=True), "no_uncached_evaluations"),
        (source.assign(duration_s=-1.0), "uncached_duration_s_contains_missing_or_invalid_values"),
    )

    for frame, reason in cases:
        cost = prepare_evaluation_dataset(frame, result).manifest["source_evaluation_cost_s"]
        assert cost["status"] == "unavailable"
        assert cost["reason"] == reason
