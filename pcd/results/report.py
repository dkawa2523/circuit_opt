"""Compact summaries of persisted study results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pcd.core.models import CandidateResult, StudySpec

from .record import read_saved_result
from .table import read_evaluation_csv


def study_artifact_path(study_root: str | Path, name: str) -> Path | None:
    """Resolve one artifact from the committed study generation."""

    record = read_saved_result(study_root, study_only=True)
    return record.artifact_path(name) if record is not None else None


def read_best_candidate(study_root: str | Path) -> dict[str, Any]:
    """Load the selected candidate's structured solver evidence."""

    path = study_artifact_path(study_root, "best_candidate")
    if path is None or not path.is_file():
        raise FileNotFoundError(f"study does not declare a selected candidate: {study_root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"selected candidate must be a JSON object: {path}")
    return payload


def read_evaluation_table(study_root: str | Path) -> pd.DataFrame:
    """Load the complete Candidate x Scenario x Control audit table."""

    path = study_artifact_path(study_root, "evaluation_table")
    if path is None or not path.is_file():
        raise FileNotFoundError(f"study does not declare an evaluation table: {study_root}")
    return read_evaluation_csv(path)


def selected_evaluation(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Read the selected trial from a current candidate result."""

    trials = scenario.get("trials")
    index = scenario.get("selected_trial")
    if not isinstance(trials, list) or not isinstance(index, int) or not 0 <= index < len(trials):
        raise ValueError("candidate scenario does not declare a valid selected trial")
    selected = trials[index]
    if not isinstance(selected, Mapping):
        raise ValueError("selected candidate trial must be a mapping")
    return dict(selected)


def best_decision_summary(
    study: StudySpec,
    best: CandidateResult,
    *,
    n_failed_evaluations: int = 0,
    missing_acceptance_evidence: bool = False,
) -> dict[str, Any]:
    """Project the selected candidate into a compact engineering decision summary."""

    successful = best.successful
    if not successful:
        status = "incomplete_evidence"
        limitation = "selected_candidate_failed_conditions"
    elif any(not evaluation.evidence_available for evaluation in best.evaluations):
        status = "incomplete_evidence"
        limitation = "selected_candidate_missing_acceptance_evidence"
    elif best.feasible:
        status = "meets_declared_acceptance"
        limitation = "none"
    else:
        status = "does_not_meet_declared_acceptance"
        limitation = "control_margin_only" if best.edge_limited else "declared_constraints"

    objective_names = tuple(objective.metric for objective in study.objectives)
    conditions: list[dict[str, Any]] = []
    for scenario_result in best.scenarios:
        selected = scenario_result.selected
        if not selected.raw.ok:
            condition_status = "failed"
        elif not selected.evidence_available:
            condition_status = "incomplete_evidence"
        elif selected.feasible:
            condition_status = "accepted"
        elif scenario_result.edge_limited:
            condition_status = "control_margin_only"
        else:
            condition_status = "not_accepted"
        conditions.append(
            {
                "scenario_id": scenario_result.scenario.scenario_id,
                "values": scenario_result.scenario.to_dict()["values"],
                "status": condition_status,
                "selected_control": selected.request.control.to_dict()["values"],
                "objectives": {name: selected.metrics.values.get(name) for name in objective_names},
                "constraint_margins": selected.constraint_margins,
                "control_margin": scenario_result.control_margin,
                "edge_limited": scenario_result.edge_limited,
                "failed_constraints": [
                    constraint.to_dict() for constraint in selected.constraints if not constraint.satisfied
                ],
            }
        )

    evidence_limitation = "none"
    if n_failed_evaluations or not successful:
        evidence_limitation = "failed_evaluations"
    elif missing_acceptance_evidence or limitation == "selected_candidate_missing_acceptance_evidence":
        evidence_limitation = "missing_acceptance_evidence"

    return {
        "candidate": best.candidate.to_dict(),
        "aggregates": dict(best.aggregates),
        "feasible_fraction": best.feasible_fraction,
        "success_fraction": best.success_fraction,
        "total_violation": best.total_violation,
        "constraint_margins": best.constraint_margins,
        "control_margin": best.control_margin,
        "edge_limited": best.edge_limited,
        "status": status,
        "limitation": limitation,
        "evaluation_evidence": "complete" if evidence_limitation == "none" else "incomplete_evidence",
        "evidence_limitation": evidence_limitation,
        "coverage": {
            "conditions": len(conditions),
            "solved": sum(item["status"] != "failed" for item in conditions),
            "accepted": sum(item["status"] == "accepted" for item in conditions),
        },
        "conditions": conditions,
    }


def _candidate_history(history_path: Path | None) -> list[dict[str, Any]]:
    if history_path is None or not history_path.is_file():
        return []
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or any(not isinstance(item, Mapping) for item in history):
        raise ValueError(f"study history must be a JSON array of objects: {history_path}")
    return [dict(item) for item in history]


def _candidate_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data in history:
        row: dict[str, Any] = {
            "candidate_id": data.get("candidate_id"),
            "feasible_fraction": data.get("feasible_fraction"),
            "success_fraction": data.get("success_fraction"),
            "total_violation": data.get("total_violation"),
            "control_margin": data.get("control_margin"),
            "edge_limited": data.get("edge_limited", False),
        }
        row.update({f"design.{name}": value for name, value in (data.get("params", {}) or {}).items()})
        row.update({f"objective.{name}": value for name, value in (data.get("aggregates", {}) or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _sorted_candidate_frame(
    frame: pd.DataFrame, payload: Mapping[str, Any], history: list[dict[str, Any]]
) -> pd.DataFrame:
    selected_id = ((payload.get("best") or {}).get("candidate") or {}).get("candidate_id")
    if selected_id is not None:
        frame.insert(1, "selected", frame["candidate_id"].eq(selected_id))
    if all("rank" in row for row in history):
        ranks = [tuple(float("inf") if value is None else float(value) for value in row["rank"]) for row in history]
        positions = sorted(range(len(ranks)), key=ranks.__getitem__)
        return frame.iloc[positions].reset_index(drop=True)

    # Older result files may not record the engine's rank.
    objectives = [name for name in frame.columns if name.startswith("objective.")]
    directions = {
        f"objective.{objective.get('metric', '')}": str(objective.get("direction", "minimize"))
        for objective in (payload.get("study") or {}).get("objectives") or []
    }
    order = ["success_fraction", "feasible_fraction", "total_violation", *objectives, "control_margin"]
    ascending = [False, False, True]
    ascending.extend(directions.get(name, "minimize") != "maximize" for name in objectives)
    ascending.append(False)
    return frame.sort_values(order, ascending=ascending, na_position="last").reset_index(drop=True)


def candidate_summary(study_root: str | Path) -> pd.DataFrame:
    """Read the compact all-candidate comparison from study history."""

    record = read_saved_result(study_root, study_only=True)
    if record is None:
        return pd.DataFrame()
    history = _candidate_history(record.artifact_path("history"))
    frame = _candidate_frame(history)
    return frame if frame.empty else _sorted_candidate_frame(frame, record.payload, history)


def pareto_front_table(
    study: StudySpec,
    front: tuple[CandidateResult, ...],
    *,
    selected_candidate_id: str,
    dataset_id: str,
) -> pd.DataFrame:
    """Project an observed nondominated set into one concise decision table."""

    design_names = sorted({name for result in front for name in result.candidate.values})
    objective_names = [objective.metric for objective in study.objectives]
    columns = [
        "table_schema",
        "dataset_id",
        "candidate_id",
        "selected",
        *(f"design.{name}" for name in design_names),
        *(f"objective.{name}" for name in objective_names),
        "control_margin",
    ]
    rows = []
    for result in front:
        rows.append(
            {
                "table_schema": "pareto_front.v1",
                "dataset_id": dataset_id,
                "candidate_id": result.candidate.candidate_id,
                "selected": result.candidate.candidate_id == selected_candidate_id,
                **{f"design.{name}": result.candidate.values.get(name) for name in design_names},
                **{f"objective.{name}": result.aggregates.get(name) for name in objective_names},
                "control_margin": result.control_margin,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["selected", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    return frame
