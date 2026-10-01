"""Compact summaries of persisted study results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from pcd.core.models import CandidateResult, StudySpec


def _study_result(root: Path) -> dict[str, Any]:
    path = root / "study_result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def study_artifact_path(study_root: str | Path, name: str) -> Path | None:
    """Resolve one artifact from the committed study generation."""

    root = Path(study_root)
    declared = (_study_result(root).get("artifacts") or {}).get(name)
    if isinstance(declared, str) and declared:
        path = Path(declared)
        return path if path.is_absolute() else root / path
    return None


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
    return pd.read_csv(path)


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
) -> dict[str, Any]:
    """Project the selected candidate into a compact engineering decision summary."""

    if best.success_fraction < 1.0:
        status = "incomplete_evidence"
        limitation = "selected_candidate_failed_conditions"
    elif best.feasible_fraction == 1.0 and best.success_fraction == 1.0:
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
        "search_completeness": "complete" if n_failed_evaluations == 0 else "incomplete_evidence",
        "search_limitation": "none" if n_failed_evaluations == 0 else "failed_evaluations",
        "coverage": {
            "conditions": len(conditions),
            "solved": sum(item["status"] != "failed" for item in conditions),
            "accepted": sum(item["status"] == "accepted" for item in conditions),
        },
        "conditions": conditions,
    }


def _candidate_history(root: Path) -> list[dict[str, Any]]:
    history_path = study_artifact_path(root, "history")
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


def _sorted_candidate_frame(frame: pd.DataFrame, payload: Mapping[str, Any]) -> pd.DataFrame:
    objectives = [name for name in frame.columns if name.startswith("objective.")]
    directions = {
        f"objective.{objective.get('metric', '')}": str(objective.get("direction", "minimize"))
        for objective in (payload.get("study") or {}).get("objectives") or []
    }
    selected_id = ((payload.get("best") or {}).get("candidate") or {}).get("candidate_id")
    if selected_id is not None:
        frame.insert(1, "selected", frame["candidate_id"].eq(selected_id))
    order = ["success_fraction", "feasible_fraction", "total_violation", *objectives, "control_margin"]
    ascending = [False, False, True]
    ascending.extend(directions.get(name, "minimize") != "maximize" for name in objectives)
    ascending.append(False)
    return frame.sort_values(order, ascending=ascending, na_position="last").reset_index(drop=True)


def candidate_summary(study_root: str | Path) -> pd.DataFrame:
    """Read the compact all-candidate comparison from study history."""

    root = Path(study_root)
    frame = _candidate_frame(_candidate_history(root))
    return frame if frame.empty else _sorted_candidate_frame(frame, _study_result(root))


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
