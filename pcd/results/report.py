"""Compact summaries of persisted study results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pcd.core.models import CandidateResult, StudySpec


def best_decision_summary(
    study: StudySpec,
    best: CandidateResult,
    *,
    n_failed_evaluations: int = 0,
) -> dict[str, Any]:
    """Project the selected candidate into a compact engineering decision summary."""

    if n_failed_evaluations:
        status = "incomplete_evidence"
        limitation = "failed_evaluations"
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
        "control_margin": best.control_margin,
        "edge_limited": best.edge_limited,
        "status": status,
        "limitation": limitation,
        "coverage": {
            "conditions": len(conditions),
            "solved": sum(item["status"] != "failed" for item in conditions),
            "accepted": sum(item["status"] == "accepted" for item in conditions),
        },
        "conditions": conditions,
    }


def candidate_summary(study_root: str | Path) -> pd.DataFrame:
    root = Path(study_root)
    directory = root / "candidates"
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        candidate = data.get("candidate", {}) or {}
        row: dict[str, Any] = {
            "candidate_id": candidate.get("candidate_id"),
            "feasible_fraction": data.get("feasible_fraction"),
            "success_fraction": data.get("success_fraction"),
            "total_violation": data.get("total_violation"),
            "control_margin": data.get("control_margin"),
            "edge_limited": data.get("edge_limited", False),
        }
        row.update({f"design.{name}": value for name, value in (candidate.get("values", {}) or {}).items()})
        row.update({f"objective.{name}": value for name, value in (data.get("aggregates", {}) or {}).items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        objectives = [str(name) for name in frame.columns if str(name).startswith("objective.")]
        directions: dict[str, str] = {}
        selected_id: str | None = None
        study_result = root / "study_result.json"
        if study_result.is_file():
            payload = json.loads(study_result.read_text(encoding="utf-8"))
            selected_id = ((payload.get("best") or {}).get("candidate") or {}).get("candidate_id")
            for objective in (payload.get("study") or {}).get("objectives") or []:
                metric = str(objective.get("metric", ""))
                directions[f"objective.{metric}"] = str(objective.get("direction", "minimize"))
        if selected_id is not None:
            frame.insert(1, "selected", frame["candidate_id"].eq(selected_id))
        order = ["success_fraction", "feasible_fraction", "total_violation", *objectives, "control_margin"]
        ascending = [False, False, True]
        ascending.extend(directions.get(name, "minimize") != "maximize" for name in objectives)
        ascending.append(False)
        frame = frame.sort_values(order, ascending=ascending, na_position="last").reset_index(drop=True)
    return frame
