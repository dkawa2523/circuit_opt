"""Compact summaries of persisted study results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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
        study_result = root / "study_result.json"
        if study_result.is_file():
            payload = json.loads(study_result.read_text(encoding="utf-8"))
            for objective in (payload.get("study") or {}).get("objectives") or []:
                metric = str(objective.get("metric", ""))
                directions[f"objective.{metric}"] = str(objective.get("direction", "minimize"))
        order = ["success_fraction", "feasible_fraction", "total_violation", *objectives, "control_margin"]
        ascending = [False, False, True]
        ascending.extend(directions.get(name, "minimize") != "maximize" for name in objectives)
        ascending.append(False)
        frame = frame.sort_values(order, ascending=ascending, na_position="last").reset_index(drop=True)
    return frame
