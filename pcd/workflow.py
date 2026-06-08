from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Case, write_json
from .ml_core import create_optimizer, score_record
from .sim_core import prepare_candidate_table, simulate_case, simulate_candidate_table


def prepare_candidates(case: Case, candidates: pd.DataFrame, run_root: str | Path) -> list[dict[str, Any]]:
    return prepare_candidate_table(case, candidates, run_root)


def simulate_candidates(case: Case, candidates: pd.DataFrame, run_root: str | Path, solver_override: str | None = None) -> list[dict[str, Any]]:
    return simulate_candidate_table(case, candidates, run_root, solver_override=solver_override)


def optimize_closed_loop(
    case: Case,
    n_trials: int,
    run_root: str | Path,
    optimizer_name: str | None = None,
    solver_override: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """The only built-in closed-loop coupling: ask -> simulate -> score -> tell."""

    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    opt = create_optimizer(case, optimizer_name, seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    n_failed_trials = 0
    for i in range(int(n_trials)):
        params = opt.ask()
        rec = simulate_case(case, params=params, run_root=run_root, solver_override=solver_override, run_id=f"trial_{i:04d}")
        metrics = score_record(case, rec.manifest())
        if rec.status != "ok" or metrics.get("status") == "failed":
            n_failed_trials += 1
        opt.tell(params, metrics)
        item = {"trial": i, "params": params, "metrics": metrics, "run_dir": str(rec.run_dir)}
        history.append(item)
        if best is None or metrics.get("loss", float("inf")) < best.get("metrics", {}).get("loss", float("inf")):
            best = item
    result = {
        "schema": "closed_loop_result.v1",
        "n_trials": int(n_trials),
        "n_failed_trials": int(n_failed_trials),
        "best": best,
        "optimizer_state": opt.state(),
    }
    write_json(run_root / "optimization_result.json", result)
    pd.DataFrame(history).to_json(run_root / "optimization_history.json", orient="records", indent=2, force_ascii=False)
    return result
