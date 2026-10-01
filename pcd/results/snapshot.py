"""Concise time-axis projection for independent quasi-static AC snapshots."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pcd.core.models import CandidateResult

_METRIC_COLUMNS = {
    "frequency_Hz": "match_frequency_Hz",
    "input_resistance_ohm": "resistance_ohm",
    "input_reactance_ohm": "reactance_ohm",
    "reflection_magnitude": "reflection_magnitude",
    "reflected_power_fraction": "reflected_power_fraction",
    "forward_power_W": "forward_power_W",
    "reflected_power_W": "reflected_power_W",
    "load_real_power_W": "load_real_power_W",
    "network_loss_W": "network_loss_W",
    "transfer_efficiency": "transfer_efficiency",
    "source_real_power_W": "source_real_power_W",
    "source_voltage_rms_V": "source_voltage_rms_V",
}


def quasi_static_snapshot_table(result: CandidateResult) -> pd.DataFrame:
    """Return the selected candidate's one-row-per-time electrical response.

    Rows remain independent AC operating points.  This projection does not
    interpolate between them or imply a propagated plasma/circuit state.
    """

    rows: list[dict[str, Any]] = []
    design = {f"design.{name}": value for name, value in result.candidate.values.items()}
    for scenario_result in result.scenarios:
        scenario = scenario_result.scenario
        values = scenario.values
        missing = [
            name for name in ("snapshot_time_s", "load_resistance_ohm", "load_reactance_ohm") if name not in values
        ]
        if missing:
            raise ValueError(f"quasi-static scenario {scenario.scenario_id!r} is missing values {missing}")

        selected = scenario_result.selected
        metrics = selected.metrics.values
        row: dict[str, Any] = {
            "time_s": values["snapshot_time_s"],
            "scenario_id": scenario.scenario_id,
            "candidate_id": result.candidate.candidate_id,
            "status": selected.raw.status,
            "accepted": selected.feasible,
            "total_violation": selected.total_violation,
            "control_margin": scenario_result.control_margin,
            "error": selected.raw.error,
            "load_resistance_ohm": values["load_resistance_ohm"],
            "load_reactance_ohm": values["load_reactance_ohm"],
            **design,
            **{f"control.{name}": value for name, value in selected.request.control.values.items()},
        }
        row.update({output: metrics.get(metric) for output, metric in _METRIC_COLUMNS.items()})
        rows.append(row)

    return pd.DataFrame(rows).sort_values("time_s", kind="stable").reset_index(drop=True)
