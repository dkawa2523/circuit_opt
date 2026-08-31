"""Pure circuit measurements and explicit case constraints.

Simulation owns waveforms and AC sweeps.  This module reads those immutable
artifacts and returns numbers; it never writes a second metrics file and never
adds constraint penalties to an objective.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    AC_LOAD_VOLTAGE,
    DEFAULT_Z0,
    ac_component_metrics,
    ac_power_flow,
    ac_probe_plan,
    at_frequency,
    component_loss_balance,
    input_impedance,
    load_current,
    read_ac,
    rf_port_metrics,
    transient_component_metrics,
)
from .case import Case, resolve_path
from .component_models import observed_components
from .core.models import ConstraintResult, EvaluationRequest, MetricSet, RawResult
from .metric_registry import get as get_metric
from .metric_registry import load_plugins, register
from .records import frequency_response_path, load_waveform, read_sim_record
from .spice import fundamental_hz


def load_target_waveform(case: Case) -> pd.DataFrame:
    cfg = case.data.get("target", {}) or {}
    path = cfg.get("waveform_file")
    if path is None:
        raise ValueError("target.waveform_file is required for waveform metrics")
    frame = pd.read_csv(resolve_path(case, path))
    if not {"time_s", "voltage_V"}.issubset(frame.columns):
        raise ValueError("target waveform must contain time_s and voltage_V")
    return frame


def interpolate_to_target(target: pd.DataFrame, waveform: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_time, target_voltage = _clean_time_value(target, "time_s", "voltage_V")
    if len(target_time) == 0:
        return target_time, target_voltage, np.asarray([], dtype=float)
    if waveform.empty or "time_s" not in waveform or "voltage_V" not in waveform:
        return target_time, target_voltage, np.full_like(target_time, np.nan, dtype=float)
    time_s, voltage = _clean_time_value(waveform, "time_s", "voltage_V")
    if len(time_s) == 0:
        return target_time, target_voltage, np.full_like(target_time, np.nan, dtype=float)
    aligned = np.interp(target_time, time_s, voltage, left=voltage[0], right=voltage[-1])
    return target_time, target_voltage, np.asarray(aligned, dtype=float)


def _clean_time_value(frame: pd.DataFrame, time_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    if time_col not in frame.columns or value_col not in frame.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    clean = pd.DataFrame(
        {
            "time": pd.to_numeric(frame[time_col], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    )
    finite = np.isfinite(clean["time"].to_numpy(float)) & np.isfinite(clean["value"].to_numpy(float))
    clean = clean.loc[finite]
    if clean.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    clean = clean.groupby("time", as_index=False, sort=True)["value"].mean()
    return clean["time"].to_numpy(float), clean["value"].to_numpy(float)


def _peak_voltage(waveform: pd.DataFrame) -> float | None:
    if waveform.empty or "voltage_V" not in waveform:
        return None
    voltage = np.asarray(pd.to_numeric(waveform["voltage_V"], errors="coerce"), dtype=float)
    finite = voltage[np.isfinite(voltage)]
    return float(np.max(np.abs(finite))) if len(finite) else None


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


@register("waveform_l2")
def waveform_l2(case: Case, record: dict[str, Any], waveform: pd.DataFrame) -> dict[str, Any]:
    del record
    target = load_target_waveform(case)
    _time, target_voltage, voltage = interpolate_to_target(target, waveform)
    if len(voltage) == 0 or np.isnan(voltage).all():
        raise ValueError("waveform is empty or has no finite voltage samples")
    reference = float(np.sqrt(np.mean(target_voltage**2)) + 1e-12)
    rmse = float(np.sqrt(np.mean((voltage - target_voltage) ** 2)))
    normalized = rmse / reference
    return {
        "loss": normalized,
        "normalized_rmse": normalized,
        "rmse_V": rmse,
        "objective": "waveform_l2",
    }


@register("impedance_match")
def impedance_match(case: Case, record: dict[str, Any], waveform: pd.DataFrame) -> dict[str, Any]:
    del waveform
    path = frequency_response_path(record)
    if path is None or not path.exists():
        raise ValueError("no frequency response; set solver.ac in the case")
    reference = float((case.data.get("measurement", {}) or {}).get("reference_impedance_ohm", DEFAULT_Z0))
    params = record.get("params") or {}
    target_frequency = fundamental_hz(case, params)
    components = observed_components(case, params)
    ac_columns = ac_probe_plan(case)[1]
    extras = [AC_LOAD_VOLTAGE, *ac_columns] if ac_columns else None
    # Interpolate the complex voltage/current first. Reflection and impedance
    # are nonlinear ratios, so interpolating those derived fields would be a
    # subtly different calculation.
    response = at_frequency(read_ac(path, extras), target_frequency)
    row = input_impedance(pd.DataFrame([response]), reference).iloc[0]
    reflection = float(row["reflection_magnitude"])
    metrics: dict[str, Any] = {
        "loss": reflection,
        "objective": "impedance_match",
        "reflection_magnitude": reflection,
        "reflection_db": _finite_or_none(row["reflection_db"]),
        "vswr": _finite_or_none(row["vswr"]),
        "resistance_ohm": _finite_or_none(row["resistance_ohm"]),
        "reactance_ohm": _finite_or_none(row["reactance_ohm"]),
        "match_frequency_Hz": float(row["frequency_Hz"]),
        "reference_impedance_ohm": reference,
    }
    metrics.update(ac_power_flow(response, load_current(case)))
    metrics.update(ac_component_metrics(response, components))
    metrics.update(component_loss_balance(metrics))
    return metrics


@register("rf_load")
def rf_load(case: Case, record: dict[str, Any], waveform: pd.DataFrame) -> dict[str, Any]:
    """Electrical load-port voltage, current, power, and harmonic metrics."""

    params = record.get("params") or {}
    frequency = fundamental_hz(case, params)
    values: dict[str, Any] = rf_port_metrics(waveform, frequency, load_current(case))
    values.update(transient_component_metrics(waveform, frequency, observed_components(case, params)))
    values.update(component_loss_balance(values))
    return {"objective": "rf_load", **values}


def measure_record(case: Case, record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Measure one successful simulation record without mutating its artifacts."""

    load_plugins(case.data.get("plugins"), case.base_dir)
    record = read_sim_record(record_or_path)
    if record.get("status") != "ok":
        raise ValueError(f"cannot measure simulation record with status {record.get('status')!r}")
    metric_name = str((case.data.get("target", {}) or {}).get("objective", "waveform_l2"))
    waveform = load_waveform(record)
    values = dict(get_metric(metric_name)(case, record, waveform))
    if not values:
        raise ValueError(f"metric '{metric_name}' returned no values")
    if "loss" in values:
        try:
            loss = float(values["loss"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric '{metric_name}' returned a non-numeric loss") from exc
        if not math.isfinite(loss):
            raise ValueError(f"metric '{metric_name}' returned a non-finite loss")
        values["loss"] = loss
    peak = _peak_voltage(waveform)
    if peak is not None:
        values.setdefault("peak_abs_voltage_V", peak)
    return values


class MetricLimitConstraint:
    """A declared engineering limit, kept separate from objective ranking."""

    def __init__(self, metric: str, bound: str, limit: float, *, name: str | None = None) -> None:
        if bound not in {"min", "max"}:
            raise ValueError("metric constraint bound must be 'min' or 'max'")
        self.metric = str(metric)
        self.bound = bound
        self.limit = float(limit)
        self.name = name or f"{bound}_{self.metric}"
        if not self.metric.strip() or not math.isfinite(self.limit):
            raise ValueError("metric constraint requires a metric name and finite limit")

    def evaluate(self, request: EvaluationRequest, raw: RawResult, metrics: MetricSet) -> ConstraintResult:
        del request, raw
        raw_value = metrics.values.get(self.metric)
        try:
            value = float(raw_value) if raw_value is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            return ConstraintResult(
                self.name,
                False,
                violation=1.0,
                limit=self.limit,
                detail=f"metric {self.metric!r} is unavailable or non-finite",
            )

        distance = max(0.0, value - self.limit) if self.bound == "max" else max(0.0, self.limit - value)
        return ConstraintResult(
            self.name,
            distance == 0.0,
            violation=distance / max(abs(self.limit), 1e-12),
            value=value,
            limit=self.limit,
        )


def constraints_from_case(case: Case) -> tuple[MetricLimitConstraint, ...]:
    raw_cfg = (case.data.get("target", {}) or {}).get("constraints")
    cfg = {} if raw_cfg is None else raw_cfg
    if not isinstance(cfg, dict):
        raise ValueError("target.constraints must be a mapping")

    constraints: list[MetricLimitConstraint] = []
    raw_bounds = cfg.get("metric_bounds")
    bounds = {} if raw_bounds is None else raw_bounds
    if not isinstance(bounds, dict):
        raise ValueError("target.constraints.metric_bounds must be a mapping")
    for metric, raw_spec in bounds.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"metric bound for {metric!r} must be a mapping")
        if not ({"min", "max"} & set(raw_spec)):
            raise ValueError(f"metric bound for {metric!r} must declare min or max")
        if "min" in raw_spec:
            constraints.append(MetricLimitConstraint(str(metric), "min", float(raw_spec["min"])))
        if "max" in raw_spec:
            constraints.append(MetricLimitConstraint(str(metric), "max", float(raw_spec["max"])))
    return tuple(constraints)
