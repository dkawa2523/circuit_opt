"""Pure circuit measurements and explicit case constraints.

Simulation owns waveforms and AC sweeps.  This module reads those immutable
artifacts and returns numbers; it never writes a second metrics file and never
adds constraint penalties to an objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis.ac import DEFAULT_Z0
from .analysis.measurements import ac_operating_metrics, periodic_operating_metrics
from .analysis.transient import rf_measurement_options
from .case import Case, resolve_path
from .component_models import observed_components
from .core.models import ConstraintResult, EvaluationRequest, MeasurementUnavailableError, MetricSet, RawResult
from .inputs import reject_unknown
from .metric_registry import get as get_metric
from .metric_registry import load_plugins, register
from .records import load_simulation_result, read_sim_record
from .simulation import SimulationResult
from .simulation_input import build_probe_plan
from .spice import ac_operating_frequency_hz, fundamental_hz
from .tables import read_csv_rows


def validate_metric_settings(case: Case) -> None:
    target = case.data.get("target") or {}
    name = str(target.get("objective", "waveform_l2"))
    fields = {
        "waveform_l2": {"waveform_file"},
        "impedance_match": set(),
        "terminal_vi_fit": {"impedance_scale_ohm"},
        "rf_load": set(),
    }
    if name in fields:
        reject_unknown(target, {"objective", "constraints", "fundamental_Hz"} | fields[name], f"target {name}")


def load_target_waveform(case: Case) -> pd.DataFrame:
    cfg = case.data.get("target", {}) or {}
    path = cfg.get("waveform_file")
    if path is None:
        raise ValueError("target.waveform_file is required for waveform metrics")
    columns, rows = read_csv_rows(resolve_path(case, path), ("time_s", "voltage_V"))
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("time_s", "voltage_V"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if len(frame) < 2 or not np.all(np.isfinite(frame[["time_s", "voltage_V"]].to_numpy())):
        raise ValueError("target waveform requires at least two finite time/voltage rows")
    if np.any(np.diff(frame["time_s"].to_numpy()) <= 0):
        raise ValueError("target waveform time_s must be strictly increasing")
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
    aligned = np.full_like(target_time, np.nan, dtype=float)
    scale = max(abs(float(time_s[0])), abs(float(time_s[-1])), np.finfo(float).tiny)
    tolerance = 64.0 * np.finfo(float).eps * scale
    covered = (target_time >= time_s[0] - tolerance) & (target_time <= time_s[-1] + tolerance)
    aligned[covered] = np.interp(np.clip(target_time[covered], time_s[0], time_s[-1]), time_s, voltage)
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


@register("waveform_l2")
def waveform_l2(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    del params
    waveform = response.as_frame()
    target = load_target_waveform(case)
    _time, target_voltage, voltage = interpolate_to_target(target, waveform)
    if len(voltage) == 0 or np.isnan(voltage).all():
        raise ValueError("waveform is empty or has no finite voltage samples")
    if np.isnan(voltage).any():
        raise ValueError("waveform time range does not cover the target waveform")
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
def impedance_match(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    metrics = _terminal_impedance_metrics(case, params, response)
    reflection = float(metrics["reflection_magnitude"])
    return {"loss": reflection, "objective": "impedance_match", **metrics}


def _terminal_impedance_metrics(
    case: Case,
    params: dict[str, Any],
    response: SimulationResult,
) -> dict[str, Any]:
    """Measure the complex impedance seen at the calibrated source plane."""

    frequency_response = response.frequency_response
    if frequency_response is None:
        raise ValueError("no frequency response; set solver.ac in the case")
    reference = float((case.data.get("measurement", {}) or {}).get("reference_impedance_ohm", DEFAULT_Z0))
    target_frequency = ac_operating_frequency_hz(case, params)
    probes = build_probe_plan(case)
    metrics = ac_operating_metrics(
        frequency_response, target_frequency, reference, probes.load_current_column, observed_components(case, params)
    )
    metrics["match_frequency_Hz"] = metrics.pop("frequency_Hz")
    return metrics


@register("terminal_vi_fit")
def terminal_vi_fit(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    """Fit source-plane impedance derived from a measured complex V/I pair."""

    target_impedance = _target_impedance(params)
    scale = _target_impedance_scale(case)
    metrics = _terminal_impedance_metrics(case, params, response)
    if metrics.get("resistance_ohm") is None or metrics.get("reactance_ohm") is None:
        raise MeasurementUnavailableError(
            "terminal_vi_fit requires finite predicted impedance; the solved port is open or undefined"
        )
    resistance_error = float(metrics["resistance_ohm"]) - target_impedance.real
    reactance_error = float(metrics["reactance_ohm"]) - target_impedance.imag
    error = math.hypot(resistance_error, reactance_error) / scale
    return {
        "loss": error,
        "objective": "terminal_vi_fit",
        "normalized_impedance_error": error,
        "target_resistance_ohm": target_impedance.real,
        "target_reactance_ohm": target_impedance.imag,
        "residual_resistance_ohm": resistance_error,
        "residual_reactance_ohm": reactance_error,
        "normalized_residual_resistance": resistance_error / scale,
        "normalized_residual_reactance": reactance_error / scale,
        **metrics,
    }


TERMINAL_VI_PARAMETERS = (
    "target_voltage_re_V",
    "target_voltage_im_V",
    "target_current_re_A",
    "target_current_im_A",
)


def _target_impedance(params: Mapping[str, Any]) -> complex:
    try:
        values = {name: float(params[name]) for name in TERMINAL_VI_PARAMETERS}
    except KeyError as exc:
        raise ValueError(f"terminal_vi_fit requires scenario value {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal_vi_fit target V/I values must be numeric") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("terminal_vi_fit target V/I values must be finite")

    target_voltage = complex(values["target_voltage_re_V"], values["target_voltage_im_V"])
    target_current = complex(values["target_current_re_A"], values["target_current_im_A"])
    if abs(target_current) <= 1e-30:
        raise ValueError("terminal_vi_fit target delivered current must be nonzero")
    return target_voltage / target_current


def _target_impedance_scale(case: Case) -> float:
    target_cfg = case.data.get("target") or {}
    try:
        scale = float(target_cfg["impedance_scale_ohm"])
    except KeyError as exc:
        raise ValueError("terminal_vi_fit requires target.impedance_scale_ohm") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("target.impedance_scale_ohm must be numeric") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("target.impedance_scale_ohm must be positive and finite")
    return scale


@register("rf_load")
def rf_load(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    """Electrical load-port voltage, current, power, and harmonic metrics."""

    waveform = response.as_frame()
    frequency = fundamental_hz(case, params)
    options = rf_measurement_options(case.data.get("measurement"))
    probes = build_probe_plan(case)
    if not probes.load_current_column:
        raise ValueError("RF-port metrics require measurement.load_current")
    values = periodic_operating_metrics(
        waveform, frequency, probes.load_current_column, observed_components(case, params), options
    )
    return {"objective": "rf_load", **values}


def _detached_response(response: SimulationResult) -> SimulationResult:
    return SimulationResult(
        time_s=np.array(response.time_s, dtype=float, copy=True),
        voltage_V=np.array(response.voltage_V, dtype=float, copy=True),
        current_A=(None if response.current_A is None else np.array(response.current_A, dtype=float, copy=True)),
        status=response.status,
        log=response.log,
        diagnostics=deepcopy(response.diagnostics),
        frequency_response=(
            None if response.frequency_response is None else response.frequency_response.copy(deep=True)
        ),
        probes={name: np.array(values, dtype=float, copy=True) for name, values in response.probes.items()},
    )


def measure_response(
    case: Case,
    response: SimulationResult,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one canonical solver response without reading or writing artifacts."""

    validate_metric_settings(case)
    load_plugins(case.data.get("plugins"), case.base_dir)
    if response.status != "ok":
        raise ValueError(f"cannot measure simulation response with status {response.status!r}")
    metric_name = str((case.data.get("target", {}) or {}).get("objective", "waveform_l2"))
    detached = _detached_response(response)
    values = dict(get_metric(metric_name)(case.detached(), deepcopy(params or {}), detached))
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
    peak = _peak_voltage(detached.as_frame())
    if peak is not None:
        values.setdefault("peak_abs_voltage_V", peak)
    return values


def measure_record(case: Case, record_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Restore and measure one successful saved simulation response."""

    record = read_sim_record(record_or_path)
    if record.get("status") != "ok":
        raise ValueError(f"cannot measure simulation record with status {record.get('status')!r}")
    return measure_response(case, load_simulation_result(record), dict(record.get("params") or {}))


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
            reason = metrics.values.get(f"{self.metric}_unavailable_reason")
            return ConstraintResult(
                self.name,
                False,
                violation=1.0,
                limit=self.limit,
                detail=f"metric {self.metric!r} is unavailable or non-finite" + (f": {reason}" if reason else ""),
            )

        distance = max(0.0, value - self.limit) if self.bound == "max" else max(0.0, self.limit - value)
        return ConstraintResult(
            self.name,
            distance == 0.0,
            violation=distance / max(abs(self.limit), 1e-12),
            value=value,
            limit=self.limit,
            margin=self.limit - value if self.bound == "max" else value - self.limit,
        )


def constraints_from_case(case: Case) -> tuple[MetricLimitConstraint, ...]:
    raw_cfg = (case.data.get("target", {}) or {}).get("constraints")
    cfg = {} if raw_cfg is None else raw_cfg
    if not isinstance(cfg, dict):
        raise ValueError("target.constraints must be a mapping")
    reject_unknown(cfg, {"metric_bounds"}, "target.constraints")

    constraints: list[MetricLimitConstraint] = []
    raw_bounds = cfg.get("metric_bounds")
    bounds = {} if raw_bounds is None else raw_bounds
    if not isinstance(bounds, dict):
        raise ValueError("target.constraints.metric_bounds must be a mapping")
    for metric, raw_spec in bounds.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"metric bound for {metric!r} must be a mapping")
        reject_unknown(raw_spec, {"min", "max"}, f"target.constraints.metric_bounds.{metric}")
        if not ({"min", "max"} & set(raw_spec)):
            raise ValueError(f"metric bound for {metric!r} must declare min or max")
        if "min" in raw_spec:
            constraints.append(MetricLimitConstraint(str(metric), "min", float(raw_spec["min"])))
        if "max" in raw_spec:
            constraints.append(MetricLimitConstraint(str(metric), "max", float(raw_spec["max"])))
        if "min" in raw_spec and "max" in raw_spec and float(raw_spec["min"]) > float(raw_spec["max"]):
            raise ValueError(f"metric bound for {metric!r} requires min <= max")
    return tuple(constraints)
