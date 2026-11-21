"""Pure circuit measurements and explicit case constraints.

Simulation supplies canonical waveforms and AC sweeps. Measurements and
engineering constraints consume those observations as separate operations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .analysis.ac import reference_impedance_ohm
from .analysis.measurements import ac_operating_metrics, periodic_operating_metrics
from .analysis.transient import rf_measurement_options
from .case import Case, resolve_path
from .component_models import observed_components
from .core.models import ConstraintResult, EvaluationRequest, MeasurementUnavailableError, MetricSet, RawResult
from .inputs import reject_unknown
from .metric_registry import MetricFunction, load_plugins, register
from .metric_registry import get as get_metric
from .signals.series import clean_series, interpolate_sides, interval_observed, ordered_samples
from .simulation import SimulationResult
from .simulation_input import build_probe_plan
from .sources import ac_port_issue, ac_power_balance_issue, transient_power_balance_issue
from .spice import ac_operating_frequency_hz, fundamental_hz
from .tables import read_csv_rows


def resolve_objective_metric(case: Case) -> tuple[str, MetricFunction]:
    """Resolve the same case-scoped objective for planning and measurement."""
    target = case.data.get("target") or {}
    name = str(target.get("objective", "waveform_l2"))
    fields = {
        "waveform_l2": {"waveform_file"},
        "impedance_match": set(),
        "terminal_vi_fit": {"impedance_scale_ohm", "observation_frequency_parameter"},
        "rf_load": set(),
    }
    if name in fields:
        reject_unknown(target, {"objective", "constraints", "fundamental_Hz"} | fields[name], f"target {name}")
    paths = load_plugins(case.data.get("plugins"), case.base_dir)
    return name, get_metric(name, plugins=paths)


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
    time_s = pd.to_numeric(waveform["time_s"], errors="coerce").to_numpy(float)
    voltage = pd.to_numeric(waveform["voltage_V"], errors="coerce").to_numpy(float)
    finite = np.isfinite(time_s) & np.isfinite(voltage)
    time_s, voltage = ordered_samples(time_s[finite], voltage[finite])
    if len(time_s) == 0:
        return target_time, target_voltage, np.full_like(target_time, np.nan, dtype=float)
    aligned = np.full_like(target_time, np.nan, dtype=float)
    scale = max(abs(float(time_s[0])), abs(float(time_s[-1])), np.finfo(float).tiny)
    tolerance = 64.0 * np.finfo(float).eps * scale
    covered = (target_time >= time_s[0] - tolerance) & (target_time <= time_s[-1] + tolerance)
    left, right = interpolate_sides(time_s, voltage, target_time[covered])
    # A point target has no event-side annotation. Preserve the midpoint only
    # at exact ties; the adjoining intervals retain their own observed limits.
    aligned[covered] = 0.5 * left + 0.5 * right
    return target_time, target_voltage, np.asarray(aligned, dtype=float)


def _clean_time_value(frame: pd.DataFrame, time_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    if time_col not in frame.columns or value_col not in frame.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(float)
    clean_time, clean_values = clean_series(time, values)
    return clean_time, clean_values


@register("waveform_l2")
def waveform_l2(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    del params
    waveform = response.as_frame()
    target = load_target_waveform(case)
    time, target_voltage, voltage = interpolate_to_target(target, waveform)
    if waveform.empty or "voltage_V" not in waveform or np.isnan(voltage).all():
        raise MeasurementUnavailableError("waveform is empty or has no finite voltage samples")
    if np.isnan(voltage).any() or not interval_observed(
        waveform["time_s"].to_numpy(float), waveform["voltage_V"].to_numpy(float), time[0], time[-1]
    ):
        raise MeasurementUnavailableError("observed waveform does not cover the target interval without gaps")
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
    if reason := metrics.get("reflection_magnitude_unavailable_reason"):
        metrics["loss_unavailable_reason"] = reason
    return {"loss": metrics["reflection_magnitude"], "objective": "impedance_match", **metrics}


def _terminal_impedance_metrics(
    case: Case,
    params: dict[str, Any],
    response: SimulationResult,
    *,
    frequency_hz: float | None = None,
) -> dict[str, Any]:
    """Measure the complex impedance seen at the calibrated source plane."""

    frequency_response = response.frequency_response
    if frequency_response is None:
        raise ValueError("no frequency response; set solver.ac in the case")
    reference = reference_impedance_ohm(case.data.get("measurement"))
    target_frequency = ac_operating_frequency_hz(case, params) if frequency_hz is None else frequency_hz
    probes = build_probe_plan(case)
    ac = (case.data.get("solver") or {}).get("ac") or {}
    source_scope = response.diagnostics.get("source_scope")
    reason = ac_port_issue(case, probes.source_name, ac.get("source"), source_scope=source_scope)
    if reason:
        raise MeasurementUnavailableError(reason)
    metrics = ac_operating_metrics(
        frequency_response,
        target_frequency,
        reference,
        response.diagnostics.get("load_current_column", probes.load_current_column),
        observed_components(case, params),
        require_simulated=True,
        power_balance_unavailable_reason=ac_power_balance_issue(source_scope),
    )
    metrics["match_frequency_Hz"] = metrics.pop("frequency_Hz")
    return metrics


@register("terminal_vi_fit")
def terminal_vi_fit(case: Case, params: dict[str, Any], response: SimulationResult) -> dict[str, Any]:
    """Fit source-plane impedance derived from a measured complex V/I pair."""

    target_impedance = terminal_observation_impedance(params)
    scale = _target_impedance_scale(case)
    frequency_parameter = terminal_observation_frequency_parameter(case)
    observed_frequency = (
        terminal_observation_frequency_hz(frequency_parameter, params) if frequency_parameter is not None else None
    )
    metrics = _terminal_impedance_metrics(case, params, response, frequency_hz=observed_frequency)
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
        **({"observed_frequency_Hz": observed_frequency} if observed_frequency is not None else {}),
        **metrics,
    }


TERMINAL_VI_PARAMETERS = (
    "target_voltage_re_V",
    "target_voltage_im_V",
    "target_current_re_A",
    "target_current_im_A",
)


def terminal_observation_frequency_parameter(case: Case) -> str | None:
    """Read the declared binding between observations and operating frequency."""

    name = (case.data.get("target") or {}).get("observation_frequency_parameter")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("target.observation_frequency_parameter must name an operating parameter")
        return name
    return None


def terminal_observation_frequency_hz(name: str, params: Mapping[str, Any]) -> float:
    """Read a finite observation frequency from the supplied row values."""

    if name not in params:
        raise ValueError(f"terminal_vi_fit requires explicit observation frequency {name!r}")
    raw = params[name]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observation frequency {name!r} must be numeric") from exc
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"observation frequency {name!r} must be positive and finite")
    return value


def terminal_observation_impedance(params: Mapping[str, Any]) -> complex:
    """Validate an observed peak-phasor pair and return its impedance."""

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
    impedance = target_voltage / target_current
    if not math.isfinite(impedance.real) or not math.isfinite(impedance.imag):
        raise ValueError("terminal_vi_fit target impedance must be finite")
    return impedance


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
    load_current = response.diagnostics.get("load_current_column", probes.load_current_column)
    if not load_current:
        raise ValueError("RF-port metrics require measurement.load_current")
    source_scope = response.diagnostics.get("source_scope")
    balance_issue = transient_power_balance_issue(case, probes.source_name, source_scope=source_scope)
    values = periodic_operating_metrics(
        waveform,
        frequency,
        load_current,
        observed_components(case, params),
        options,
        power_balance_unavailable_reason=balance_issue,
        source_port_unavailable_reason=source_scope.get("transient_port_unavailable_reason") if source_scope else None,
    )
    return {"objective": "rf_load", "observed_source": probes.source_name, **values}


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

    if response.status != "ok":
        raise ValueError(f"cannot measure simulation response with status {response.status!r}")
    time, voltage = np.asarray(response.time_s, dtype=float), np.asarray(response.voltage_V, dtype=float)
    if time.ndim != 1 or time.shape != voltage.shape:
        raise ValueError("canonical waveform time and voltage must be aligned one-dimensional arrays")
    metric_name, metric = resolve_objective_metric(case)
    detached = _detached_response(response)
    values = dict(metric(case.detached(), deepcopy(params or {}), detached))
    if not values:
        raise ValueError(f"metric '{metric_name}' returned no values")
    if values.get("loss") is not None:
        try:
            loss = float(values["loss"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric '{metric_name}' returned a non-numeric loss") from exc
        if not math.isfinite(loss):
            raise ValueError(f"metric '{metric_name}' returned a non-finite loss")
        values["loss"] = loss
    if time.size:
        observed = np.isfinite(time).all() and np.isfinite(voltage).all()
        values["peak_abs_voltage_V"] = float(np.max(np.abs(voltage))) if observed else None
        if not observed:
            values["peak_abs_voltage_V_unavailable_reason"] = (
                "finite time and voltage are required for the full waveform"
            )
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
            reason = metrics.values.get(f"{self.metric}_unavailable_reason")
            return ConstraintResult(
                self.name,
                False,
                violation=1.0,
                limit=self.limit,
                detail=f"metric {self.metric!r} is unavailable or non-finite" + (f": {reason}" if reason else ""),
                evidence_available=False,
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
