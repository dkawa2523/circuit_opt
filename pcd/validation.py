from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import Case, resolve_path, variable_specs


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    path: str = "$"

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message, "path": self.path}


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    strict: bool = False

    @property
    def ok(self) -> bool:
        bad_levels = {"error", "warning"} if self.strict else {"error"}
        return not any(issue.level in bad_levels for issue in self.issues)

    def add(self, level: str, code: str, message: str, path: str = "$") -> None:
        self.issues.append(ValidationIssue(level=level, code=code, message=message, path=path))

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "strict": self.strict, "issues": [issue.to_dict() for issue in self.issues]}

    def format_text(self) -> str:
        if not self.issues:
            return "OK"
        return "\n".join(f"{i.level.upper()} {i.code} {i.path}: {i.message}" for i in self.issues)


def validate_case(case: Case, strict: bool = False) -> ValidationReport:
    report = ValidationReport(strict=strict)
    data = case.data
    if not isinstance(data, dict):
        report.add("error", "case.root_not_mapping", "case root must be a mapping")
        return report

    _validate_sources(data, report)
    _validate_variables(case, report)
    _validate_solver(case, report)
    _validate_plugins(case, report)
    _validate_target(case, report)
    _validate_registry_conflicts(report)
    return report


def validate_waveform(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    report = ValidationReport(strict=strict)
    if df.empty:
        report.add("error", "waveform.empty", "waveform is empty")
    required = {"time_s", "voltage_V"}
    missing = sorted(required - set(df.columns))
    if missing:
        report.add("error", "waveform.missing_columns", f"missing required columns: {missing}")
        return report

    time = pd.to_numeric(df["time_s"], errors="coerce")
    voltage = pd.to_numeric(df["voltage_V"], errors="coerce")
    _check_numeric_series(time, report, "time_s", "$.time_s")
    _check_numeric_series(voltage, report, "voltage_V", "$.voltage_V")
    finite_time = np.isfinite(time.to_numpy(float))
    finite_voltage = np.isfinite(voltage.to_numpy(float))
    finite = finite_time & finite_voltage
    if not finite.any():
        report.add("error", "waveform.no_finite_samples", "waveform has no finite time/voltage samples")
        return report

    clean_time = time.loc[finite]
    if clean_time.duplicated().any():
        report.add("warning", "waveform.duplicate_time", "duplicate time_s samples will be averaged")
    if not clean_time.is_monotonic_increasing:
        report.add("warning", "waveform.non_monotonic_time", "time_s samples are not monotonic increasing")

    if "current_A" in df.columns:
        current = pd.to_numeric(df["current_A"], errors="coerce")
        _check_numeric_series(current, report, "current_A", "$.current_A", required=False)
    return report


def _validate_sources(data: dict[str, Any], report: ValidationReport) -> None:
    if not data.get("source") and not data.get("sources"):
        report.add("warning", "case.no_source", "no source/source list found; netlist may have no independent source")
    if data.get("source") is not None and not isinstance(data.get("source"), dict):
        report.add("error", "case.source_not_mapping", "source must be a mapping", "$.source")
    if data.get("sources") is not None and not isinstance(data.get("sources"), list):
        report.add("error", "case.sources_not_list", "sources must be a list", "$.sources")


def _validate_variables(case: Case, report: ValidationReport) -> None:
    for name, spec in variable_specs(case).items():
        path = f"$.variables.{name}"
        if not isinstance(spec, dict):
            report.add("error", "variable.spec_not_mapping", "variable spec must be a mapping", path)
            continue
        choices = spec.get("choices")
        if choices is not None:
            if not isinstance(choices, list) or not choices:
                report.add("error", "variable.empty_choices", "choices must be a non-empty list", path)
            elif "default" in spec and spec["default"] not in choices:
                report.add("warning", "variable.default_not_in_choices", "default is not present in choices", path)
        bounds = spec.get("bounds")
        if bounds is not None:
            if not isinstance(bounds, list) or len(bounds) != 2:
                report.add("error", "variable.invalid_bounds", "bounds must be a two-item list", path)
                continue
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
            except (TypeError, ValueError):
                report.add("error", "variable.non_numeric_bounds", "bounds must be numeric", path)
                continue
            if lo > hi:
                report.add("error", "variable.bounds_reversed", "lower bound must be <= upper bound", path)
            if spec.get("scale") == "log" and (lo <= 0 or hi <= 0):
                report.add("error", "variable.log_bounds_non_positive", "log-scale bounds must be positive", path)
            if "default" in spec:
                try:
                    default = float(spec["default"])
                    if default < lo or default > hi:
                        report.add("warning", "variable.default_out_of_bounds", "default is outside bounds", path)
                except (TypeError, ValueError):
                    pass


def _validate_solver(case: Case, report: ValidationReport) -> None:
    solver = case.data.get("solver", {}) or {}
    if not isinstance(solver, dict):
        report.add("error", "solver.not_mapping", "solver must be a mapping", "$.solver")
        return
    name = str(solver.get("name", "dummy"))
    if name == "dummy":
        report.add("warning", "solver.dummy", "dummy solver is for research screening and is not physical validation", "$.solver.name")
    tran = solver.get("tran", {}) or {}
    if not isinstance(tran, dict):
        report.add("error", "solver.tran_not_mapping", "solver.tran must be a mapping", "$.solver.tran")
        return
    step = tran.get("step_s", 1e-9)
    stop = tran.get("stop_s", 1e-6)
    try:
        step_f, stop_f = float(step), float(stop)
    except (TypeError, ValueError):
        report.add("error", "solver.tran_non_numeric", "tran step_s and stop_s must be numeric", "$.solver.tran")
        return
    if step_f <= 0 or stop_f <= 0:
        report.add("error", "solver.tran_non_positive", "tran step_s and stop_s must be positive", "$.solver.tran")
    if step_f > stop_f:
        report.add("warning", "solver.step_exceeds_stop", "tran step_s exceeds stop_s", "$.solver.tran")
    timeout = solver.get("timeout_s")
    if timeout is not None:
        try:
            if float(timeout) <= 0:
                report.add("error", "solver.timeout_non_positive", "timeout_s must be positive", "$.solver.timeout_s")
        except (TypeError, ValueError):
            report.add("error", "solver.timeout_non_numeric", "timeout_s must be numeric", "$.solver.timeout_s")


def _validate_plugins(case: Case, report: ValidationReport) -> None:
    plugins = case.data.get("plugins") or []
    if not isinstance(plugins, list):
        report.add("error", "plugins.not_list", "plugins must be a list", "$.plugins")
        return
    for i, raw in enumerate(plugins):
        path = Path(raw)
        if not path.is_absolute():
            path = case.base_dir / path
        if not path.exists():
            report.add("error", "plugin.not_found", f"plugin not found: {path}", f"$.plugins[{i}]")
    if not any(issue.level == "error" for issue in report.issues):
        try:
            from .ml_registry import load_plugins as load_ml_plugins
            from .sim_registry import load_plugins as load_sim_plugins

            load_sim_plugins(plugins, case.base_dir)
            load_ml_plugins(plugins, case.base_dir)
        except Exception as exc:
            report.add("error", "plugin.load_failed", f"{type(exc).__name__}: {exc}", "$.plugins")


def _validate_target(case: Case, report: ValidationReport) -> None:
    target = case.data.get("target", {}) or {}
    if not target:
        return
    if not isinstance(target, dict):
        report.add("error", "target.not_mapping", "target must be a mapping", "$.target")
        return
    raw = target.get("waveform_file")
    if raw is None:
        report.add("warning", "target.no_waveform", "target.waveform_file is not set", "$.target.waveform_file")
        return
    path = resolve_path(case, raw)
    if not path.exists():
        report.add("error", "target.waveform_not_found", f"target waveform not found: {path}", "$.target.waveform_file")


def _validate_registry_conflicts(report: ValidationReport) -> None:
    from .ml_registry import conflicts as ml_conflicts
    from .sim_registry import conflicts as sim_conflicts

    for conflict in sim_conflicts():
        report.add("warning", "registry.sim_conflict", _format_conflict(conflict), "$.plugins")
    for conflict in ml_conflicts():
        report.add("warning", "registry.ml_conflict", _format_conflict(conflict), "$.plugins")


def _format_conflict(conflict: dict[str, str]) -> str:
    return (
        f"{conflict.get('kind')}:{conflict.get('name')} keeps "
        f"{conflict.get('existing')} and ignores {conflict.get('new')}"
    )


def _check_numeric_series(
    series: pd.Series,
    report: ValidationReport,
    name: str,
    path: str,
    required: bool = True,
) -> None:
    arr = series.to_numpy(float)
    if required and len(arr) == 0:
        report.add("error", f"waveform.{name}.empty", f"{name} is empty", path)
        return
    if np.isnan(arr).any():
        report.add("warning", f"waveform.{name}.nan", f"{name} contains NaN or non-numeric values", path)
    if np.isinf(arr).any():
        report.add("warning", f"waveform.{name}.inf", f"{name} contains infinite values", path)
