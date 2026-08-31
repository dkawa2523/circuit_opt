from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .case import NO_SOURCE_WARNING, Case, resolve_path, variable_specs


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
    _validate_load(case, report)
    _validate_measurement(case, report)
    _validate_plugins(case, report)
    _validate_target(case, report)
    _validate_study(case, report)
    return report


def _validate_sources(data: dict[str, Any], report: ValidationReport) -> None:
    if not data.get("source") and not data.get("sources"):
        report.add("warning", "case.no_source", NO_SOURCE_WARNING)
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
        _validate_variable_choices(spec, report, path)
        _validate_variable_bounds(spec, report, path)


def _validate_variable_choices(spec: dict[str, Any], report: ValidationReport, path: str) -> None:
    choices = spec.get("choices")
    if choices is None:
        return
    if not isinstance(choices, list) or not choices:
        report.add("error", "variable.empty_choices", "choices must be a non-empty list", path)
    elif "default" in spec and spec["default"] not in choices:
        report.add("warning", "variable.default_not_in_choices", "default is not present in choices", path)


def _parse_bounds(bounds: Any, report: ValidationReport, path: str) -> tuple[float, float] | None:
    """Return numeric (lo, hi), or None after reporting why it is unusable."""

    if not isinstance(bounds, list) or len(bounds) != 2:
        report.add("error", "variable.invalid_bounds", "bounds must be a two-item list", path)
        return None
    try:
        return float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError):
        report.add("error", "variable.non_numeric_bounds", "bounds must be numeric", path)
        return None


def _validate_variable_bounds(spec: dict[str, Any], report: ValidationReport, path: str) -> None:
    if spec.get("bounds") is None:
        return
    parsed = _parse_bounds(spec.get("bounds"), report, path)
    if parsed is None:
        return
    lo, hi = parsed
    if lo > hi:
        report.add("error", "variable.bounds_reversed", "lower bound must be <= upper bound", path)
    if spec.get("scale") == "log" and (lo <= 0 or hi <= 0):
        report.add("error", "variable.log_bounds_non_positive", "log-scale bounds must be positive", path)
    _validate_default_within_bounds(spec, lo, hi, report, path)


def _validate_default_within_bounds(
    spec: dict[str, Any],
    lo: float,
    hi: float,
    report: ValidationReport,
    path: str,
) -> None:
    if "default" not in spec:
        return
    try:
        default = float(spec["default"])
    except (TypeError, ValueError):
        # A categorical default alongside bounds is not a range violation.
        return
    if default < lo or default > hi:
        report.add("warning", "variable.default_out_of_bounds", "default is outside bounds", path)


def _validate_solver(case: Case, report: ValidationReport) -> None:
    solver = case.data.get("solver", {}) or {}
    if not isinstance(solver, dict):
        report.add("error", "solver.not_mapping", "solver must be a mapping", "$.solver")
        return
    if str(solver.get("name", "dummy")) == "dummy":
        report.add(
            "warning",
            "solver.dummy",
            "dummy solver is for research screening and is not physical validation",
            "$.solver.name",
        )
        if "ac" in solver:
            report.add(
                "error",
                "solver.dummy_ac_unsupported",
                "dummy solver does not synthesize an AC frequency response",
                "$.solver.ac",
            )
    if "tran" in solver or "ac" not in solver:
        _validate_tran(solver.get("tran", {}) or {}, report)
    if "ac" in solver:
        _validate_ac(solver.get("ac"), report)
    _validate_timeout(solver.get("timeout_s"), report)


def _validate_tran(tran: Any, report: ValidationReport) -> None:
    """The transient window: both bounds positive, and step inside stop."""

    if not isinstance(tran, dict):
        report.add("error", "solver.tran_not_mapping", "solver.tran must be a mapping", "$.solver.tran")
        return
    try:
        step = float(tran.get("step_s", 1e-9))
        stop = float(tran.get("stop_s", 1e-6))
    except (TypeError, ValueError):
        report.add("error", "solver.tran_non_numeric", "tran step_s and stop_s must be numeric", "$.solver.tran")
        return
    if step <= 0 or stop <= 0:
        report.add("error", "solver.tran_non_positive", "tran step_s and stop_s must be positive", "$.solver.tran")
    if step > stop:
        report.add("warning", "solver.step_exceeds_stop", "tran step_s exceeds stop_s", "$.solver.tran")


def _validate_ac(ac: Any, report: ValidationReport) -> None:
    if not isinstance(ac, dict):
        report.add("error", "solver.ac_not_mapping", "solver.ac must be a mapping", "$.solver.ac")
        return
    if "frequency_Hz" in ac:
        _validate_ac_point(ac, report)
        return
    if str(ac.get("sweep", "dec")) not in {"lin", "dec", "oct"}:
        report.add("error", "solver.ac_invalid_sweep", "ac sweep must be lin, dec, or oct", "$.solver.ac.sweep")
    try:
        points = int(ac.get("points", 20))
        start = float(ac.get("start_Hz", 1e6))
        stop = float(ac.get("stop_Hz", 1e8))
    except (TypeError, ValueError):
        report.add("error", "solver.ac_non_numeric", "ac points/start_Hz/stop_Hz must be numeric", "$.solver.ac")
        return
    if points <= 0 or start <= 0 or stop <= 0 or stop < start:
        report.add(
            "error",
            "solver.ac_invalid_range",
            "ac points and frequencies must be positive, with stop_Hz >= start_Hz",
            "$.solver.ac",
        )


def _validate_ac_point(ac: dict[str, Any], report: ValidationReport) -> None:
    conflicting = sorted({"sweep", "points", "start_Hz", "stop_Hz"} & set(ac))
    if conflicting:
        report.add(
            "error",
            "solver.ac_point_conflict",
            f"ac.frequency_Hz is a one-point analysis and cannot be combined with {conflicting}",
            "$.solver.ac",
        )
    raw = ac["frequency_Hz"]
    try:
        frequency = float(raw)
    except (TypeError, ValueError):
        if not isinstance(raw, str) or not raw.strip():
            report.add(
                "error",
                "solver.ac_non_numeric",
                "ac.frequency_Hz must be positive numeric data or a parameter reference",
                "$.solver.ac.frequency_Hz",
            )
    else:
        if frequency <= 0 or not math.isfinite(frequency):
            report.add(
                "error",
                "solver.ac_invalid_range",
                "ac.frequency_Hz must be positive and finite",
                "$.solver.ac.frequency_Hz",
            )


def _validate_load(case: Case, report: ValidationReport) -> None:
    cfg = case.data.get("load", {}) or {}
    if not isinstance(cfg, dict):
        report.add("error", "load.not_mapping", "load must be a mapping", "$.load")
        return
    name = str(cfg.get("name", cfg.get("model", "none")))
    fields = {
        "impedance_point": ("resistance_ohm", "reactance_ohm", "model_frequency_Hz"),
        "ccp_lumped": ("R_eff_ohm", "L_eff_H", "C_sheath_eq_F"),
    }
    required = fields.get(name)
    if required is None and name != "icp_transformer":
        return
    if name == "icp_transformer":
        _validate_icp_parameters(cfg, report)
    else:
        missing = [field for field in required or () if field not in cfg]
        if missing:
            report.add("error", "load.missing_parameters", f"{name} is missing parameters: {missing}", "$.load")
    if not str(cfg.get("reference_plane", "")).strip():
        report.add("error", "load.missing_reference_plane", f"{name} requires load.reference_plane", "$.load")
    if not isinstance(cfg.get("characterization"), dict):
        report.add(
            "warning",
            "load.missing_characterization",
            f"{name} has no characterization mapping; results are usable, but applicability is not established",
            "$.load.characterization",
        )
    if name == "impedance_point":
        ac = (case.data.get("solver", {}) or {}).get("ac")
        if isinstance(ac, dict) and "frequency_Hz" not in ac:
            report.add(
                "error",
                "load.impedance_point_requires_ac_point",
                "impedance_point is exact at one frequency; use solver.ac.frequency_Hz instead of a sweep",
                "$.solver.ac",
            )


def _validate_icp_parameters(cfg: dict[str, Any], report: ValidationReport) -> None:
    required = {
        "R_coil_ohm",
        "L_coil_H",
        "reflected_inductance_H",
        "secondary_damping_rate_rad_s",
    }
    missing = sorted(required - set(cfg))
    if missing:
        report.add(
            "error",
            "load.missing_parameters",
            f"icp_transformer is missing parameters: {missing}",
            "$.load",
        )


def _validate_timeout(timeout: Any, report: ValidationReport) -> None:
    if timeout is None:
        return
    try:
        if float(timeout) <= 0:
            report.add("error", "solver.timeout_non_positive", "timeout_s must be positive", "$.solver.timeout_s")
    except (TypeError, ValueError):
        report.add("error", "solver.timeout_non_numeric", "timeout_s must be numeric", "$.solver.timeout_s")


MIN_MEASUREMENT_CYCLES = 3


def _validate_measurement(case: Case, report: ValidationReport) -> None:
    """`load_current: auto` meters the load, so there has to be a load."""

    if (case.data.get("measurement", {}) or {}).get("load_current") != "auto":
        return
    if str((case.data.get("load", {}) or {}).get("name", "none")) == "none":
        report.add(
            "error",
            "measurement.auto_meter_without_load",
            "measurement.load_current: auto inserts an ammeter in series with the load, but no load is declared",
            "$.measurement.load_current",
        )
    _validate_measurement_duration(case, report)


def _validate_measurement_duration(case: Case, report: ValidationReport) -> None:
    """Require enough history for the shared three-cycle measurement window."""

    tran = (case.data.get("solver", {}) or {}).get("tran", {}) or {}
    try:
        stop_s = float(tran.get("stop_s", 0.0))
        frequency = float((case.data.get("source", {}) or {}).get("frequency_Hz", 0.0))
    except (TypeError, ValueError):
        return  # a design-variable reference; nothing to check statically
    if stop_s <= 0 or frequency <= 0:
        return

    cycles = stop_s * frequency
    if cycles < MIN_MEASUREMENT_CYCLES:
        report.add(
            "warning",
            "solver.insufficient_rf_cycles",
            f"solver.tran.stop_s spans only {cycles:.3f} RF cycles; periodic power and harmonic measurement "
            f"needs at least {MIN_MEASUREMENT_CYCLES} cycles ({MIN_MEASUREMENT_CYCLES / frequency:.7g} s)",
            "$.solver.tran.stop_s",
        )


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


def _validate_study(case: Case, report: ValidationReport) -> None:
    """Check that scenarios and the exact control grid can be constructed."""

    try:
        from .core.models import Candidate
        from .study_config import CaseControlPolicy, candidate_case, study_spec_from_case

        study = study_spec_from_case(case)
        projected_case = candidate_case(case)
        candidate = Candidate(
            "validation", {name: spec.get("default") for name, spec in variable_specs(projected_case).items()}
        )
        policy = CaseControlPolicy(case)
        for scenario in study.scenarios:
            policy.controls(study, candidate, scenario)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report.add("error", "study.invalid", str(exc), "$.study")


def _validate_target(case: Case, report: ValidationReport) -> None:
    target = case.data.get("target", {}) or {}
    if not target:
        return
    if not isinstance(target, dict):
        report.add("error", "target.not_mapping", "target must be a mapping", "$.target")
        return
    objective = str(target.get("objective", "waveform_l2"))
    solver = case.data.get("solver", {}) or {}
    if objective == "impedance_match" and "ac" not in solver:
        report.add(
            "error",
            "target.impedance_without_ac",
            "impedance_match requires solver.ac",
            "$.target.objective",
        )
    if objective == "rf_load":
        if "tran" not in solver and "ac" in solver:
            report.add("error", "target.rf_load_without_tran", "rf_load requires solver.tran", "$.target.objective")
        if not (case.data.get("measurement", {}) or {}).get("load_current"):
            report.add(
                "error",
                "target.rf_load_without_current",
                "rf_load requires measurement.load_current; use auto for a built-in load",
                "$.measurement.load_current",
            )
    raw = target.get("waveform_file")
    if raw is None:
        if objective.startswith("waveform_"):
            report.add("warning", "target.no_waveform", "target.waveform_file is not set", "$.target.waveform_file")
        return
    path = resolve_path(case, raw)
    if not path.exists():
        report.add("error", "target.waveform_not_found", f"target waveform not found: {path}", "$.target.waveform_file")
