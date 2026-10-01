from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .case import NO_SOURCE_WARNING, Case, default_params, resolve_path, variable_specs
from .netlist import NetlistInputs, build_netlist_inputs, load_config
from .simulation_input import ResolvedSimulationCase, resolve_simulation_case
from .spice import fundamental_hz


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


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_case(case: Case, strict: bool = False) -> ValidationReport:
    report = ValidationReport(strict=strict)
    data = case.data
    if not isinstance(data, dict):
        report.add("error", "case.root_not_mapping", "case root must be a mapping")
        return report

    if data.get("source") is None and not data.get("sources"):
        report.add("warning", "case.no_source", NO_SOURCE_WARNING)
    _validate_variables(case, report)
    plugins_ok = _validate_plugins(case, report)
    simulation = _validate_simulation_input(case, report)
    netlist_inputs = _validate_netlist_inputs(case, report) if plugins_ok and simulation is not None else None
    _validate_load_applicability(case, report, simulation, netlist_inputs)
    _validate_measurement(case, report, simulation)
    _validate_target(case, report)
    _validate_study(case, report)
    return report


def _validate_variables(case: Case, report: ValidationReport) -> None:
    for name, spec in variable_specs(case).items():
        path = f"$.variables.{name}"
        if not isinstance(spec, dict):
            report.add("error", "variable.spec_not_mapping", "variable spec must be a mapping", path)
            continue
        _validate_variable_choices(spec, report, path)
        _validate_variable_bounds(spec, report, path)


def _validate_variable_choices(spec: dict[str, Any], report: ValidationReport, path: str) -> None:
    if "default" in spec:
        _validate_finite_tree(spec["default"], report, f"{path}.default")
    choices = spec.get("choices")
    if choices is None:
        return
    if not isinstance(choices, list) or not choices:
        report.add("error", "variable.empty_choices", "choices must be a non-empty list", path)
    else:
        _validate_finite_tree(choices, report, f"{path}.choices")
        if "default" in spec and spec["default"] not in choices:
            report.add("warning", "variable.default_not_in_choices", "default is not present in choices", path)


def _validate_finite_tree(value: Any, report: ValidationReport, path: str) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _validate_finite_tree(item, report, f"{path}.{name}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_finite_tree(item, report, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        report.add("error", "variable.non_finite_value", "numeric variable values must be finite", path)


def _parse_bounds(bounds: Any, report: ValidationReport, path: str) -> tuple[float, float] | None:
    """Return numeric (lo, hi), or None after reporting why it is unusable."""

    if not isinstance(bounds, list) or len(bounds) != 2:
        report.add("error", "variable.invalid_bounds", "bounds must be a two-item list", path)
        return None
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError):
        report.add("error", "variable.non_numeric_bounds", "bounds must be numeric", path)
        return None
    if not math.isfinite(lo) or not math.isfinite(hi):
        report.add("error", "variable.non_finite_bounds", "bounds must be finite", path)
        return None
    return lo, hi


def _validate_variable_bounds(spec: dict[str, Any], report: ValidationReport, path: str) -> None:
    if spec.get("bounds") is None:
        return
    parsed = _parse_bounds(spec.get("bounds"), report, path)
    if parsed is None:
        return
    lo, hi = parsed
    if lo > hi:
        report.add("error", "variable.bounds_reversed", "lower bound must be <= upper bound", path)
    if spec.get("type") == "int" and math.ceil(lo) > math.floor(hi):
        report.add("error", "variable.empty_integer_bounds", "integer bounds must contain an integer", path)
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


def _validate_simulation_input(case: Case, report: ValidationReport) -> ResolvedSimulationCase | None:
    """Compile the same input used by execution, then check solver availability."""

    try:
        simulation = resolve_simulation_case(case, default_params(case))
    except (TypeError, ValueError) as exc:
        report.add("error", "simulation.invalid_input", str(exc))
        return None

    from .sim_registry import available

    known = available()["solver"]
    if simulation.solver.name not in known:
        report.add(
            "error",
            "solver.unknown",
            f"unknown solver {simulation.solver.name!r}; available={known}",
            "$.solver.name",
        )
    return simulation


def _validate_netlist_inputs(case: Case, report: ValidationReport) -> NetlistInputs | None:
    """Build the same in-memory circuit/load model used by execution."""

    try:
        return build_netlist_inputs(case, default_params(case))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        report.add("error", "model.invalid_input", str(exc))
        return None


def _validate_load_applicability(
    case: Case,
    report: ValidationReport,
    simulation: ResolvedSimulationCase | None,
    netlist_inputs: NetlistInputs | None,
) -> None:
    if netlist_inputs is None or netlist_inputs.load_name not in {"impedance_point", "ccp_lumped", "icp_transformer"}:
        return
    cfg = load_config(case)
    name = netlist_inputs.load_name
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
        ac = simulation.analysis.ac if simulation is not None else None
        if ac is not None and (ac.points != 1 or ac.start_hz != ac.stop_hz):
            report.add(
                "error",
                "load.impedance_point_requires_ac_point",
                "impedance_point is exact at one frequency; use solver.ac.frequency_Hz instead of a sweep",
                "$.solver.ac",
            )


def _validate_measurement(
    case: Case,
    report: ValidationReport,
    simulation: ResolvedSimulationCase | None,
) -> None:
    """`load_current: auto` meters the load, so there has to be a load."""

    measurement = case.data.get("measurement")
    if measurement is None:
        measurement = {}
    if not isinstance(measurement, dict):
        return
    _validate_reference_impedance(measurement, report)
    try:
        from .analysis.transient import rf_measurement_options

        rf_measurement_options(measurement)
    except ValueError as exc:
        report.add("error", "measurement.invalid_periodic_options", str(exc), "$.measurement")
    if measurement.get("load_current") == "auto":
        load = case.data.get("load", {}) or {}
        if isinstance(load, dict) and str(load.get("name", "none")) == "none":
            report.add(
                "error",
                "measurement.auto_meter_without_load",
                "measurement.load_current: auto inserts an ammeter in series with the load, but no load is declared",
                "$.measurement.load_current",
            )
    _validate_measurement_duration(case, report, simulation)


def _validate_reference_impedance(measurement: dict[str, Any], report: ValidationReport) -> None:
    if "reference_impedance_ohm" not in measurement:
        return
    try:
        reference = float(measurement["reference_impedance_ohm"])
    except (TypeError, ValueError):
        report.add(
            "error",
            "measurement.invalid_reference_impedance",
            "reference_impedance_ohm must be numeric",
            "$.measurement.reference_impedance_ohm",
        )
        return
    if not math.isfinite(reference) or reference <= 0:
        report.add(
            "error",
            "measurement.invalid_reference_impedance",
            "reference_impedance_ohm must be positive and finite",
            "$.measurement.reference_impedance_ohm",
        )


def _validate_measurement_duration(
    case: Case,
    report: ValidationReport,
    simulation: ResolvedSimulationCase | None,
) -> None:
    """Require enough history for the configured periodic measurement window."""

    source = case.data.get("source", {}) or {}
    target = case.data.get("target", {}) or {}
    if simulation is None or not isinstance(source, dict) or not isinstance(target, dict):
        return
    if "frequency_Hz" not in source and "fundamental_Hz" not in target:
        return
    params = default_params(case)
    try:
        frequency = fundamental_hz(case, params)
    except (TypeError, ValueError):
        return  # the typed boundary reports invalid or unresolved input
    if simulation.analysis.transient is None or frequency <= 0:
        return
    stop_s = simulation.analysis.transient.stop_s

    from .analysis.transient import rf_measurement_options

    try:
        options = rf_measurement_options(case.data.get("measurement"))
    except ValueError:
        return
    required_cycles = max(int(options["periodic_cycles"]), int(options["settling_comparisons"]) + 1)
    cycles = stop_s * frequency
    if cycles < required_cycles:
        report.add(
            "warning",
            "solver.insufficient_rf_cycles",
            f"solver.tran.stop_s spans only {cycles:.3f} RF cycles; periodic power and harmonic measurement "
            f"needs at least {required_cycles} cycles ({required_cycles / frequency:.7g} s)",
            "$.solver.tran.stop_s",
        )


def _validate_plugins(case: Case, report: ValidationReport) -> bool:
    plugins = case.data.get("plugins")
    if plugins is None:
        plugins = []
    if not isinstance(plugins, list):
        report.add("error", "plugins.not_list", "plugins must be a list", "$.plugins")
        return False
    valid = True
    for i, raw in enumerate(plugins):
        try:
            path = Path(raw)
        except TypeError:
            report.add("error", "plugin.invalid_path", "plugin path must be a string", f"$.plugins[{i}]")
            valid = False
            continue
        if not path.is_absolute():
            path = case.base_dir / path
        if not path.is_file():
            report.add("error", "plugin.not_found", f"plugin not found: {path}", f"$.plugins[{i}]")
            valid = False
    if not valid or not plugins:
        return valid
    try:
        from .sim_registry import load_plugins

        load_plugins(plugins, case.base_dir)
    except Exception as exc:
        report.add("error", "plugin.load_failed", f"{type(exc).__name__}: {exc}", "$.plugins")
        return False
    return True


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
    target = case.data.get("target")
    if target is None:
        return
    if not isinstance(target, dict):
        report.add("error", "target.not_mapping", "target must be a mapping", "$.target")
        return
    if not target:
        return
    objective = str(target.get("objective", "waveform_l2"))
    solver = _mapping_or_empty(case.data.get("solver"))
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
        measurement = case.data.get("measurement", {}) or {}
        if isinstance(measurement, dict) and not measurement.get("load_current"):
            report.add(
                "error",
                "target.rf_load_without_current",
                "rf_load requires measurement.load_current; use auto for a built-in load",
                "$.measurement.load_current",
            )
    _validate_target_waveform(case, target, objective, report)


def _validate_target_waveform(
    case: Case,
    target: dict[str, Any],
    objective: str,
    report: ValidationReport,
) -> None:
    raw = target.get("waveform_file")
    if raw is None:
        if objective.startswith("waveform_"):
            report.add("warning", "target.no_waveform", "target.waveform_file is not set", "$.target.waveform_file")
        return
    try:
        path = resolve_path(case, raw)
    except TypeError:
        report.add(
            "error",
            "target.invalid_waveform_path",
            "target.waveform_file must be a path string",
            "$.target.waveform_file",
        )
        return
    if not path.exists():
        report.add("error", "target.waveform_not_found", f"target waveform not found: {path}", "$.target.waveform_file")
