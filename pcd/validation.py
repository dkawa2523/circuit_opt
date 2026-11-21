from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .case import Case, default_params, validate_case_fields
from .core.models import StudySpec
from .netlist import NetlistInputs, compile_netlist
from .problem import ParameterSet, candidate_role, resolve_parameter_set
from .simulation_input import ResolvedSimulationCase, resolve_simulation_case


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
    plan: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        bad_levels = {"error", "warning"} if self.strict else {"error"}
        return not any(issue.level in bad_levels for issue in self.issues)

    def add(self, level: str, code: str, message: str, path: str = "$") -> None:
        self.issues.append(ValidationIssue(level=level, code=code, message=message, path=path))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.ok,
            "strict": self.strict,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.plan is not None:
            result["plan"] = self.plan
        return result

    def format_text(self) -> str:
        if not self.issues:
            return "OK"
        return "\n".join(f"{i.level.upper()} {i.code} {i.path}: {i.message}" for i in self.issues)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_case(case: Case, strict: bool = False, *, include_plan: bool = False) -> ValidationReport:
    report = ValidationReport(strict=strict)
    data = case.data
    if not isinstance(data, dict):
        report.add("error", "case.root_not_mapping", "case root must be a mapping")
        return report
    try:
        validate_case_fields(data)
    except ValueError as exc:
        report.add("error", "case.unsupported_fields", str(exc))
        return report

    parameters = _validate_parameters(case, report)
    defaults = default_params(case) if parameters is not None else None
    plugins = _validate_plugins(case, report)
    simulation = _validate_simulation_input(case, report, defaults, plugins) if defaults is not None else None
    compiled = None
    if defaults is not None and plugins is not None and simulation is not None:
        compiled = _validate_netlist(case, report, defaults, simulation)
    measurement_options = _validate_measurement(case, report)
    scoring = not case.is_forward_simulation
    _validate_target(
        case,
        report,
        compiled[1] if compiled else simulation,
        defaults,
        measurement_options,
        scoring=scoring and plugins is not None,
    )
    try:
        from .output import output_settings

        output_settings(data.get("output"))
    except (TypeError, ValueError) as exc:
        report.add("error", "output.invalid", str(exc), "$.output")
    if data.get("identification") is not None:
        _validate_identification(case, report)
    if parameters is not None and defaults is not None and plugins is not None:
        study = _validate_study(case, report, parameters, scoring)
        if include_plan and report.ok and compiled is not None and study is not None:
            from .study_config import case_plan_summary

            inputs, resolved = compiled
            spec, counts = study
            report.plan = case_plan_summary(case, spec, counts, defaults, resolved, inputs.circuit.output_node)
    return report


def _validate_parameters(case: Case, report: ValidationReport) -> ParameterSet | None:
    """Project the shared input contract into the preflight report."""
    try:
        parameters = resolve_parameter_set(case)
    except (TypeError, ValueError) as exc:
        report.add("error", "parameter.invalid", str(exc))
        return None
    for definition in parameters.definitions:
        spec = definition.spec
        path = f"$.variables.{definition.name}"
        if "default" in spec:
            _warn_on_default(spec, report, path)
    return parameters


def _warn_on_default(spec: dict[str, Any], report: ValidationReport, path: str) -> None:
    choices = spec.get("choices", spec.get("values"))
    if choices is not None and spec["default"] not in choices:
        report.add("warning", "variable.default_not_in_choices", "default is not present in choices", path)
    if "bounds" not in spec:
        return
    try:
        default = float(spec["default"])
    except (TypeError, ValueError, OverflowError):
        return
    low, high = map(float, spec["bounds"])
    if default < low or default > high:
        report.add("warning", "variable.default_out_of_bounds", "default is outside bounds", path)


def _validate_simulation_input(
    case: Case, report: ValidationReport, defaults: dict[str, Any], plugins: tuple[Path, ...] | None
) -> ResolvedSimulationCase | None:
    """Compile the same input used by execution, then check solver availability."""

    try:
        simulation = resolve_simulation_case(case, defaults)
    except (OSError, TypeError, ValueError) as exc:
        report.add("error", "simulation.invalid_input", str(exc))
        return None

    if plugins is not None:
        from .sim_registry import get

        try:
            get("solver", simulation.solver.name, plugins=plugins)
        except (KeyError, RuntimeError, ValueError) as exc:
            report.add("error", "solver.unknown", str(exc), "$.solver.name")
    return simulation


def _validate_netlist(
    case: Case, report: ValidationReport, defaults: dict[str, Any], simulation: ResolvedSimulationCase
) -> tuple[NetlistInputs, ResolvedSimulationCase] | None:
    try:
        inputs, resolved, _ = compile_netlist(case, defaults, simulation)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        report.add("error", "model.invalid_input", str(exc))
        return None
    return inputs, resolved


def _validate_measurement(case: Case, report: ValidationReport) -> dict[str, int | float] | None:
    """Check reporting options; probe bindings belong to the simulation adapter."""

    measurement = case.data.get("measurement")
    if measurement is None:
        measurement = {}
    if not isinstance(measurement, dict):
        return None
    from .analysis.ac import reference_impedance_ohm
    from .analysis.transient import rf_measurement_options

    try:
        reference_impedance_ohm(measurement)
    except ValueError as exc:
        report.add(
            "error", "measurement.invalid_reference_impedance", str(exc), "$.measurement.reference_impedance_ohm"
        )

    try:
        return rf_measurement_options(measurement)
    except ValueError as exc:
        report.add("error", "measurement.invalid_periodic_options", str(exc), "$.measurement")
        return None


def _validate_rf_load_period(
    case: Case,
    report: ValidationReport,
    defaults: dict[str, Any],
    stop_s: float,
    options: dict[str, int | float] | None,
) -> None:
    """Check the default point for a metric that requires settled RF observations."""
    from .signals.windows import required_periodic_cycles
    from .spice import periodic_frequency_hz

    try:
        frequency = periodic_frequency_hz(case, defaults)
    except (TypeError, ValueError):
        # Parameter references can depend on the selected condition or candidate.
        return
    if frequency is None:
        report.add(
            "error",
            "target.rf_load_without_fundamental",
            "rf_load requires a periodic fundamental",
            "$.target.fundamental_Hz",
        )
        return
    if options is None:
        return
    required_cycles = required_periodic_cycles(int(options["periodic_cycles"]), int(options["settling_comparisons"]))
    cycles = stop_s / (1.0 / frequency)
    if cycles + 1e-12 < required_cycles:
        report.add(
            "warning",
            "solver.insufficient_rf_cycles",
            f"default transient spans only {cycles:.3f} RF cycles; periodic power and harmonic measurement "
            f"needs at least {required_cycles} cycles ({required_cycles / frequency:.7g} s)",
            "$.solver.tran.stop_s",
        )


def _validate_plugins(case: Case, report: ValidationReport) -> tuple[Path, ...] | None:
    from .sim_registry import load_plugins

    try:
        return load_plugins(case.data.get("plugins"), case.base_dir)
    except Exception as exc:
        report.add("error", "plugin.invalid", f"{type(exc).__name__}: {exc}", "$.plugins")
        return None


def _validate_study(
    case: Case, report: ValidationReport, parameters: ParameterSet, scoring: bool
) -> tuple[StudySpec, dict[str, int]] | None:
    """Check that scenarios and the exact control grid can be constructed."""

    if scoring:
        from .search import validate_optimizer_plan

        try:
            validate_optimizer_plan(case)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            report.add("error", "optimizer.invalid", str(exc), "$.optimizer")

    from .study_config import control_state_counts, study_spec_from_case

    try:
        if not scoring:
            from .search import validate_search_settings

            validate_search_settings(case)
        study = study_spec_from_case(case, parameters=parameters)
        candidate_role(case)
        return study, control_state_counts(case, study)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report.add("error", "study.invalid", str(exc), "$.study")
        return None


def _validate_target(
    case: Case,
    report: ValidationReport,
    simulation: ResolvedSimulationCase | None,
    defaults: dict[str, Any] | None,
    measurement_options: dict[str, int | float] | None,
    *,
    scoring: bool,
) -> None:
    target = case.data.get("target")
    if target is not None and not isinstance(target, dict):
        report.add("error", "target.not_mapping", "target must be a mapping", "$.target")
        return
    if not scoring:
        return
    from .metrics import constraints_from_case, resolve_objective_metric

    try:
        objective, _ = resolve_objective_metric(case)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report.add("error", "target.invalid_objective", str(exc), "$.target.objective")
        return
    try:
        constraints_from_case(case)
    except (TypeError, ValueError) as exc:
        report.add("error", "target.invalid_constraints", str(exc), "$.target.constraints")
    if (
        simulation is not None
        and objective in {"impedance_match", "terminal_vi_fit"}
        and simulation.analysis.ac is None
    ):
        report.add("error", "target.impedance_without_ac", f"{objective} requires solver.ac", "$.target.objective")
    if simulation is not None and objective == "rf_load":
        if simulation.analysis.transient is None:
            report.add("error", "target.rf_load_without_tran", "rf_load requires solver.tran", "$.target.objective")
        elif defaults is not None:
            _validate_rf_load_period(case, report, defaults, simulation.analysis.transient.stop_s, measurement_options)
        if not simulation.probes.load_current_column:
            report.add(
                "error",
                "target.rf_load_without_current",
                "rf_load requires measurement.load_current; use auto for an attached load",
                "$.measurement.load_current",
            )
    if objective == "waveform_l2":
        _validate_target_waveform(case, target or {}, report, simulation, defaults)


def _validate_identification(case: Case, report: ValidationReport) -> None:
    try:
        from .identification import identification_inputs

        identification_inputs(case)
    except (OSError, TypeError, ValueError) as exc:
        report.add("error", "identification.invalid", str(exc), "$.identification")


def _validate_target_waveform(
    case: Case,
    target: dict[str, Any],
    report: ValidationReport,
    simulation: ResolvedSimulationCase | None,
    defaults: dict[str, Any] | None,
) -> None:
    if target.get("waveform_file") is None:
        report.add("error", "target.no_waveform", "target.waveform_file is required", "$.target.waveform_file")
        return
    from .metrics import load_target_waveform

    try:
        waveform = load_target_waveform(case)
    except (OSError, TypeError, ValueError) as exc:
        report.add("error", "target.invalid_waveform", str(exc), "$.target.waveform_file")
        return
    if simulation is None:
        return
    transient = simulation.analysis.transient
    if transient is None:
        report.add("error", "target.waveform_without_tran", "waveform_l2 requires transient analysis", "$.solver")
        return
    declared = _mapping_or_empty(_mapping_or_empty(case.data.get("solver")).get("tran")).get("stop_s")
    if isinstance(declared, str) and declared.removeprefix("$") in (defaults or {}):
        return  # A scenario, control, or candidate may change the resolved duration.
    start_s, stop_s = float(waveform["time_s"].iloc[0]), float(waveform["time_s"].iloc[-1])
    if start_s < 0 or (stop_s > transient.stop_s and not math.isclose(stop_s, transient.stop_s, rel_tol=1e-14)):
        report.add(
            "error",
            "target.waveform_outside_transient",
            "target waveform must lie inside the declared transient time span",
            "$.target.waveform_file",
        )
