"""Resolve one case into the small typed input needed by a solver run.

Only this adapter reads the case's solver and measurement declarations for
simulation execution.  Circuit and load construction remain separate; this is
deliberately not a dataclass containing every possible case feature.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case import Case
from .component_models import observed_components
from .probes import LOAD_AMMETER, RESERVED_PROBE_COLUMNS, NamedProbe, ProbePlan
from .simulation import LOAD_CURRENT_COLUMN, SOURCE_VOLTAGE_COLUMN, AnalysisRequest

DEFAULT_SOLVER_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class SolverSettings:
    """Execution settings shared by provenance and a solver adapter."""

    name: str = "ngspice_cli"
    executable: str | None = None
    timeout_s: float = DEFAULT_SOLVER_TIMEOUT_S

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("solver.name must not be empty")
        timeout = self.timeout_s
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("solver.timeout_s must be positive and finite")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class MeasurementReference:
    """Resolved output port and named electrical reference plane."""

    voltage_node: str | None = None
    load_positive: str | None = None
    load_negative: str = "0"
    reference_plane: str = "load_ports"


@dataclass(frozen=True)
class ResolvedSimulationCase:
    """The solver-facing subset of a case, resolved for one design point."""

    solver: SolverSettings
    analysis: AnalysisRequest
    probes: ProbePlan
    measurement: MeasurementReference
    netlist_options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class SolverRunRequest:
    """Paths and typed settings supplied to a modern solver adapter."""

    netlist_path: Path
    run_dir: Path
    simulation: ResolvedSimulationCase


def _mapping_section(case: Case, name: str) -> Mapping[str, Any]:
    config = case.data.get(name)
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return config


def _solver_config(case: Case) -> Mapping[str, Any]:
    return _mapping_section(case, "solver")


def _solver_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("solver.timeout_s must be positive and finite") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("solver.timeout_s must be positive and finite")
    return timeout


def resolve_solver_settings(case: Case, solver_override: str | None = None) -> SolverSettings:
    """Resolve solver selection once without interpreting analysis details."""

    config = _solver_config(case)
    executable = config.get("executable")
    return SolverSettings(
        name=str(solver_override or config.get("name", "ngspice_cli")),
        executable=str(executable) if executable else None,
        timeout_s=_solver_timeout(config.get("timeout_s", DEFAULT_SOLVER_TIMEOUT_S)),
    )


def resolve_analysis_request(
    case: Case,
    params: Mapping[str, Any] | None = None,
) -> AnalysisRequest:
    """Resolve and validate the solver's analysis declaration once."""

    return AnalysisRequest.from_config(_solver_config(case), params)


def resolve_source_specs(case: Case) -> tuple[dict[str, Any], ...]:
    """Resolve the mutually exclusive single/list source declarations once."""

    declared = case.data.get("sources")
    source = case.data.get("source")
    if declared is not None and source is not None:
        raise ValueError("use either source or sources, not both")
    if declared is not None:
        if not isinstance(declared, list):
            raise TypeError("sources must be a list of mappings")
        sources: list[dict[str, Any]] = []
        for index, source in enumerate(declared):
            if not isinstance(source, Mapping):
                raise TypeError(f"sources[{index}] must be a mapping")
            sources.append(dict(source))
        return tuple(sources)

    if source is None:
        return ()
    if not isinstance(source, Mapping):
        raise TypeError("source must be a mapping")
    return (dict(source),)


def _effective_source_name(source: Mapping[str, Any], index: int) -> str:
    return str(source.get("name", "Vsrc" if index == 0 else f"Vsrc{index}"))


def _source_name(sources: tuple[dict[str, Any], ...], measurement: Mapping[str, Any]) -> str:
    structured = [
        (_effective_source_name(source, index), source) for index, source in enumerate(sources) if "raw" not in source
    ]
    names = [name for name, _source in structured]
    if any(not name.strip() for name in names):
        raise ValueError("structured source names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("structured source names must be unique")
    requested = str(measurement.get("current_source", "")).strip()
    if requested:
        if names and requested not in names:
            raise ValueError(f"measurement.current_source {requested!r} is not a declared structured source")
        return requested
    return names[0] if names else "Vsrc"


def _active_source(sources: tuple[dict[str, Any], ...], name: str) -> Mapping[str, Any] | None:
    for index, source in enumerate(sources):
        if "raw" not in source and _effective_source_name(source, index) == name:
            return source
    return None


def _source_voltage_vector(sources: tuple[dict[str, Any], ...], name: str) -> str:
    source = _active_source(sources, name)
    if source is None:
        return "v(src)"
    positive, negative = str(source.get("p", "src")), str(source.get("n", "0"))
    return f"v({positive})" if negative == "0" else f"v({positive},{negative})"


def _load_current_column(measurement: Mapping[str, Any]) -> str | None:
    declared = measurement.get("load_current")
    if not declared:
        return None
    return LOAD_CURRENT_COLUMN if str(declared).lower() == "auto" else str(declared)


def _declared_probes(case: Case, measurement: Mapping[str, Any]) -> tuple[NamedProbe, ...]:
    raw = measurement.get("probes")
    if raw is None:
        raw = []
    if isinstance(raw, Mapping):
        pairs = [(str(vector), str(name)) for name, vector in raw.items()]
    elif isinstance(raw, list):
        pairs = [(str(vector), str(vector)) for vector in raw]
    else:
        raise ValueError("measurement.probes must be a list or a name-to-vector mapping")

    for component in observed_components(case):
        pairs.extend(
            [
                (component.voltage_vector, component.voltage_column),
                (component.current_vector, component.current_column),
            ]
        )
    by_column: dict[str, str] = {}
    for vector, column in pairs:
        if column in RESERVED_PROBE_COLUMNS:
            raise ValueError(f"probe column {column!r} is reserved by the waveform/AC result format")
        if column in by_column and by_column[column] != vector:
            raise ValueError(f"probe column {column!r} names both {by_column[column]!r} and {vector!r}")
        by_column[column] = vector
    return tuple(NamedProbe(vector, column) for column, vector in by_column.items())


def _build_probe_plan(
    case: Case,
    sources: tuple[dict[str, Any], ...],
    measurement: Mapping[str, Any],
) -> ProbePlan:
    source = _source_name(sources, measurement)
    declared = list(_declared_probes(case, measurement))
    load_current = _load_current_column(measurement)
    if load_current == LOAD_CURRENT_COLUMN:
        declared.append(NamedProbe(f"i({LOAD_AMMETER})", LOAD_CURRENT_COLUMN))
    ac = tuple(declared)
    source_voltage = _source_voltage_vector(sources, source)
    transient = (*ac, NamedProbe(source_voltage, SOURCE_VOLTAGE_COLUMN))
    return ProbePlan(
        source_name=source,
        source_voltage_vector=source_voltage,
        load_current_column=load_current,
        transient=transient,
        ac=ac,
    )


def build_probe_plan(case: Case) -> ProbePlan:
    """Compile all observation declarations exactly once for a case."""

    return _build_probe_plan(
        case,
        resolve_source_specs(case),
        _mapping_section(case, "measurement"),
    )


def _measurement_reference(
    measurement: Mapping[str, Any],
    load: Mapping[str, Any],
) -> MeasurementReference:
    ports = load.get("ports")
    if ports is None:
        ports = {}
    if not isinstance(ports, Mapping):
        raise TypeError("load.ports must be a mapping")
    return MeasurementReference(
        voltage_node=str(measurement["voltage_node"]) if "voltage_node" in measurement else None,
        load_positive=str(ports["p"]) if "p" in ports else None,
        load_negative=str(ports.get("n", "0")),
        reference_plane=str(load.get("reference_plane", "load_ports")),
    )


def resolve_simulation_case(
    case: Case,
    params: Mapping[str, Any] | None = None,
    solver_override: str | None = None,
) -> ResolvedSimulationCase:
    """Compile the solver-facing case subset at the orchestration boundary."""

    config = _solver_config(case)
    options = config.get("options")
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("solver.options must be a mapping")
    solver = resolve_solver_settings(case, solver_override)
    analysis = resolve_analysis_request(case, params)
    sources = resolve_source_specs(case)
    measurement = _mapping_section(case, "measurement")
    load = _mapping_section(case, "load")
    return ResolvedSimulationCase(
        solver=solver,
        analysis=analysis,
        probes=_build_probe_plan(case, sources, measurement),
        measurement=_measurement_reference(measurement, load),
        netlist_options=tuple((str(name), value) for name, value in options.items()),
    )
