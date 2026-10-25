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
from .inputs import reject_unknown
from .probes import LOAD_AMMETER, RESERVED_PROBE_COLUMNS, NamedProbe, ProbePlan
from .simulation import LOAD_CURRENT_COLUMN, SOURCE_VOLTAGE_COLUMN, AnalysisRequest
from .sources import ac_source_name, source_vectors

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
    if name == "measurement":
        reject_unknown(
            config,
            {
                "voltage_node",
                "current_source",
                "source_voltage",
                "load_current",
                "probes",
                "reference_impedance_ohm",
                "periodic_cycles",
                "settling_comparisons",
                "settling_tolerance",
                "harmonic_count",
            },
            "measurement",
        )
        for field in ("voltage_node", "current_source", "source_voltage", "load_current"):
            if field in config and (not isinstance(config[field], str) or not config[field].strip()):
                raise ValueError(f"measurement.{field} must be a nonempty string")
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
        name=str(config.get("name", "ngspice_cli") if solver_override is None else solver_override),
        executable=str(executable) if executable else None,
        timeout_s=_solver_timeout(config.get("timeout_s", DEFAULT_SOLVER_TIMEOUT_S)),
    )


def resolve_analysis_request(
    case: Case,
    params: Mapping[str, Any] | None = None,
) -> AnalysisRequest:
    """Resolve and validate the solver's analysis declaration once."""

    return AnalysisRequest.from_config(_solver_config(case), params)


def _declared_probes(case: Case, measurement: Mapping[str, Any]) -> tuple[NamedProbe, ...]:
    raw = measurement.get("probes")
    if raw is None:
        raw = []
    pairs: list[tuple[Any, str]]
    if isinstance(raw, Mapping):
        pairs = [(vector, str(name)) for name, vector in raw.items()]
    elif isinstance(raw, list):
        pairs = [(str(vector), str(vector)) for vector in raw]
    else:
        raise ValueError("measurement.probes must be a list or a name-to-vector mapping")

    for component in observed_components(case):
        pairs.extend(
            [
                (
                    {
                        "vector": component.voltage_vector,
                        "unit": "V",
                        "label": f"{component.reference} voltage ({component.p} to {component.n})",
                    },
                    component.voltage_column,
                ),
                (
                    {
                        "vector": component.current_vector,
                        "unit": "A",
                        "label": f"{component.reference} current ({component.p} to {component.n})",
                    },
                    component.current_column,
                ),
            ]
        )
    by_column: dict[str, NamedProbe] = {}
    for raw_vector, column in pairs:
        if not column.strip() or any(char in column for char in "\r\n"):
            raise ValueError("probe names must be nonempty single-line column names")
        config = dict(raw_vector) if isinstance(raw_vector, Mapping) else {"vector": raw_vector}
        reject_unknown(config, {"vector", "unit", "label"}, f"measurement.probes.{column}")
        vector = config.get("vector")
        if not isinstance(vector, str) or not vector.strip() or any(char in vector for char in "\r\n;"):
            raise ValueError(f"probe {column!r} requires one nonempty SPICE vector")
        probe = NamedProbe(vector.strip(), column, config.get("unit"), config.get("label"))
        if probe.label is not None and (not isinstance(probe.label, str) or not probe.label.strip()):
            raise ValueError(f"probe {column!r} label must be a nonempty string")
        if probe.unit is not None and (not isinstance(probe.unit, str) or probe.unit != probe.resolved_unit):
            raise ValueError(f"probe {column!r} unit does not match its voltage/current vector")
        if probe.resolved_unit is None:
            raise ValueError(f"probe {column!r} needs an explicit unit for a general SPICE expression")
        if column in RESERVED_PROBE_COLUMNS:
            raise ValueError(f"probe column {column!r} is reserved by the waveform/AC result format")
        if column in by_column and by_column[column].vector != vector:
            raise ValueError(f"probe column {column!r} names both {by_column[column]!r} and {vector!r}")
        by_column[column] = probe
    return tuple(by_column.values())


def _build_probe_plan(
    case: Case,
    measurement: Mapping[str, Any],
) -> ProbePlan:
    source, source_voltage, source_current = source_vectors(case)
    declared = list(_declared_probes(case, measurement))
    load_current = measurement.get("load_current")
    if load_current in {"auto", LOAD_CURRENT_COLUMN}:
        load = _mapping_section(case, "load")
        if load.get("name", "none") == "none":
            raise ValueError(
                "measurement.load_current: auto requires a declared load; use a named current probe for an embedded load"
            )
        load_current = LOAD_CURRENT_COLUMN
        declared.append(NamedProbe(f"i({LOAD_AMMETER})", LOAD_CURRENT_COLUMN))
    elif load_current is not None:
        probe = next((probe for probe in declared if probe.column == load_current), None)
        if probe is None or not probe.is_current:
            raise ValueError(
                f"measurement.load_current {load_current!r} must name a declared current probe (i(branch) or @device[i]); "
                "its positive direction must enter the load at the declared load ports"
            )
    ac = tuple(declared)
    transient = (*ac, NamedProbe(source_voltage, SOURCE_VOLTAGE_COLUMN))
    return ProbePlan(
        source_name=source,
        source_voltage_vector=source_voltage,
        source_current_vector=source_current,
        load_current_column=load_current,
        transient=transient,
        ac=ac,
    )


def build_probe_plan(case: Case) -> ProbePlan:
    """Compile all observation declarations exactly once for a case."""

    return _build_probe_plan(
        case,
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
    if analysis.ac is not None:
        ac_source_name(case, analysis.ac.source)
    measurement = _mapping_section(case, "measurement")
    load = _mapping_section(case, "load")
    return ResolvedSimulationCase(
        solver=solver,
        analysis=analysis,
        probes=_build_probe_plan(case, measurement),
        measurement=_measurement_reference(measurement, load),
        netlist_options=tuple((str(name), value) for name, value in options.items()),
    )
