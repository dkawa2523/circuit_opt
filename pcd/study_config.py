"""Translate advanced case study inputs into Candidate/Scenario/Control roles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .case import Case, default_params, resolve_path
from .core.models import Candidate, ControlState, Objective, Scenario, StudySpec
from .core.spaces import grid_values, parameter_grid
from .inputs import condition_value, integer, load_plane_label, reject_unknown
from .problem import ParameterRole, ParameterSet, resolve_parameter_set, validate_parameter_value
from .simulation_input import ResolvedSimulationCase
from .tables import read_csv_rows


def mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _scenario_specs(case: Case, parameters: ParameterSet | None = None) -> tuple[Scenario, ...]:
    parameters = resolve_parameter_set(case) if parameters is None else parameters
    specs = {item.name: item.spec for item in parameters.definitions}
    study_cfg = mapping(case.data.get("study"), "study")
    raw = study_cfg.get("scenarios")
    table = study_cfg.get("scenario_table")
    if raw is not None and table is not None:
        raise ValueError("study.scenarios and study.scenario_table are mutually exclusive")
    if table is not None:
        return _scenario_table_specs(case, mapping(table, "study.scenario_table"), specs)
    if raw is None:
        return (Scenario("nominal"),)
    if not isinstance(raw, list) or not raw:
        raise ValueError("study.scenarios must be a non-empty list")
    scenarios: list[Scenario] = []
    for index, item in enumerate(raw):
        cfg = mapping(item, f"study.scenarios[{index}]")
        reject_unknown(cfg, {"id", "scenario_id", "values", "weight", "controls"}, f"study.scenarios[{index}]")
        if "id" in cfg and "scenario_id" in cfg:
            raise ValueError(f"study.scenarios[{index}] must use only one of id or scenario_id")
        scenario_id = str(cfg.get("id", cfg.get("scenario_id", f"scenario_{index:03d}")))
        values = {
            name: condition_value(value, name, specs.get(name, {}), f"study.scenarios[{index}].values.{name}")
            for name, value in mapping(cfg.get("values"), f"study.scenarios[{index}].values").items()
        }
        scenarios.append(Scenario(scenario_id, values, float(cfg.get("weight", 1.0))))
    return tuple(scenarios)


def _scenario_table_specs(
    case: Case, cfg: dict[str, Any], specs: Mapping[str, Mapping[str, Any]]
) -> tuple[Scenario, ...]:
    reject_unknown(cfg, {"table_file", "values", "id_column", "weight_column", "defaults"}, "study.scenario_table")
    raw_path = cfg.get("table_file")
    if not raw_path:
        raise ValueError("study.scenario_table.table_file is required")
    value_columns = mapping(cfg.get("values"), "study.scenario_table.values")
    if not value_columns and not (case.data.get("study", {}).get("controls") or {}).get("by_scenario"):
        raise ValueError("study.scenario_table.values must map parameter names to CSV columns")
    id_column = str(cfg.get("id_column", "scenario_id"))
    weight_column = str(cfg.get("weight_column", "weight"))
    path = resolve_path(case, raw_path)
    required = set(map(str, value_columns.values()))
    if "id_column" in cfg:
        required.add(id_column)
    if "weight_column" in cfg:
        required.add(weight_column)
    _columns, rows = read_csv_rows(path, required)
    defaults = mapping(cfg.get("defaults"), "study.scenario_table.defaults")
    scenarios: list[Scenario] = []
    for index, row in enumerate(rows, start=2):
        scenario_id = row[id_column].strip() if id_column in row else f"point_{index - 1}"
        if not scenario_id:
            raise ValueError(f"scenario table row {index} has an empty {id_column}")
        values = {
            name: condition_value(value, name, specs.get(name, {}), f"{path}:{index}:{name}")
            for name, value in defaults.items()
            if name not in value_columns
        }
        for name, column in value_columns.items():
            raw_value = row[str(column)].strip()
            if not raw_value:
                raise ValueError(f"scenario value is empty at {path}:{index}:{column}")
            values[name] = condition_value(raw_value, name, specs.get(name, {}), f"{path}:{index}:{column}")
        weight_raw = row.get(weight_column, "").strip()
        if weight_column in row and not weight_raw:
            raise ValueError(f"scenario weight is empty at {path}:{index}:{weight_column}")
        scenarios.append(Scenario(scenario_id, values, float(weight_raw) if weight_raw else 1.0))
    if not scenarios:
        raise ValueError(f"scenario table is empty: {path}")
    return tuple(scenarios)


def _objective_specs(case: Case) -> tuple[Objective, ...]:
    study_cfg = mapping(case.data.get("study"), "study")
    raw = study_cfg.get("objectives")
    if raw is None:
        return (
            Objective(
                metric="loss",
                direction="minimize",
                aggregation=str(study_cfg.get("aggregation", "worst")),
                cvar_alpha=float(study_cfg.get("cvar_alpha", 0.1)),
            ),
        )
    if not isinstance(raw, list) or not raw:
        raise ValueError("study.objectives must be a non-empty list")
    return tuple(Objective.from_dict(mapping(item, f"study.objectives[{index}]")) for index, item in enumerate(raw))


def _electrical_context(case: Case) -> dict[str, Any]:
    """Return the small physical context needed to interpret a study result."""

    solver = mapping(case.data.get("solver"), "solver")
    measurement = mapping(case.data.get("measurement"), "measurement")
    load = mapping(case.data.get("load"), "load")
    analyses = [name for name in ("ac", "tran") if name in solver]
    context: dict[str, Any] = {
        "analysis": "+".join("transient" if name == "tran" else name for name in analyses) or "unspecified",
        "reference_plane": load_plane_label(load.get("reference_plane")),
        "reference_impedance_ohm": float(measurement.get("reference_impedance_ohm", 50.0)),
    }

    ac = mapping(solver.get("ac"), "solver.ac")
    source = mapping(case.data.get("source"), "source")
    raw_frequency = ac.get("frequency_Hz", source.get("frequency_Hz"))
    if isinstance(raw_frequency, int | float) and not isinstance(raw_frequency, bool):
        frequency = float(raw_frequency)
        if math.isfinite(frequency) and frequency > 0:
            context["frequency_Hz"] = frequency
    return context


def study_spec_from_case(case: Case, *, parameters: ParameterSet | None = None) -> StudySpec:
    study_cfg = mapping(case.data.get("study"), "study")
    target_cfg = mapping(case.data.get("target"), "target")
    if parameters is None:
        parameters = resolve_parameter_set(case)
    analysis_mode = str(study_cfg.get("analysis_mode", "")).strip()
    reject_unknown(
        study_cfg,
        {
            "scenarios",
            "scenario_table",
            "objectives",
            "aggregation",
            "cvar_alpha",
            "controls",
            "control_margin_min",
            "design_variables",
            "candidate_role",
            "candidate_enumeration",
            "analysis_mode",
        },
        "study",
    )
    if study_cfg.get("candidate_enumeration", "exact") != "exact":
        raise ValueError("study.candidate_enumeration must be exact when declared")
    if "objectives" in study_cfg and ({"aggregation", "cvar_alpha"} & set(study_cfg)):
        raise ValueError("put aggregation and cvar_alpha inside each declared objective")
    return StudySpec(
        study_id=case.case_id,
        scenarios=_scenario_specs(case, parameters),
        objectives=_objective_specs(case),
        control_margin_min=(
            float(study_cfg["control_margin_min"]) if study_cfg.get("control_margin_min") is not None else None
        ),
        metadata={
            "case_path": str(case.path),
            "case_schema": str(case.authored_data.get("schema", "case_yaml.v1")),
            "resolved_case_schema": str(case.data.get("schema", "case_yaml.v1")),
            "objective_adapter": str(target_cfg.get("objective", "waveform_l2")),
            "parameter_roles": parameters.role_names(),
            **_electrical_context(case),
            **({"analysis_mode": analysis_mode} if analysis_mode else {}),
        },
    )


def select_simulation_inputs(
    case: Case,
    params: Mapping[str, Any] | None = None,
    *,
    scenario_id: str | None = None,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one physical point explicitly when a case describes several."""

    supplied = dict(params or {})
    parameters = resolve_parameter_set(case)
    definitions = parameters.definitions
    specs = {item.name: item.spec for item in definitions}
    if unknown := sorted(set(supplied) - set(specs)):
        raise ValueError(f"unknown simulation parameters {unknown}")
    study = study_spec_from_case(case, parameters=parameters)
    if scenario_id is None and len(study.scenarios) != 1:
        raise ValueError("sim-run needs --scenario for a case with multiple conditions; use run for the complete study")
    scenario = next((row for row in study.scenarios if scenario_id is None or row.scenario_id == scenario_id), None)
    if scenario is None:
        raise ValueError(f"unknown scenario {scenario_id!r}")
    for item in definitions:
        if item.role not in {ParameterRole.DESIGN, ParameterRole.LATENT} or item.name in supplied:
            continue
        spec = item.spec
        values = spec.get("choices", spec.get("values"))
        bounds = spec.get("bounds")
        if (values is not None and len(values) > 1) or (bounds is not None and bounds[0] != bounds[1]):
            raise ValueError(
                f"sim-run needs an explicit value for {item.name!r} via --params; use run for candidate search"
            )
        if values:
            supplied[item.name] = values[0]
        elif bounds:
            supplied[item.name] = bounds[0]
    policy = CaseControlPolicy(case, study.scenarios)
    available = policy.controls(study, Candidate("single_point", supplied), scenario)
    requested_controls = dict(controls or {})
    matches = [
        control
        for control in available
        if all(control.values.get(name) == value for name, value in requested_controls.items())
    ]
    if not matches:
        raise ValueError("requested simulation controls do not belong to the declared condition/control grid")
    if len(matches) != 1:
        raise ValueError("sim-run needs --controls for a case with multiple control states; use run for control search")
    selected = dict(matches[0].values)
    if overlap := sorted(set(supplied) & (set(scenario.values) | set(selected))):
        raise ValueError(f"select operating/control inputs through --scenario/--controls, not --params: {overlap}")
    resolved = {**default_params(case), **supplied, **scenario.values, **selected}
    for name, spec in specs.items():
        if name not in resolved:
            raise ValueError(f"simulation parameter {name!r} has no selected value")
        resolved[name] = validate_parameter_value(name, resolved[name], spec)
    return resolved


def describe_case(case: Case) -> dict[str, Any]:
    """Describe the declared study and the default simulation point."""

    from .netlist import build_netlist_inputs
    from .simulation_input import bind_load_probe, resolve_simulation_case

    study = study_spec_from_case(case)
    defaults = default_params(case)
    simulation = resolve_simulation_case(case, defaults)
    netlist = build_netlist_inputs(case, defaults)
    simulation = replace(simulation, probes=bind_load_probe(simulation.probes, attached=bool(netlist.load_subckt)))
    return case_plan_summary(
        case, study, control_state_counts(case, study), defaults, simulation, netlist.circuit.output_node
    )


def control_state_counts(case: Case, study: StudySpec) -> dict[str, int]:
    policy = CaseControlPolicy(case, study.scenarios)
    candidate = Candidate("preview")
    return {scenario.scenario_id: len(policy.controls(study, candidate, scenario)) for scenario in study.scenarios}


def case_plan_summary(
    case: Case,
    study: StudySpec,
    counts: dict[str, int],
    defaults: dict[str, Any],
    simulation: ResolvedSimulationCase,
    output_node: str,
) -> dict[str, Any]:
    from .artifacts import data_file_references

    probes = simulation.probes
    return {
        "output": dict(case.data.get("output") or {}),
        "workflow": "simulation"
        if case.is_forward_simulation
        else ("identification" if case.data.get("identification") is not None else "study"),
        "parameter_roles": study.metadata["parameter_roles"],
        "condition_ids": [scenario.scenario_id for scenario in study.scenarios],
        "control_states_per_condition": counts,
        "candidate_budget": (case.data.get("run") or {}).get("trials", 1),
        "complete_candidate_enumeration": case.has_exact_candidate_enumeration,
        "solver": dict(case.data.get("solver") or {}),
        "measurement": {
            **dict(case.data.get("measurement") or {}),
            "current_source": probes.source_name,
            "source_voltage_vector": probes.source_voltage_vector,
            "load_current_column": probes.load_current_column,
            "voltage_node": simulation.measurement.voltage_node or output_node,
            "load_ports": {
                "p": simulation.measurement.load_positive or output_node,
                "n": simulation.measurement.load_negative,
            },
            "load_reference_plane": simulation.measurement.reference_plane,
        },
        "preview_params": defaults,
        "preview_scope": "default point only; sim-run requires selection when multiple points are declared",
        "analysis_mode": study.metadata.get("analysis_mode", "circuit_response"),
        "dependencies": [str(resolve_path(case, path)) for _field, path in data_file_references(case.data)],
        "inferences": list((case.resolved_plan or {}).get("inferences") or []),
    }


class CaseControlPolicy:
    """Expand tunable controls separately from each candidate design."""

    def __init__(self, case: Case, scenarios: tuple[Scenario, ...] | None = None) -> None:
        study_cfg = mapping(case.data.get("study"), "study")
        control_cfg = mapping(study_cfg.get("controls"), "study.controls")
        reject_unknown(control_cfg, {"defaults", "variables", "budget", "by_scenario"}, "study.controls")
        self.defaults = mapping(control_cfg.get("defaults"), "study.controls.defaults")
        self.variables = {
            str(name): mapping(spec, f"study.controls.variables.{name}")
            for name, spec in mapping(control_cfg.get("variables"), "study.controls.variables").items()
        }
        self.budget = integer(control_cfg.get("budget", 27), "study.controls.budget", minimum=1)
        self.axis_values = {}
        self.numeric_ranges = {}
        for name, spec in self.variables.items():
            if spec.get("role", "control") != "control":
                raise ValueError(f"study.controls.variables.{name}.role must be control")
            values = grid_values(spec)
            self.axis_values[name] = values
            if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
                self.numeric_ranges[name] = (min(values), max(values))
        self.by_scenario = {
            str(name): mapping(values, f"study.controls.by_scenario.{name}")
            for name, values in mapping(control_cfg.get("by_scenario"), "study.controls.by_scenario").items()
        }
        for index, item in enumerate(study_cfg.get("scenarios") or []):
            cfg = mapping(item, f"study.scenarios[{index}]")
            scenario_id = str(cfg.get("id", cfg.get("scenario_id", f"scenario_{index:03d}")))
            inline = mapping(cfg.get("controls"), f"study.scenarios[{index}].controls")
            if inline:
                self.by_scenario[scenario_id] = {**self.by_scenario.get(scenario_id, {}), **inline}
        ids = {scenario.scenario_id for scenario in (_scenario_specs(case) if scenarios is None else scenarios)}
        if unknown := sorted(set(self.by_scenario) - ids):
            raise ValueError(f"study.controls.by_scenario references unknown scenarios {unknown}")

    def controls(
        self,
        study: StudySpec,
        candidate: Candidate,
        scenario: Scenario,
    ) -> tuple[ControlState, ...]:
        del study, candidate
        observed = {
            name: condition_value(
                value, name, self.variables[name], f"scenario {scenario.scenario_id!r} control {name!r}"
            )
            if name in self.variables
            else value
            for name, value in self.by_scenario.get(scenario.scenario_id, {}).items()
        }
        # Observed condition settings pin axes; they must never be replaced
        # by a more favorable grid setting during evaluation or identification.
        for name, value in observed.items():
            if name in self.axis_values and value not in self.axis_values[name]:
                raise ValueError(f"scenario {scenario.scenario_id!r} control {name!r}={value!r} is outside its grid")
        remaining = {name: values for name, values in self.axis_values.items() if name not in observed}
        grid = parameter_grid(remaining, budget=self.budget)
        return tuple(ControlState({**self.defaults, **point, **observed}, self.numeric_ranges) for point in grid)
