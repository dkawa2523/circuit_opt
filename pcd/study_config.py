"""Translate advanced case study inputs into Candidate/Scenario/Control roles."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .case import Case, resolve_path, variable_specs
from .core.models import Candidate, ControlState, Objective, Scenario, StudySpec
from .core.spaces import parameter_grid


def mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _scenario_specs(case: Case) -> tuple[Scenario, ...]:
    study_cfg = mapping(case.data.get("study"), "study")
    raw = study_cfg.get("scenarios")
    table = study_cfg.get("scenario_table")
    if raw is not None and table is not None:
        raise ValueError("study.scenarios and study.scenario_table are mutually exclusive")
    if table is not None:
        return _scenario_table_specs(case, mapping(table, "study.scenario_table"))
    if raw is None:
        return (Scenario("nominal"),)
    if not isinstance(raw, list) or not raw:
        raise ValueError("study.scenarios must be a non-empty list")
    scenarios: list[Scenario] = []
    for index, item in enumerate(raw):
        cfg = mapping(item, f"study.scenarios[{index}]")
        scenario_id = str(cfg.get("id", cfg.get("scenario_id", f"scenario_{index:03d}")))
        values = mapping(cfg.get("values"), f"study.scenarios[{index}].values")
        scenarios.append(Scenario(scenario_id, values, float(cfg.get("weight", 1.0))))
    return tuple(scenarios)


def _scenario_table_specs(case: Case, cfg: dict[str, Any]) -> tuple[Scenario, ...]:
    raw_path = cfg.get("table_file")
    if not raw_path:
        raise ValueError("study.scenario_table.table_file is required")
    value_columns = mapping(cfg.get("values"), "study.scenario_table.values")
    if not value_columns:
        raise ValueError("study.scenario_table.values must map parameter names to CSV columns")
    id_column = str(cfg.get("id_column", "scenario_id"))
    weight_column = str(cfg.get("weight_column", "weight"))
    path = resolve_path(case, raw_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {id_column, *map(str, value_columns.values())}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"scenario table is missing columns {missing}: {path}")
        scenarios: list[Scenario] = []
        for index, row in enumerate(reader):
            scenario_id = str(row.get(id_column, "")).strip()
            if not scenario_id:
                raise ValueError(f"scenario table row {index + 2} has an empty {id_column}")
            values = {
                parameter: _scenario_cell(row[str(column)], f"{path}:{index + 2}:{column}")
                for parameter, column in value_columns.items()
            }
            weight_raw = row.get(weight_column, "") if weight_column in columns else ""
            weight = float(weight_raw) if str(weight_raw).strip() else 1.0
            scenarios.append(Scenario(scenario_id, values, weight))
    if not scenarios:
        raise ValueError(f"scenario table is empty: {path}")
    return tuple(scenarios)


def _scenario_cell(raw: Any, location: str) -> Any:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"scenario value is empty at {location}")
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        raise ValueError(f"scenario value must be finite at {location}")
    return value


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


def study_spec_from_case(case: Case) -> StudySpec:
    study_cfg = mapping(case.data.get("study"), "study")
    if "fidelities" in study_cfg:
        raise ValueError(
            "study.fidelities is no longer supported; select one solver in solver.name "
            "and run separate explicit studies when two numerical methods must be compared"
        )
    return StudySpec(
        study_id=case.case_id,
        scenarios=_scenario_specs(case),
        objectives=_objective_specs(case),
        control_margin_min=(
            float(study_cfg["control_margin_min"]) if study_cfg.get("control_margin_min") is not None else None
        ),
        metadata={
            "case_path": str(case.path),
            "case_schema": str(case.authored_data.get("schema", "case_yaml.v1")),
            "resolved_case_schema": str(case.data.get("schema", "case_yaml.v1")),
            "objective_adapter": str((case.data.get("target", {}) or {}).get("objective", "waveform_l2")),
        },
    )


def _role_variable_names(case: Case) -> set[str]:
    study_cfg = mapping(case.data.get("study"), "study")
    names: set[str] = set()
    for index, item in enumerate(study_cfg.get("scenarios") or []):
        cfg = mapping(item, f"study.scenarios[{index}]")
        names.update(mapping(cfg.get("values"), f"study.scenarios[{index}].values"))
        names.update(mapping(cfg.get("controls"), f"study.scenarios[{index}].controls"))
    scenario_table = mapping(study_cfg.get("scenario_table"), "study.scenario_table")
    names.update(mapping(scenario_table.get("values"), "study.scenario_table.values"))
    controls = mapping(study_cfg.get("controls"), "study.controls")
    names.update(mapping(controls.get("defaults"), "study.controls.defaults"))
    names.update(mapping(controls.get("variables"), "study.controls.variables"))
    for scenario_id, values in mapping(controls.get("by_scenario"), "study.controls.by_scenario").items():
        names.update(mapping(values, f"study.controls.by_scenario.{scenario_id}"))
    return names


def candidate_case(case: Case) -> Case:
    """Project all variable declarations onto fixed-design variables only."""

    study_cfg = mapping(case.data.get("study"), "study")
    declared = variable_specs(case)
    explicit = study_cfg.get("design_variables")
    if isinstance(explicit, list):
        missing = sorted({str(item) for item in explicit} - set(declared))
        if missing:
            raise ValueError(f"study.design_variables are not declared: {missing}")
        specs = {str(name): declared[str(name)] for name in explicit}
    elif isinstance(explicit, Mapping):
        specs = {str(name): mapping(spec, f"study.design_variables.{name}") for name, spec in explicit.items()}
    elif explicit is not None:
        raise ValueError("study.design_variables must be a list or mapping")
    else:
        role_names = _role_variable_names(case)
        specs = {name: spec for name, spec in declared.items() if name not in role_names}

    data = deepcopy(case.data)
    data["variables"] = specs
    for section in ("source", "circuit", "load"):
        if isinstance(data.get(section), dict):
            data[section].pop("variables", None)
    for source in data.get("sources") or []:
        if isinstance(source, dict):
            source.pop("variables", None)
    return Case(path=case.path, data=data)


class CaseControlPolicy:
    """Expand explicitly tunable controls separately from fixed designs."""

    def __init__(self, case: Case) -> None:
        study_cfg = mapping(case.data.get("study"), "study")
        control_cfg = mapping(study_cfg.get("controls"), "study.controls")
        self.defaults = mapping(control_cfg.get("defaults"), "study.controls.defaults")
        self.variables = {
            str(name): mapping(spec, f"study.controls.variables.{name}")
            for name, spec in mapping(control_cfg.get("variables"), "study.controls.variables").items()
        }
        self.budget = int(control_cfg.get("budget", 27))
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

    def controls(
        self,
        study: StudySpec,
        candidate: Candidate,
        scenario: Scenario,
    ) -> tuple[ControlState, ...]:
        del study, candidate
        fixed = {**self.defaults, **self.by_scenario.get(scenario.scenario_id, {})}
        grid = parameter_grid(self.variables, budget=self.budget)
        return tuple(ControlState({**fixed, **point}) for point in grid)
