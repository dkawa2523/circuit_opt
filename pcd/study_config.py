"""Translate advanced case study inputs into Candidate/Scenario/Control roles."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from typing import Any

from .case import Case, resolve_path
from .core.models import Candidate, ControlState, Objective, Scenario, StudySpec
from .core.spaces import parameter_grid
from .problem import project_candidate_case, resolve_parameter_set


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


def _electrical_context(case: Case) -> dict[str, Any]:
    """Return the small physical context needed to interpret a study result."""

    solver = mapping(case.data.get("solver"), "solver")
    measurement = mapping(case.data.get("measurement"), "measurement")
    load = mapping(case.data.get("load"), "load")
    analyses = [name for name in ("ac", "tran") if name in solver]
    context: dict[str, Any] = {
        "analysis": "+".join("transient" if name == "tran" else name for name in analyses) or "unspecified",
        "reference_plane": str(load.get("reference_plane", "load_ports")),
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


def study_spec_from_case(case: Case) -> StudySpec:
    study_cfg = mapping(case.data.get("study"), "study")
    target_cfg = mapping(case.data.get("target"), "target")
    parameters = resolve_parameter_set(case)
    analysis_mode = str(study_cfg.get("analysis_mode", "")).strip()
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
            "objective_adapter": str(target_cfg.get("objective", "waveform_l2")),
            "parameter_roles": parameters.role_names(),
            **_electrical_context(case),
            **({"analysis_mode": analysis_mode} if analysis_mode else {}),
        },
    )


def candidate_case(case: Case) -> Case:
    """Project a case onto the parameters that an optimizer may change."""

    return project_candidate_case(case)


class CaseControlPolicy:
    """Expand tunable controls separately from each candidate design."""

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
