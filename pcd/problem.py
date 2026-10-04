"""Resolve case parameters into one explicit, non-overlapping role set.

The executable case remains the single circuit problem description.  This
module adds the one piece that a plain mapping cannot express reliably: who
owns each parameter.  Design search, terminal identification, operating
scenarios, and controls consume role-filtered views of the same set instead
of reinterpreting YAML dictionaries independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .case import Case


class ParameterRole(str, Enum):
    """The single owner of a parameter during a circuit study."""

    FIXED = "fixed"
    CALIBRATION = "calibration"
    OPERATING = "operating"
    CONTROL = "control"
    DESIGN = "design"
    LATENT = "latent"


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """One named parameter, its owner, and its existing case specification."""

    name: str
    role: ParameterRole
    spec: dict[str, Any]
    declared_at: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        object.__setattr__(self, "spec", deepcopy(self.spec))


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """Role-explicit parameters resolved once from an executable case."""

    definitions: tuple[ParameterDefinition, ...]

    def names(self, role: ParameterRole) -> tuple[str, ...]:
        return tuple(item.name for item in self.definitions if item.role is role)

    def specs(self, role: ParameterRole) -> dict[str, dict[str, Any]]:
        return {item.name: deepcopy(item.spec) for item in self.definitions if item.role is role}

    def defaults(self, role: ParameterRole) -> dict[str, Any]:
        return {
            item.name: deepcopy(item.spec["default"])
            for item in self.definitions
            if item.role is role and "default" in item.spec
        }

    def role_names(self) -> dict[str, list[str]]:
        """Return a small result-ready summary in stable role order."""

        return {role.value: list(self.names(role)) for role in ParameterRole}


def candidate_role(case: Case) -> ParameterRole:
    """Return the parameter role proposed by this case's optimizer."""

    study = _mapping(case.data.get("study"), "study")
    raw = str(study.get("candidate_role", ParameterRole.DESIGN.value))
    try:
        role = ParameterRole(raw)
    except ValueError as exc:
        raise ValueError("study.candidate_role must be 'design' or 'latent'") from exc
    if role not in {ParameterRole.DESIGN, ParameterRole.LATENT}:
        raise ValueError("study.candidate_role must be 'design' or 'latent'")
    return role


def project_candidate_case(case: Case, role: ParameterRole | None = None) -> Case:
    """Return the optimizer-facing case containing one candidate role only."""

    from .case import Case

    selected = role or candidate_role(case)
    if selected not in {ParameterRole.DESIGN, ParameterRole.LATENT}:
        raise ValueError("optimizer candidates must use the design or latent role")
    data = deepcopy(case.data)
    data["variables"] = resolve_parameter_set(case).specs(selected)
    for section in ("source", "circuit", "load"):
        if isinstance(data.get(section), dict):
            data[section].pop("variables", None)
    for source in data.get("sources") or []:
        if isinstance(source, dict):
            source.pop("variables", None)
    study = _mapping(data.get("study"), "study")
    study["candidate_role"] = selected.value
    if selected is ParameterRole.LATENT:
        study.pop("design_variables", None)
    data["study"] = study
    return Case(path=case.path, data=data)


_VARIABLE_SECTIONS = ("variables", "source", "circuit", "load")


def declared_variable_specs(case: Case) -> dict[str, dict[str, Any]]:
    """Collect declared parameter specs and reject ambiguous redeclarations."""

    specs, _origins = _declared_parameters(case)
    return specs


def resolve_parameter_set(case: Case) -> ParameterSet:
    """Classify every declared or study-owned parameter exactly once."""

    declared, origins = _declared_parameters(case)
    implied, implied_at, inline_design, has_explicit_design = _study_roles(case)
    definitions: list[ParameterDefinition] = []

    for name, declared_spec in declared.items():
        spec = inline_design.get(name, declared_spec)
        explicit = _explicit_role(declared_spec, name)
        inferred = implied.get(name)
        if explicit is not None and inferred is not None and explicit is not inferred:
            raise ValueError(
                f"parameter {name!r} is declared as {explicit.value!r} but {implied_at[name]} assigns "
                f"the {inferred.value!r} role"
            )
        role = explicit or inferred or (ParameterRole.FIXED if has_explicit_design else _default_role(spec))
        definitions.append(ParameterDefinition(name, role, spec, origins[name]))

    for name, role in implied.items():
        if name in declared:
            continue
        if role is ParameterRole.DESIGN and name not in inline_design:
            raise ValueError(f"study.design_variables parameter {name!r} is not declared")
        spec = inline_design.get(name, {})
        definitions.append(ParameterDefinition(name, role, spec, implied_at[name]))

    return ParameterSet(tuple(definitions))


def _variable_sections(case: Case) -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    for section in _VARIABLE_SECTIONS:
        block = case.data.get(section)
        if section == "variables":
            variables = block
            label = "variables"
        elif isinstance(block, Mapping):
            variables = block.get("variables")
            label = f"{section}.variables"
        else:
            continue
        if isinstance(variables, Mapping):
            found.append((label, variables))

    sources = case.data.get("sources") or []
    if isinstance(sources, list):
        for index, source in enumerate(sources):
            if isinstance(source, Mapping) and isinstance(source.get("variables"), Mapping):
                found.append((f"sources[{index}].variables", source["variables"]))
    return found


def _declared_parameters(case: Case) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    specs: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for label, variables in _variable_sections(case):
        for raw_name, raw_spec in variables.items():
            name = str(raw_name)
            if name in specs:
                raise ValueError(f"parameter {name!r} is declared in both {origins[name]} and {label}; declare it once")
            specs[name] = dict(raw_spec) if isinstance(raw_spec, Mapping) else {"default": raw_spec}
            origins[name] = label
    return specs, origins


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return {str(name): item for name, item in value.items()}


def _study_roles(
    case: Case,
) -> tuple[dict[str, ParameterRole], dict[str, str], dict[str, dict[str, Any]], bool]:
    study = _mapping(case.data.get("study"), "study")
    roles: dict[str, ParameterRole] = {}
    locations: dict[str, str] = {}
    inline_design, has_explicit_design = _design_roles(study, roles, locations)
    _scenario_roles(study, roles, locations)
    _scenario_table_roles(study, roles, locations)
    _control_roles(study, roles, locations)
    return roles, locations, inline_design, has_explicit_design


def _design_roles(
    study: Mapping[str, Any],
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], bool]:
    design = study.get("design_variables")
    inline: dict[str, dict[str, Any]] = {}
    if isinstance(design, list):
        for index, raw_name in enumerate(design):
            _assign_role(roles, locations, str(raw_name), ParameterRole.DESIGN, f"study.design_variables[{index}]")
    elif isinstance(design, Mapping):
        for raw_name, raw_spec in design.items():
            name = str(raw_name)
            inline[name] = _mapping(raw_spec, f"study.design_variables.{name}")
            _assign_role(roles, locations, name, ParameterRole.DESIGN, f"study.design_variables.{name}")
    elif design is not None:
        raise ValueError("study.design_variables must be a list or mapping")
    return inline, design is not None


def _scenario_roles(
    study: Mapping[str, Any],
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
) -> None:
    scenarios = study.get("scenarios")
    if scenarios is None:
        return
    if not isinstance(scenarios, list):
        raise ValueError("study.scenarios must be a list")
    for index, raw in enumerate(scenarios):
        scenario = _mapping(raw, f"study.scenarios[{index}]")
        _assign_mapping_names(
            roles,
            locations,
            scenario.get("values"),
            ParameterRole.OPERATING,
            f"study.scenarios[{index}].values",
        )
        _assign_mapping_names(
            roles,
            locations,
            scenario.get("controls"),
            ParameterRole.CONTROL,
            f"study.scenarios[{index}].controls",
        )


def _scenario_table_roles(
    study: Mapping[str, Any],
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
) -> None:
    table = _mapping(study.get("scenario_table"), "study.scenario_table")
    _assign_mapping_names(roles, locations, table.get("values"), ParameterRole.OPERATING, "study.scenario_table.values")
    _assign_mapping_names(
        roles,
        locations,
        table.get("defaults"),
        ParameterRole.OPERATING,
        "study.scenario_table.defaults",
    )


def _control_roles(
    study: Mapping[str, Any],
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
) -> None:
    controls = _mapping(study.get("controls"), "study.controls")
    for field in ("defaults", "variables"):
        _assign_mapping_names(
            roles,
            locations,
            controls.get(field),
            ParameterRole.CONTROL,
            f"study.controls.{field}",
        )
    for scenario_id, raw_values in _mapping(controls.get("by_scenario"), "study.controls.by_scenario").items():
        _assign_mapping_names(
            roles,
            locations,
            raw_values,
            ParameterRole.CONTROL,
            f"study.controls.by_scenario.{scenario_id}",
        )


def _assign_mapping_names(
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
    raw_values: Any,
    role: ParameterRole,
    path: str,
) -> None:
    for name in _mapping(raw_values, path):
        _assign_role(roles, locations, name, role, path)


def _assign_role(
    roles: dict[str, ParameterRole],
    locations: dict[str, str],
    name: str,
    role: ParameterRole,
    path: str,
) -> None:
    current = roles.get(name)
    if current is not None and current is not role:
        raise ValueError(
            f"parameter {name!r} has both {current.value!r} ({locations[name]}) and {role.value!r} ({path}) roles"
        )
    roles[name] = role
    locations.setdefault(name, path)


def _explicit_role(spec: Mapping[str, Any], name: str) -> ParameterRole | None:
    raw = spec.get("role")
    if raw is None:
        return None
    try:
        return ParameterRole(str(raw))
    except ValueError as exc:
        available = ", ".join(role.value for role in ParameterRole)
        raise ValueError(f"parameter {name!r} role must be one of: {available}") from exc


def _default_role(spec: Mapping[str, Any]) -> ParameterRole:
    if "default" not in spec:
        return ParameterRole.DESIGN
    if "bounds" in spec:
        return ParameterRole.DESIGN
    choices = spec.get("choices")
    if isinstance(choices, list) and len(choices) > 1:
        return ParameterRole.DESIGN
    return ParameterRole.FIXED
