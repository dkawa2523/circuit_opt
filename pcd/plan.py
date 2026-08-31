"""Compile the small RF study input into one explicit executable case.

The public file describes an engineering decision: fixed hardware, optional
design search, settings that may be retuned, an electrical load envelope, and
acceptance limits.  The rest of PCD consumes one explicit executable mapping.
Keeping that translation here gives validation, simulation, caching, and
reporting exactly the same resolved values without spreading aliases and
defaults through the numerical code.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PUBLIC_SCHEMA = "pcd.rf.v1"
RESOLVED_SCHEMA = "resolved_rf_plan.v1"
EXECUTABLE_SCHEMA = "case_yaml.v1"

_TOPOLOGY_COMPONENTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "l_match": (("L1", "src", "electrode"), ("C1", "electrode", "0")),
    "pi_match": (
        ("C1", "src", "0"),
        ("L1", "src", "electrode"),
        ("C2", "electrode", "0"),
    ),
    "pi_match_harmonic": (
        ("C1", "src", "0"),
        ("L1", "src", "electrode"),
        ("C2", "electrode", "0"),
        ("Lh", "electrode", "harmonic_mid"),
        ("Ch", "harmonic_mid", "0"),
    ),
}

_COMPONENT_LIMIT_METRICS = {
    "current_rms_A_max": "current_rms_A",
    "current_peak_A_max": "current_peak_A",
    "voltage_rms_V_max": "voltage_rms_V",
    "voltage_peak_V_max": "voltage_peak_V",
    "loss_W_max": "loss_W",
}

_SOURCE_LIMIT_METRICS = {
    "current_rms_A_max": "source_current_rms_A",
    "apparent_power_VA_max": "source_apparent_power_VA",
}


@dataclass(frozen=True)
class ResolvedPlan:
    """Immutable description of what an authored RF input will execute."""

    case: dict[str, Any]
    inferences: tuple[str, ...]
    trials: int
    optimizer: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESOLVED_SCHEMA,
            "source_schema": PUBLIC_SCHEMA,
            "execution": {
                "solver": str((self.case.get("solver") or {}).get("name")),
                "optimizer": self.optimizer,
                "trials": self.trials,
                "seed": self.seed,
            },
            "inferences": list(self.inferences),
            "case": deepcopy(self.case),
        }


@dataclass(frozen=True)
class _NetworkPlan:
    topology: str
    variables: dict[str, dict[str, Any]]
    search_variables: dict[str, dict[str, Any]]
    controls: dict[str, dict[str, list[Any]]]
    control_states: int
    metric_bounds: dict[str, dict[str, float]]
    observed_refs: set[str]
    needs_absolute_drive: bool
    losses: dict[str, float]
    circuit: dict[str, Any]


@dataclass(frozen=True)
class _OperatingPlan:
    frequency: float | str
    drive: float | str
    load: dict[str, Any]
    table: dict[str, Any] | None
    conditions: list[dict[str, Any]]
    has_absolute_drive: bool


@dataclass(frozen=True)
class _ExecutionPlan:
    solver: str
    optimizer: str
    trials: int
    seed: int


def compile_rf_case(data: Mapping[str, Any], base_dir: Path) -> ResolvedPlan:
    """Resolve a ``pcd.rf.v1`` mapping without executing plugins or solvers."""

    authored = _mapping(data, "$", required=True)
    _reject_unknown(
        authored,
        {
            "schema",
            "case_id",
            "description",
            "frequency_Hz",
            "drive_peak_V",
            "network",
            "load",
            "conditions",
            "acceptance",
            "execution",
        },
        "$",
    )
    if not str(authored.get("case_id", "")).strip():
        raise ValueError("case_id is required")

    inferences: list[str] = []
    load = _mapping(authored.get("load"), "load", required=True)
    acceptance = _mapping(authored.get("acceptance"), "acceptance", required=True)
    execution = _mapping(authored.get("execution"), "execution")
    _reject_unknown(
        acceptance,
        {
            "reflected_power_fraction_max",
            "reflection_magnitude_max",
            "component_limits",
            "source_limits",
            "loss_balance_fraction_max",
            "control_margin_min",
        },
        "acceptance",
    )
    _reject_unknown(execution, {"solver", "candidate_state_limit", "control_state_limit"}, "execution")
    network = _network_plan(authored.get("network"), acceptance, execution, inferences)
    operating = _operating_plan(authored, load, base_dir, network.needs_absolute_drive, inferences)
    network = _observe_absolute_stress(network, operating.has_absolute_drive, inferences)
    run = _execution_plan(execution, network.search_variables, inferences)
    variables = {**network.variables, **_operating_defaults(operating, inferences)}

    study: dict[str, Any] = {
        "design_variables": list(network.variables),
        "candidate_enumeration": "exact",
        "objectives": [{"metric": "reflection_magnitude", "direction": "minimize", "aggregation": "worst"}],
    }
    if operating.table:
        study["scenario_table"] = operating.table
    elif operating.conditions:
        study["scenarios"] = operating.conditions
    if network.controls:
        study["controls"] = {"variables": network.controls, "budget": network.control_states}
    margin = _control_margin_limit(acceptance, bool(network.controls))
    if margin is not None:
        study["control_margin_min"] = margin
        inferences.append(f"require at least {margin:g} normalized tuning headroom in every condition")

    measurement: dict[str, Any] = {
        "voltage_node": "electrode",
        "current_source": "Vsrc",
        "reference_impedance_ohm": 50.0,
    }
    if network.observed_refs or "loss_balance_fraction_max" in acceptance:
        measurement["load_current"] = "auto"

    resolved = {
        "schema": EXECUTABLE_SCHEMA,
        "case_id": str(authored["case_id"]),
        "variables": variables,
        "source": {
            "type": "sine_voltage",
            "name": "Vsrc",
            "p": "src",
            "n": "0",
            "amplitude_V": operating.drive,
            "frequency_Hz": operating.frequency,
        },
        "circuit": network.circuit,
        "load": operating.load,
        "measurement": measurement,
        "solver": {"name": run.solver, "ac": {"frequency_Hz": operating.frequency}},
        "target": {
            "objective": "impedance_match",
            "constraints": {"metric_bounds": network.metric_bounds},
        },
        "study": study,
        "optimizer": {"name": run.optimizer, "seed": run.seed},
        "run": {"trials": run.trials},
    }
    return ResolvedPlan(resolved, tuple(inferences), run.trials, run.optimizer, run.seed)


def _network_plan(
    raw_network: Any,
    acceptance: Mapping[str, Any],
    execution: Mapping[str, Any],
    inferences: list[str],
) -> _NetworkPlan:
    network = _mapping(raw_network, "network", required=True)
    _reject_unknown(network, {"type", "fixed", "search", "tuning", "loss_ohm"}, "network")
    topology = str(network.get("type", ""))
    if topology not in _TOPOLOGY_COMPONENTS:
        raise ValueError(f"network.type must be one of {sorted(_TOPOLOGY_COMPONENTS)}, got {topology!r}")
    required_refs = {ref for ref, _n1, _n2 in _TOPOLOGY_COMPONENTS[topology]}
    fixed, search, tuning = _network_roles(network, topology, required_refs)
    variables = {name: _fixed_spec(name, value) for name, value in fixed.items()}
    search_variables = {name: _search_spec(name, spec) for name, spec in search.items()}
    variables.update(search_variables)
    controls = {name: _tuning_spec(name, spec) for name, spec in tuning.items()}
    states = _control_state_count(controls, execution, inferences)
    bounds, observed, needs_drive = _acceptance_bounds(acceptance, required_refs)
    loss = _loss_values(network.get("loss_ohm"), topology, required_refs)
    _validate_loss_requirements(acceptance, loss)
    observed.update(loss)
    circuit = _resolved_circuit(topology, loss, observed)
    if observed:
        inferences.append("expand the named network so requested component stress and effective loss are observable")
    return _NetworkPlan(
        topology,
        variables,
        search_variables,
        controls,
        states,
        bounds,
        observed,
        needs_drive,
        loss,
        circuit,
    )


def _observe_absolute_stress(network: _NetworkPlan, has_absolute_drive: bool, inferences: list[str]) -> _NetworkPlan:
    """Expose all named matching components when their stress has physical scale."""

    if not has_absolute_drive:
        return network
    all_refs = {ref for ref, _n1, _n2 in _TOPOLOGY_COMPONENTS[network.topology]}
    if all_refs <= network.observed_refs:
        return network
    observed = set(network.observed_refs) | all_refs
    inferences.append("observe every matching-network component because an absolute drive is declared")
    return replace(
        network,
        observed_refs=observed,
        circuit=_resolved_circuit(network.topology, network.losses, observed),
    )


def _network_roles(
    network: Mapping[str, Any], topology: str, required_refs: set[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixed = _mapping(network.get("fixed"), "network.fixed")
    search = _mapping(network.get("search"), "network.search")
    tuning = _mapping(network.get("tuning"), "network.tuning")
    _reject_role_overlap(fixed, search, tuning)
    supplied = set(fixed) | set(search) | set(tuning)
    if missing := sorted(required_refs - supplied):
        raise ValueError(f"network is missing component values for {missing}")
    if extra := sorted(supplied - required_refs):
        raise ValueError(f"network contains components not used by {topology}: {extra}")
    return fixed, search, tuning


def _control_state_count(
    controls: Mapping[str, Mapping[str, list[Any]]], execution: Mapping[str, Any], inferences: list[str]
) -> int:
    states = math.prod(len(spec["values"]) for spec in controls.values()) if controls else 1
    limit = _positive_int(execution.get("control_state_limit", 250), "execution.control_state_limit")
    if states > limit:
        raise ValueError(
            f"network.tuning declares {states} exact states, exceeding the safety limit {limit}; "
            "reduce the value sets or deliberately raise execution.control_state_limit"
        )
    if controls:
        inferences.append(f"enumerate all {states} tuning states for every condition")
    return states


def _loss_values(raw: Any, topology: str, required_refs: set[str]) -> dict[str, float]:
    values = _mapping(raw, "network.loss_ohm")
    if unknown := sorted(set(values) - required_refs):
        raise ValueError(f"network.loss_ohm names components not used by {topology}: {unknown}")
    losses = {ref: _finite_number(value, f"network.loss_ohm.{ref}") for ref, value in values.items()}
    if negative := sorted(ref for ref, value in losses.items() if value < 0):
        raise ValueError(f"network.loss_ohm must be non-negative for {negative}")
    return losses


def _validate_loss_requirements(acceptance: Mapping[str, Any], losses: Mapping[str, float]) -> None:
    if "loss_balance_fraction_max" in acceptance and not losses:
        raise ValueError("network.loss_ohm is required when acceptance.loss_balance_fraction_max is declared")
    limits = _mapping(acceptance.get("component_limits"), "acceptance.component_limits")
    missing = sorted(
        ref
        for ref, raw in limits.items()
        if "loss_W_max" in _mapping(raw, f"acceptance.component_limits.{ref}") and ref not in losses
    )
    if missing:
        raise ValueError(f"network.loss_ohm is required for component loss limits on {missing}")


def _operating_plan(
    authored: Mapping[str, Any],
    load: Mapping[str, Any],
    base_dir: Path,
    needs_absolute_drive: bool,
    inferences: list[str],
) -> _OperatingPlan:
    frequency, table = _load_frequency_and_scenarios(authored, load, base_dir, inferences)
    conditions = _conditions(authored.get("conditions"))
    if table and conditions:
        raise ValueError(
            "load.type: impedance_table and conditions cannot be combined; put each electrical point in the table"
        )
    resolved_frequency = _resolved_frequency(frequency, table, conditions)
    drive, has_absolute_drive = _resolved_drive(authored, table, conditions, needs_absolute_drive, inferences)
    return _OperatingPlan(
        resolved_frequency,
        drive,
        _resolved_load(load, resolved_frequency),
        table,
        conditions,
        has_absolute_drive,
    )


def _resolved_frequency(
    frequency: Any, table: Mapping[str, Any] | None, conditions: list[dict[str, Any]]
) -> float | str:
    table_frequency = bool(table and "rf_frequency_Hz" in table["values"])
    condition_frequency = any("rf_frequency_Hz" in item["values"] for item in conditions)
    if condition_frequency and not all("rf_frequency_Hz" in item["values"] for item in conditions):
        raise ValueError("every conditions item must declare frequency_Hz when any one does")
    return "rf_frequency_Hz" if table_frequency or condition_frequency else _positive_number(frequency, "frequency_Hz")


def _resolved_drive(
    authored: Mapping[str, Any],
    table: Mapping[str, Any] | None,
    conditions: list[dict[str, Any]],
    needs_absolute_drive: bool,
    inferences: list[str],
) -> tuple[float | str, bool]:
    table_drive = bool(table and "drive_amplitude_V" in table["values"])
    if table_drive and "drive_peak_V" in authored:
        raise ValueError("drive_peak_V is already supplied by every impedance-table row; remove the top-level value")
    if table_drive:
        return "drive_amplitude_V", True
    condition_drive = any("drive_amplitude_V" in item["values"] for item in conditions)
    if condition_drive and not all("drive_amplitude_V" in item["values"] for item in conditions):
        raise ValueError("every conditions item must declare drive_peak_V when any one does")
    if condition_drive:
        return "drive_amplitude_V", True
    if "drive_peak_V" in authored:
        return _positive_number(authored["drive_peak_V"], "drive_peak_V"), True
    if needs_absolute_drive:
        raise ValueError(
            "drive_peak_V (or drive_peak_V in every condition) is required for absolute component or source limits"
        )
    inferences.append("use a 1 V peak AC source because reflection is amplitude independent")
    return 1.0, False


def _execution_plan(
    execution: Mapping[str, Any], search: Mapping[str, Mapping[str, Any]], inferences: list[str]
) -> _ExecutionPlan:
    solver = str(execution.get("solver", "ngspice_cli"))
    trials = math.prod(len(spec["choices"]) for spec in search.values()) if search else 1
    limit = _positive_int(execution.get("candidate_state_limit", 250), "execution.candidate_state_limit")
    if trials > limit:
        raise ValueError(
            f"network.search declares {trials} exact candidates, exceeding the safety limit {limit}; "
            "reduce the value sets or deliberately raise execution.candidate_state_limit"
        )
    if search:
        inferences.append(f"compare all {trials} discrete hardware candidates exactly once")
    else:
        inferences.append("evaluate the one declared fixed hardware candidate")
    return _ExecutionPlan(solver, "grid", trials, 0)


def _operating_defaults(operating: _OperatingPlan, inferences: list[str]) -> dict[str, dict[str, Any]]:
    values: Mapping[str, Any] = {}
    if operating.table:
        values = _mapping(operating.table.get("defaults"), "resolved scenario-table defaults")
    elif operating.conditions:
        values = operating.conditions[0]["values"]
    if values:
        inferences.append("use the first declared condition only as the standalone netlist preview default")
    return {name: {"default": value} for name, value in values.items()}


def _mapping(value: Any, path: str, *, required: bool = False) -> dict[str, Any]:
    if value is None:
        if required:
            raise ValueError(f"{path} must be a mapping")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], path: str) -> None:
    if unknown := sorted(set(values) - allowed):
        raise ValueError(f"{path} contains unsupported fields {unknown}")


def _reject_role_overlap(fixed: Mapping[str, Any], search: Mapping[str, Any], tuning: Mapping[str, Any]) -> None:
    memberships: dict[str, list[str]] = {}
    for role, values in (("fixed", fixed), ("search", search), ("tuning", tuning)):
        for name in values:
            memberships.setdefault(name, []).append(role)
    overlap = {name: roles for name, roles in memberships.items() if len(roles) > 1}
    if overlap:
        detail = ", ".join(f"{name} ({'/'.join(roles)})" for name, roles in sorted(overlap.items()))
        raise ValueError(f"each network parameter must have one role; duplicated: {detail}")


def _fixed_spec(name: str, value: Any) -> dict[str, Any]:
    number = _positive_number(value, f"network.fixed.{name}")
    return {"choices": [number], "default": number}


def _search_spec(name: str, value: Any) -> dict[str, Any]:
    cfg = _mapping(value, f"network.search.{name}", required=True)
    _reject_unknown(cfg, {"values", "default"}, f"network.search.{name}")
    choices = cfg.get("values")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"network.search.{name}.values must be a non-empty list")
    normalized = [
        _positive_number(item, f"network.search.{name}.values[{index}]") for index, item in enumerate(choices)
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"network.search.{name}.values must not contain duplicates")
    resolved: dict[str, Any] = {"choices": normalized}
    if "default" in cfg:
        default = _positive_number(cfg["default"], f"network.search.{name}.default")
        if default not in normalized:
            raise ValueError(f"network.search.{name}.default must be one of its values")
        resolved["default"] = default
    else:
        resolved["default"] = normalized[0]
    return resolved


def _tuning_spec(name: str, value: Any) -> dict[str, list[Any]]:
    if isinstance(value, Mapping):
        _reject_unknown(value, {"values"}, f"network.tuning.{name}")
        values = value.get("values")
    else:
        values = value
    if not isinstance(values, list) or not values:
        raise ValueError(f"network.tuning.{name} must be a non-empty list of exact settings")
    normalized = [_positive_number(item, f"network.tuning.{name}.values[{index}]") for index, item in enumerate(values)]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"network.tuning.{name} must not contain duplicate settings")
    return {"values": normalized}


def _acceptance_bounds(
    acceptance: Mapping[str, Any], required_refs: set[str]
) -> tuple[dict[str, dict[str, float]], set[str], bool]:
    bounds: dict[str, dict[str, float]] = {"reflection_magnitude": {"max": _reflection_limit(acceptance)}}
    component_bounds, observed = _component_limit_bounds(acceptance, required_refs)
    source_bounds = _source_limit_bounds(acceptance)
    bounds.update(component_bounds)
    bounds.update(source_bounds)
    if "loss_balance_fraction_max" in acceptance:
        limit = _finite_number(acceptance["loss_balance_fraction_max"], "acceptance.loss_balance_fraction_max")
        if limit < 0:
            raise ValueError("acceptance.loss_balance_fraction_max must be non-negative")
        bounds["component_loss_balance_fraction_of_source"] = {"max": limit}
    return bounds, observed, bool(component_bounds or source_bounds)


def _source_limit_bounds(acceptance: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw_limits = acceptance.get("source_limits")
    if raw_limits is None:
        return {}
    limits = _mapping(raw_limits, "acceptance.source_limits", required=True)
    if not limits:
        raise ValueError("acceptance.source_limits must contain at least one limit")
    bounds: dict[str, dict[str, float]] = {}
    for key, raw_limit in limits.items():
        metric = _SOURCE_LIMIT_METRICS.get(key)
        if metric is None:
            raise ValueError(f"unknown source limit {key!r}; available={sorted(_SOURCE_LIMIT_METRICS)}")
        limit = _finite_number(raw_limit, f"acceptance.source_limits.{key}")
        if limit < 0:
            raise ValueError(f"acceptance.source_limits.{key} must be non-negative")
        bounds[metric] = {"max": limit}
    return bounds


def _control_margin_limit(acceptance: Mapping[str, Any], has_controls: bool) -> float | None:
    raw = acceptance.get("control_margin_min")
    if raw is None:
        return None
    if not has_controls:
        raise ValueError("acceptance.control_margin_min requires network.tuning")
    margin = _finite_number(raw, "acceptance.control_margin_min")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("acceptance.control_margin_min must be between 0 and 1")
    return margin


def _reflection_limit(acceptance: Mapping[str, Any]) -> float:
    power_limit = acceptance.get("reflected_power_fraction_max")
    gamma_limit = acceptance.get("reflection_magnitude_max")
    if (power_limit is None) == (gamma_limit is None):
        raise ValueError(
            "acceptance must declare exactly one of reflected_power_fraction_max or reflection_magnitude_max"
        )
    if power_limit is not None:
        fraction = _finite_number(power_limit, "acceptance.reflected_power_fraction_max")
        if not 0 <= fraction <= 1:
            raise ValueError("acceptance.reflected_power_fraction_max must be between 0 and 1")
        return math.sqrt(fraction)
    else:
        reflection = _finite_number(gamma_limit, "acceptance.reflection_magnitude_max")
        if not 0 <= reflection <= 1:
            raise ValueError("acceptance.reflection_magnitude_max must be between 0 and 1")
        return reflection


def _component_limit_bounds(
    acceptance: Mapping[str, Any], required_refs: set[str]
) -> tuple[dict[str, dict[str, float]], set[str]]:
    bounds: dict[str, dict[str, float]] = {}
    observed: set[str] = set()
    component_limits = _mapping(acceptance.get("component_limits"), "acceptance.component_limits")
    for ref, raw_limits in component_limits.items():
        if ref not in required_refs:
            raise ValueError(f"acceptance.component_limits names a component not used by the network: {ref}")
        limits = _mapping(raw_limits, f"acceptance.component_limits.{ref}", required=True)
        if not limits:
            raise ValueError(f"acceptance.component_limits.{ref} must contain at least one limit")
        observed.add(ref)
        for key, raw_limit in limits.items():
            suffix = _COMPONENT_LIMIT_METRICS.get(key)
            if suffix is None:
                raise ValueError(f"unknown component limit {key!r}; available={sorted(_COMPONENT_LIMIT_METRICS)}")
            limit = _finite_number(raw_limit, f"acceptance.component_limits.{ref}.{key}")
            if limit < 0:
                raise ValueError(f"acceptance.component_limits.{ref}.{key} must be non-negative")
            bounds[f"component_{ref}_{suffix}"] = {"max": limit}
    return bounds, observed


def _load_frequency_and_scenarios(
    authored: Mapping[str, Any], load: Mapping[str, Any], base_dir: Path, inferences: list[str]
) -> tuple[Any, dict[str, Any] | None]:
    load_type = str(load.get("type", ""))
    if load_type != "impedance_table":
        frequency = authored.get("frequency_Hz")
        raw_conditions = authored.get("conditions") or []
        condition_frequency = any(isinstance(item, Mapping) and "frequency_Hz" in item for item in raw_conditions)
        if frequency is None and not condition_frequency:
            raise ValueError("frequency_Hz is required unless the impedance table or every condition supplies it")
        return frequency, None

    raw_file = load.get("file")
    if not isinstance(raw_file, str) or not raw_file.strip():
        raise ValueError("load.file is required for load.type: impedance_table")
    path = Path(raw_file)
    path = path if path.is_absolute() else base_dir / path
    if not path.is_file():
        raise ValueError(f"impedance table not found: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        first_row = next(reader, None)
    required = {"scenario_id", "resistance_ohm", "reactance_ohm"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"impedance table is missing canonical columns {missing}: {path.resolve()}")
    if first_row is None:
        raise ValueError(f"impedance table is empty: {path.resolve()}")
    has_frequency = "frequency_Hz" in columns
    frequency = authored.get("frequency_Hz")
    if has_frequency and frequency is not None:
        raise ValueError("frequency_Hz is already supplied by every impedance-table row; remove the top-level value")
    if not has_frequency and frequency is None:
        raise ValueError("frequency_Hz is required when the impedance table has no frequency_Hz column")
    values = _table_value_columns(columns, inferences)
    defaults = {
        parameter: _table_cell(first_row[column], f"{path.resolve()}:{column}") for parameter, column in values.items()
    }
    return frequency, {
        "table_file": raw_file,
        "values": values,
        "id_column": "scenario_id",
        "weight_column": "weight",
        "defaults": defaults,
    }


def _table_value_columns(columns: set[str], inferences: list[str]) -> dict[str, str]:
    values = {
        "load_resistance_ohm": "resistance_ohm",
        "load_reactance_ohm": "reactance_ohm",
    }
    if "frequency_Hz" in columns:
        values = {"rf_frequency_Hz": "frequency_Hz", **values}
        inferences.append("run one exact-frequency AC solve for every impedance-table row")
    if "drive_peak_V" in columns:
        values = {"drive_amplitude_V": "drive_peak_V", **values}
    return values


def _table_cell(raw: Any, path: str) -> Any:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"impedance-table value is empty at {path}")
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        raise ValueError(f"impedance-table value must be finite at {path}")
    return value


def _conditions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("conditions must be a non-empty list")
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        cfg = _mapping(raw, f"conditions[{index}]", required=True)
        _reject_unknown(cfg, {"id", "weight", "drive_peak_V", "frequency_Hz"}, f"conditions[{index}]")
        condition_id = str(cfg.pop("id", "")).strip()
        if not condition_id:
            raise ValueError(f"conditions[{index}].id is required")
        weight = float(cfg.pop("weight", 1.0))
        values: dict[str, Any] = {}
        for name, item in cfg.items():
            internal_name = {"drive_peak_V": "drive_amplitude_V", "frequency_Hz": "rf_frequency_Hz"}.get(name, name)
            values[internal_name] = item
        out.append({"id": condition_id, "values": values, "weight": weight})
    return out


def _resolved_load(load: Mapping[str, Any], frequency: float | str) -> dict[str, Any]:
    load_type = _validated_load_type(load)
    common = _load_common(load)
    if load_type == "impedance_table":
        characterization = common.get("characterization")
        if isinstance(characterization, dict):
            characterization.setdefault("table", str(load["file"]))
        return {
            "name": "impedance_point",
            **common,
            "resistance_ohm": "load_resistance_ohm",
            "reactance_ohm": "load_reactance_ohm",
            "model_frequency_Hz": frequency,
        }
    if load_type == "impedance_point":
        return {"name": load_type, **common, **_point_parameters(load), "model_frequency_Hz": frequency}
    return {"name": load_type, **common, **_effective_load_parameters(load_type, load)}


def _validated_load_type(load: Mapping[str, Any]) -> str:
    load_type = str(load.get("type", ""))
    allowed = {
        "impedance_table": {"type", "file", "reference_plane", "evidence"},
        "impedance_point": {"type", "resistance_ohm", "reactance_ohm", "reference_plane", "evidence"},
        "ccp_lumped": {"type", "parameters", "reference_plane", "evidence"},
        "icp_transformer": {"type", "parameters", "reference_plane", "evidence"},
    }.get(load_type)
    if allowed is None:
        raise ValueError("load.type must be impedance_table, impedance_point, ccp_lumped, or icp_transformer")
    _reject_unknown(load, allowed, "load")
    return load_type


def _load_common(load: Mapping[str, Any]) -> dict[str, Any]:
    reference_plane = str(load.get("reference_plane", "")).strip()
    if not reference_plane:
        raise ValueError("load.reference_plane is required")
    evidence = load.get("evidence")
    if evidence is not None and not isinstance(evidence, Mapping):
        raise ValueError("load.evidence must be a mapping")

    common: dict[str, Any] = {
        "ports": {"p": "electrode", "n": "0"},
        "reference_plane": reference_plane,
    }
    if evidence is not None:
        common["characterization"] = deepcopy(dict(evidence))
    return common


def _point_parameters(load: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("resistance_ohm", "reactance_ohm"):
        if field not in load:
            raise ValueError(f"load.{field} is required for load.type: impedance_point")
    return {"resistance_ohm": load["resistance_ohm"], "reactance_ohm": load["reactance_ohm"]}


def _effective_load_parameters(load_type: str, load: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _mapping(load.get("parameters"), "load.parameters", required=True)
    if load_type == "ccp_lumped":
        required = {"R_eff_ohm", "L_eff_H", "C_sheath_eq_F"}
        _reject_unknown(parameters, required, "load.parameters")
        if missing := sorted(required - set(parameters)):
            raise ValueError(f"load.parameters is missing {missing} for {load_type}")
        return parameters

    required = {
        "R_coil_ohm",
        "L_coil_H",
        "reflected_inductance_H",
        "secondary_damping_rate_rad_s",
    }
    _reject_unknown(parameters, required | {"C_parallel_F"}, "load.parameters")
    if missing := sorted(required - set(parameters)):
        raise ValueError(f"load.parameters is missing {missing} for {load_type}")
    return parameters


def _resolved_circuit(topology: str, loss_ohm: Mapping[str, Any], observed_refs: set[str]) -> dict[str, Any]:
    if not observed_refs:
        return {"builder": topology, "output_node": "electrode"}
    components = []
    for ref, n1, n2 in _TOPOLOGY_COMPONENTS[topology]:
        item: dict[str, Any] = {"ref": ref, "n1": n1, "n2": n2, "value": ref}
        if ref in loss_ohm:
            item["series_resistance_ohm"] = loss_ohm[ref]
        if ref in observed_refs:
            item["observe"] = True
        components.append(item)
    return {"builder": "from_yaml", "output_node": "electrode", "components": components}


def _finite_number(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _positive_number(value: Any, path: str) -> float:
    number = _finite_number(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be positive")
    return number


def _positive_int(value: Any, path: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
    if number < 1:
        raise ValueError(f"{path} must be positive")
    return number
