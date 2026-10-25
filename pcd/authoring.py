"""Public circuit authoring translated into the existing executable case.

This module owns no solver, measurement, or optimization implementation.
Named RF networks and authored circuits share the same public schema and
the same downstream Case/StudyRunner path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

from .core.spaces import grid_values
from .inputs import integer, parameter_spec, reject_unknown
from .tables import read_csv_rows


def _mapping(raw: Any, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(raw)


def _parameters(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = _mapping(raw, "parameters")
    reject_unknown(groups, {"fixed", "calibration", "operating", "control", "design", "latent"}, "parameters")
    variables: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for role, block in groups.items():
        for name, value in _mapping(block, f"parameters.{role}").items():
            if name in variables:
                raise ValueError(f"parameter {name!r} occurs in more than one role")
            spec = parameter_spec(value, f"parameters.{role}.{name}", allow_grid=role == "control")
            if "role" in spec:
                raise ValueError(f"parameters.{role}.{name}: placement already declares its role")
            if role in {"fixed", "calibration", "operating"} and set(spec) & {
                "bounds",
                "choices",
                "values",
                "grid",
                "scale",
            }:
                raise ValueError(
                    f"parameters.{role}.{name} is fixed per condition; put ranges in design/control or enumerate tolerance conditions"
                )
            if role == "control":
                controls[name] = spec
                variables[name] = {"role": role, "default": spec.get("default", grid_values(spec)[0])}
            else:
                variables[name] = {**spec, "role": role}
    return variables, controls


def _circuit(raw: Any) -> dict[str, Any]:
    cfg = _mapping(raw, "circuit")
    reject_unknown(cfg, {"file", "format", "source_policy", "output_node", "components"}, "circuit")
    if ("file" in cfg) == ("components" in cfg):
        raise ValueError("circuit requires exactly one of file or components")
    if "output_node" not in cfg:
        raise ValueError("circuit.output_node is required; choose the observed circuit node")
    if "file" in cfg:
        return {
            "builder": "from_netlist",
            "netlist_file": cfg["file"],
            "netlist_mode": cfg.get("format", "deck"),
            "source_policy": cfg.get("source_policy", "preserve"),
            "output_node": cfg["output_node"],
        }
    if set(cfg) & {"format", "source_policy"}:
        raise ValueError("circuit.format/source_policy apply only to circuit.file")
    return {"builder": "from_yaml", "components": cfg["components"], "output_node": cfg["output_node"]}


def _rows(
    raw: Any, base: Path, *, observations: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    cfg: dict[str, Any] = (
        {"rows": raw} if isinstance(raw, list) else _mapping(raw, "observations" if observations else "conditions")
    )
    reject_unknown(
        cfg,
        {"rows", "file", "columns", "id_column", "partition_column"},
        "observations" if observations else "conditions",
    )
    if ("rows" in cfg) == ("file" in cfg):
        raise ValueError("conditions/observations require exactly one of rows or file")
    if "file" in cfg:
        path = Path(cfg["file"])
        _header, raw_rows = read_csv_rows(path if path.is_absolute() else base / path)
    else:
        raw_rows = cfg["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("conditions/observations must contain at least one row")
    rows = [_mapping(row, f"row {index + 1}") for index, row in enumerate(raw_rows)]
    columns = _mapping(cfg.get("columns"), "columns")
    if observations:
        defaults = {f"target_{name}": name for name in ("voltage_re_V", "voltage_im_V", "current_re_A", "current_im_A")}
        if "frequency_Hz" in rows[0]:
            defaults["frequency_Hz"] = "frequency_Hz"
        columns = {**defaults, **columns}
    elif not columns:
        columns = {name: name for name in rows[0] if name not in {"id", "scenario_id", "weight", "controls"}}
    id_column = str(cfg.get("id_column", "scenario_id" if "scenario_id" in rows[0] else "id"))
    partition_column = str(cfg.get("partition_column", "partition"))
    scenarios = []
    for index, row in enumerate(rows):
        values = {}
        for parameter, column in columns.items():
            if column not in row or row[column] == "":
                raise ValueError(f"row {index + 1} is missing {column!r}")
            value = row[column]
            if isinstance(value, str):
                with suppress(ValueError):
                    value = float(value)
            values[parameter] = value
        if observations:
            for name in ("target_voltage_re_V", "target_voltage_im_V", "target_current_re_A", "target_current_im_A"):
                if name not in values:
                    raise ValueError(f"observations require a column for {name}")
        scenarios.append(
            {
                "id": str(row.get(id_column, f"point_{index + 1}")),
                "values": values,
                "weight": float(row.get("weight", 1)),
                **({"controls": row["controls"]} if "controls" in row else {}),
            }
        )
    table = None
    if "file" in cfg:
        if any("controls" in row for row in rows):
            raise ValueError(
                "CSV control settings must be numeric columns with explicitly declared operating/control ownership"
            )
        table = {
            "table_file": cfg["file"],
            "values": columns,
            **({"id_column": id_column} if id_column in rows[0] else {}),
            **({"weight_column": "weight"} if "weight" in rows[0] else {}),
        }
    partitions = {"fit": [], "holdout": []}
    if observations:
        for row, scenario in zip(rows, scenarios, strict=True):
            partition = row.get(partition_column)
            if partition not in partitions:
                raise ValueError(f"each observation needs {partition_column}: fit or holdout")
            partitions[partition].append(scenario["id"])
    return scenarios, table, partitions


def _objective(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = _mapping(raw, "objective")
    kind = cfg.pop("type", "impedance_match")
    metric = cfg.pop("metric", "load_real_power_W" if kind == "rf_load" else "loss")
    direction = cfg.pop("direction", "maximize" if kind == "rf_load" else "minimize")
    aggregation = cfg.pop("aggregation", "worst")
    allowed = {"impedance_match": set(), "rf_load": set(), "waveform_l2": {"waveform_file"}}.get(kind)
    if allowed is None:
        raise ValueError(
            "objective.type must be impedance_match, rf_load, or waveform_l2; identification owns terminal fitting"
        )
    reject_unknown(cfg, allowed, "objective")
    return {"objective": kind, **cfg}, [{"metric": metric, "direction": direction, "aggregation": aggregation}]


def compile_circuit_case(authored: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], list[str]]:
    """Compile optional purpose-specific blocks without exposing study internals."""

    reject_unknown(
        authored,
        {
            "schema",
            "case_id",
            "description",
            "circuit",
            "source",
            "sources",
            "analysis",
            "parameters",
            "measurement",
            "conditions",
            "objective",
            "acceptance",
            "execution",
            "identification",
            "load",
            "reference_impedance_ohm",
            "output",
        },
        "case",
    )
    if not str(authored.get("case_id", "")).strip():
        raise ValueError("case_id is required")
    variables, controls = _parameters(authored.get("parameters"))
    circuit = _circuit(authored["circuit"])
    execution = _mapping(authored.get("execution"), "execution")
    reject_unknown(
        execution,
        {"solver", "executable", "timeout_s", "optimizer", "trials", "seed", "control_state_limit"},
        "execution",
    )
    if "analysis" not in authored:
        raise ValueError("analysis.ac or analysis.tran is required")
    analysis = _mapping(authored["analysis"], "analysis")
    reject_unknown(analysis, {"ac", "tran", "options"}, "analysis")
    data: dict[str, Any] = {
        "schema": "case_yaml.v1",
        "case_id": authored["case_id"],
        "circuit": circuit,
        "variables": variables,
        "solver": {
            "name": execution.get("solver", "ngspice_cli"),
            **analysis,
            **{key: execution[key] for key in ("executable", "timeout_s") if key in execution},
        },
        "measurement": _mapping(authored.get("measurement"), "measurement"),
        "study": {},
    }
    for field in ("source", "sources"):
        if field in authored:
            data[field] = deepcopy(authored[field])
    if "output" in authored:
        data["output"] = deepcopy(authored["output"])
    load = _mapping(authored.get("load"), "load")
    if load:
        if "name" in load:
            raise ValueError("public load uses type, not the internal name field")
        data["load"] = {**load, "name": load.pop("type", "none")}
        data["load"].pop("type", None)
        if "evidence" in data["load"]:
            data["load"]["characterization"] = data["load"].pop("evidence")
        if "parameters" in data["load"]:
            for key, value in _mapping(data["load"].pop("parameters"), "load.parameters").items():
                if key in data["load"]:
                    raise ValueError(f"load parameter {key!r} is declared twice")
                data["load"][key] = value
    if "reference_impedance_ohm" in authored:
        if "reference_impedance_ohm" in data["measurement"]:
            raise ValueError("declare reference_impedance_ohm once at the public case root")
        data["measurement"]["reference_impedance_ohm"] = authored["reference_impedance_ohm"]
    scenarios, table = [], None
    if "conditions" in authored:
        scenarios, table, _ = _rows(authored["conditions"], base)
    fitting = "identification" in authored
    fit_cfg = _mapping(authored.get("identification"), "identification")
    if fitting:
        if "conditions" in authored or "objective" in authored:
            raise ValueError(
                "identification owns observations and the fit objective; do not also declare conditions/objective"
            )
        observations = fit_cfg.pop("observations", None)
        scenarios, table, partitions = _rows(observations, base, observations=True)
        scale = float(fit_cfg.pop("error_scale_ohm", 100.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("identification.error_scale_ohm must be positive and finite")
        reject_unknown(
            fit_cfg,
            {
                "fit_loss_max",
                "holdout_loss_max",
                "holdout_mode",
                "condition_parameters",
                "sensitivity_step",
                "condition_number_max",
                "rank_tolerance",
            },
            "identification",
        )
        data["identification"] = {
            **fit_cfg,
            "fit_scenarios": partitions["fit"],
            "holdout_scenarios": partitions["holdout"],
            "observed_metrics": ["resistance_ohm", "reactance_ohm"],
            "metric_scales": {"resistance_ohm": scale, "reactance_ohm": scale},
        }
        data["target"] = {"objective": "terminal_vi_fit", "impedance_scale_ohm": scale}
        data["study"].update(
            {
                "candidate_role": "latent",
                "objectives": [{"metric": "normalized_impedance_error", "direction": "minimize", "aggregation": "rms"}],
            }
        )
    elif "objective" in authored:
        data["target"], data["study"]["objectives"] = _objective(authored["objective"])
    elif "acceptance" in authored:
        raise ValueError("acceptance requires an objective; omit it for a forward simulation")
    else:
        data["metadata"] = {"workflow": "simulation"}
        if any(spec["role"] in {"design", "latent"} for spec in variables.values()):
            raise ValueError(
                "simulation parameters must be fixed/calibration/operating; declare an objective for design or identification for latent fitting"
            )
    if table:
        for name in controls:
            table["values"].pop(name, None)
        data["study"]["scenario_table"] = table
    elif scenarios:
        data["study"]["scenarios"] = scenarios
    for row in scenarios:
        for name in controls:
            if name in row["values"]:
                row.setdefault("controls", {})[name] = row["values"].pop(name)
        for name, value in row["values"].items():
            if name in variables and variables[name]["role"] != "operating":
                raise ValueError(f"condition {name!r} must have the operating role")
            variables.setdefault(name, {"role": "operating", "default": value})
    if controls:
        data["study"]["controls"] = {
            "variables": controls,
            "budget": execution.get("control_state_limit", 250),
            "by_scenario": {row["id"]: row["controls"] for row in scenarios if row.get("controls")},
        }
    elif "control_state_limit" in execution:
        raise ValueError("execution.control_state_limit requires control parameters")
    bounds = _mapping(authored.get("acceptance"), "acceptance")
    if bounds:
        reject_unknown(bounds, {"metric_bounds"}, "acceptance")
        data.setdefault("target", {})["constraints"] = bounds
    proposed = "latent" if fitting else "design"
    specs = [spec for spec in variables.values() if spec["role"] == proposed]
    discrete = all("bounds" not in spec for spec in specs)
    optimizer = execution.get(
        "optimizer", "least_squares" if fitting else ("grid" if discrete else "differential_evolution")
    )
    trials = (
        math.prod(len(spec.get("choices", spec.get("values", [spec.get("default")]))) for spec in specs)
        if discrete
        else 32
    )
    data["optimizer"] = {"name": optimizer, "seed": integer(execution.get("seed", 0), "execution.seed")}
    data["run"] = {"trials": integer(execution.get("trials", trials), "execution.trials", minimum=1)}
    if optimizer == "grid":
        if not discrete or data["run"]["trials"] != trials:
            raise ValueError(f"grid enumeration requires finite choices and exactly {trials} trials")
        data["study"]["candidate_enumeration"] = "exact"
    return data, [
        "public circuit input; parameters are owned by their declared group",
        "conditions are explicit rows; no implicit Cartesian product",
        "SI values are resolved before simulation",
    ]
