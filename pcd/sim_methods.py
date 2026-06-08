from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Case, pick_value, resolve_path, spice_value
from .sim_core import Circuit, SimulationResult, dummy_waveform, ngspice_cli
from .sim_registry import register


# -----------------------------------------------------------------------------
# Circuit builders
# -----------------------------------------------------------------------------


def _circuit_cfg(case: Case) -> dict[str, Any]:
    return case.data.get("circuit", {}) or {}


@register("circuit", "from_yaml")
def circuit_from_yaml(case: Case, params: dict[str, Any]) -> Circuit:
    cfg = _circuit_cfg(case)
    c = Circuit(output_node=str(cfg.get("output_node", "out")))
    c.params.update(params)
    for item in cfg.get("components", []) or []:
        if "raw" in item:
            c.raw(str(item["raw"]))
        else:
            c.add(str(item["ref"]), str(item["n1"]), str(item["n2"]), item["value"])
    return c


@register("circuit", "l_match")
def circuit_l_match(case: Case, params: dict[str, Any]) -> Circuit:
    out = str(_circuit_cfg(case).get("output_node", "electrode"))
    c = Circuit(output_node=out)
    c.params.update(params)
    c.add("L1", "src", out, "L1")
    c.add("C1", out, "0", "C1")
    return c


@register("circuit", "pi_match")
def circuit_pi_match(case: Case, params: dict[str, Any]) -> Circuit:
    out = str(_circuit_cfg(case).get("output_node", "electrode"))
    c = Circuit(output_node=out)
    c.params.update(params)
    c.add("C1", "src", "0", "C1")
    c.add("L1", "src", out, "L1")
    c.add("C2", out, "0", "C2")
    return c


@register("circuit", "pi_match_harmonic")
def circuit_pi_match_harmonic(case: Case, params: dict[str, Any]) -> Circuit:
    c = circuit_pi_match(case, params)
    out = c.output_node
    c.add("Lh", out, "harmonic_mid", "Lh")
    c.add("Ch", "harmonic_mid", "0", "Ch")
    c.notes.append("series LC shunt branch for harmonic shaping")
    return c


# -----------------------------------------------------------------------------
# Load builders. Each returns either '' or a subckt named load_model.
# -----------------------------------------------------------------------------


def _load_cfg(case: Case) -> dict[str, Any]:
    return case.data.get("load", {}) or {}


@register("load", "none")
def load_none(case: Case, params: dict[str, Any]) -> str:
    return ""


@register("load", "resistor")
def load_resistor(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    r = pick_value(cfg, "R_ohm", params, "Rload")
    return f"""
* load model: resistor
.subckt load_model p n
Rload p n {spice_value(r)}
.ends load_model
""".strip()


@register("load", "parallel_rc")
def load_parallel_rc(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    r = pick_value(cfg, "R_ohm", params, "Rload")
    c = pick_value(cfg, "C_F", params, "Cload")
    return f"""
* load model: parallel_rc
.subckt load_model p n
Rload p n {spice_value(r)}
Cload p n {spice_value(c)}
.ends load_model
""".strip()


@register("load", "series_rlc")
def load_series_rlc(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    r = pick_value(cfg, "R_ohm", params, "Rload")
    l = pick_value(cfg, "L_H", params, "Lload")
    c = pick_value(cfg, "C_F", params, "Cload")
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    return f"""
* load model: series_rlc
.subckt load_model p n
Rload p nl {spice_value(r)}
Lload nl nc {spice_value(l)}
Cload nc n {spice_value(c)}
Rleak p n {spice_value(rleak)}
.ends load_model
""".strip()


@register("load", "electrode_stray")
def load_electrode_stray(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    cap = pick_value(cfg, "C_F", params, "Ce")
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    return f"""
* load model: electrode_stray
.subckt load_model p n
Rleak p n {spice_value(rleak)}
Celectrode p n {spice_value(cap)}
.ends load_model
""".strip()


@register("load", "from_yaml")
def load_from_yaml(case: Case, params: dict[str, Any]) -> str:
    lines = ["* load model: from_yaml", ".subckt load_model p n"]
    for item in _load_cfg(case).get("components", []) or []:
        if "raw" in item:
            lines.append(str(item["raw"]))
        else:
            lines.append(f"{item['ref']} {item['n1']} {item['n2']} {spice_value(pick_value(item, 'value', params, item['value']))}")
    lines.append(".ends load_model")
    return "\n".join(lines)


@register("load", "plasma_fixed_rlc")
def load_plasma_fixed_rlc(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    rp = pick_value(cfg, "Rp_ohm", params, "Rp")
    lp = pick_value(cfg, "Lp_H", params, "Lp")
    csh = pick_value(cfg, "Csh_F", params, "Csh")
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    return f"""
* load model: plasma_fixed_rlc
.subckt load_model p n
Rbulk p nb {spice_value(rp)}
Lbulk nb ns {spice_value(lp)}
Csh ns n {spice_value(csh)}
Rleak p n {spice_value(rleak)}
.ends load_model
""".strip()


@register("load", "plasma_state_rlc")
def load_plasma_state_rlc(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    ne = float(pick_value(cfg, "electron_density_m3", params, 1e16))
    nu = float(pick_value(cfg, "momentum_collision_Hz", params, 5e7))
    length = float(pick_value(cfg, "bulk_length_m", params, 0.03))
    area = float(pick_value(cfg, "area_m2", params, 0.01))
    csh = pick_value(cfg, "Csh_F", params, 2e-10)
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    e = 1.602176634e-19
    me = 9.1093837015e-31
    lp = length * me / max(area * ne * e * e, 1e-300)
    rp = nu * lp
    return f"""
* load model: plasma_state_rlc
.subckt load_model p n
Rbulk p nb {rp:.12g}
Lbulk nb ns {lp:.12g}
Csh ns n {spice_value(csh)}
Rleak p n {spice_value(rleak)}
.ends load_model
""".strip()


def _pwl_time_expr(pairs: list[tuple[float, float]]) -> str:
    parts: list[str] = []
    for t, v in pairs:
        parts.extend([f"{float(t):.12g}", f"{float(v):.12g}"])
    return "pwl(time, " + ", ".join(parts) + ")"


@register("load", "plasma_table_rlcq")
def load_plasma_table_rlcq(case: Case, params: dict[str, Any]) -> str:
    cfg = _load_cfg(case)
    table = resolve_path(case, cfg["table_file"])
    df = pd.read_csv(table)
    required = {"time_s", "Rp_ohm", "Lp_H", "Csh_F"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"plasma table is missing columns {missing}: {table}")
    df = df.sort_values("time_s")
    rp = _pwl_time_expr(list(zip(df["time_s"], df["Rp_ohm"])))
    lp = _pwl_time_expr(list(zip(df["time_s"], df["Lp_H"])))
    csh = _pwl_time_expr(list(zip(df["time_s"], df["Csh_F"])))
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    return f"""
* load model: plasma_table_rlcq
.subckt load_model p n
Rbulk p nb R = '{rp}'
Lbulk nb ns L = '{lp}'
Csh ns n Q = '({csh})*V(ns,n)'
Rleak p n {spice_value(rleak)}
.ends load_model
""".strip()


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------


@register("solver", "dummy")
def solver_dummy(netlist_path: Path, run_dir: Path, case: Case, params: dict[str, Any]) -> SimulationResult:
    wf = dummy_waveform(case, params)
    return SimulationResult(
        time_s=wf["time_s"].to_numpy(float),
        voltage_V=wf["voltage_V"].to_numpy(float),
        current_A=wf["current_A"].to_numpy(float),
        status="ok",
        log="dummy solver: no SPICE execution",
    )


@register("solver", "ngspice_cli")
def solver_ngspice_cli(netlist_path: Path, run_dir: Path, case: Case, params: dict[str, Any]) -> SimulationResult:
    return ngspice_cli(netlist_path, run_dir, case, params)
