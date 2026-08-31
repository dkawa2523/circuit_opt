from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .analysis import ac_sweep
from .case import Case, resolve_path
from .component_models import core_node, loss_reference, meter_node, meter_reference, series_resistance_ohm
from .netlist import Circuit
from .netlist_parse import split_netlist
from .rf_loads import (
    ccp_lumped_impedance,
    icp_effective_impedance,
    impedance_point,
    impedance_point_reactive_element,
)
from .sim_registry import register
from .solver import SimulationResult, dummy_waveform, ngspice_cli
from .spice import fundamental_hz, pick_value, spice_value

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
            if item.get("observe") or "series_resistance_ohm" in item:
                raise ValueError(
                    "raw circuit components cannot use observe or series_resistance_ohm; "
                    "declare ref/n1/n2/value explicitly"
                )
            c.raw(str(item["raw"]))
        else:
            _add_declared_component(c, item, params)
    return c


def _add_declared_component(circuit: Circuit, item: dict[str, Any], params: dict[str, Any]) -> None:
    """Add an ideal or effective-series-loss two-terminal component."""

    reference = str(item["ref"])
    n1, n2 = str(item["n1"]), str(item["n2"])
    start = n1
    if item.get("observe"):
        observed_node = meter_node(reference)
        circuit.raw(f"{meter_reference(reference)} {n1} {observed_node} DC 0")
        start = observed_node

    resistance = series_resistance_ohm(item, params)
    if resistance is not None and resistance > 0:
        internal = core_node(reference)
        circuit.raw(f"{loss_reference(reference)} {start} {internal} {spice_value(resistance)}")
        start = internal
    circuit.add(reference, start, n2, item["value"])


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
    inductance = pick_value(cfg, "L_H", params, "Lload")
    c = pick_value(cfg, "C_F", params, "Cload")
    rleak = pick_value(cfg, "Rleak_ohm", params, 1e12)
    return f"""
* load model: series_rlc
.subckt load_model p n
Rload p nl {spice_value(r)}
Lload nl nc {spice_value(inductance)}
Cload nc n {spice_value(c)}
Rleak p n {spice_value(rleak)}
.ends load_model
""".strip()


@register("load", "impedance_point")
def load_impedance_point(case: Case, params: dict[str, Any]) -> str:
    """Realize a measured ``R+jX`` at one declared model frequency."""

    cfg = _load_cfg(case)
    resistance = float(pick_value(cfg, "resistance_ohm", params, "Rload"))
    reactance = float(pick_value(cfg, "reactance_ohm", params, "Xload"))
    frequency = float(pick_value(cfg, "model_frequency_Hz", params, fundamental_hz(case, params)))
    drive_frequency = fundamental_hz(case, params)
    if not math.isclose(frequency, drive_frequency, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "impedance_point model_frequency_Hz must equal the run's fundamental frequency; "
            "use independent scenarios for independent measured frequency points"
        )
    impedance_point(resistance, reactance)
    reactive = impedance_point_reactive_element(reactance, frequency)
    lines = [
        "* load model: impedance_point",
        f"* exact at {frequency:.12g} Hz; no broadband plasma behavior is implied",
        ".subckt load_model p n",
    ]
    if reactive is None:
        lines.append(f"Rpoint p n {spice_value(resistance)}")
    else:
        kind, value = reactive
        lines.append(f"Rpoint p nx {spice_value(resistance)}")
        lines.append(f"{kind}point nx n {spice_value(value)}")
    lines.append(".ends load_model")
    return "\n".join(lines)


@register("load", "ccp_lumped")
def load_ccp_lumped(case: Case, params: dict[str, Any]) -> str:
    """Effective CCP one-port for a qualified frequency range."""

    cfg = _load_cfg(case)
    resistance = pick_value(cfg, "R_eff_ohm", params, "R_eff")
    inductance = pick_value(cfg, "L_eff_H", params, "L_eff")
    capacitance = pick_value(cfg, "C_sheath_eq_F", params, "C_sheath_eq")
    # Validate numeric values before emitting a netlist.  The frequency value
    # is immaterial to positivity, so the case fundamental is sufficient.
    ccp_lumped_impedance(fundamental_hz(case, params), resistance, inductance, capacitance)
    return f"""
* load model: ccp_lumped (effective port R-L-C, not a plasma-state solver)
.subckt load_model p n
Reffective p nb {spice_value(resistance)}
Leffective nb ns {spice_value(inductance)}
Csheath_eq ns n {spice_value(capacitance)}
.ends load_model
""".strip()


@register("load", "icp_transformer")
def load_icp_transformer(case: Case, params: dict[str, Any]) -> str:
    """ICP coil one-port using identifiable reflected-load parameters."""

    cfg = _load_cfg(case)
    rc = float(pick_value(cfg, "R_coil_ohm", params, "R_coil"))
    lc = float(pick_value(cfg, "L_coil_H", params, "L_coil"))
    cp = float(pick_value(cfg, "C_parallel_F", params, 0.0))
    reflected = float(pick_value(cfg, "reflected_inductance_H", params, "L_reflected"))
    damping = float(pick_value(cfg, "secondary_damping_rate_rad_s", params, "secondary_damping_rate"))
    icp_effective_impedance(fundamental_hz(case, params), rc, lc, reflected, damping, cp)
    # Any L_secondary scale gives the same terminal response.  Choosing
    # L_secondary=L_coil yields a well-scaled numerical realization; these
    # secondary element values are not physical parameters.
    ls = lc
    rs = damping * ls
    coupling = math.sqrt(reflected / lc)
    lines = [
        "* load model: icp_transformer (identifiable effective coil-port fit)",
        f"* reflected_inductance_H={reflected:.12g} secondary_damping_rate_rad_s={damping:.12g}",
        ".subckt load_model p n",
    ]
    coil_node = "np" if rc > 0.0 else "p"
    if rc > 0.0:
        lines.append(f"Rcoil p np {spice_value(rc)}")
    lines += [
        f"Lcoil {coil_node} n {spice_value(lc)}",
        # Reference one point of the otherwise isolated secondary loop to the
        # load return.  A very large resistor leaves the loop almost floating;
        # ngspice's conductance floor can then create several ohms of apparent
        # primary loss for high-Q cases.  A one-point connection fixes only the
        # common-mode voltage and does not add a galvanic current path through
        # the ideal transformer.
        f"Lsecondary n nr {spice_value(ls)}",
        f"Rsecondary nr n {spice_value(rs)}",
        f"Kload Lcoil Lsecondary {spice_value(coupling)}",
    ]
    if float(cp) > 0:
        lines.append(f"Cparallel p n {spice_value(cp)}")
    lines.append(".ends load_model")
    return "\n".join(lines)


@register("load", "from_yaml")
def load_from_yaml(case: Case, params: dict[str, Any]) -> str:
    lines = ["* load model: from_yaml", ".subckt load_model p n"]
    for item in _load_cfg(case).get("components", []) or []:
        if "raw" in item:
            lines.append(str(item["raw"]))
        else:
            lines.append(
                f"{item['ref']} {item['n1']} {item['n2']} {spice_value(pick_value(item, 'value', params, item['value']))}"
            )
    lines.append(".ends load_model")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------


@register("solver", "dummy")
def solver_dummy(netlist_path: Path, run_dir: Path, case: Case, params: dict[str, Any]) -> SimulationResult:
    if ac_sweep(case.data.get("solver", {}) or {}, params) is not None:
        empty = dummy_waveform(case, params).iloc[:0]
        return SimulationResult(
            time_s=empty["time_s"].to_numpy(float),
            voltage_V=empty["voltage_V"].to_numpy(float),
            current_A=empty["current_A"].to_numpy(float),
            status="failed",
            log="dummy solver does not synthesize an AC frequency response",
            diagnostics={"unsupported_analysis": "ac"},
        )
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


@register("circuit", "from_netlist")
def circuit_from_netlist(case: Case, params: dict[str, Any]) -> Circuit:
    """Use an existing SPICE netlist file as the circuit.

    The component lines are taken verbatim, so nothing is lost in translation.
    The case file still supplies the source, the analysis, and the design
    parameters -- which is what lets a hand-written or exported netlist be
    simulated, scored and optimized by the rest of the platform unchanged.

    A source in the file is dropped only when it would actually fight the
    case's source -- same name, or driving the same node.  Other sources are
    kept: a 0 V source is the standard way to write an ammeter, and deleting it
    would silently remove a measurement point.
    """

    cfg = _circuit_cfg(case)
    path = resolve_path(case, cfg["netlist_file"])
    top_lines, subckts = split_netlist(path.read_text(encoding="utf-8"))

    circuit = Circuit(output_node=str(cfg.get("output_node", "out")))
    circuit.params.update(params)
    for name, (ports, body) in subckts.items():
        circuit.raw(f".subckt {name} {' '.join(ports)}")
        for line in body:
            circuit.raw(line)
        circuit.raw(f".ends {name}")

    for line in top_lines:
        if _conflicts_with_case_source(case, line):
            circuit.notes.append(f"ignored conflicting source line from netlist: {line}")
            continue
        circuit.raw(line)
    return circuit


def _conflicts_with_case_source(case: Case, line: str) -> bool:
    """True when this netlist line would drive the same net as the case source."""

    parts = line.split()
    if len(parts) < 3 or parts[0][:1].upper() not in {"V", "I"}:
        return False
    data = case.data
    sources = data.get("sources") or ([data["source"]] if data.get("source") else [])
    for src in sources:
        if not isinstance(src, dict):
            continue
        if parts[0].lower() == str(src.get("name", "Vsrc")).lower():
            return True
        if parts[1] == str(src.get("p", "src")):
            return True
    return False
