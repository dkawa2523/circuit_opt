from __future__ import annotations

import math
from typing import Any

from .case import Case, resolve_path
from .component_models import series_resistance_ohm
from .component_profiles import profiled_resistor_line
from .netlist import (
    Circuit,
    case_spice_names,
    circuit_config,
    load_config,
    reject_generated_name_collisions,
    spice_names,
)
from .netlist_import import NETLIST_IMPORT_MODES, SOURCE_POLICIES, executable_netlist_lines, flatten_netlist_file
from .probes import core_node, loss_reference, meter_node, meter_reference
from .rf_loads import (
    EFFECTIVE_LOAD_PARAMETERS,
    effective_load_parameters,
    impedance_point,
    impedance_point_reactive_element,
)
from .sim_registry import register, register_solver
from .simulation import SimulationResult
from .simulation_input import SolverRunRequest, resolve_analysis_request
from .solver import ngspice_cli
from .sources import source_name, source_specs
from .spice import fundamental_hz, pick_value, resolve_value, spice_value
from .topology_catalog import matching_topology

# -----------------------------------------------------------------------------
# Circuit builders
# -----------------------------------------------------------------------------


def _component_specs(config: dict[str, Any], section: str) -> list[dict[str, Any]]:
    declared = config.get("components")
    if declared is None:
        return []
    if not isinstance(declared, list):
        raise TypeError(f"{section}.components must be a list of mappings")
    specs: list[dict[str, Any]] = []
    for index, item in enumerate(declared):
        if not isinstance(item, dict):
            raise TypeError(f"{section}.components[{index}] must be a mapping")
        if "raw" not in item:
            missing = sorted({"ref", "n1", "n2", "value"} - set(item))
            if missing:
                raise ValueError(f"{section}.components[{index}] is missing {missing}")
        specs.append(item)
    return specs


@register("circuit", "from_yaml")
def circuit_from_yaml(case: Case, params: dict[str, Any]) -> Circuit:
    cfg = circuit_config(case)
    c = Circuit(output_node=str(cfg.get("output_node", "out")))
    c.params.update(params)
    items = _component_specs(cfg, "circuit")
    generated: list[str] = []
    for item in items:
        if "raw" in item:
            if item.get("observe") or "series_resistance_ohm" in item:
                raise ValueError(
                    "raw circuit components cannot use observe or series_resistance_ohm; "
                    "declare ref/n1/n2/value explicitly"
                )
            c.raw(str(item["raw"]))
        else:
            _add_declared_component(case, c, item, params, generated)
    if generated:
        authored = case_spice_names(case, c.output_node)
        for item in items:
            if "raw" in item:
                authored.update(spice_names(str(item["raw"])))
            else:
                authored.update(str(item[key]) for key in ("ref", "n1", "n2"))
        reject_generated_name_collisions(generated, authored)
    return c


def _add_declared_component(
    case: Case, circuit: Circuit, item: dict[str, Any], params: dict[str, Any], generated: list[str]
) -> None:
    """Add an ideal or effective-series-loss two-terminal component."""

    reference = str(item["ref"])
    n1, n2 = str(item["n1"]), str(item["n2"])
    start = n1
    if item.get("observe"):
        observed_node = meter_node(reference)
        meter = meter_reference(reference)
        generated.extend((meter, observed_node))
        circuit.raw(f"{meter} {n1} {observed_node} DC 0")
        start = observed_node

    resistance = series_resistance_ohm(item, params)
    if resistance is not None and resistance > 0:
        internal = core_node(reference)
        loss = loss_reference(reference)
        generated.extend((loss, internal))
        circuit.raw(f"{loss} {start} {internal} {spice_value(resistance)}")
        start = internal
    profiled = profiled_resistor_line(case, params, reference, start, n2, item["value"])
    if profiled is None:
        circuit.add(reference, start, n2, item["value"])
    else:
        circuit.raw(profiled)


@register("circuit", "l_match")
def circuit_l_match(case: Case, params: dict[str, Any]) -> Circuit:
    return _matching_circuit(case, params, "l_match")


@register("circuit", "pi_match")
def circuit_pi_match(case: Case, params: dict[str, Any]) -> Circuit:
    return _matching_circuit(case, params, "pi_match")


@register("circuit", "pi_match_harmonic")
def circuit_pi_match_harmonic(case: Case, params: dict[str, Any]) -> Circuit:
    return _matching_circuit(case, params, "pi_match_harmonic")


def _matching_circuit(case: Case, params: dict[str, Any], name: str) -> Circuit:
    out = str(circuit_config(case).get("output_node", "electrode"))
    circuit = Circuit(output_node=out)
    circuit.params.update(params)
    for reference, n1, n2 in matching_topology(name).connections(out):
        circuit.add(reference, n1, n2, reference)
    if name == "pi_match_harmonic":
        circuit.notes.append("series LC shunt branch for harmonic shaping")
    return circuit


# -----------------------------------------------------------------------------
# Load builders. Each returns either '' or a subckt named load_model.
# -----------------------------------------------------------------------------


def _impedance_point_number(config: dict[str, Any], params: dict[str, Any], field: str) -> float:
    if field not in config:
        raise ValueError(f"load.{field} is required for impedance_point")
    value = resolve_value(config[field], params)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"load.{field} must resolve to a number, got {value!r}") from exc


@register("load", "none")
def load_none(case: Case, params: dict[str, Any]) -> str:
    return ""


@register("load", "resistor")
def load_resistor(case: Case, params: dict[str, Any]) -> str:
    cfg = load_config(case)
    r = pick_value(cfg, "R_ohm", params, "Rload")
    return f"""
* load model: resistor
.subckt load_model p n
Rload p n {spice_value(r)}
.ends load_model
""".strip()


@register("load", "impedance_point")
def load_impedance_point(case: Case, params: dict[str, Any]) -> str:
    """Realize a measured ``R+jX`` at one declared model frequency."""

    cfg = load_config(case)
    resistance = _impedance_point_number(cfg, params, "resistance_ohm")
    reactance = _impedance_point_number(cfg, params, "reactance_ohm")
    frequency = _impedance_point_number(cfg, params, "model_frequency_Hz")
    analysis = resolve_analysis_request(case, params)
    if analysis.ac is not None:
        if analysis.ac.start_hz != analysis.ac.stop_hz:
            raise ValueError("impedance_point requires a single AC frequency; use solver.ac.frequency_Hz")
        if not math.isclose(frequency, analysis.ac.start_hz, rel_tol=1e-12):
            raise ValueError("impedance_point model_frequency_Hz must equal the requested AC frequency")
    if analysis.transient is not None and not math.isclose(frequency, fundamental_hz(case, params), rel_tol=1e-12):
        raise ValueError("impedance_point model_frequency_Hz must equal the transient fundamental frequency")
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

    values = _resolved_effective_load_parameters(case, params, "ccp_lumped")
    resistance = values["R_eff_ohm"]
    inductance = values["L_eff_H"]
    capacitance = values["C_sheath_eq_F"]
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

    values = _resolved_effective_load_parameters(case, params, "icp_transformer")
    rc = values["R_coil_ohm"]
    lc = values["L_coil_H"]
    reflected = values["reflected_inductance_H"]
    damping = values["secondary_damping_rate_rad_s"]
    cp = values["C_parallel_F"]
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
    if cp > 0:
        lines.append(f"Cparallel p n {spice_value(cp)}")
    lines.append(".ends load_model")
    return "\n".join(lines)


def _resolved_effective_load_parameters(case: Case, params: dict[str, Any], model: str) -> dict[str, float]:
    cfg = load_config(case)
    values = {name: resolve_value(cfg[name], params) for name in EFFECTIVE_LOAD_PARAMETERS[model] if name in cfg}
    return effective_load_parameters(model, values)


@register("load", "from_yaml")
def load_from_yaml(case: Case, params: dict[str, Any]) -> str:
    lines = ["* load model: from_yaml", ".subckt load_model p n"]
    for item in _component_specs(load_config(case), "load"):
        if "raw" in item:
            lines.append(str(item["raw"]))
        else:
            reference, n1, n2 = str(item["ref"]), str(item["n1"]), str(item["n2"])
            profiled = profiled_resistor_line(case, params, reference, n1, n2, item["value"])
            lines.append(
                profiled or f"{reference} {n1} {n2} {spice_value(pick_value(item, 'value', params, item['value']))}"
            )
    lines.append(".ends load_model")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------


@register_solver("ngspice_cli")
def solver_ngspice_cli(request: SolverRunRequest) -> SimulationResult:
    return ngspice_cli(request)


@register("circuit", "from_netlist")
def circuit_from_netlist(case: Case, params: dict[str, Any]) -> Circuit:
    """Use an existing SPICE netlist file as the circuit.

    Circuit statements are retained in authored order, including model,
    parameter, include, conditional, and subcircuit statements.  The case file
    supplies the source, the analysis, and final design-parameter overrides --
    which is what lets a hand-written or exported netlist be simulated, scored
    and optimized by the rest of the platform unchanged.

    ``netlist_mode`` distinguishes a complete SPICE deck (whose first line is
    a title) from a circuit fragment. ``source_policy: replace_named`` replaces
    only independent sources whose names match generated case sources. Other
    sources are retained: node-sharing alone does not prove a conflict, and a
    zero-volt source is the standard way to write an ammeter.
    """

    cfg = circuit_config(case)
    declared = cfg.get("netlist_file")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("circuit.netlist_file is required for from_netlist")
    mode = str(cfg.get("netlist_mode", "fragment")).strip().lower()
    if mode not in NETLIST_IMPORT_MODES:
        raise ValueError(f"circuit.netlist_mode must be one of {sorted(NETLIST_IMPORT_MODES)}")
    source_policy = str(cfg.get("source_policy", "replace_named")).strip().lower()
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(f"circuit.source_policy must be one of {sorted(SOURCE_POLICIES)}")
    path = resolve_path(case, declared)
    flattened, _dependencies = flatten_netlist_file(path, mode=mode)
    imported = executable_netlist_lines(flattened, mode=mode)

    circuit = Circuit(output_node=str(cfg.get("output_node", "out")))
    circuit.params.update(params)
    for item in imported:
        if item.top_level and source_policy == "replace_named" and _matches_case_source_name(case, item.statement):
            circuit.notes.append(f"replaced same-named source line from netlist: {item.text}")
            continue
        circuit.preamble_raw(item.text)
    return circuit


def _matches_case_source_name(case: Case, line: str) -> bool:
    """Whether an independent source has the exact name of a case source."""

    parts = line.split()
    if len(parts) < 3 or parts[0][:1].upper() not in {"V", "I"}:
        return False
    names = {source_name(src, index).lower() for index, src in enumerate(source_specs(case)) if "raw" not in src}
    return parts[0].lower() in names
