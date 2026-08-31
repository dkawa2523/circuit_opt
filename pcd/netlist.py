"""Turning a case into ngspice netlist text.

This is the first half of the simulation pipeline:

    case.yaml -> circuit + load -> netlist text

Nothing here executes anything or touches the filesystem.  Running the netlist
is :mod:`pcd.solver`; recording the run is :mod:`pcd.sim_core`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .analysis import (
    LOAD_AMMETER,
    LOAD_CURRENT,
    ac_probe_plan,
    ac_sweep,
    control_lines,
    load_current,
    probe_plan,
    source_node,
)
from .case import Case
from .sim_registry import get as get_sim_method
from .sim_registry import load_plugins as load_sim_plugins
from .spice import param_ref_or_value, should_emit_spice_param, spice_value


@dataclass
class Component:
    """One SPICE line: either a two-terminal element or a verbatim line."""

    ref: str | None = None
    n1: str | None = None
    n2: str | None = None
    value: Any = None
    raw: str | None = None

    def to_spice(self) -> str:
        if self.raw is not None:
            return self.raw
        if self.ref is None or self.n1 is None or self.n2 is None or self.value is None:
            raise ValueError(f"invalid component: {self}")
        return f"{self.ref} {self.n1} {self.n2} {spice_value(self.value)}"


@dataclass
class Circuit:
    """What a circuit builder returns.  This is the plugin-facing type."""

    components: list[Component] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    output_node: str = "out"
    ground: str = "0"
    notes: list[str] = field(default_factory=list)

    def add(self, ref: str, n1: str, n2: str, value: Any) -> None:
        self.components.append(Component(ref=str(ref), n1=str(n1), n2=str(n2), value=value))

    def raw(self, line: str) -> None:
        self.components.append(Component(raw=str(line)))

    def couple(self, ref: str, first: str, second: str, coefficient: float) -> None:
        """Magnetically couple two inductors, i.e. make them a transformer."""

        self.components.append(Component(raw=f"{ref} {first} {second} {spice_value(coefficient)}"))

    def nodes(self) -> set[str]:
        nodes = {self.ground}
        for comp in self.components:
            if comp.n1:
                nodes.add(comp.n1)
            if comp.n2:
                nodes.add(comp.n2)
        return nodes

    def warnings(self) -> list[str]:
        refs = [c.ref for c in self.components if c.ref]
        out: list[str] = []
        if len(refs) != len(set(refs)):
            out.append("duplicate component reference names detected")
        if self.output_node not in self.nodes():
            out.append(f"output_node '{self.output_node}' does not appear in two-terminal components")
        return out


# -----------------------------------------------------------------------------
# Choosing and building the circuit and load
# -----------------------------------------------------------------------------


def _selected_name(cfg: dict[str, Any], params: dict[str, Any], key: str, variable_key: str, default: str) -> str:
    """Resolve a method name that may be fixed, `$param`, or named by a variable.

    This is what makes topology and load model usable as categorical design
    variables rather than fixed settings.
    """

    if variable_key in cfg:
        return str(params.get(str(cfg[variable_key]), cfg.get(key, default)))
    raw = cfg.get(key, default)
    if isinstance(raw, str) and raw.startswith("$"):
        return str(params.get(raw[1:], default))
    return str(raw)


def select_circuit_name(case: Case, params: dict[str, Any]) -> str:
    cfg = case.data.get("circuit", {}) or {}
    key = "builder" if "builder" in cfg or "topology" not in cfg else "topology"
    return _selected_name(cfg, params, key, "builder_variable", "from_yaml")


def select_load_name(case: Case, params: dict[str, Any]) -> str:
    cfg = case.data.get("load", {}) or {}
    if not cfg:
        return "none"
    return _selected_name(cfg, params, "name", "name_variable", "none")


def build_circuit(case: Case, params: dict[str, Any]) -> tuple[str, Circuit]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_circuit_name(case, params)
    circuit = get_sim_method("circuit", name)(case, params)
    if not isinstance(circuit, Circuit):
        raise TypeError(f"circuit builder '{name}' must return Circuit")
    return name, circuit


def build_load_subckt(case: Case, params: dict[str, Any]) -> tuple[str, str]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_load_name(case, params)
    subckt = get_sim_method("load", name)(case, params)
    return name, "" if subckt is None else str(subckt).strip()


# -----------------------------------------------------------------------------
# Sources
#
# One renderer per source type.  Adding a source type means adding a function
# and one SOURCE_RENDERERS entry -- no branch to extend.
# -----------------------------------------------------------------------------


def _pick(src: dict[str, Any], params: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    """First present key among aliases, resolved through the design params."""

    for key in keys:
        if key in src:
            return param_ref_or_value(src[key], params)
    return param_ref_or_value(default, params)


def _render_sine(name: str, p: str, n: str, src: dict[str, Any], params: dict[str, Any]) -> str:
    dc = _pick(src, params, "dc_V")
    amp = _pick(src, params, "amplitude_V", "amplitude", default="Vamp")
    freq = _pick(src, params, "frequency_Hz", "frequency", default="freq")
    phase = _pick(src, params, "phase_deg")
    return f"{name} {p} {n} SIN({spice_value(dc)} {spice_value(amp)} {spice_value(freq)} 0 0 {spice_value(phase)})"


def _render_dc_voltage(name: str, p: str, n: str, src: dict[str, Any], params: dict[str, Any]) -> str:
    return f"{name} {p} {n} DC {spice_value(_pick(src, params, 'voltage_V', 'value_V', 'value'))}"


def _render_dc_current(name: str, p: str, n: str, src: dict[str, Any], params: dict[str, Any]) -> str:
    return f"{name} {p} {n} DC {spice_value(_pick(src, params, 'current_A', 'value_A', 'value'))}"


def _render_pulse(name: str, p: str, n: str, src: dict[str, Any], params: dict[str, Any]) -> str:
    fields = [
        ("v1_V", 0.0),
        ("v2_V", 1.0),
        ("delay_s", 0.0),
        ("rise_s", 1e-9),
        ("fall_s", 1e-9),
        ("width_s", 1e-6),
        ("period_s", 2e-6),
    ]
    values = " ".join(spice_value(_pick(src, params, key, default=default)) for key, default in fields)
    return f"{name} {p} {n} PULSE({values})"


SourceRenderer = Callable[[str, str, str, dict[str, Any], dict[str, Any]], str]

SOURCE_RENDERERS: dict[str, SourceRenderer] = {
    "sine_voltage": _render_sine,
    "rf_voltage": _render_sine,
    "voltage_sine": _render_sine,
    "sine": _render_sine,
    "dc_voltage": _render_dc_voltage,
    "voltage_dc": _render_dc_voltage,
    "dc": _render_dc_voltage,
    "voltage_pulse": _render_pulse,
    "pulse": _render_pulse,
    "current_dc": _render_dc_current,
}


def _case_sources(case: Case) -> list[dict[str, Any]]:
    """A case may declare `sources:` (a list) or `source:` (a single mapping)."""

    if "sources" in case.data:
        return case.data.get("sources") or []
    if "source" in case.data:
        return [case.data.get("source") or {}]
    return []


def _source_ac_suffix(src: dict[str, Any], params: dict[str, Any]) -> str:
    """Use an explicit AC magnitude, or the sine peak amplitude, for stress."""

    if "ac_magnitude_V" in src:
        magnitude = param_ref_or_value(src["ac_magnitude_V"], params)
    elif str(src.get("type", "sine_voltage")) in {"sine_voltage", "rf_voltage", "voltage_sine", "sine"}:
        magnitude = _pick(src, params, "amplitude_V", "amplitude", default=1.0)
    else:
        magnitude = 1.0
    return f" AC {spice_value(magnitude)}"


def render_source(case: Case, params: dict[str, Any], ac_enabled: bool = False) -> list[str]:
    """Render each source, using physical peak magnitude for AC when enabled.

    The impedance ratio is independent of source magnitude, while component
    voltage, current, and loss are not.  A sine therefore uses its declared
    peak amplitude unless ``ac_magnitude_V`` explicitly overrides it.
    """

    lines: list[str] = []
    for i, src in enumerate(_case_sources(case)):
        if not src:
            continue
        if "raw" in src:
            lines.append(str(src["raw"]))
            continue
        typ = str(src.get("type", "sine_voltage"))
        render = SOURCE_RENDERERS.get(typ)
        if render is None:
            raise ValueError(f"unknown source type: {typ}. available={sorted(SOURCE_RENDERERS)}")
        line = render(str(src.get("name", f"Vsrc{i}")), str(src.get("p", "src")), str(src.get("n", "0")), src, params)
        lines.append(line + (_source_ac_suffix(src, params) if ac_enabled else ""))
    return lines


# -----------------------------------------------------------------------------
# Netlist assembly
# -----------------------------------------------------------------------------


def render_ngspice_netlist(
    case: Case,
    circuit: Circuit,
    load_subckt: str,
    params: dict[str, Any],
    waveform_file: str = "waveform.csv",
) -> str:
    """Assemble the full netlist, including the .control transient block."""

    solver_cfg = case.data.get("solver", {}) or {}
    meas = case.data.get("measurement", {}) or {}
    output_node = str(meas.get("voltage_node", circuit.output_node))
    source = str(meas.get("current_source", "Vsrc"))

    ports = (case.data.get("load", {}) or {}).get("ports", {}) or {}
    load_p = str(ports.get("p", circuit.output_node))
    load_n = str(ports.get("n", "0"))
    if "voltage_node" in meas:
        output_vector = f"v({output_node})"
    else:
        output_vector = f"v({load_p})" if load_n == "0" else f"v({load_p},{load_n})"

    lines = [f"* Auto-generated simulation netlist for case: {case.case_id}"]
    lines += _header_lines(circuit, params, solver_cfg)
    lines += ["", "* Sources", *render_source(case, params, ac_enabled=bool(ac_sweep(solver_cfg, params)))]
    lines += ["", "* Circuit", *(comp.to_spice() for comp in circuit.components)]
    if load_subckt:
        lines += ["", "* Optional load", load_subckt, *_load_instance(case, circuit.output_node)]
    vectors, _columns = probe_plan(case)
    ac_vectors, _ac_columns = ac_probe_plan(case)
    lines += ["", *control_lines(solver_cfg, output_vector, source, source_node(case), vectors, params, ac_vectors)]
    return "\n".join(lines) + "\n"


def _header_lines(circuit: Circuit, params: dict[str, Any], solver_cfg: dict[str, Any]) -> list[str]:
    """`.param` for every design value, plus any `.options` the case sets."""

    lines = [
        f".param {name}={spice_value(value)}"
        for name, value in sorted({**circuit.params, **params}.items())
        if should_emit_spice_param(str(name), value)
    ]
    options = solver_cfg.get("options", {}) or {}
    if options:
        lines.append(".options " + " ".join(f"{k}={v}" for k, v in options.items()))
    return lines


def _load_instance(case: Case, output_node: str) -> list[str]:
    """Wire the load subcircuit in, optionally through an ammeter.

    `measurement.load_current: auto` inserts a zero-volt source in series with
    the load.  It changes no voltage, and it is the only way to measure the
    current actually entering the electrode: the standard `i(Vsrc)` channel
    also carries whatever the matching network's shunt elements draw.
    """

    ports = (case.data.get("load", {}) or {}).get("ports", {}) or {}
    p, n = str(ports.get("p", output_node)), str(ports.get("n", "0"))
    if load_current(case) != LOAD_CURRENT:
        return [f"Xload {p} {n} load_model"]
    metered = f"{p}_metered"
    return [f"{LOAD_AMMETER} {p} {metered} DC 0", f"Xload {metered} {n} load_model"]
