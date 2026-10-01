"""Turning a case into ngspice netlist text.

This is the first half of the simulation pipeline:

    case.yaml -> circuit + load -> netlist text

Nothing here executes anything or touches the filesystem.  Running the netlist
is :mod:`pcd.solver`; recording the run is :mod:`pcd.sim_core`.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .case import Case
from .circuit_graph import CircuitGraph, GraphComponent, GraphPort, GraphTerminal
from .probes import LOAD_AMMETER, ProbePlan
from .sim_registry import get as get_sim_method
from .sim_registry import load_plugins as load_sim_plugins
from .simulation import AC_FILE, LOAD_CURRENT_COLUMN, WAVEFORM_FILE, AcSweep, AnalysisRequest, TransientAnalysis
from .simulation_input import MeasurementReference, ResolvedSimulationCase, resolve_source_specs
from .spice import param_ref_or_value, should_emit_spice_param, spice_value


@dataclass
class Component:
    """One SPICE line: either a two-terminal element or a verbatim line."""

    ref: str | None = None
    n1: str | None = None
    n2: str | None = None
    value: Any = None
    raw: str | None = None
    graph_neutral: bool = False

    def to_spice(self) -> str:
        if self.raw is not None:
            return self.raw
        if self.ref is None or self.n1 is None or self.n2 is None or self.value is None:
            raise ValueError(f"invalid component: {self}")
        return f"{self.ref} {self.n1} {self.n2} {spice_value(self.value)}"


@dataclass(frozen=True)
class PhysicalComponent:
    """One logical circuit element before solver instrumentation is added."""

    ref: str
    n1: str
    n2: str
    value: Any
    series_resistance_ohm: float | None = None


@dataclass
class Circuit:
    """What a circuit builder returns.  This is the plugin-facing type."""

    components: list[Component] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    output_node: str = "out"
    ground: str = "0"
    notes: list[str] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)
    graph_components: list[PhysicalComponent] = field(default_factory=list)
    graph_issues: list[str] = field(default_factory=list)

    def add(self, ref: str, n1: str, n2: str, value: Any, *, include_in_graph: bool = True) -> None:
        self.components.append(
            Component(
                ref=ref,
                n1=n1,
                n2=n2,
                value=value,
                graph_neutral=not include_in_graph,
            )
        )
        if include_in_graph:
            self.add_graph_component(ref, n1, n2, value)

    def add_graph_component(
        self,
        ref: str,
        n1: str,
        n2: str,
        value: Any,
        *,
        series_resistance_ohm: float | None = None,
    ) -> None:
        """Declare one physical element independently of its SPICE rendering."""

        self.graph_components.append(
            PhysicalComponent(
                ref=ref,
                n1=n1,
                n2=n2,
                value=value,
                series_resistance_ohm=series_resistance_ohm,
            )
        )

    def mark_graph_unsupported(self, reason: str) -> None:
        reason = reason.strip()
        if reason and reason not in self.graph_issues:
            self.graph_issues.append(reason)

    def raw(self, line: str, *, graph_neutral: bool = False) -> None:
        self.components.append(Component(raw=line, graph_neutral=graph_neutral))
        if not graph_neutral:
            self.mark_graph_unsupported("raw SPICE component has no declared graph semantics")

    def preamble_raw(self, line: str) -> None:
        """Add an authored statement before generated case parameters."""

        self.preamble.append(line)
        self.mark_graph_unsupported("imported SPICE preamble has no declared graph semantics")

    def couple(self, ref: str, first: str, second: str, coefficient: float) -> None:
        """Magnetically couple two inductors, i.e. make them a transformer."""

        self.components.append(Component(raw=f"{ref} {first} {second} {spice_value(coefficient)}"))
        self.mark_graph_unsupported(f"magnetic coupling {ref!r} is not represented by circuit_graph.v1")

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
        # Imported/raw SPICE carries topology outside ``components``.  In that
        # mode this check cannot prove that the output is absent, so avoid a
        # misleading warning while retaining the check for structured cases.
        has_unparsed_topology = bool(self.preamble) or any(
            component.raw is not None and not component.graph_neutral for component in self.components
        )
        if self.output_node not in self.nodes() and not has_unparsed_topology:
            out.append(f"output_node '{self.output_node}' does not appear in two-terminal components")
        return out


@dataclass(frozen=True)
class NetlistInputs:
    """In-memory circuit and load produced before allocating run artifacts."""

    circuit_name: str
    circuit: Circuit
    load_name: str
    load_subckt: str


_GRAPH_COMPONENT_KINDS = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "D": "diode",
    "V": "voltage_source",
    "I": "current_source",
}


def circuit_to_graph(
    circuit: Circuit,
    topology_family: str,
    *,
    source_node: str = "src",
) -> CircuitGraph:
    """Project a structured circuit builder result into the graph contract.

    The projection uses physical component declarations rather than rendered
    SPICE lines.  This keeps zero-volt observation sources and internal loss
    nodes out of the learned topology.  Authored raw SPICE remains unsupported
    unless a builder supplies equivalent physical declarations explicitly.
    """

    issues = _graph_issues(circuit)
    if issues:
        detail = "; ".join(dict.fromkeys(issues))
        raise ValueError(f"circuit graph is unavailable: {detail}")
    return CircuitGraph(
        topology_family=topology_family,
        components=tuple(_graph_component(circuit, component) for component in _physical_components(circuit)),
        ports=(
            GraphPort("source", source_node),
            GraphPort("load", circuit.output_node),
            GraphPort("ground", circuit.ground),
        ),
    )


def _graph_issues(circuit: Circuit) -> list[str]:
    issues = list(circuit.graph_issues)
    if any(component.raw is not None and not component.graph_neutral for component in circuit.components):
        issues.append("raw SPICE component has no declared graph semantics")
    return issues


def _physical_components(circuit: Circuit) -> list[PhysicalComponent]:
    physical = list(circuit.graph_components)
    declared_refs = {component.ref for component in physical}
    for component in circuit.components:
        if component.graph_neutral or component.raw is not None or component.ref in declared_refs:
            continue
        if component.ref is None or component.n1 is None or component.n2 is None or component.value is None:
            raise ValueError(f"circuit graph cannot represent incomplete component: {component}")
        physical.append(PhysicalComponent(component.ref, component.n1, component.n2, component.value))
        declared_refs.add(component.ref)
    return physical


def _graph_component(circuit: Circuit, component: PhysicalComponent) -> GraphComponent:
    value = (
        circuit.params.get(component.value, component.value) if isinstance(component.value, str) else component.value
    )
    return GraphComponent(
        reference=component.ref,
        kind=_GRAPH_COMPONENT_KINDS.get(component.ref[:1].upper(), "component"),
        terminals=(GraphTerminal("p", component.n1), GraphTerminal("n", component.n2)),
        value=value,
        series_resistance_ohm=component.series_resistance_ohm,
    )


# -----------------------------------------------------------------------------
# Choosing and building the circuit and load
# -----------------------------------------------------------------------------


def circuit_config(case: Case) -> dict[str, Any]:
    config = case.data.get("circuit")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError("circuit must be a mapping")
    return config


def load_config(case: Case) -> dict[str, Any]:
    config = case.data.get("load")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError("load must be a mapping")
    return config


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
    cfg = circuit_config(case)
    key = "builder" if "builder" in cfg or "topology" not in cfg else "topology"
    return _selected_name(cfg, params, key, "builder_variable", "from_yaml")


def select_load_name(case: Case, params: dict[str, Any]) -> str:
    cfg = load_config(case)
    if not cfg:
        return "none"
    return _selected_name(cfg, params, "name", "name_variable", "none")


def build_circuit(case: Case, params: dict[str, Any]) -> tuple[str, Circuit]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_circuit_name(case, params)
    circuit = get_sim_method("circuit", name)(case.detached(), deepcopy(params))
    if not isinstance(circuit, Circuit):
        raise TypeError(f"circuit builder '{name}' must return Circuit")
    return name, circuit


def build_load_subckt(case: Case, params: dict[str, Any]) -> tuple[str, str]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_load_name(case, params)
    subckt = get_sim_method("load", name)(case.detached(), deepcopy(params))
    if subckt is not None and not isinstance(subckt, str):
        raise TypeError(f"load builder '{name}' must return str or None")
    return name, "" if subckt is None else subckt.strip()


def build_netlist_inputs(case: Case, params: dict[str, Any]) -> NetlistInputs:
    """Build the complete in-memory model before any run directory exists."""

    circuit_name, circuit = build_circuit(case, params)
    load_name, load_subckt = build_load_subckt(case, params)
    return NetlistInputs(circuit_name, circuit, load_name, load_subckt)


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


def _source_ac_suffix(src: dict[str, Any], params: dict[str, Any]) -> str:
    """Use an explicit AC magnitude, or the sine peak amplitude, for stress."""

    if "ac_magnitude" in src:
        magnitude = param_ref_or_value(src["ac_magnitude"], params)
    elif "ac_magnitude_V" in src:
        magnitude = param_ref_or_value(src["ac_magnitude_V"], params)
    elif "ac_magnitude_A" in src:
        magnitude = param_ref_or_value(src["ac_magnitude_A"], params)
    elif str(src.get("type", "sine_voltage")) in {"sine_voltage", "rf_voltage", "voltage_sine", "sine"}:
        magnitude = _pick(src, params, "amplitude_V", "amplitude", default=1.0)
    else:
        magnitude = 1.0
    phase = param_ref_or_value(src.get("ac_phase_deg", 0.0), params)
    return f" AC {spice_value(magnitude)} {spice_value(phase)}"


def render_source(
    case: Case,
    params: dict[str, Any],
    active_source: str | None = None,
) -> list[str]:
    """Render each source, using physical peak magnitude for AC when enabled.

    The impedance ratio is independent of source magnitude, while component
    voltage, current, and loss are not.  A sine therefore uses its declared
    peak amplitude unless ``ac_magnitude_V`` explicitly overrides it.
    """

    lines: list[str] = []
    for i, src in enumerate(resolve_source_specs(case)):
        if not src:
            continue
        if "raw" in src:
            lines.append(str(src["raw"]))
            continue
        typ = str(src.get("type", "sine_voltage"))
        render = SOURCE_RENDERERS.get(typ)
        if render is None:
            raise ValueError(f"unknown source type: {typ}. available={sorted(SOURCE_RENDERERS)}")
        name = str(src.get("name", "Vsrc" if i == 0 else f"Vsrc{i}"))
        line = render(name, str(src.get("p", "src")), str(src.get("n", "0")), src, params)
        if active_source is not None:
            line += _source_ac_suffix(src, params) if name == active_source else " AC 0"
        lines.append(line)
    return lines


# -----------------------------------------------------------------------------
# Netlist assembly
# -----------------------------------------------------------------------------


def render_ngspice_netlist(
    case: Case,
    circuit: Circuit,
    load_subckt: str,
    params: dict[str, Any],
    simulation: ResolvedSimulationCase,
) -> str:
    """Assemble the full netlist, including its typed analysis request."""

    request = simulation.analysis
    probes = simulation.probes
    measurement = simulation.measurement
    output_node = measurement.voltage_node or circuit.output_node
    load_p = measurement.load_positive or circuit.output_node
    load_n = measurement.load_negative
    if measurement.voltage_node is not None:
        output_vector = f"v({output_node})"
    else:
        output_vector = f"v({load_p})" if load_n == "0" else f"v({load_p},{load_n})"

    lines = [f"* Auto-generated simulation netlist for case: {case.case_id}"]
    if circuit.preamble:
        lines += ["", "* Imported circuit deck", *circuit.preamble]
    lines += _header_lines(circuit, params, simulation.netlist_options)
    lines += [
        "",
        "* Sources",
        *render_source(
            case,
            params,
            active_source=probes.source_name if request.ac is not None else None,
        ),
    ]
    lines += ["", "* Circuit", *(comp.to_spice() for comp in circuit.components)]
    if load_subckt:
        lines += [
            "",
            "* Optional load",
            load_subckt,
            *_load_instance(circuit.output_node, probes.load_current_column, measurement),
        ]
    lines += ["", *render_control_lines(request, output_vector, probes)]
    return "\n".join(lines) + "\n"


def _render_transient(analysis: TransientAnalysis) -> str:
    # Repeat the requested sample step as ngspice's explicit tmax.  This makes
    # the numerical resolution independent of user/system `nostepsizelimit`
    # settings and gives time-profile components a checkable upper bound.
    return f"tran {analysis.step_s:g} {analysis.stop_s:g} 0 {analysis.step_s:g}"


def _render_ac(sweep: AcSweep) -> str:
    return f"ac {sweep.sweep} {sweep.points} {sweep.start_hz:g} {sweep.stop_hz:g}"


def render_control_lines(request: AnalysisRequest, output_vector: str, probes: ProbePlan) -> list[str]:
    """Render ngspice control syntax from solver-neutral request/probe types."""

    voltage = output_vector if output_vector.strip().lower().startswith("v(") else f"v({output_vector})"
    source_current = f"i({probes.source_name})"
    transient_vectors = [voltage, source_current, *probes.transient_vectors]
    ac_vectors = [probes.source_voltage_vector, source_current, voltage, *probes.ac_vectors]

    saved: list[str] = []
    if request.transient is not None:
        saved.extend(transient_vectors)
    if request.ac is not None:
        saved.extend(ac_vectors)
    saved = list(dict.fromkeys(saved))

    lines = [
        f".save {' '.join(saved)}",
        ".control",
        # Impedance divides voltage by current.  The default wrdata precision
        # loses the small in-phase current of high-Q RF loads.
        "set numdgt=15",
    ]
    if request.transient is not None:
        lines += [_render_transient(request.transient), f"wrdata {WAVEFORM_FILE} time {' '.join(transient_vectors)}"]
    if request.ac is not None:
        lines += [_render_ac(request.ac), f"wrdata {AC_FILE} {' '.join(ac_vectors)}"]
    lines += ["quit", ".endc", ".end"]
    return lines


def _header_lines(
    circuit: Circuit,
    params: dict[str, Any],
    options: tuple[tuple[str, Any], ...],
) -> list[str]:
    """`.param` for every design value, plus any `.options` the case sets."""

    lines = [
        f".param {name}={spice_value(value)}"
        for name, value in sorted({**circuit.params, **params}.items())
        if should_emit_spice_param(name, value)
    ]
    if options:
        lines.append(".options " + " ".join(f"{name}={value}" for name, value in options))
    return lines


def _load_instance(
    output_node: str,
    load_current_column: str | None,
    measurement: MeasurementReference,
) -> list[str]:
    """Wire the load subcircuit in, optionally through an ammeter.

    `measurement.load_current: auto` inserts a zero-volt source in series with
    the load.  It changes no voltage, and it is the only way to measure the
    current actually entering the electrode: the standard `i(Vsrc)` channel
    also carries whatever the matching network's shunt elements draw.
    """

    p = measurement.load_positive or output_node
    n = measurement.load_negative
    if load_current_column != LOAD_CURRENT_COLUMN:
        return [f"Xload {p} {n} load_model"]
    metered = f"{p}_metered"
    return [f"{LOAD_AMMETER} {p} {metered} DC 0", f"Xload {metered} {n} load_model"]
