"""Turning a case into ngspice netlist text.

This is the first half of the simulation pipeline:

    case.yaml -> circuit + load -> netlist text

Nothing here executes anything or touches the filesystem.  Running the netlist
is :mod:`pcd.solver`; recording the run is :mod:`pcd.sim_core`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from .case import Case
from .netlist_import import executable_netlist_lines
from .probes import LOAD_AMMETER, ProbePlan, source_meter_node, source_meter_reference
from .sim_registry import get as get_sim_method
from .sim_registry import load_plugins as load_sim_plugins
from .simulation import AC_FILE, LOAD_CURRENT_COLUMN, WAVEFORM_FILE, AcSweep, AnalysisRequest, TransientAnalysis
from .simulation_input import MeasurementReference, ResolvedSimulationCase, bind_load_probe
from .sources import ac_source_name, source_name, source_specs
from .spice import resolve_value, should_emit_spice_param, spice_value


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
    preamble: list[str] = field(default_factory=list)

    def add(self, ref: str, n1: str, n2: str, value: Any) -> None:
        self.components.append(Component(ref=ref, n1=n1, n2=n2, value=value))

    def raw(self, line: str) -> None:
        self.components.append(Component(raw=line))

    def preamble_raw(self, line: str) -> None:
        """Add an authored statement before generated case parameters."""

        self.preamble.append(line)

    def couple(self, ref: str, first: str, second: str, coefficient: float) -> None:
        """Magnetically couple two inductors, i.e. make them a transformer."""

        self.raw(f"{ref} {first} {second} {spice_value(coefficient)}")

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
        has_unparsed_topology = bool(self.preamble) or any(component.raw is not None for component in self.components)
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
        parameter = str(cfg[variable_key])
        if parameter not in params:
            raise ValueError(f"{variable_key} references missing parameter {parameter!r}")
        return str(params[parameter])
    raw = cfg.get(key, default)
    if isinstance(raw, str) and raw.startswith("$"):
        if raw[1:] not in params:
            raise ValueError(f"{key} references missing parameter {raw!r}")
        return str(params[raw[1:]])
    return str(raw)


def select_circuit_name(case: Case, params: dict[str, Any]) -> str:
    cfg = circuit_config(case)
    return _selected_name(cfg, params, "builder", "builder_variable", "from_yaml")


def select_load_name(case: Case, params: dict[str, Any]) -> str:
    cfg = load_config(case)
    if not cfg:
        return "none"
    return _selected_name(cfg, params, "name", "name_variable", "none")


def build_circuit(case: Case, params: dict[str, Any]) -> tuple[str, Circuit]:
    plugins = load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_circuit_name(case, params)
    method = get_sim_method("circuit", name, plugins=plugins)
    circuit = method(case.detached(), deepcopy(params))
    if not isinstance(circuit, Circuit):
        raise TypeError(f"circuit builder '{name}' must return Circuit")
    return name, circuit


def build_load_subckt(case: Case, params: dict[str, Any]) -> tuple[str, str]:
    plugins = load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_load_name(case, params)
    method = get_sim_method("load", name, plugins=plugins)
    subckt = method(case.detached(), deepcopy(params))
    if subckt is not None and not isinstance(subckt, str):
        raise TypeError(f"load builder '{name}' must return str or None")
    return name, "" if subckt is None else subckt.strip()


def build_netlist_inputs(case: Case, params: dict[str, Any]) -> NetlistInputs:
    """Build the complete in-memory model before any run directory exists."""

    circuit_name, circuit = build_circuit(case, params)
    load_name, load_subckt = build_load_subckt(case, params)
    _check_assembly_names(case, circuit, load_subckt, params)
    return NetlistInputs(circuit_name, circuit, load_name, load_subckt)


def compile_netlist(
    case: Case, params: dict[str, Any], simulation: ResolvedSimulationCase
) -> tuple[NetlistInputs, ResolvedSimulationCase, str]:
    """Compile the same executable deck for preflight, preview and execution."""

    inputs = build_netlist_inputs(case, params)
    probes = bind_load_probe(simulation.probes, attached=bool(inputs.load_subckt))
    simulation = replace(simulation, probes=probes)
    text = render_ngspice_netlist(case, inputs.circuit, inputs.load_subckt, params, simulation)
    return inputs, simulation, text


def spice_names(text: str) -> set[str]:
    """Find authored identifier tokens conservatively, including in raw statements."""

    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("*"):
            continue
        content = re.split(r"\s[;$]", line, maxsplit=1)[0]
        names.update(re.split(r"[\s(),{}=+*/<>;\[\]@'\"]+", content))
    return names


def case_spice_names(case: Case, output_node: str) -> set[str]:
    """Names authored outside the circuit's component declarations."""

    names = {output_node}
    for index, source in enumerate(source_specs(case)):
        if "raw" in source:
            names.update(spice_names(str(source["raw"])))
        else:
            names.update((source_name(source, index), str(source.get("p", "src")), str(source.get("n", "0"))))
    ports = load_config(case).get("ports")
    if isinstance(ports, Mapping):
        names.update(str(ports[key]) for key in ("p", "n") if key in ports)
    return names


def reject_generated_name_collisions(generated: Iterable[str], authored: Iterable[str]) -> None:
    """Keep generated branches and nodes distinct in SPICE's caseless namespace."""

    used = {name.lower() for name in authored}
    for name in generated:
        key = name.lower()
        if key in used:
            raise ValueError(f"generated SPICE name {name!r} collides with a circuit name; rename the conflicting name")
        used.add(key)


def _load_meter_node(positive: str) -> str:
    return f"{positive}_metered"


def _check_load_definition_names(circuit_text: str, params: dict[str, Any]) -> None:
    device_names: set[str] = set()
    # ngspice also reserves subcircuit definition names for parameter lookup.
    model_names = {name for name, value in params.items() if should_emit_spice_param(name, value)}
    subcircuit_depth = 0
    for item in executable_netlist_lines(circuit_text):
        name, *fields = item.statement.split()
        directive = name.lower()
        if directive == ".subckt":
            if not subcircuit_depth and fields:
                model_names.add(fields[0])
            subcircuit_depth += 1
        elif directive == ".ends":
            subcircuit_depth = max(0, subcircuit_depth - 1)
        elif item.top_level:
            if directive == ".param" and re.search(r"(?i)\bload_model\s*=", item.statement):
                model_names.add("load_model")
            else:
                device_names.add(name)
    reject_generated_name_collisions(("Xload",), device_names)
    reject_generated_name_collisions(("load_model",), model_names)


def _check_assembly_names(case: Case, circuit: Circuit, load_subckt: str, params: dict[str, Any]) -> None:
    generated = []
    sources = source_specs(case)
    for index, source in enumerate(sources):
        if source.get("type") == "current_dc":
            name = source_name(source, index)
            generated.extend((source_meter_node(name), source_meter_reference(name)))
    if load_subckt:
        measurement = case.data.get("measurement") or {}
        if measurement.get("load_current") is not None:
            ports = load_config(case).get("ports") or {}
            positive = str(ports.get("p", circuit.output_node))
            generated.extend((LOAD_AMMETER, _load_meter_node(positive)))
    if not generated and not load_subckt:
        return
    circuit_text = "\n".join(
        [
            *circuit.preamble,
            *(str(source["raw"]) for source in sources if "raw" in source),
            *(item.to_spice() for item in circuit.components),
        ]
    )
    if generated:
        authored = case_spice_names(case, circuit.output_node)
        authored.update(spice_names(circuit_text))
        reject_generated_name_collisions(generated, authored)
    if load_subckt:
        _check_load_definition_names(circuit_text, {**circuit.params, **params})


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
            return resolve_value(src[key], params)
    return resolve_value(default, params)


def _sine_peak(src: dict[str, Any], params: dict[str, Any]) -> Any:
    return _pick(src, params, "amplitude_V", "amplitude", default="Vamp")


def _render_sine(name: str, p: str, n: str, src: dict[str, Any], params: dict[str, Any]) -> str:
    dc = _pick(src, params, "dc_V")
    amp = _sine_peak(src, params)
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
        magnitude = resolve_value(src["ac_magnitude"], params)
    elif "ac_magnitude_V" in src:
        magnitude = resolve_value(src["ac_magnitude_V"], params)
    elif "ac_magnitude_A" in src:
        magnitude = resolve_value(src["ac_magnitude_A"], params)
    elif str(src.get("type", "sine_voltage")) in {"sine_voltage", "rf_voltage", "voltage_sine", "sine"}:
        magnitude = _sine_peak(src, params)
    else:
        magnitude = 1.0
    phase = resolve_value(src.get("ac_phase_deg", 0.0), params)
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
    for i, src in enumerate(source_specs(case)):
        if "raw" in src:
            lines.append(str(src["raw"]))
            continue
        typ = str(src.get("type", "sine_voltage"))
        render = SOURCE_RENDERERS.get(typ)
        if render is None:
            raise ValueError(f"unknown source type: {typ}. available={sorted(SOURCE_RENDERERS)}")
        name = source_name(src, i)
        prefix = "I" if typ == "current_dc" else "V"
        if not name.upper().startswith(prefix):
            raise ValueError(f"source {name!r} of type {typ!r} requires a {prefix}-prefixed SPICE name")
        positive = str(src.get("p", "src"))
        measured_positive = source_meter_node(name) if typ == "current_dc" else positive
        line = render(name, measured_positive, str(src.get("n", "0")), src, params)
        if active_source is not None:
            line += _source_ac_suffix(src, params) if name == active_source else " AC 0"
        lines.append(line)
        if typ == "current_dc":
            # @I[current] remains the DC setting during AC analysis.
            # A series zero-volt ammeter measures the actual complex current.
            lines.append(f"{source_meter_reference(name)} {positive} {measured_positive} DC 0")
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
    """Render assembled circuit inputs with their typed analysis request."""

    request = simulation.analysis
    probes = simulation.probes
    measurement = simulation.measurement
    output_node = measurement.voltage_node or circuit.output_node
    load_p = measurement.load_positive or circuit.output_node
    load_n = measurement.load_negative
    load_voltage_vector = f"v({load_p})" if load_n == "0" else f"v({load_p},{load_n})"
    output_vector = f"v({output_node})" if measurement.voltage_node is not None else load_voltage_vector

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
            active_source=ac_source_name(case, request.ac.source) if request.ac is not None else None,
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
    lines += ["", *render_control_lines(request, output_vector, probes, load_voltage_vector=load_voltage_vector)]
    return "\n".join(lines) + "\n"


def _render_transient(analysis: TransientAnalysis) -> str:
    # Repeat the requested sample step as ngspice's explicit tmax.  This makes
    # the numerical resolution independent of user/system `nostepsizelimit`
    # settings and gives time-profile components a checkable upper bound.
    step, stop = spice_value(analysis.step_s), spice_value(analysis.stop_s)
    return f"tran {step} {stop} 0 {step}"


def _render_ac(sweep: AcSweep) -> str:
    return f"ac {sweep.sweep} {sweep.points} {spice_value(sweep.start_hz)} {spice_value(sweep.stop_hz)}"


_SIMPLE_PROBE = re.compile(r"([+-]?)([vi])\(\s*([^(),\s]+)\s*(?:,\s*([^(),\s]+)\s*)?\)", re.IGNORECASE)
_DEVICE_PROBE = re.compile(r"@[^\[\]\s(),+*/]+\[[^\[\]\s]+\]")


def _probe_expression(vector: str) -> tuple[str, tuple[str, ...]]:
    """Return a probe expression and the raw vectors ngspice must save."""

    direct = _SIMPLE_PROBE.fullmatch(vector.strip())
    if direct is None:
        return vector, ("all", *_DEVICE_PROBE.findall(vector))
    sign, kind, positive, negative = direct.groups()
    if kind.lower() == "v":
        if positive == "0":
            if negative in {None, "0"}:
                return "0", ()
            node = f"v({negative})"
            return (node if sign == "-" else f"-{node}"), (node,)
        if negative == "0":
            negative = None
    raw = f"{kind}({positive},{negative})" if negative is not None else f"{kind}({positive})"
    return f"{sign}{raw}", (raw,)


def _wrdata_line(filename: str, vectors: list[str], scale: str, saved: list[str]) -> str:
    fields = [scale] if scale == "time" else []
    for vector in vectors:
        expression, dependencies = _probe_expression(vector)
        saved.extend(dependencies)
        if dependencies == (expression,):
            fields.append(expression)
        else:
            # Parentheses isolate signs from the preceding vector. Adding the
            # zero scale broadcasts constants and promotes real AC expressions
            # to the same complex layout as voltage and branch current.
            fields.append(f"(({expression}) + 0 * {scale})")
    return f"wrdata {filename} {' '.join(fields)}"


def render_control_lines(
    request: AnalysisRequest,
    output_vector: str,
    probes: ProbePlan,
    *,
    load_voltage_vector: str | None = None,
) -> list[str]:
    """Render ngspice control syntax from solver-neutral request/probe types."""

    voltage = output_vector if output_vector.strip().lower().startswith("v(") else f"v({output_vector})"
    source_current = probes.source_current
    transient_vectors = [voltage]
    if source_current is not None:
        transient_vectors.append(source_current)
    transient_vectors.extend((*probes.transient_vectors, load_voltage_vector or voltage))

    saved: list[str] = []
    body: list[str] = []
    if request.transient is not None:
        body += [_render_transient(request.transient), _wrdata_line(WAVEFORM_FILE, transient_vectors, "time", saved)]
    if request.ac is not None:
        source_voltage = probes.source_voltage_vector
        if source_voltage is None or source_current is None:
            raise ValueError("AC analysis requires an observed source voltage and current")
        ac_vectors = [source_voltage, source_current, load_voltage_vector or voltage, voltage, *probes.ac_vectors]
        sweep = request.ac
        write = _wrdata_line(AC_FILE, ac_vectors, "frequency", saved)
        if sweep.sweep == "lin" and sweep.points == 2 and sweep.start_hz != sweep.stop_hz:
            # ngspice 46 collapses LIN 2 to one point. Execute the requested
            # endpoints explicitly, preserving the canonical response domain.
            for index, frequency in enumerate((sweep.start_hz, sweep.stop_hz)):
                point = spice_value(frequency)
                body += [f"ac lin 1 {point} {point}", *(["set appendwrite"] if index else []), write]
            body.append("unset appendwrite")
        else:
            body += [_render_ac(sweep), write]
    lines = [
        f".save {' '.join(dict.fromkeys(saved))}",
        ".control",
        # Impedance divides voltage by current.  The default wrdata precision
        # loses the small in-phase current of high-Q RF loads.
        "set numdgt=15",
    ]
    lines += [*body, "quit", ".endc", ".end"]
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

    When load-current measurement is requested, the zero-volt ammeter measures
    current entering this load separately from any upstream parallel paths.
    """

    p = measurement.load_positive or output_node
    n = measurement.load_negative
    if load_current_column != LOAD_CURRENT_COLUMN:
        return [f"Xload {p} {n} load_model"]
    metered = _load_meter_node(p)
    return [f"{LOAD_AMMETER} {p} {metered} DC 0", f"Xload {metered} {n} load_model"]
