from __future__ import annotations

import pytest

from pcd.circuit_graph import CircuitGraph, GraphComponent, GraphPort, GraphTerminal
from pcd.netlist import Circuit, build_circuit, circuit_to_graph


def _component(reference: str, n1: str, n2: str, value: float) -> GraphComponent:
    return GraphComponent(
        reference=reference,
        kind={"R": "resistor", "L": "inductor", "C": "capacitor"}[reference[0]],
        terminals=(GraphTerminal("p", n1), GraphTerminal("n", n2)),
        value=value,
    )


def _graph(*components: GraphComponent) -> CircuitGraph:
    return CircuitGraph(
        topology_family="l_match",
        components=components,
        ports=(GraphPort("source", "src"), GraphPort("load", "out"), GraphPort("ground", "0")),
    )


def test_graph_expands_components_terminals_and_nets_without_framework_types():
    graph = _graph(_component("L1", "src", "out", 1e-6), _component("C1", "out", "0", 1e-9))

    assert graph.net_names == ("0", "out", "src")
    assert [node.kind for node in graph.nodes].count("component") == 2
    assert [node.kind for node in graph.nodes].count("terminal") == 4
    assert [node.kind for node in graph.nodes].count("net") == 3
    assert len(graph.edges) == 8
    assert graph.to_dict()["schema"] == "circuit_graph.v1"


def test_graph_fingerprints_separate_wiring_from_component_values():
    first = _graph(_component("L1", "src", "out", 1e-6), _component("C1", "out", "0", 1e-9))
    reordered = _graph(_component("C1", "out", "0", 2e-9), _component("L1", "src", "out", 2e-6))

    assert first.topology_fingerprint == reordered.topology_fingerprint
    assert first.instance_fingerprint != reordered.instance_fingerprint
    assert first.to_dict() == _graph(*first.components).to_dict()


def test_series_loss_changes_instance_but_not_topology_identity():
    shunt = _component("C1", "out", "0", 1e-9)
    ideal = _graph(_component("L1", "src", "out", 1e-6), shunt)
    lossy_component = GraphComponent(
        reference="L1",
        kind="inductor",
        terminals=(GraphTerminal("p", "src"), GraphTerminal("n", "out")),
        value=1e-6,
        series_resistance_ohm=0.4,
    )
    lossy = _graph(lossy_component, shunt)

    assert ideal.topology_fingerprint == lossy.topology_fingerprint
    assert ideal.instance_fingerprint != lossy.instance_fingerprint
    lossy_record = next(item for item in lossy.to_dict()["components"] if item["reference"] == "L1")
    assert lossy_record["series_resistance_ohm"] == 0.4


def test_structured_circuit_adapter_resolves_design_values_and_ports():
    circuit = Circuit(output_node="electrode", params={"L1": 1e-6, "C1": 1e-9})
    circuit.add("L1", "src", "electrode", "L1")
    circuit.add("C1", "electrode", "0", "C1")

    graph = circuit_to_graph(circuit, "l_match")

    assert [component.value for component in graph.components] == [1e-6, 1e-9]
    assert {port.role: port.net for port in graph.ports} == {
        "source": "src",
        "load": "electrode",
        "ground": "0",
    }


@pytest.mark.parametrize(
    ("topology", "component_count"),
    [("l_match", 2), ("pi_match", 3), ("pi_match_harmonic", 5)],
)
def test_builtin_matching_templates_are_graph_ready(make_case, topology: str, component_count: int):
    case = make_case({"case_id": topology, "circuit": {"builder": topology, "output_node": "electrode"}})
    params = {"L1": 1e-6, "C1": 1e-9, "C2": 2e-9, "Lh": 3e-6, "Ch": 4e-9}
    name, circuit = build_circuit(case, params)

    graph = circuit_to_graph(circuit, name)

    assert graph.topology_family == topology
    assert len(graph.components) == component_count
    assert graph.to_dict()["topology_fingerprint"] == graph.topology_fingerprint


def test_observation_devices_do_not_change_the_physical_graph(make_case):
    case = make_case(
        {
            "case_id": "observed",
            "circuit": {
                "builder": "from_yaml",
                "output_node": "out",
                "components": [
                    {
                        "ref": "L1",
                        "n1": "src",
                        "n2": "out",
                        "value": "L1",
                        "series_resistance_ohm": 0.4,
                        "observe": True,
                    },
                    {"ref": "C1", "n1": "out", "n2": "0", "value": "C1", "observe": True},
                ],
            },
        }
    )
    _name, circuit = build_circuit(case, {"L1": 1e-6, "C1": 1e-9})

    graph = circuit_to_graph(circuit, "l_match")

    assert [component.reference for component in graph.components] == ["L1", "C1"]
    assert graph.components[0].terminals == (GraphTerminal("p", "src"), GraphTerminal("n", "out"))
    assert graph.components[0].series_resistance_ohm == 0.4
    assert all(not component.reference.startswith("Vobserve") for component in graph.components)


def test_graph_contract_rejects_untyped_spice_and_invalid_boundaries():
    circuit = Circuit(output_node="out")
    circuit.raw("B1 src out V=1")
    with pytest.raises(ValueError, match="raw SPICE component has no declared graph semantics"):
        circuit_to_graph(circuit, "custom")

    with pytest.raises(ValueError, match="unknown nets"):
        CircuitGraph(
            topology_family="broken",
            components=(_component("R1", "src", "0", 50.0),),
            ports=(GraphPort("source", "src"), GraphPort("load", "missing"), GraphPort("ground", "0")),
        )


def test_graph_contract_validates_identifiers_values_and_required_structure():
    with pytest.raises(ValueError, match="terminal name must be non-empty"):
        GraphTerminal("", "0")
    with pytest.raises(ValueError, match="at least two terminals"):
        GraphComponent("R1", "resistor", (GraphTerminal("p", "src"),), 50.0)
    with pytest.raises(ValueError, match="duplicate terminal names"):
        GraphComponent(
            "R1",
            "resistor",
            (GraphTerminal("p", "src"), GraphTerminal("p", "0")),
            50.0,
        )
    with pytest.raises(ValueError, match="numeric value must be finite"):
        _component("R1", "src", "0", float("nan"))
    with pytest.raises(ValueError, match="series resistance must be finite and non-negative"):
        GraphComponent(
            "R1",
            "resistor",
            (GraphTerminal("p", "src"), GraphTerminal("n", "0")),
            50.0,
            series_resistance_ohm=-1.0,
        )

    component = _component("R1", "src", "0", 50.0)
    ports = (GraphPort("source", "src"), GraphPort("load", "src"), GraphPort("ground", "0"))
    with pytest.raises(ValueError, match="must contain at least one component"):
        CircuitGraph("empty", (), ports)
    with pytest.raises(ValueError, match="references must be unique"):
        CircuitGraph("duplicate", (component, component), ports)
    with pytest.raises(ValueError, match="port roles must be unique"):
        CircuitGraph(
            "duplicate-port",
            (component,),
            (GraphPort("source", "src"), GraphPort("source", "0"), GraphPort("load", "src"), GraphPort("ground", "0")),
        )
    with pytest.raises(ValueError, match="missing required ports"):
        CircuitGraph("missing-port", (component,), (GraphPort("source", "src"), GraphPort("ground", "0")))
