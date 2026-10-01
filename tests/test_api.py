"""The small, documented Python and plugin extension surface."""

from __future__ import annotations

from pcd.api import (
    AnalysisRequest,
    Case,
    Circuit,
    MeasurementReference,
    NamedProbe,
    ProbePlan,
    ResolvedSimulationCase,
    SimulationResult,
    SolverRunRequest,
    SolverSettings,
    register_simulation,
    register_solver,
)
from pcd.netlist import build_circuit


def test_public_api_exposes_core_extension_types():
    assert Case.__name__ == "Case"
    assert Circuit.__name__ == "Circuit"
    assert MeasurementReference.__name__ == "MeasurementReference"
    assert NamedProbe.__name__ == "NamedProbe"
    assert ProbePlan.__name__ == "ProbePlan"
    assert AnalysisRequest.__name__ == "AnalysisRequest"
    assert SimulationResult.__name__ == "SimulationResult"
    assert ResolvedSimulationCase.__name__ == "ResolvedSimulationCase"
    assert SolverRunRequest.__name__ == "SolverRunRequest"
    assert SolverSettings.__name__ == "SolverSettings"
    assert callable(register_solver)


def test_a_case_owns_the_mapping_passed_to_its_constructor(tmp_path):
    source = {"case_id": "owned", "circuit": {"builder": "from_yaml"}}
    case = Case(tmp_path / "case.yaml", source)

    source["circuit"]["builder"] = "changed"

    assert case.data["circuit"]["builder"] == "from_yaml"


def test_extension_code_receives_an_isolated_case_and_params(make_case):
    @register_simulation("circuit", "test_mutating_extension")
    def mutating_extension(case, params):
        case.data["mutated"] = True
        params["mutated"] = True
        return Circuit(output_node="out")

    case = make_case({"circuit": {"builder": "test_mutating_extension"}})
    params = {"R": 1.0}

    build_circuit(case, params)

    assert "mutated" not in case.data
    assert "mutated" not in params
