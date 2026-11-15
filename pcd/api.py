"""Supported extension surface for PCD plugins and Python callers.

Imports from this module are kept stable within the current major version.
Implementation modules remain available for application development, but
third-party plugins should not depend on their private helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .analysis.network import PortExcitation, PortReference, ScatteringResult, solve_scattering
from .case import Case, load_case
from .core import (
    Candidate,
    CandidateResult,
    ConstraintResult,
    ControlState,
    EvaluationRequest,
    EvaluationResult,
    MetricSet,
    Objective,
    RawResult,
    Scenario,
    ScenarioResult,
    StudySpec,
)
from .identification import run_identification
from .metric_registry import register as register_metric
from .metrics import measure_response
from .netlist import Circuit, Component
from .network import run_network
from .probes import NamedProbe, ProbePlan
from .search import BaseOptimizer
from .search_registry import register as register_optimizer
from .sim_core import SimRecord, SimulationRun, execute_case, prepare_case, write_sim_record
from .sim_registry import register as register_simulation
from .sim_registry import register_solver
from .simulation import AnalysisRequest, SimulationResult
from .simulation_input import MeasurementReference, ResolvedSimulationCase, SolverRunRequest, SolverSettings
from .study import run_case_study
from .study_config import describe_case, select_simulation_inputs


def simulate_case(
    case: Case,
    params: Mapping[str, Any] | None = None,
    run_root: str | Path | None = None,
    solver_override: str | None = None,
    run_id: str | None = None,
    case_archive_root: str | Path | None = None,
    *,
    scenario_id: str | None = None,
    controls: Mapping[str, Any] | None = None,
) -> SimRecord:
    """Run one selected point; the public API never silently reduces a study."""

    from .output import output_settings

    output_settings(case.data.get("output"))
    record = execute_case(
        case,
        run_root=run_root,
        solver_override=solver_override,
        run_id=run_id,
        case_archive_root=case_archive_root,
        select_inputs=lambda snapshot: select_simulation_inputs(
            snapshot, params, scenario_id=scenario_id, controls=controls
        ),
    ).record
    if record.status == "ok" and "output" in case.data:
        from .reporting import analyze_run, render_requested_output

        def render() -> dict[str, Any]:
            analysis = analyze_run(record.run_dir)
            summary = Path(analysis["output_dir"]).relative_to(record.run_dir) / "summary.json"
            return {"status": analysis["status"], "summary": summary.as_posix()}

        record.output_result = render_requested_output(render)
        write_sim_record(record)
    return record


__all__ = [
    "AnalysisRequest",
    "BaseOptimizer",
    "Candidate",
    "CandidateResult",
    "Case",
    "Circuit",
    "Component",
    "ConstraintResult",
    "ControlState",
    "EvaluationRequest",
    "EvaluationResult",
    "MeasurementReference",
    "MetricSet",
    "NamedProbe",
    "Objective",
    "PortExcitation",
    "PortReference",
    "ProbePlan",
    "RawResult",
    "ResolvedSimulationCase",
    "ScatteringResult",
    "Scenario",
    "ScenarioResult",
    "SimRecord",
    "SimulationResult",
    "SimulationRun",
    "SolverRunRequest",
    "SolverSettings",
    "StudySpec",
    "describe_case",
    "execute_case",
    "load_case",
    "measure_response",
    "prepare_case",
    "register_metric",
    "register_optimizer",
    "register_simulation",
    "register_solver",
    "run_case_study",
    "run_identification",
    "run_network",
    "simulate_case",
    "solve_scattering",
]
