"""Supported extension surface for PCD plugins and Python callers.

Imports from this module are kept stable within the current major version.
Implementation modules remain available for application development, but
third-party plugins should not depend on their private helpers.
"""

from __future__ import annotations

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
from .probes import NamedProbe, ProbePlan
from .search import BaseOptimizer
from .search_registry import register as register_optimizer
from .sim_core import SimRecord, SimulationRun, execute_case, prepare_case, simulate_case
from .sim_registry import register as register_simulation
from .sim_registry import register_solver
from .simulation import AnalysisRequest, SimulationResult
from .simulation_input import MeasurementReference, ResolvedSimulationCase, SolverRunRequest, SolverSettings
from .study import run_case_study

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
    "ProbePlan",
    "RawResult",
    "ResolvedSimulationCase",
    "Scenario",
    "ScenarioResult",
    "SimRecord",
    "SimulationResult",
    "SimulationRun",
    "SolverRunRequest",
    "SolverSettings",
    "StudySpec",
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
    "simulate_case",
]
