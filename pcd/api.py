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
from .metric_registry import register as register_metric
from .netlist import Circuit, Component
from .search import BaseOptimizer
from .search_registry import register as register_optimizer
from .sim_core import SimRecord, prepare_case, simulate_case
from .sim_registry import register as register_simulation
from .solver import SimulationResult
from .study import run_case_study

__all__ = [
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
    "MetricSet",
    "Objective",
    "RawResult",
    "Scenario",
    "ScenarioResult",
    "SimRecord",
    "SimulationResult",
    "StudySpec",
    "load_case",
    "prepare_case",
    "register_metric",
    "register_optimizer",
    "register_simulation",
    "run_case_study",
    "simulate_case",
]
