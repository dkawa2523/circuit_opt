"""Generic design-study execution primitives.

The core intentionally knows nothing about circuits, ngspice, RF-load models,
or optimizers. Applications translate their own inputs into these types and
provide evaluator and metric implementations.
"""

from .models import (
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
from .pipeline import StudyRunner

__all__ = [
    "Candidate",
    "CandidateResult",
    "ConstraintResult",
    "ControlState",
    "EvaluationRequest",
    "EvaluationResult",
    "MetricSet",
    "Objective",
    "RawResult",
    "Scenario",
    "ScenarioResult",
    "StudyRunner",
    "StudySpec",
]
