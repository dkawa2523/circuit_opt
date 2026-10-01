"""Small extension protocols for the study engine."""

from __future__ import annotations

from typing import Protocol

from .models import (
    Candidate,
    ConstraintResult,
    ControlState,
    EvaluationRequest,
    MetricSet,
    RawResult,
    Scenario,
    StudySpec,
)


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> RawResult: ...


class MetricCalculator(Protocol):
    def compute(self, request: EvaluationRequest, raw: RawResult) -> MetricSet: ...


class Constraint(Protocol):
    def evaluate(self, request: EvaluationRequest, raw: RawResult, metrics: MetricSet) -> ConstraintResult: ...


class ControlPolicy(Protocol):
    def controls(
        self,
        study: StudySpec,
        candidate: Candidate,
        scenario: Scenario,
    ) -> tuple[ControlState, ...]: ...


class ResultStore(Protocol):
    """Reusable physical-result storage used while evaluating a study."""

    def raw_key(self, request: EvaluationRequest) -> str: ...

    def load_raw(self, request: EvaluationRequest) -> RawResult | None: ...

    def save_raw(self, request: EvaluationRequest, raw: RawResult) -> None: ...


class FixedControlPolicy:
    """Use the same explicitly allowed controls for every scenario."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self._control = ControlState(values or {})

    def controls(
        self,
        study: StudySpec,
        candidate: Candidate,
        scenario: Scenario,
    ) -> tuple[ControlState, ...]:
        del study, candidate, scenario
        return (self._control,)
