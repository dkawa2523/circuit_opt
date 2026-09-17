"""Small extension protocols for the study engine."""

from __future__ import annotations

from typing import Protocol

from .models import (
    Candidate,
    CandidateResult,
    ConstraintResult,
    ControlState,
    EvaluationRequest,
    EvaluationResult,
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
    def key(self, request: EvaluationRequest) -> str: ...

    def raw_key(self, request: EvaluationRequest) -> str: ...

    def load_raw(self, request: EvaluationRequest) -> RawResult | None: ...

    def save_raw(self, request: EvaluationRequest, raw: RawResult) -> None: ...

    def save(self, result: EvaluationResult) -> None: ...

    def save_candidate(self, result: CandidateResult) -> None: ...


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
