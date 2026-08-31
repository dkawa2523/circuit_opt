"""The one execution path shared by direct, coupled, and replay studies."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from .aggregation import aggregate_candidate, candidate_rank_key, control_margin, evaluation_rank_key
from .models import (
    Candidate,
    CandidateResult,
    ConstraintResult,
    EvaluationRequest,
    EvaluationResult,
    MetricSet,
    RawResult,
    ScenarioResult,
    StudySpec,
)
from .protocols import Constraint, ControlPolicy, Evaluator, FixedControlPolicy, MetricCalculator, ResultStore


class StudyRunner:
    """Evaluate fixed designs across scenarios through one explicit pipeline."""

    def __init__(
        self,
        study: StudySpec,
        evaluator: Evaluator,
        metrics: tuple[MetricCalculator, ...],
        constraints: tuple[Constraint, ...] = (),
        control_policy: ControlPolicy | None = None,
        store: ResultStore | None = None,
    ) -> None:
        self.study = study
        self.evaluator = evaluator
        self.metric_calculators = metrics
        self.constraints = constraints
        self.control_policy = control_policy or FixedControlPolicy()
        self.store = store

    def evaluate_candidate(self, candidate: Candidate) -> CandidateResult:
        scenarios: list[ScenarioResult] = []
        for scenario in self.study.scenarios:
            controls = self.control_policy.controls(self.study, candidate, scenario)
            if not controls:
                raise ValueError(f"control policy produced no operating point for scenario {scenario.scenario_id!r}")
            trial_rows = tuple(
                (
                    self._with_control_margin(self._evaluate_control(candidate, scenario, control), margin),
                    margin,
                )
                for control in controls
                for margin in (control_margin(control, controls),)
            )
            trials = tuple(result for result, _margin in trial_rows)
            if self.store:
                for trial in trials:
                    self.store.save(trial)
            selected, margin = min(
                trial_rows,
                key=lambda row: self._control_rank(*row),
            )
            scenarios.append(ScenarioResult(scenario, selected, trials, margin))

        candidate_result = aggregate_candidate(self.study, candidate, tuple(scenarios))
        if self.store:
            self.store.save_candidate(candidate_result)
        return candidate_result

    def run(self, candidates: list[Candidate]) -> list[CandidateResult]:
        results = [self.evaluate_candidate(candidate) for candidate in candidates]
        return sorted(results, key=lambda item: candidate_rank_key(self.study, item))

    def _control_rank(self, result: EvaluationResult, margin: float | None) -> tuple[float, ...]:
        if self.study.control_margin_min is not None:
            electrical = tuple(item for item in result.constraints if item.name != "min_control_margin")
            electrical_feasible = result.raw.ok and all(item.satisfied for item in electrical)
            electrical_violation = sum(item.violation for item in electrical)
            margin_constraint = next(
                (item for item in result.constraints if item.name == "min_control_margin"),
                None,
            )
            margin_violation = 1.0 if margin_constraint is None else margin_constraint.violation
            objective_rank = evaluation_rank_key(self.study, result)[2:]
            return (
                0.0 if result.feasible else 1.0,
                0.0 if electrical_feasible else 1.0,
                electrical_violation,
                margin_violation,
                *objective_rank,
                float("inf") if margin is None else -margin,
            )
        return (*evaluation_rank_key(self.study, result), float("inf") if margin is None else -margin)

    def _with_control_margin(self, result: EvaluationResult, margin: float | None) -> EvaluationResult:
        """Apply optional tuning-headroom acceptance after the grid is known."""

        limit = self.study.control_margin_min
        if limit is None:
            return result
        if margin is None:
            constraint = ConstraintResult(
                "min_control_margin",
                False,
                violation=1.0,
                limit=limit,
                detail="control margin is unavailable because the control grid has no numeric axis",
            )
        else:
            distance = max(0.0, limit - margin)
            constraint = ConstraintResult(
                "min_control_margin",
                distance == 0.0,
                violation=distance / max(limit, 1e-12),
                value=margin,
                limit=limit,
            )
        return replace(result, constraints=(*result.constraints, constraint))

    def _evaluate_control(self, candidate, scenario, control) -> EvaluationResult:
        started = time.perf_counter()
        request = EvaluationRequest(candidate, scenario, control)
        # Collision checking is a semantic guard, not optional validation:
        # one value must never silently be both a design and an environment.
        request.merged_inputs()
        raw = self.store.load_raw(request) if self.store else None
        from_cache = raw is not None
        if raw is None:
            raw = self._evaluate_raw(request)
            if self.store:
                self.store.save_raw(request, raw)
        result = self._finish_evaluation(request, raw, started, from_cache)
        return result

    def _evaluate_raw(self, request: EvaluationRequest) -> RawResult:
        try:
            return self.evaluator.evaluate(request)
        except Exception as exc:  # an evaluation failure is data, not a lost study
            return RawResult(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                diagnostics={"stage": "evaluate", "exception_type": type(exc).__name__},
            )

    def _finish_evaluation(
        self,
        request: EvaluationRequest,
        raw: RawResult,
        started: float,
        from_cache: bool,
    ) -> EvaluationResult:
        metrics = MetricSet()
        constraints: list[ConstraintResult] = []
        if raw.ok:
            try:
                metrics = self._compute_metrics(request, raw)
                self._require_objectives(metrics)
                constraints.extend(item.evaluate(request, raw, metrics) for item in self.constraints)
            except Exception as exc:
                raw = RawResult(
                    status="failed",
                    observations=raw.observations,
                    artifacts=raw.artifacts,
                    diagnostics={**dict(raw.diagnostics), "stage": "measure", "exception_type": type(exc).__name__},
                    error=f"{type(exc).__name__}: {exc}",
                )
                metrics = MetricSet()

        constraints.append(
            ConstraintResult(
                name="evaluation_success",
                satisfied=raw.ok,
                violation=0.0 if raw.ok else 1.0,
                detail=None if raw.ok else raw.error or raw.status,
            )
        )
        cache_key = self.store.key(request) if self.store else ""
        raw_cache_key = self.store.raw_key(request) if self.store else ""
        return EvaluationResult(
            request=request,
            raw=raw,
            metrics=metrics,
            constraints=tuple(constraints),
            cache_key=cache_key,
            raw_cache_key=raw_cache_key,
            duration_s=time.perf_counter() - started,
            from_cache=from_cache,
        )

    def _compute_metrics(self, request: EvaluationRequest, raw: RawResult) -> MetricSet:
        merged: dict[str, object] = {}
        for calculator in self.metric_calculators:
            current = calculator.compute(request, raw)
            overlap = set(merged) & set(current.values)
            if overlap:
                raise ValueError(f"metric calculators produced duplicate values: {sorted(overlap)}")
            merged.update(current.values)
        return MetricSet(merged)

    def _require_objectives(self, metrics: MetricSet) -> None:
        """A candidate may only be ranked from complete objective evidence."""

        for objective in self.study.objectives:
            raw = metrics.values.get(objective.metric)
            try:
                value = float(raw) if raw is not None else float("nan")
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                raise ValueError(f"objective metric {objective.metric!r} is unavailable or non-finite")
