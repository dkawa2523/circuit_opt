"""Bounded complex-terminal fitting through the shared study evaluator."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .case import Case
from .continuous_search import continuous_space
from .core.models import CandidateResult
from .problem import ParameterRole, candidate_role
from .search import BaseOptimizer
from .study_config import _objective_specs


class _BudgetExhausted(Exception):
    """Stop a numerical method before it consumes another solver evaluation."""


class _ResidualUnavailable(Exception):
    """The model did not yield usable residuals at a proposed point."""


class LeastSquaresOptimizer(BaseOptimizer):
    """Fit normalized complex V/I residuals; final selection remains in StudyRunner."""

    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case, seed)
        if candidate_role(case) is not ParameterRole.LATENT:
            raise ValueError("least_squares is for latent terminal identification; use a design optimizer for run")
        if (case.data.get("target") or {}).get("objective") != "terminal_vi_fit":
            raise ValueError("least_squares requires target.objective: terminal_vi_fit")
        objectives = _objective_specs(case)
        if (
            len(objectives) != 1
            or objectives[0].metric not in {"loss", "normalized_impedance_error"}
            or (objectives[0].direction != "minimize" or objectives[0].aggregation != "rms")
        ):
            raise ValueError("least_squares requires one minimized terminal-error objective with aggregation: rms")
        self.axes, self.fixed = continuous_space(case, "least_squares")
        self._phase = "fit"

    def feedback_details(self, result: CandidateResult) -> dict[str, Any]:
        if not result.successful:
            return {"residuals": None}
        weight_scale = max(item.scenario.weight for item in result.scenarios)
        weights = [item.scenario.weight / weight_scale for item in result.scenarios]
        weight_sum = math.fsum(weights)
        residuals = [
            float(scenario.selected.metrics.values[name]) * math.sqrt(weight / weight_sum)
            for scenario, weight in zip(result.scenarios, weights, strict=True)
            for name in ("normalized_residual_resistance", "normalized_residual_reactance")
        ]
        return {"residuals": residuals}

    def optimize(self, evaluate: Callable[[dict[str, Any]], dict[str, Any]], budget: int) -> None:
        initial = [axis.encode(self.fixed.get(axis.name)) for axis in self.axes]
        # An exact bound can give TRF a near-zero initial trust radius and false convergence.
        declared = np.asarray([0.5 if value is None else value for value in initial])
        start = declared.copy()
        start[start == 0.0] = 0.01
        start[start == 1.0] = 0.99
        cache: dict[tuple[float, ...], np.ndarray] = {}
        evaluations = 0

        def residual(vector: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            key = tuple(float(value) for value in vector)
            if key in cache:
                return cache[key]
            if evaluations >= budget:
                raise _BudgetExhausted
            params = {
                **self.fixed,
                **{axis.name: axis.decode(float(value)) for axis, value in zip(self.axes, vector, strict=True)},
            }
            evaluations += 1
            feedback = evaluate(params)
            if feedback["residuals"] is None:
                raise _ResidualUnavailable
            values = np.asarray(feedback["residuals"], dtype=float)
            cache[key] = values
            return values

        try:
            if not np.array_equal(start, declared):
                # Preserve the authored endpoint as a candidate even though the local solver starts inside.
                with suppress(_ResidualUnavailable):
                    residual(declared)
            result = least_squares(residual, start, bounds=(0.0, 1.0), diff_step=1e-4, gtol=1e-12, max_nfev=budget)
            self.outcome = {
                "termination": "converged" if result.success else "iteration_budget_exhausted",
                "message": result.message,
            }
        except _BudgetExhausted:
            self._phase = "budget_exhausted"
            self.outcome = {"termination": "candidate_budget_exhausted"}
        except _ResidualUnavailable:
            self._phase = "residual_unavailable"
            self.outcome = {"termination": "residual_unavailable"}

    def proposal_metadata(self) -> dict[str, Any]:
        return {"method": "bounded_least_squares", "phase": self._phase}
