"""Bounded complex-terminal fitting through the shared study evaluator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .case import Case
from .continuous_search import continuous_space
from .problem import ParameterRole, candidate_role
from .search import BaseOptimizer
from .study_config import study_spec_from_case


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
        objectives = study_spec_from_case(case).objectives
        if (
            len(objectives) != 1
            or objectives[0].metric not in {"loss", "normalized_impedance_error"}
            or (objectives[0].direction != "minimize" or objectives[0].aggregation != "rms")
        ):
            raise ValueError("least_squares requires one minimized terminal-error objective with aggregation: rms")
        self.axes, self.fixed = continuous_space(case, "least_squares")
        self._phase = "fit"

    def optimize(self, evaluate: Callable[[dict[str, Any]], dict[str, Any]], budget: int) -> None:
        initial = [axis.encode(self.fixed.get(axis.name)) for axis in self.axes]
        start = np.asarray([0.5 if value is None else value for value in initial])
        cache: dict[tuple[float, ...], np.ndarray] = {}

        def residual(vector: np.ndarray) -> np.ndarray:
            key = tuple(float(value) for value in vector)
            if key in cache:
                return cache[key]
            if len(cache) >= budget:
                raise _BudgetExhausted
            params = {
                **self.fixed,
                **{axis.name: axis.decode(float(value)) for axis, value in zip(self.axes, vector, strict=True)},
            }
            feedback = evaluate(params)
            if feedback["residuals"] is None:
                raise _ResidualUnavailable
            values = np.asarray(feedback["residuals"], dtype=float)
            cache[key] = values
            return values

        try:
            result = least_squares(residual, start, bounds=(0.0, 1.0), diff_step=1e-4, max_nfev=budget)
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
