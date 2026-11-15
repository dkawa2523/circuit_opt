"""Candidate generators for design studies and terminal identification.

Search proposes exactly one declared candidate role: ``design`` for hardware
sizing or ``latent`` for identification. Scenario values and per-scenario
controls are owned by the StudyRunner and cannot leak into this space.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any

import numpy as np

from .case import Case
from .core.models import CandidateResult
from .inputs import integer, reject_unknown
from .problem import candidate_role, project_candidate_case, resolve_parameter_set, validate_parameter_value
from .search_registry import OptimizerFactory, load_plugins, register
from .search_registry import get as get_optimizer


def feedback_rank(feedback: dict[str, Any]) -> tuple[float, ...] | None:
    """Read the feasibility-first rank supplied by the study orchestration."""

    rank = feedback.get("rank")
    if isinstance(rank, list | tuple) and rank:
        try:
            return tuple(float(item) for item in rank)
        except (TypeError, ValueError):
            return None
    loss = feedback.get("loss")
    if loss is None:
        return None
    try:
        return (0.0, float(loss))
    except (TypeError, ValueError):
        return None


def sample_param(rng: np.random.Generator, spec: dict[str, Any]) -> Any:
    """Draw from a resolved parameter domain without changing its YAML type."""

    if "choices" in spec or "values" in spec:
        choices = list(spec["choices"] if "choices" in spec else spec["values"])
        return choices[int(rng.integers(len(choices)))]
    if spec.get("type") == "bool":
        return bool(rng.integers(0, 2))
    if "bounds" not in spec:
        if "default" not in spec:
            raise ValueError("random parameter requires choices, bounds, a bool type, or a default")
        return spec["default"]
    lo, hi = spec["bounds"]
    if spec.get("type") == "int":
        low, high = math.ceil(float(lo)), math.floor(float(hi))
        return int(rng.integers(low, high + 1))
    if spec.get("scale", "linear") == "log":
        lo_f, hi_f = float(lo), float(hi)
        value = float(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f)))
        return min(max(value, lo_f), hi_f)
    return float(rng.uniform(float(lo), float(hi)))


def candidate_specs(case: Case) -> dict[str, dict[str, Any]]:
    """Return the only parameter space an optimizer is allowed to propose."""

    return resolve_parameter_set(case).specs(candidate_role(case))


def validate_proposal(case: Case, proposal: Any) -> dict[str, Any]:
    """Normalize one optimizer proposal and enforce the declared candidate space.

    Optimizers may omit variables that have defaults, but may not introduce
    undeclared axes or return values outside a variable's type/domain.  This is
    deliberately the single runtime check between ``ask`` and simulation.
    """

    if not isinstance(proposal, Mapping):
        raise TypeError("optimizer.ask() must return a mapping")
    specs = candidate_specs(case)
    extras = sorted(str(name) for name in proposal if name not in specs)
    if extras:
        raise ValueError(f"optimizer proposed undeclared candidate variables: {extras}")
    missing = sorted(name for name, spec in specs.items() if name not in proposal and "default" not in spec)
    if missing:
        raise ValueError(f"optimizer proposal is missing candidate variables without defaults: {missing}")
    return {
        name: validate_parameter_value(name, proposal.get(name, spec.get("default")), spec)
        for name, spec in specs.items()
    }


@dataclass
class BaseOptimizer:
    case: Case
    seed: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict, init=False)

    def optimize(self, evaluate: Callable[[dict[str, Any]], dict[str, Any]], budget: int) -> None:
        """Propose values; the shared callback owns evaluation and feedback."""

        for _ in range(budget):
            evaluate(self.ask())
        self.outcome = {"termination": "candidate_budget_exhausted"}

    def ask(self) -> dict[str, Any]:
        raise NotImplementedError

    def tell(self, params: dict[str, Any], feedback: dict[str, Any]) -> None:
        self.history.append({"params": dict(params), "feedback": dict(feedback)})

    def feedback_details(self, result: CandidateResult) -> dict[str, Any]:
        """Extract method-specific feedback from a completed evaluation."""

        return {}

    def proposal_metadata(self) -> dict[str, Any]:
        """Describe the latest proposal for the persisted study trace."""

        return {}

    def state(self) -> dict[str, Any]:
        ranked = [(feedback_rank(item["feedback"]), item) for item in self.history]
        usable = [(rank, item) for rank, item in ranked if rank is not None]
        best = min(usable, key=lambda pair: pair[0])[1] if usable else None
        return {"type": type(self).__name__, "n_observations": len(self.history), "best": best}


def create_optimizer(case: Case, optimizer_name: str | None = None, seed: int | None = None) -> BaseOptimizer:
    factory = resolve_optimizer_factory(case, optimizer_name)
    return factory(project_candidate_case(case), seed=seed)


def resolve_optimizer_factory(case: Case, optimizer_name: str | None = None) -> OptimizerFactory:
    """Validate the search settings and resolve the optimizer for this case."""

    validate_search_settings(case, optimizer_name)
    paths = load_plugins(case.data.get("plugins"), case.base_dir)
    config = case.data.get("optimizer", {}) or {}
    name = str(config.get("name", "random")) if optimizer_name is None else optimizer_name
    return get_optimizer(name, plugins=paths)


def validate_optimizer_plan(case: Case) -> None:
    """Resolve the method and check the exact size of a built-in grid."""
    if resolve_optimizer_factory(case) is grid_optimizer:
        budget = int((case.data.get("run") or {}).get("trials", 1))
        GridOptimizer(case).validate_budget(budget)


def validate_search_settings(case: Case, optimizer_name: str | None = None) -> None:
    config, run = case.data.get("optimizer") or {}, case.data.get("run") or {}
    if not isinstance(config, Mapping) or not isinstance(run, Mapping):
        raise ValueError("optimizer and run must be mappings")
    name = optimizer_name if optimizer_name is not None else str(config.get("name", "random"))
    if name in {"random", "grid", "differential_evolution", "least_squares"}:
        reject_unknown(config, {"name", "seed"}, f"optimizer {name}")
    reject_unknown(run, {"trials", "root"}, "run")
    integer(run.get("trials", 1), "run.trials", minimum=1)
    integer(config.get("seed", 0), "optimizer.seed", minimum=0)


class RandomOptimizer(BaseOptimizer):
    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        configured_seed = (case.data.get("optimizer", {}) or {}).get("seed", 0)
        self.rng = np.random.default_rng(seed if seed is not None else configured_seed)
        self.specs = candidate_specs(case)

    def ask(self) -> dict[str, Any]:
        return {name: sample_param(self.rng, spec) for name, spec in self.specs.items()}


@register("random")
def random_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    return RandomOptimizer(case, seed=seed)


class GridOptimizer(BaseOptimizer):
    """Enumerate a finite hardware shortlist exactly once.

    Grid search is deliberately limited to categorical ``choices`` axes. A
    continuous interval has no complete finite enumeration and belongs to a
    stochastic or model-based optimizer instead.
    """

    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        axes: list[tuple[str, list[Any]]] = []
        for name, spec in candidate_specs(case).items():
            if "choices" in spec or "values" in spec:
                choices = list(spec["choices"] if "choices" in spec else spec["values"])
            elif "default" in spec and "bounds" not in spec:
                choices = [spec["default"]]
            else:
                raise ValueError(
                    f"grid optimizer requires finite choices for every design variable; {name!r} is continuous"
                )
            axes.append((name, choices))
        self.axes = tuple(axes)
        self.n_points = prod(len(choices) for _name, choices in self.axes)
        self._points = iter(product(*(choices for _name, choices in self.axes)))
        self._asked = 0

    def validate_budget(self, budget: int) -> None:
        if budget != self.n_points:
            raise ValueError(f"grid optimizer requires exactly {self.n_points} trials, got {budget}")

    def optimize(self, evaluate: Callable[[dict[str, Any]], dict[str, Any]], budget: int) -> None:
        super().optimize(evaluate, budget)
        if self._asked == self.n_points:
            self.outcome = {"termination": "declared_grid_completed"}

    def ask(self) -> dict[str, Any]:
        try:
            values = next(self._points)
        except StopIteration as exc:
            raise RuntimeError(f"grid optimizer exhausted its {self.n_points} unique candidates") from exc
        self._asked += 1
        return dict(zip((name for name, _choices in self.axes), values, strict=True))

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "n_points": self.n_points,
            "complete": self._asked == self.n_points,
        }


@register("grid")
def grid_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    return GridOptimizer(case, seed=seed)


@register("differential_evolution")
def differential_evolution_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    """Build the advanced continuous-space optimizer without a heavy dependency."""

    from .continuous_search import DifferentialEvolutionOptimizer

    return DifferentialEvolutionOptimizer(case, seed=seed)


@register("least_squares")
def least_squares_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    """Bounded terminal fitting using the ordinary evaluation callback."""

    from .least_squares_search import LeastSquaresOptimizer

    return LeastSquaresOptimizer(case, seed=seed)
