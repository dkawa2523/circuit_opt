"""Candidate generators for case studies.

Search proposes fixed designs only. Scenario values and per-scenario controls
are owned by the StudyRunner and can never leak into this parameter space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from math import prod
from numbers import Integral, Real
from typing import Any

import numpy as np

from .case import Case, default_params, variable_specs
from .search_registry import get as get_optimizer
from .search_registry import load_plugins, register


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
    """Draw one declared parameter value without changing its YAML type."""

    if "choices" in spec:
        choices = list(spec["choices"])
        return choices[int(rng.integers(len(choices)))]
    if spec.get("type") == "bool":
        return bool(rng.integers(0, 2))
    lo, hi = spec.get("bounds", [0.0, 1.0])
    if spec.get("type") == "int":
        low, high = math.ceil(float(lo)), math.floor(float(hi))
        if low > high:
            raise ValueError(f"integer bounds contain no integer: {spec}")
        return int(rng.integers(low, high + 1))
    if spec.get("scale", "linear") == "log":
        lo_f, hi_f = float(lo), float(hi)
        if lo_f <= 0 or hi_f <= 0:
            raise ValueError(f"log-scale bounds must be positive: {spec}")
        value = float(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f)))
        return min(max(value, lo_f), hi_f)
    return float(rng.uniform(float(lo), float(hi)))


def validate_proposal(case: Case, proposal: Any) -> dict[str, Any]:
    """Normalize one optimizer proposal and enforce the declared design space.

    Optimizers may omit variables that have defaults, but may not introduce
    undeclared axes or return values outside a variable's type/domain.  This is
    deliberately the single runtime check between ``ask`` and simulation.
    """

    if not isinstance(proposal, Mapping):
        raise TypeError("optimizer.ask() must return a mapping")
    specs = variable_specs(case)
    extras = sorted(str(name) for name in proposal if name not in specs)
    if extras:
        raise ValueError(f"optimizer proposed undeclared design variables: {extras}")
    values = {**default_params(case), **dict(proposal)}
    missing = sorted(name for name in specs if name not in values)
    if missing:
        raise ValueError(f"optimizer proposal is missing design variables without defaults: {missing}")
    return {name: _validate_proposed_value(name, values[name], spec) for name, spec in specs.items()}


def _validate_proposed_value(name: str, value: Any, spec: dict[str, Any]) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    value = _normalize_proposed_type(name, value, str(spec.get("type", "")))
    _validate_proposed_domain(name, value, spec)
    return value


def _normalize_proposed_type(name: str, value: Any, kind: str) -> Any:
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"optimizer variable {name!r} must be bool")
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"optimizer variable {name!r} must be int")
        value = int(value)
    elif kind in {"float", "number"}:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"optimizer variable {name!r} must be numeric")
        value = float(value)
    elif kind in {"str", "string"} and not isinstance(value, str):
        raise ValueError(f"optimizer variable {name!r} must be a string")
    return value


def _validate_proposed_domain(name: str, value: Any, spec: dict[str, Any]) -> None:
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        raise ValueError(f"optimizer variable {name!r} must be finite")
    choices = spec.get("choices")
    if choices is not None and value not in choices:
        raise ValueError(f"optimizer variable {name!r} must be one of {list(choices)!r}; got {value!r}")
    if "bounds" in spec:
        try:
            number = float(value)
            lo, hi = (float(item) for item in spec["bounds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"optimizer variable {name!r} must be numeric within its bounds") from exc
        if not math.isfinite(number) or number < lo or number > hi:
            raise ValueError(f"optimizer variable {name!r}={value!r} is outside [{lo:g}, {hi:g}]")
    if spec.get("scale") == "log" and isinstance(value, Real) and float(value) <= 0:
        raise ValueError(f"optimizer variable {name!r} must be positive on a log scale")


@dataclass
class BaseOptimizer:
    case: Case
    seed: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def ask(self) -> dict[str, Any]:
        raise NotImplementedError

    def tell(self, params: dict[str, Any], feedback: dict[str, Any]) -> None:
        self.history.append({"params": dict(params), "feedback": dict(feedback)})

    def state(self) -> dict[str, Any]:
        ranked = [(feedback_rank(item["feedback"]), item) for item in self.history]
        usable = [(rank, item) for rank, item in ranked if rank is not None]
        best = min(usable, key=lambda pair: pair[0])[1] if usable else None
        return {"type": type(self).__name__, "n_observations": len(self.history), "best": best}


def create_optimizer(case: Case, optimizer_name: str | None = None, seed: int | None = None) -> BaseOptimizer:
    load_plugins(case.data.get("plugins"), case.base_dir)
    config = case.data.get("optimizer", {}) or {}
    name = optimizer_name or str(config.get("name", "random"))
    factory = get_optimizer(name)
    return factory(case.detached(), seed=seed)


class RandomOptimizer(BaseOptimizer):
    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        configured_seed = (case.data.get("optimizer", {}) or {}).get("seed", 0)
        self.rng = np.random.default_rng(seed if seed is not None else configured_seed)
        self.specs = variable_specs(case)

    def ask(self) -> dict[str, Any]:
        params = default_params(self.case)
        for name, spec in self.specs.items():
            params[name] = sample_param(self.rng, spec)
        return params


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
        for name, spec in variable_specs(case).items():
            if "choices" in spec:
                choices = list(spec["choices"])
            elif "default" in spec and "bounds" not in spec:
                choices = [spec["default"]]
            else:
                raise ValueError(
                    f"grid optimizer requires finite choices for every design variable; {name!r} is continuous"
                )
            if not choices:
                raise ValueError(f"grid optimizer variable {name!r} has no choices")
            axes.append((name, choices))
        self.axes = tuple(axes)
        self.n_points = prod(len(choices) for _name, choices in self.axes)
        self._points = iter(product(*(choices for _name, choices in self.axes)))
        self._asked = 0

    def ask(self) -> dict[str, Any]:
        try:
            values = next(self._points)
        except StopIteration as exc:
            raise RuntimeError(f"grid optimizer exhausted its {self.n_points} unique candidates") from exc
        self._asked += 1
        return {**default_params(self.case), **dict(zip((name for name, _choices in self.axes), values, strict=True))}

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "n_points": self.n_points,
            "complete": self._asked == self.n_points,
        }


@register("grid")
def grid_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    return GridOptimizer(case, seed=seed)
