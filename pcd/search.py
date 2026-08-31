"""Candidate generators for case studies.

Search proposes fixed designs only. Scenario values and per-scenario controls
are owned by the StudyRunner and can never leak into this parameter space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from math import prod
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
        return int(rng.integers(int(lo), int(hi) + 1))
    if spec.get("scale", "linear") == "log":
        lo_f, hi_f = float(lo), float(hi)
        if lo_f <= 0 or hi_f <= 0:
            raise ValueError(f"log-scale bounds must be positive: {spec}")
        value = float(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f)))
        return min(max(value, lo_f), hi_f)
    return float(rng.uniform(float(lo), float(hi)))


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
    return factory(case, seed=seed)


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


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        try:
            import optuna
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("optuna is optional. install with: pip install -e '.[optuna]'") from exc
        configured_seed = (case.data.get("optimizer", {}) or {}).get("seed", 0)
        sampler = optuna.samplers.TPESampler(seed=seed if seed is not None else configured_seed)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.specs = variable_specs(case)
        self._pending: dict[int, Any] = {}

    def ask(self) -> dict[str, Any]:
        trial = self.study.ask()
        params = default_params(self.case)
        for name, spec in self.specs.items():
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
                continue
            low, high = spec.get("bounds", [0.0, 1.0])
            logarithmic = bool(spec.get("scale") == "log")
            if spec.get("type") == "int":
                params[name] = trial.suggest_int(name, int(low), int(high), log=logarithmic)
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), log=logarithmic)
        self._pending[id(params)] = trial
        return params

    def tell(self, params: dict[str, Any], feedback: dict[str, Any]) -> None:
        trial = self._pending.pop(id(params), None)
        value = float(feedback.get("loss", 1e30))
        if trial is not None:
            self.study.tell(trial, value)
        super().tell(params, feedback)

    def state(self) -> dict[str, Any]:
        try:
            best = {"optimizer_loss": self.study.best_value, "params": self.study.best_params}
        except Exception:
            best = None
        return {"type": type(self).__name__, "n_observations": len(self.history), "best": best}


@register("optuna")
def optuna_optimizer(case: Case, seed: int | None = None) -> BaseOptimizer:
    return OptunaOptimizer(case, seed=seed)
