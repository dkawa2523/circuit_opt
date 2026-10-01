"""Reproducible continuous candidate search for advanced studies.

The implementation is deliberately limited to bounded numeric design axes.
Scenarios, controls, simulation, and engineering constraints remain owned by
the study pipeline; this module only proposes fixed candidate values and uses
the pipeline's feasibility-first rank for selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .case import Case, default_params, variable_specs
from .search import BaseOptimizer, feedback_rank

_MUTATION = 0.8
_CROSSOVER = 0.7


@dataclass(frozen=True, slots=True)
class _Axis:
    name: str
    low: float
    high: float
    logarithmic: bool

    def decode(self, unit_value: float) -> float:
        if self.logarithmic:
            low = math.log10(self.low)
            return 10.0 ** (low + unit_value * (math.log10(self.high) - low))
        return self.low + unit_value * (self.high - self.low)

    def encode(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not self.low <= number <= self.high:
            return None
        if self.logarithmic:
            return (math.log10(number) - math.log10(self.low)) / (math.log10(self.high) - math.log10(self.low))
        return (number - self.low) / (self.high - self.low)


def _continuous_space(case: Case) -> tuple[tuple[_Axis, ...], dict[str, Any]]:
    axes: list[_Axis] = []
    fixed = default_params(case)
    for name, spec in variable_specs(case).items():
        choices = spec.get("choices")
        if choices is not None:
            fixed[name] = _single_choice(name, choices)
            continue
        if "bounds" not in spec:
            if "default" not in spec:
                raise ValueError(f"differential_evolution variable {name!r} needs numeric bounds or a fixed default")
            continue
        axes.append(_continuous_axis(name, spec))
    if not axes:
        raise ValueError("differential_evolution requires at least one bounded continuous design variable")
    return tuple(axes), fixed


def _single_choice(name: str, choices: Any) -> Any:
    values = list(choices)
    if len(values) != 1:
        raise ValueError(
            "differential_evolution requires bounded continuous design variables; "
            f"{name!r} declares {len(values)} choices"
        )
    return values[0]


def _continuous_axis(name: str, spec: dict[str, Any]) -> _Axis:
    if spec.get("type") in {"bool", "int", "str", "string"}:
        raise ValueError(f"differential_evolution variable {name!r} must be continuous, not {spec.get('type')!r}")
    try:
        low, high = (float(item) for item in spec["bounds"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"differential_evolution variable {name!r} needs two finite numeric bounds") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"differential_evolution variable {name!r} needs increasing finite bounds")
    scale = str(spec.get("scale", "linear"))
    if scale not in {"linear", "log"}:
        raise ValueError(f"differential_evolution variable {name!r} scale must be linear or log")
    if scale == "log" and low <= 0:
        raise ValueError(f"differential_evolution variable {name!r} needs positive log-scale bounds")
    return _Axis(name, low, high, scale == "log")


def _trial_budget(case: Case) -> int:
    raw = (case.data.get("run") or {}).get("trials", 0)
    try:
        budget = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("differential_evolution requires an integer run.trials budget") from exc
    return budget


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """Sequential ask/tell implementation of bounded DE/rand/1/bin."""

    def __init__(self, case: Case, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        self.axes, self.fixed = _continuous_space(case)
        self.population_size = max(4, 4 * len(self.axes))
        budget = _trial_budget(case)
        if budget < 2 * self.population_size:
            raise ValueError(
                "differential_evolution run.trials must cover initialization and one full generation; "
                f"need at least {2 * self.population_size} for {len(self.axes)} continuous variables"
            )
        configured_seed = (case.data.get("optimizer") or {}).get("seed", 0)
        self.seed = int(seed if seed is not None else configured_seed)
        self.rng = np.random.default_rng(self.seed)
        self.population = self._initial_population()
        self.ranks: list[tuple[float, ...] | None] = [None] * self.population_size
        self.generation = 0
        self._target = 0
        self._pending: tuple[str, int, np.ndarray] | None = None
        self._next_population: np.ndarray | None = None
        self._next_ranks: list[tuple[float, ...] | None] | None = None
        self._last_metadata: dict[str, Any] = {}

    def _initial_population(self) -> np.ndarray:
        population = np.empty((self.population_size, len(self.axes)), dtype=float)
        for column in range(len(self.axes)):
            strata = self.rng.permutation(self.population_size) + self.rng.random(self.population_size)
            population[:, column] = strata / self.population_size
        encoded_defaults = [axis.encode(self.fixed.get(axis.name)) for axis in self.axes]
        if all(value is not None for value in encoded_defaults):
            population[0] = np.asarray(encoded_defaults, dtype=float)
        return population

    def _decode(self, vector: np.ndarray) -> dict[str, Any]:
        return {
            **self.fixed,
            **{axis.name: axis.decode(float(value)) for axis, value in zip(self.axes, vector, strict=True)},
        }

    def ask(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("differential_evolution.ask() requires tell() for the previous proposal")
        if self.generation == 0:
            vector = self.population[self._target].copy()
            self._pending = ("initialization", self._target, vector)
            self._last_metadata = {
                "phase": "initialization",
                "generation": 0,
                "population_index": self._target,
            }
            return self._decode(vector)

        target = self._target
        candidates = np.delete(np.arange(self.population_size), target)
        first, second, third = self.rng.choice(candidates, size=3, replace=False)
        mutant = np.clip(
            self.population[first] + _MUTATION * (self.population[second] - self.population[third]),
            0.0,
            1.0,
        )
        crossover = self.rng.random(len(self.axes)) < _CROSSOVER
        crossover[int(self.rng.integers(len(self.axes)))] = True
        vector = np.where(crossover, mutant, self.population[target])
        self._pending = ("evolution", target, vector)
        self._last_metadata = {
            "phase": "evolution",
            "strategy": "DE/rand/1/bin",
            "generation": self.generation,
            "population_index": target,
            "parents": [int(first), int(second), int(third)],
        }
        return self._decode(vector)

    def tell(self, params: dict[str, Any], feedback: dict[str, Any]) -> None:
        if self._pending is None:
            raise RuntimeError("differential_evolution.tell() requires a pending proposal")
        super().tell(params, feedback)
        phase, target, vector = self._pending
        rank = feedback_rank(feedback)
        accepted = phase == "initialization"
        if phase == "initialization":
            self.ranks[target] = rank
        else:
            parent_rank = self.ranks[target]
            accepted = rank is not None and (parent_rank is None or rank < parent_rank)
            if accepted:
                population, ranks = self._generation_buffers()
                population[target] = vector
                ranks[target] = rank
        self._last_metadata["accepted"] = accepted
        self._pending = None
        self._advance()

    def _advance(self) -> None:
        self._target += 1
        if self._target < self.population_size:
            return
        if self.generation > 0:
            self.population, self.ranks = self._generation_buffers()
        self.generation += 1
        self._target = 0
        self._next_population = self.population.copy()
        self._next_ranks = list(self.ranks)

    def _generation_buffers(self) -> tuple[np.ndarray, list[tuple[float, ...] | None]]:
        if self._next_population is None or self._next_ranks is None:
            raise RuntimeError("differential_evolution generation buffers are not initialized")
        return self._next_population, self._next_ranks

    def proposal_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "strategy": "DE/rand/1/bin",
            "seed": self.seed,
            "continuous_variables": [axis.name for axis in self.axes],
            "population_size": self.population_size,
            "generation": self.generation,
        }
