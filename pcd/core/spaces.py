"""Small, dependency-free parameter-space utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pcd.inputs import integer, parameter_spec


@dataclass(frozen=True, slots=True)
class ContinuousAxis:
    """One bounded numeric coordinate, shared by search and sensitivity."""

    name: str
    low: float
    high: float
    logarithmic: bool

    def decode(self, unit_value: float) -> float:
        if unit_value <= 0.0:
            return self.low
        if unit_value >= 1.0:
            return self.high
        if self.logarithmic:
            low, high = math.log10(self.low), math.log10(self.high)
            if low < high:
                exponent = low + unit_value * (high - low)
                value = self.high if exponent >= high else 10.0**exponent
                return min(max(value, self.low), self.high)
        width = self.high - self.low
        value = (
            self.low + unit_value * width
            if math.isfinite(width)
            else (1.0 - unit_value) * self.low + unit_value * self.high
        )
        return min(max(value, self.low), self.high)

    def encode(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not self.low <= number <= self.high:
            return None
        if number == self.low:
            return 0.0
        if number == self.high:
            return 1.0
        if self.logarithmic:
            low, high = math.log10(self.low), math.log10(self.high)
            if low < high:
                return (math.log10(number) - low) / (high - low)
        width = self.high - self.low
        if math.isfinite(width):
            return (number - self.low) / width
        return (number / 2.0 - self.low / 2.0) / (self.high / 2.0 - self.low / 2.0)


def grid_values(spec: Mapping[str, Any], default_levels: int = 3) -> list[Any]:
    spec = parameter_spec(spec, "parameter grid", allow_grid=True)
    if (choices := spec.get("choices", spec.get("values"))) is not None:
        return list(choices)
    if "bounds" not in spec:
        if "default" not in spec:
            raise ValueError("parameter grid requires choices, values, bounds, or a default")
        return [spec["default"]]
    low, high = map(float, spec["bounds"])
    levels = integer(spec.get("grid", default_levels), "parameter grid levels", minimum=1)
    if spec.get("type") == "int":
        return sorted({round(item) for item in np.linspace(math.ceil(low), math.floor(high), levels)})
    if spec.get("scale") == "log":
        return [min(max(float(item), low), high) for item in np.geomspace(low, high, levels)]
    return [float(item) for item in np.linspace(low, high, levels)]


def parameter_grid(
    axes: Mapping[str, Sequence[Any]],
    *,
    budget: int = 27,
) -> list[dict[str, Any]]:
    """Enumerate the complete discrete space from resolved axis values.

    This function is used for equipment controls.  A sampled inner search
    cannot prove that no feasible tuner setting exists, so an oversized space
    is rejected instead of being silently thinned.
    """

    if budget < 1:
        raise ValueError("parameter-grid budget must be positive")
    total = math.prod(map(len, axes.values()))
    if total > budget:
        raise ValueError(
            f"complete control grid has {total} states, exceeding budget {budget}; "
            "increase study.controls.budget or reduce the declared control values"
        )
    return [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
