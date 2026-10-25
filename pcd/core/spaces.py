"""Small, dependency-free parameter-space utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
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


def grid_values(spec: Mapping[str, Any], default_levels: int = 3) -> list[Any]:
    spec = parameter_spec(spec, "parameter grid", allow_grid=True)
    if spec.get("type", "") not in {"", "float", "number", "int", "bool", "str", "string"}:
        raise ValueError("unsupported parameter type")
    if spec.get("scale", "linear") not in {"linear", "log"}:
        raise ValueError("parameter scale must be linear or log")
    if "choices" in spec and "values" in spec:
        raise ValueError("use only one of choices or values for a parameter grid")
    if "choices" in spec:
        values = list(spec["choices"])
    elif "values" in spec:
        values = list(spec["values"])
    elif "bounds" not in spec:
        if "default" not in spec:
            raise ValueError("parameter grid requires choices, values, bounds, or a default")
        values = [spec["default"]]
    else:
        low, high = spec["bounds"]
        levels = integer(spec.get("grid", default_levels), "parameter grid levels", minimum=1)
        if spec.get("type") == "int":
            low_int, high_int = math.ceil(float(low)), math.floor(float(high))
            if low_int > high_int:
                raise ValueError(f"integer bounds contain no integer: {dict(spec)}")
            values = sorted({round(item) for item in np.linspace(low_int, high_int, levels)})
        elif spec.get("scale") == "log":
            low_f, high_f = float(low), float(high)
            if low_f <= 0 or high_f <= 0:
                raise ValueError(f"log-scale bounds must be positive: {dict(spec)}")
            values = [min(max(float(item), low_f), high_f) for item in np.geomspace(low_f, high_f, levels)]
        else:
            values = [float(item) for item in np.linspace(float(low), float(high), levels)]
    if not values:
        raise ValueError(f"parameter space has no values: {dict(spec)}")
    return values


def parameter_grid(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    budget: int = 27,
) -> list[dict[str, Any]]:
    """Enumerate the complete declared discrete space.

    This function is used for equipment controls.  A sampled inner search
    cannot prove that no feasible tuner setting exists, so an oversized space
    is rejected instead of being silently thinned.
    """

    if budget < 1:
        raise ValueError("parameter-grid budget must be positive")
    if not specs:
        return [{}]
    names = list(specs)
    value_sets = [grid_values(specs[name]) for name in names]
    total = math.prod(len(items) for items in value_sets)
    if total > budget:
        raise ValueError(
            f"complete control grid has {total} states, exceeding budget {budget}; "
            "increase study.controls.budget or reduce the declared control values"
        )
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*value_sets)]
