"""Small, dependency-free parameter-space utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def grid_values(spec: Mapping[str, Any], default_levels: int = 3) -> list[Any]:
    if "choices" in spec:
        values = list(spec["choices"])
    elif "values" in spec:
        values = list(spec["values"])
    elif "bounds" not in spec:
        values = [spec.get("default")]
    else:
        low, high = spec["bounds"]
        levels = max(1, int(spec.get("grid", default_levels)))
        if spec.get("type") == "int":
            values = sorted({round(item) for item in np.linspace(int(low), int(high), levels)})
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
