"""Named circuit-metric extensions used by case studies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .registry import Registry

if TYPE_CHECKING:
    from .case import Case
    from .simulation import SimulationResult

MetricFunction = Callable[["Case", dict[str, Any], "SimulationResult"], dict[str, Any]]

_REGISTRY = Registry(label="metric", kinds=("metric",), builtins_module="pcd.metrics")


def register(name: str) -> Callable[[MetricFunction], MetricFunction]:
    # A plugin must never register before built-ins and silently take a built-in
    # name. Registry.ensure_builtins marks the import in progress before loading,
    # so decorators used by pcd.metrics can safely pass through here recursively.
    _REGISTRY.ensure_builtins()
    return _REGISTRY.register("metric", name)


def get(name: str, *, plugins: tuple[Path, ...] | None = None) -> MetricFunction:
    return _REGISTRY.get("metric", name, plugins=plugins)


def available() -> list[str]:
    return _REGISTRY.available()["metric"]


def conflicts() -> list[dict[str, str]]:
    return _REGISTRY.conflicts()


load_plugins = _REGISTRY.load_plugins
