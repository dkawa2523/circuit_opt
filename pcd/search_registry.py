"""Named candidate-search extensions used by case studies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .registry import Registry

OptimizerFactory = Callable[..., Any]

_REGISTRY = Registry(label="search", kinds=("optimizer",), builtins_module="pcd.search")


def register(name: str) -> Callable[[OptimizerFactory], OptimizerFactory]:
    _REGISTRY.ensure_builtins()
    return _REGISTRY.register("optimizer", name)


def get(name: str) -> OptimizerFactory:
    return _REGISTRY.get("optimizer", name)


def available() -> list[str]:
    return _REGISTRY.available()["optimizer"]


def conflicts() -> list[dict[str, str]]:
    return _REGISTRY.conflicts()


def load_plugins(paths: list[str] | None, base_dir: Path) -> None:
    _REGISTRY.load_plugins(paths, base_dir)
