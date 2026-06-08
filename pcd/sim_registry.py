from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

KINDS = ("circuit", "load", "solver")
REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}
_BUILTINS_LOADED = False
_LOADED_PLUGIN_PATHS: set[Path] = set()
CONFLICTS: list[dict[str, str]] = []


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if kind not in REGISTRY:
        raise KeyError(f"unknown simulation method kind: {kind}. expected one of {list(REGISTRY)}")

    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        # Idempotent registration keeps shared plugin files easy to load.
        existing = REGISTRY[kind].get(name)
        if existing is not None and existing is not func:
            conflict = {
                "kind": kind,
                "name": name,
                "existing": f"{existing.__module__}.{existing.__name__}",
                "new": f"{func.__module__}.{func.__name__}",
            }
            CONFLICTS.append(conflict)
            warnings.warn(
                f"simulation method conflict for {kind}:{name}; keeping {conflict['existing']}",
                RuntimeWarning,
                stacklevel=2,
            )
            return func
        REGISTRY[kind][name] = func
        return func

    return deco


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        importlib.import_module("pcd.sim_methods")
        _BUILTINS_LOADED = True


def get(kind: str, name: str) -> Callable[..., Any]:
    _load_builtins()
    try:
        return REGISTRY[kind][name]
    except KeyError as exc:
        raise KeyError(f"unknown {kind} method: {name}. available={sorted(REGISTRY.get(kind, {}))}") from exc


def available() -> dict[str, list[str]]:
    _load_builtins()
    return {k: sorted(v) for k, v in REGISTRY.items()}


def conflicts() -> list[dict[str, str]]:
    _load_builtins()
    return list(CONFLICTS)


def load_plugins(paths: list[str] | None, base_dir: Path) -> None:
    _load_builtins()
    for raw in paths or []:
        path = Path(raw)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if path in _LOADED_PLUGIN_PATHS:
            continue
        if not path.exists():
            raise FileNotFoundError(f"plugin not found: {path}")
        mod_name = f"pcd_plugin_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]}_{path.stem}"
        if mod_name in sys.modules:
            _LOADED_PLUGIN_PATHS.add(path)
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        _LOADED_PLUGIN_PATHS.add(path)
