"""Name-to-method lookup shared by the application layers.

Metrics, search, and simulation use the same small mechanism while keeping
their names isolated in separate :class:`Registry` instances.  Built-in
modules are named as strings and imported lazily on first use.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

Method = Callable[..., Any]

#: Plugin files already executed, shared across registries.  One plugin file
#: commonly registers both a circuit builder and an objective, so it must run
#: exactly once no matter which layer asks for it first.
_LOADED_PLUGIN_PATHS: set[Path] = set()


class Registry:
    """A set of named methods, grouped by kind."""

    def __init__(self, label: str, kinds: tuple[str, ...], builtins_module: str) -> None:
        self.label = label
        self.builtins_module = builtins_module
        self.methods: dict[str, dict[str, Method]] = {kind: {} for kind in kinds}
        self.seen_conflicts: list[dict[str, str]] = []
        self._builtins_loaded = False

    def ensure_builtins(self) -> None:
        """Import the built-in methods module once, on first use."""

        if not self._builtins_loaded:
            self._builtins_loaded = True
            importlib.import_module(self.builtins_module)

    def register(self, kind: str, name: str) -> Callable[[Method], Method]:
        """Decorator that binds a function to ``kind``/``name``.

        Registration is idempotent and first-writer-wins: re-importing the same
        plugin is harmless, and a genuine clash is recorded and warned about
        rather than silently overriding a built-in.
        """

        if kind not in self.methods:
            raise KeyError(f"unknown {self.label} method kind: {kind}. expected one of {list(self.methods)}")

        def decorate(func: Method) -> Method:
            existing = self.methods[kind].get(name)
            if existing is not None and existing is not func:
                self._record_conflict(kind, name, existing, func)
                return func
            self.methods[kind][name] = func
            return func

        return decorate

    def _record_conflict(self, kind: str, name: str, existing: Method, new: Method) -> None:
        conflict = {
            "kind": kind,
            "name": name,
            "existing": f"{existing.__module__}.{existing.__name__}",
            "new": f"{new.__module__}.{new.__name__}",
        }
        self.seen_conflicts.append(conflict)
        warnings.warn(
            f"{self.label} method conflict for {kind}:{name}; keeping {conflict['existing']}",
            RuntimeWarning,
            stacklevel=3,
        )

    def get(self, kind: str, name: str) -> Method:
        self.ensure_builtins()
        try:
            return self.methods[kind][name]
        except KeyError as exc:
            available = sorted(self.methods.get(kind, {}))
            raise KeyError(f"unknown {kind} method: {name}. available={available}") from exc

    def available(self) -> dict[str, list[str]]:
        self.ensure_builtins()
        return {kind: sorted(methods) for kind, methods in self.methods.items()}

    def conflicts(self) -> list[dict[str, str]]:
        self.ensure_builtins()
        return list(self.seen_conflicts)

    def load_plugins(self, paths: list[str] | None, base_dir: Path) -> None:
        """Load built-ins first, then any plugin files the case names.

        Order matters: built-ins register first, so a plugin cannot silently
        replace one — it raises a conflict warning instead.
        """

        self.ensure_builtins()
        load_plugin_files(paths, base_dir)


def load_plugin_files(paths: list[str] | None, base_dir: Path) -> None:
    """Execute plugin files so their ``@register`` decorators run."""

    for raw in paths or []:
        path = Path(raw)
        path = (path if path.is_absolute() else base_dir / path).resolve()
        if path in _LOADED_PLUGIN_PATHS:
            continue
        if not path.exists():
            raise FileNotFoundError(f"plugin not found: {path}")
        _import_file(path)
        _LOADED_PLUGIN_PATHS.add(path)


def _import_file(path: Path) -> None:
    # The digest disambiguates module names for identically-named plugin files
    # in different directories; it is not a security primitive.
    digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    mod_name = f"pcd_plugin_{digest}_{path.stem}"
    if mod_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
