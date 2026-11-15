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
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
from typing import Any

Method = Callable[..., Any]

#: Plugin files already executed and the exact content that was executed,
#: shared across registries.  One file commonly registers multiple methods, so
#: it must run exactly once no matter which layer asks for it first.
_LOADED_PLUGIN_PATHS: dict[Path, str] = {}
_FAILED_PLUGIN_PATHS: set[Path] = set()
_PLUGIN_BEING_LOADED: ContextVar[Path | None] = ContextVar("pcd_plugin_being_loaded", default=None)
_PLUGIN_LOCK = RLock()


def declared_plugin_paths(paths: list[str] | None, base_dir: Path) -> tuple[Path, ...]:
    if paths is None:
        return ()
    if not isinstance(paths, list):
        raise TypeError("plugins must be a list")
    resolved = []
    for index, raw in enumerate(paths):
        if not isinstance(raw, (str, Path)):
            raise TypeError(f"plugins[{index}] must be a path string")
        resolved.append((base_dir / raw).resolve())
    return tuple(resolved)


def loaded_plugin_digest(path: Path) -> str | None:
    return _LOADED_PLUGIN_PATHS.get(path)


class Registry:
    """A set of named methods, grouped by kind."""

    def __init__(self, label: str, kinds: tuple[str, ...], builtins_module: str) -> None:
        self.label = label
        self.builtins_module = builtins_module
        self.methods: dict[str, dict[str, Method]] = {kind: {} for kind in kinds}
        self._plugin_sources: dict[tuple[str, str], Path] = {}
        self._conflicting_plugins: dict[tuple[str, str], set[Path]] = {}
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
            source = _PLUGIN_BEING_LOADED.get() if func.__module__ != self.builtins_module else None
            if source is not None:
                self.ensure_builtins()
            existing = self.methods[kind].get(name)
            if existing is not None:
                if existing is not func:
                    if source is not None:
                        self._conflicting_plugins.setdefault((kind, name), set()).add(source)
                    self._record_conflict(kind, name, existing, func)
                return func
            self.methods[kind][name] = func
            if source is not None:
                self._plugin_sources.setdefault((kind, name), source)
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

    def get(self, kind: str, name: str, *, plugins: tuple[Path, ...] | None = None) -> Method:
        self.ensure_builtins()
        try:
            method = self.methods[kind][name]
        except KeyError as exc:
            available = sorted(self.methods.get(kind, {}))
            raise KeyError(f"unknown {kind} method: {name}. available={available}") from exc
        source = self._plugin_sources.get((kind, name))
        if source in _FAILED_PLUGIN_PATHS:
            raise RuntimeError(f"{kind} method {name!r} came from a failed plugin import: {source}")
        if plugins is not None:
            declared = set(plugins)
            if collisions := self._conflicting_plugins.get((kind, name), set()) & declared:
                raise ValueError(f"conflicting {kind} method {name!r} in case plugins: {sorted(map(str, collisions))}")
            if source is not None and source not in declared:
                raise ValueError(f"{kind} method {name!r} is provided by an undeclared plugin: {source}")
        return method

    def available(self) -> dict[str, list[str]]:
        self.ensure_builtins()
        return {kind: sorted(methods) for kind, methods in self.methods.items()}

    def conflicts(self) -> list[dict[str, str]]:
        self.ensure_builtins()
        return list(self.seen_conflicts)

    def load_plugins(self, paths: list[str] | None, base_dir: Path) -> tuple[Path, ...]:
        """Load built-ins first, then return the resolved paths of the case's plugins.

        Order matters: built-ins register first, so a plugin cannot silently
        replace one — it raises a conflict warning instead.
        """

        self.ensure_builtins()
        return load_plugin_files(paths, base_dir)


def load_plugin_files(paths: list[str] | None, base_dir: Path) -> tuple[Path, ...]:
    """Execute plugin registrations and return the paths checked by the loader."""

    resolved = declared_plugin_paths(paths, base_dir)
    with _PLUGIN_LOCK:
        for path in resolved:
            if path in _FAILED_PLUGIN_PATHS:
                raise RuntimeError(f"plugin import previously failed: {path}; restart the process before retrying")
            try:
                content = path.read_bytes()
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"plugin not found: {path}") from exc
            content_sha256 = hashlib.sha256(content).hexdigest()
            loaded_sha256 = _LOADED_PLUGIN_PATHS.get(path)
            if loaded_sha256 is not None:
                if loaded_sha256 != content_sha256:
                    raise RuntimeError(
                        f"plugin changed after it was loaded: {path}; restart the process before using the new code"
                    )
                continue
            _import_file(path, content)
            _LOADED_PLUGIN_PATHS[path] = content_sha256
    return resolved


def _import_file(path: Path, content: bytes) -> None:
    # The digest disambiguates module names for identically-named plugin files
    # in different directories; it is not a security primitive.
    digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    mod_name = f"pcd_plugin_{digest}_{path.stem}"
    if mod_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin: {path}")
    code = compile(content, str(path), "exec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    source = _PLUGIN_BEING_LOADED.set(path)
    try:
        exec(code, module.__dict__)  # noqa: S102  # explicit trusted plugin code
    except BaseException:
        # Decorators may have registered methods before the module raised.
        _FAILED_PLUGIN_PATHS.add(path)
        sys.modules.pop(mod_name, None)
        raise
    finally:
        _PLUGIN_BEING_LOADED.reset(source)
