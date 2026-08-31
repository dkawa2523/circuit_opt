"""Simulation method registry: circuit builders, load models, and solvers.

The mechanism lives in :mod:`pcd.registry`; this module only declares what the
simulation layer offers.  Plugins use ``@register("circuit", "my_topology")``.
"""

from __future__ import annotations

from .registry import Registry

KINDS = ("circuit", "load", "solver")

_REGISTRY = Registry(label="simulation", kinds=KINDS, builtins_module="pcd.sim_methods")

register = _REGISTRY.register
get = _REGISTRY.get
available = _REGISTRY.available
conflicts = _REGISTRY.conflicts
load_plugins = _REGISTRY.load_plugins
