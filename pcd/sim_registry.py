"""Simulation method registry: circuit builders, load models, and solvers.

The mechanism lives in :mod:`pcd.registry`; this module only declares what the
simulation layer offers.  Plugins use ``@register("circuit", "my_topology")``.
"""

from __future__ import annotations

from collections.abc import Callable

from .registry import Registry
from .simulation import SimulationResult
from .simulation_input import SolverRunRequest

KINDS = ("circuit", "load", "solver")

_REGISTRY = Registry(label="simulation", kinds=KINDS, builtins_module="pcd.sim_methods")

register = _REGISTRY.register
get = _REGISTRY.get
available = _REGISTRY.available
conflicts = _REGISTRY.conflicts
load_plugins = _REGISTRY.load_plugins

TypedSolver = Callable[[SolverRunRequest], SimulationResult]


def register_solver(name: str) -> Callable[[TypedSolver], TypedSolver]:
    """Register a solver that consumes the typed execution request."""

    def decorate(method: TypedSolver) -> TypedSolver:
        register("solver", name)(method)
        return method

    return decorate


def invoke_solver(
    name: str,
    request: SolverRunRequest,
) -> SimulationResult:
    """Invoke one solver through the single typed execution boundary."""

    method = get("solver", name)
    return method(request)
