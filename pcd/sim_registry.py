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
    """Invoke a solver and check the response supplied by an extension."""

    method = get("solver", name, plugins=request.plugin_paths)
    result = method(request)
    if not isinstance(result, SimulationResult):
        raise TypeError(f"solver '{name}' must return SimulationResult")
    # ngspice checks the numerical domain while translating external solver
    # problems into failed results. A broken extension response is a code error.
    from .sim_methods import solver_ngspice_cli

    if method is not solver_ngspice_cli:
        result.validate_for(request.simulation.analysis)
    return result
