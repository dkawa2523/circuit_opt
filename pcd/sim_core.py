"""Circuit-case adapter: prepare, execute, and record one solver run.

This is the third part of the simulation pipeline:

    netlist (pcd.netlist) -> solver (pcd.solver) -> run record (here)

The generic study pipeline calls this adapter and keeps its artifacts for
replay. Objective aggregation and scenario semantics live in :mod:`pcd.core`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    archive_data_files,
    artifact_path_segment,
    atomic_write_text,
    file_sha256,
    package_source_sha256,
    rewrite_data_file_paths,
    utc_now,
    write_json,
    yaml_dump,
)
from .case import Case, case_warnings, fill_default_params, resolve_path
from .netlist import Circuit, NetlistInputs, build_netlist_inputs, render_ngspice_netlist
from .netlist_import import flatten_netlist_file
from .sim_registry import invoke_solver
from .simulation import AC_ARTIFACT, AC_FILE, TRANSIENT_ARTIFACT, WAVEFORM_FILE, SimulationResult
from .simulation_input import (
    ResolvedSimulationCase,
    SolverRunRequest,
    SolverSettings,
    resolve_simulation_case,
)
from .solver import solver_identity

# Re-exported so plugins and callers have one import for the simulation layer.
__all__ = [
    "Circuit",
    "SimRecord",
    "SimulationResult",
    "SimulationRun",
    "archive_case_bundle",
    "archive_case_definition",
    "execute_case",
    "prepare_case",
    "simulate_case",
]

SUMMARY_FILE = "summary.json"
DEBUG_MANIFEST_FILE = "debug/manifest.json"
NETLIST_ARTIFACT = "debug/netlist.cir"
SOLVER_LOG_ARTIFACT = "debug/solver.log"


@dataclass
class SimRecord:
    """One simulation run with separate public data and replay details."""

    run_dir: Path
    case_id: str
    status: str
    params: dict[str, Any]
    circuit: str
    load: str
    solver: str
    netlist_file: str = NETLIST_ARTIFACT
    waveform_file: str = TRANSIENT_ARTIFACT
    solver_log_file: str = SOLVER_LOG_ARTIFACT
    #: Written only when the case requested an AC sweep.
    frequency_response_file: str | None = None
    created_at: str = field(default_factory=utc_now)
    run_seconds: float | None = None
    measurement: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    case_files: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return the small result a simulation user should read first."""

        solver = self.provenance.get("solver") or {}
        artifacts = {
            "waveform": self.waveform_file,
            **({"frequency_response": self.frequency_response_file} if self.frequency_response_file else {}),
            "debug_manifest": DEBUG_MANIFEST_FILE,
        }
        return {
            "schema": "simulation_summary.v1",
            "case_id": self.case_id,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "run_seconds": self.run_seconds,
            "solver": self.solver,
            "solver_version": solver.get("version"),
            "artifacts": artifacts,
            "warnings": self.warnings,
            "error": self.error,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "simulation_record.v2",
            "case_id": self.case_id,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "created_at": self.created_at,
            "run_seconds": self.run_seconds,
            "params": self.params,
            "circuit": self.circuit,
            "load": self.load,
            "solver": self.solver,
            "measurement": self.measurement,
            "artifacts": {
                **self.case_files,
                "netlist": self.netlist_file,
                "waveform": self.waveform_file,
                "solver_log": self.solver_log_file,
                **({"frequency_response": self.frequency_response_file} if self.frequency_response_file else {}),
            },
            "warnings": self.warnings,
            "error": self.error,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """One solver response together with its persisted replay record."""

    record: SimRecord
    response: SimulationResult


# -----------------------------------------------------------------------------
# Provenance: enough to reproduce a run, or to prove two runs differ
# -----------------------------------------------------------------------------


def _digest(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plugin_provenance(case: Case) -> list[dict[str, Any]]:
    out = []
    for raw in case.data.get("plugins") or []:
        path = Path(raw)
        path = (path if path.is_absolute() else case.base_dir / path).resolve()
        out.append({"path": str(path), "sha256": file_sha256(path), "exists": path.exists()})
    return out


def _build_provenance(case: Case, params: dict[str, Any], settings: SolverSettings) -> dict[str, Any]:
    provenance = {
        "platform_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "python_version": sys.version.split()[0],
        "case_path": str(case.path),
        "case_data_sha256": _digest(case.data),
        "params_sha256": _digest(params),
        "plugins": _plugin_provenance(case),
        "solver": solver_identity(settings),
    }
    if case.is_resolved_rf:
        provenance["input_data_sha256"] = _digest(case.authored_data)
        provenance["input_schema"] = str(case.authored_data.get("schema"))
    return provenance


# -----------------------------------------------------------------------------
# Run directories
# -----------------------------------------------------------------------------


def _run_root(case: Case, run_root: str | Path | None) -> Path:
    return Path(run_root or case.data.get("run", {}).get("root", "runs")).resolve()


def _make_run_dir(root: Path, run_id: str | None, params: dict[str, Any]) -> Path:
    if run_id:
        return _ensure_unique_dir(root / artifact_path_segment(run_id))
    # The digest names the directory; it is not a security primitive, and
    # usedforsecurity=False does not change it, so names stay reproducible.
    digest = hashlib.sha1(repr(sorted(params.items())).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return _ensure_unique_dir(root / f"sim_{time.strftime('%Y%m%d_%H%M%S')}_{digest}")


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(artifact_path_segment(f"{path.name}_{i:03d}"))
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot create unique run directory near {path}")


# -----------------------------------------------------------------------------
# Preparing and running one case
# -----------------------------------------------------------------------------


def archive_case_bundle(case: Case, directory: str | Path) -> tuple[Case, dict[str, str]]:
    """Persist one self-contained data snapshot and return its executable case."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    flattened_netlist: str | None = None
    netlist_dependencies: tuple[Path, ...] = ()
    circuit_cfg = case.data.get("circuit")
    if isinstance(circuit_cfg, dict) and isinstance(circuit_cfg.get("netlist_file"), str):
        source_netlist = resolve_path(case, circuit_cfg["netlist_file"]).resolve()
        if source_netlist.is_file():
            flattened_netlist, netlist_dependencies = flatten_netlist_file(source_netlist)
    dependency_references = [
        (f"$.circuit.netlist_file.dependencies[{index}]", path) for index, path in enumerate(netlist_dependencies)
    ]
    input_manifest, replacements = archive_data_files(
        case.data,
        case.base_dir,
        root,
        extra_references=dependency_references,
    )
    archived_data = rewrite_data_file_paths(case.data, case.base_dir, replacements)
    if flattened_netlist is not None:
        bundled_name = "imported_netlist.cir"
        atomic_write_text(root / bundled_name, flattened_netlist)
        archived_circuit = dict(archived_data.get("circuit") or {})
        archived_circuit["netlist_file"] = bundled_name
        archived_data["circuit"] = archived_circuit
    # Plugins are executable code rather than input data.  Preserve their
    # existing provenance while keeping relative plugin paths runnable from
    # the relocated snapshot.
    plugins = archived_data.get("plugins")
    if isinstance(plugins, list):
        archived_data["plugins"] = [
            str((Path(raw) if Path(raw).is_absolute() else case.base_dir / Path(raw)).resolve()) for raw in plugins
        ]

    (root / "case.yaml").write_text(yaml_dump(archived_data), encoding="utf-8")
    write_json(root / "input_manifest.json", input_manifest)
    files = {"case": "case.yaml", "input_manifest": "input_manifest.json"}
    if flattened_netlist is not None:
        files["imported_netlist"] = "imported_netlist.cir"
    archived_plan = deepcopy(case.resolved_plan)
    if case.is_resolved_rf:
        # input_case is the exact authored record.  The executable case and
        # resolved plan point to the immutable bundled data.
        (root / "input_case.yaml").write_text(yaml_dump(case.authored_data), encoding="utf-8")
        archived_plan = rewrite_data_file_paths(case.resolved_plan or {}, case.base_dir, replacements)
        (root / "resolved_plan.yaml").write_text(yaml_dump(archived_plan), encoding="utf-8")
        files.update({"input_case": "input_case.yaml", "resolved_plan": "resolved_plan.yaml"})
    snapshot = Case(
        path=(root / "case.yaml").resolve(),
        data=archived_data,
        source_data=case.source_data,
        resolved_plan=archived_plan,
    )
    return snapshot, files


def archive_case_definition(case: Case, directory: str | Path) -> dict[str, str]:
    """Persist the executable case, authored input, and referenced data."""

    _snapshot, files = archive_case_bundle(case, directory)
    return files


def _shared_case_files(run_dir: Path, bundle_root: str | Path) -> dict[str, str]:
    root = Path(bundle_root).resolve()
    names = {
        "case": "case.yaml",
        "input_manifest": "input_manifest.json",
        "input_case": "input_case.yaml",
        "resolved_plan": "resolved_plan.yaml",
        "imported_netlist": "imported_netlist.cir",
    }
    return {
        key: Path(os.path.relpath(root / name, run_dir)).as_posix()
        for key, name in names.items()
        if (root / name).is_file()
    }


def prepare_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    solver_name: str | None = None,
    case_archive_root: str | Path | None = None,
) -> SimRecord:
    """Write every artifact for a run without executing a solver."""

    full_params = fill_default_params(case, params)
    simulation = resolve_simulation_case(case, full_params, solver_name)
    netlist_inputs = build_netlist_inputs(case, full_params)
    run_dir = _make_run_dir(_run_root(case, run_root), run_id, full_params)
    return _prepare_case_in_dir(case, full_params, run_dir, simulation, netlist_inputs, case_archive_root)


def _prepare_case_in_dir(
    case: Case,
    full_params: dict[str, Any],
    run_dir: Path,
    simulation: ResolvedSimulationCase,
    netlist_inputs: NetlistInputs,
    case_archive_root: str | Path | None,
) -> SimRecord:
    """Prepare one already allocated run directory."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    (run_dir / "debug").mkdir(exist_ok=True)

    circuit = netlist_inputs.circuit
    netlist = render_ngspice_netlist(case, circuit, netlist_inputs.load_subckt, full_params, simulation)

    case_files = (
        _debug_case_files(archive_case_definition(case, run_dir / "debug"))
        if case_archive_root is None
        else _shared_case_files(run_dir, case_archive_root)
    )
    (run_dir / NETLIST_ARTIFACT).write_text(netlist, encoding="utf-8")
    (run_dir / SOLVER_LOG_ARTIFACT).write_text("prepared only; solver was not executed\n", encoding="utf-8")

    probes = simulation.probes
    measurement = simulation.measurement
    solver = simulation.solver.name
    record = SimRecord(
        run_dir=run_dir,
        case_id=case.case_id,
        status="prepared",
        params=full_params,
        circuit=netlist_inputs.circuit_name,
        load=netlist_inputs.load_name,
        solver=solver,
        measurement={
            "voltage_node": measurement.voltage_node or circuit.output_node,
            "current_source": probes.source_name,
            "load_ports": {
                "p": measurement.load_positive or circuit.output_node,
                "n": measurement.load_negative,
            },
            "load_current": probes.load_current_column,
            "reference_plane": measurement.reference_plane,
        },
        warnings=case_warnings(case) + circuit.warnings(),
        provenance=_build_provenance(case, full_params, simulation.solver),
        case_files=case_files,
    )
    _write_record(case, record)
    return record


def execute_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    solver_override: str | None = None,
    run_id: str | None = None,
    case_archive_root: str | Path | None = None,
) -> SimulationRun:
    """Resolve, solve, and persist one case through the canonical path.

    Solver-reported execution failures are recorded as results so a study can
    keep collecting observations.  Invalid configuration and implementation
    exceptions propagate instead of being disguised as simulation results.
    """

    start = time.perf_counter()
    full_params = fill_default_params(case, params)
    simulation = resolve_simulation_case(case, full_params, solver_override)
    netlist_inputs = build_netlist_inputs(case, full_params)
    run_dir = _make_run_dir(_run_root(case, run_root), run_id, full_params)
    prepared = _prepare_case_in_dir(case, full_params, run_dir, simulation, netlist_inputs, case_archive_root)
    result = _run_solver(prepared, simulation)
    warnings = list(prepared.warnings)
    if result.status != "ok":
        warnings.append(f"solver status: {result.status}")
    final = replace(
        prepared,
        status=result.status,
        solver=simulation.solver.name,
        warnings=warnings,
        run_seconds=time.perf_counter() - start,
        diagnostics=result.diagnostics,
        frequency_response_file=AC_ARTIFACT if result.frequency_response is not None else None,
    )
    _write_record(case, final)
    return SimulationRun(final, result)


def simulate_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    solver_override: str | None = None,
    run_id: str | None = None,
    case_archive_root: str | Path | None = None,
) -> SimRecord:
    """Run one case and return its persisted record.

    ``execute_case`` is the shared in-memory boundary used by studies and
    future identification.  This wrapper preserves the small public API used
    by direct simulation callers.
    """

    return execute_case(
        case,
        params=params,
        run_root=run_root,
        solver_override=solver_override,
        run_id=run_id,
        case_archive_root=case_archive_root,
    ).record


def _run_solver(record: SimRecord, simulation: ResolvedSimulationCase) -> SimulationResult:
    request = SolverRunRequest(
        netlist_path=record.run_dir / record.netlist_file,
        run_dir=record.run_dir,
        simulation=simulation,
    )
    result = invoke_solver(simulation.solver.name, request)
    if not isinstance(result, SimulationResult):
        raise TypeError(f"solver '{simulation.solver.name}' must return SimulationResult")
    result.as_frame().to_csv(record.run_dir / record.waveform_file, index=False)
    if result.frequency_response is not None:
        result.frequency_response.to_csv(record.run_dir / AC_ARTIFACT, index=False)
    (record.run_dir / record.solver_log_file).write_text(result.log or "", encoding="utf-8")
    _remove_solver_scratch(record.run_dir)
    return result


def _write_record(case: Case, record: SimRecord) -> None:
    del case
    write_json(record.run_dir / DEBUG_MANIFEST_FILE, record.manifest())
    write_json(record.run_dir / SUMMARY_FILE, record.summary())


def _debug_case_files(files: dict[str, str]) -> dict[str, str]:
    return {name: (Path("debug") / path).as_posix() for name, path in files.items()}


def _remove_solver_scratch(run_dir: Path) -> None:
    """Remove solver-format intermediates after canonical data is persisted."""

    for name in (WAVEFORM_FILE, AC_FILE, "solver.log"):
        (run_dir / name).unlink(missing_ok=True)
