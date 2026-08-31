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
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .analysis import AC_FILE, load_current
from .artifacts import (
    archive_data_files,
    artifact_path_segment,
    file_sha256,
    package_source_sha256,
    rewrite_data_file_paths,
    utc_now,
    write_json,
    yaml_dump,
)
from .case import Case, case_warnings, fill_default_params
from .netlist import Circuit, build_circuit, build_load_subckt, render_ngspice_netlist
from .sim_registry import get as get_sim_method
from .solver import SimulationResult, solver_identity

# Re-exported so plugins and callers have one import for the simulation layer.
__all__ = [
    "Circuit",
    "SimRecord",
    "SimulationResult",
    "archive_case_bundle",
    "archive_case_definition",
    "prepare_case",
    "simulate_case",
]


@dataclass
class SimRecord:
    """One simulation run, as written to ``sim_manifest.json``."""

    run_dir: Path
    case_id: str
    status: str
    params: dict[str, Any]
    circuit: str
    load: str
    solver: str
    netlist_file: str = "netlist.cir"
    waveform_file: str = "waveform.csv"
    solver_log_file: str = "solver.log"
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
            # Legacy flat keys keep downstream scripts simple.
            "netlist_file": self.netlist_file,
            "waveform_file": self.waveform_file,
            "solver_log_file": self.solver_log_file,
            "warnings": self.warnings,
            "error": self.error,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


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


def _build_provenance(case: Case, params: dict[str, Any], solver_name: str) -> dict[str, Any]:
    provenance = {
        "platform_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "python_version": sys.version.split()[0],
        "case_path": str(case.path),
        "case_data_sha256": _digest(case.data),
        "params_sha256": _digest(params),
        "plugins": _plugin_provenance(case),
        "solver": solver_identity(case, solver_name),
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
    input_manifest, replacements = archive_data_files(case.data, case.base_dir, root)
    archived_data = rewrite_data_file_paths(case.data, case.base_dir, replacements)
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
    run_dir = _make_run_dir(_run_root(case, run_root), run_id, full_params)
    run_dir.mkdir(parents=True, exist_ok=True)

    circuit_name, circuit = build_circuit(case, full_params)
    load_name, load_subckt = build_load_subckt(case, full_params)
    netlist = render_ngspice_netlist(case, circuit, load_subckt, full_params)

    case_files = (
        archive_case_definition(case, run_dir)
        if case_archive_root is None
        else _shared_case_files(run_dir, case_archive_root)
    )
    (run_dir / "netlist.cir").write_text(netlist, encoding="utf-8")
    (run_dir / "solver.log").write_text("prepared only; solver was not executed\n", encoding="utf-8")
    write_json(run_dir / "params.json", full_params)

    meas = case.data.get("measurement", {}) or {}
    load_cfg = case.data.get("load", {}) or {}
    load_ports = load_cfg.get("ports", {}) or {}
    solver = str(solver_name or case.data.get("solver", {}).get("name", "not_run"))
    record = SimRecord(
        run_dir=run_dir,
        case_id=case.case_id,
        status="prepared",
        params=full_params,
        circuit=circuit_name,
        load=load_name,
        solver=solver,
        measurement={
            "voltage_node": meas.get("voltage_node", circuit.output_node),
            "current_source": meas.get("current_source", "Vsrc"),
            "load_ports": {
                "p": str(load_ports.get("p", circuit.output_node)),
                "n": str(load_ports.get("n", "0")),
            },
            "load_current": load_current(case),
            "reference_plane": load_cfg.get("reference_plane", "load_ports"),
        },
        warnings=case_warnings(case) + circuit.warnings(),
        provenance=_build_provenance(case, full_params, solver),
        case_files=case_files,
    )
    _write_record(case, record)
    return record


def simulate_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    solver_override: str | None = None,
    run_id: str | None = None,
    case_archive_root: str | Path | None = None,
) -> SimRecord:
    """Prepare, run, and record one case.

    A failure is recorded rather than raised: the run directory always ends up
    with a manifest and a waveform file so an optimizer can score it and keep
    going.  ``--strict-exit`` is what turns that into a nonzero exit code.
    """

    start = time.perf_counter()
    solver_name = str(solver_override or case.data.get("solver", {}).get("name", "dummy"))
    prepared: SimRecord | None = None
    try:
        prepared = prepare_case(
            case,
            params=params,
            run_root=run_root,
            run_id=run_id,
            solver_name=solver_name,
            case_archive_root=case_archive_root,
        )
        result = _run_solver(case, prepared, solver_name)
        warnings = list(prepared.warnings)
        if result.status != "ok":
            warnings.append(f"solver status: {result.status}")
        final = replace(
            prepared,
            status=result.status,
            solver=solver_name,
            warnings=warnings,
            run_seconds=time.perf_counter() - start,
            diagnostics=result.diagnostics,
            frequency_response_file=AC_FILE if result.frequency_response is not None else None,
        )
    except Exception as exc:
        final = _record_failure(
            case,
            prepared,
            params,
            run_root,
            run_id,
            solver_name,
            exc,
            start,
            case_archive_root,
        )
    _write_record(case, final)
    return final


def _run_solver(case: Case, record: SimRecord, solver_name: str) -> SimulationResult:
    solver = get_sim_method("solver", solver_name)
    result = solver(record.run_dir / "netlist.cir", record.run_dir, case, record.params)
    if not isinstance(result, SimulationResult):
        raise TypeError(f"solver '{solver_name}' must return SimulationResult")
    result.as_frame().to_csv(record.run_dir / "waveform.csv", index=False)
    (record.run_dir / "solver.log").write_text(result.log or "", encoding="utf-8")
    return result


def _record_failure(
    case: Case,
    prepared: SimRecord | None,
    params: dict[str, Any] | None,
    run_root: str | Path | None,
    run_id: str | None,
    solver_name: str,
    exc: Exception,
    start: float,
    case_archive_root: str | Path | None,
) -> SimRecord:
    """Build a complete failed record, even if preparation itself failed."""

    full_params = fill_default_params(case, params)
    if prepared is None:
        run_dir = _make_run_dir(_run_root(case, run_root), run_id, full_params)
        run_dir.mkdir(parents=True, exist_ok=True)
        prepared = SimRecord(
            run_dir=run_dir,
            case_id=case.case_id,
            status="failed",
            params=full_params,
            circuit="unknown",
            load="unknown",
            solver=solver_name,
            warnings=case_warnings(case),
            provenance=_build_provenance(case, full_params, solver_name),
        )
        prepared.case_files = (
            archive_case_definition(case, run_dir)
            if case_archive_root is None
            else _shared_case_files(run_dir, case_archive_root)
        )
        write_json(run_dir / "params.json", full_params)

    (prepared.run_dir / "solver.log").write_text(traceback.format_exc(), encoding="utf-8")
    pd.DataFrame(columns=["time_s", "voltage_V", "current_A"]).to_csv(prepared.run_dir / "waveform.csv", index=False)
    return replace(
        prepared,
        status="failed",
        solver=solver_name,
        warnings=[*prepared.warnings, f"simulation exception: {type(exc).__name__}"],
        run_seconds=time.perf_counter() - start,
        error=f"{type(exc).__name__}: {exc}",
        diagnostics={"exception_type": type(exc).__name__, "exception": str(exc)},
    )


def _write_record(case: Case, record: SimRecord) -> None:
    del case
    write_json(record.run_dir / "sim_manifest.json", record.manifest())
