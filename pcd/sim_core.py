from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .common import (
    Case,
    case_warnings,
    default_params,
    fill_default_params,
    param_ref_or_value,
    should_emit_spice_param,
    spice_value,
    utc_now,
    write_json,
    yaml_dump,
)
from .sim_registry import get as get_sim_method, load_plugins as load_sim_plugins


@dataclass
class Component:
    ref: str | None = None
    n1: str | None = None
    n2: str | None = None
    value: Any = None
    raw: str | None = None

    def to_spice(self) -> str:
        if self.raw is not None:
            return self.raw
        if self.ref is None or self.n1 is None or self.n2 is None or self.value is None:
            raise ValueError(f"invalid component: {self}")
        return f"{self.ref} {self.n1} {self.n2} {spice_value(self.value)}"


@dataclass
class Circuit:
    components: list[Component] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    output_node: str = "out"
    ground: str = "0"
    notes: list[str] = field(default_factory=list)

    def add(self, ref: str, n1: str, n2: str, value: Any) -> None:
        self.components.append(Component(ref=str(ref), n1=str(n1), n2=str(n2), value=value))

    def raw(self, line: str) -> None:
        self.components.append(Component(raw=str(line)))

    # Plugin examples often use add_raw for readability.
    add_raw = raw

    def nodes(self) -> set[str]:
        nodes = {self.ground}
        for comp in self.components:
            if comp.n1:
                nodes.add(comp.n1)
            if comp.n2:
                nodes.add(comp.n2)
        return nodes

    def warnings(self) -> list[str]:
        refs = [c.ref for c in self.components if c.ref]
        out: list[str] = []
        if len(refs) != len(set(refs)):
            out.append("duplicate component reference names detected")
        if self.output_node not in self.nodes():
            out.append(f"output_node '{self.output_node}' does not appear in two-terminal components")
        return out


@dataclass
class SimulationResult:
    time_s: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray | None = None
    status: str = "ok"
    log: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        current = self.current_A if self.current_A is not None else np.zeros_like(self.time_s)
        return pd.DataFrame({"time_s": self.time_s, "voltage_V": self.voltage_V, "current_A": current})


@dataclass
class SimRecord:
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
    created_at: str = field(default_factory=utc_now)
    run_seconds: float | None = None
    measurement: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

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
                "netlist": self.netlist_file,
                "waveform": self.waveform_file,
                "solver_log": self.solver_log_file,
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
# Build and render
# -----------------------------------------------------------------------------


def select_circuit_name(case: Case, params: dict[str, Any]) -> str:
    cfg = case.data.get("circuit", {}) or {}
    if "builder_variable" in cfg:
        return str(params.get(str(cfg["builder_variable"]), cfg.get("builder", "from_yaml")))
    raw = cfg.get("builder", cfg.get("topology", "from_yaml"))
    if isinstance(raw, str) and raw.startswith("$"):
        return str(params.get(raw[1:], "from_yaml"))
    return str(raw)


def select_load_name(case: Case, params: dict[str, Any]) -> str:
    cfg = case.data.get("load", {}) or {}
    if not cfg:
        return "none"
    if "name_variable" in cfg:
        return str(params.get(str(cfg["name_variable"]), cfg.get("name", "none")))
    raw = cfg.get("name", "none")
    if isinstance(raw, str) and raw.startswith("$"):
        return str(params.get(raw[1:], "none"))
    return str(raw)


def build_circuit(case: Case, params: dict[str, Any]) -> tuple[str, Circuit]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_circuit_name(case, params)
    builder = get_sim_method("circuit", name)
    circuit = builder(case, params)
    if not isinstance(circuit, Circuit):
        raise TypeError(f"circuit builder '{name}' must return Circuit")
    return name, circuit


def build_load_subckt(case: Case, params: dict[str, Any]) -> tuple[str, str]:
    load_sim_plugins(case.data.get("plugins"), case.base_dir)
    name = select_load_name(case, params)
    builder = get_sim_method("load", name)
    subckt = builder(case, params)
    return name, "" if subckt is None else str(subckt).strip()


def render_source(case: Case, params: dict[str, Any]) -> list[str]:
    if "sources" in case.data:
        sources = case.data.get("sources") or []
    elif "source" in case.data:
        sources = [case.data.get("source") or {}]
    else:
        sources = []

    lines: list[str] = []
    for i, src in enumerate(sources):
        if not src:
            continue
        if "raw" in src:
            lines.append(str(src["raw"]))
            continue
        typ = str(src.get("type", "sine_voltage"))
        name = str(src.get("name", f"Vsrc{i}"))
        p = str(src.get("p", "src"))
        n = str(src.get("n", "0"))
        if typ in {"sine_voltage", "rf_voltage", "voltage_sine", "sine"}:
            amp = param_ref_or_value(src.get("amplitude_V", src.get("amplitude", "Vamp")), params)
            freq = param_ref_or_value(src.get("frequency_Hz", src.get("frequency", "freq")), params)
            phase = param_ref_or_value(src.get("phase_deg", 0.0), params)
            dc = param_ref_or_value(src.get("dc_V", 0.0), params)
            lines.append(f"{name} {p} {n} SIN({spice_value(dc)} {spice_value(amp)} {spice_value(freq)} 0 0 {spice_value(phase)})")
        elif typ in {"dc_voltage", "voltage_dc", "dc"}:
            val = param_ref_or_value(src.get("voltage_V", src.get("value_V", src.get("value", 0.0))), params)
            lines.append(f"{name} {p} {n} DC {spice_value(val)}")
        elif typ in {"voltage_pulse", "pulse"}:
            values = [
                src.get("v1_V", 0.0), src.get("v2_V", 1.0), src.get("delay_s", 0.0),
                src.get("rise_s", 1e-9), src.get("fall_s", 1e-9), src.get("width_s", 1e-6), src.get("period_s", 2e-6),
            ]
            vals = [spice_value(param_ref_or_value(v, params)) for v in values]
            lines.append(f"{name} {p} {n} PULSE({' '.join(vals)})")
        elif typ in {"current_dc"}:
            val = param_ref_or_value(src.get("current_A", src.get("value_A", src.get("value", 0.0))), params)
            lines.append(f"{name} {p} {n} DC {spice_value(val)}")
        else:
            raise ValueError(f"unknown source type: {typ}")
    return lines


def render_ngspice_netlist(
    case: Case,
    circuit: Circuit,
    load_subckt: str,
    params: dict[str, Any],
    waveform_file: str = "waveform.csv",
) -> str:
    solver_cfg = case.data.get("solver", {}) or {}
    tran = solver_cfg.get("tran", {}) or {}
    step = tran.get("step_s", 1e-9)
    stop = tran.get("stop_s", 1e-6)
    options = solver_cfg.get("options", {}) or {}
    meas = case.data.get("measurement", {}) or {}
    output_node = str(meas.get("voltage_node", circuit.output_node))
    source_name = str(meas.get("current_source", "Vsrc"))

    lines: list[str] = [f"* Auto-generated simulation netlist for case: {case.case_id}"]
    for name, value in sorted({**circuit.params, **params}.items()):
        if should_emit_spice_param(str(name), value):
            lines.append(f".param {name}={spice_value(value)}")
    if options:
        lines.append(".options " + " ".join(f"{k}={v}" for k, v in options.items()))

    lines.append("")
    lines.append("* Sources")
    lines.extend(render_source(case, params))
    lines.append("")
    lines.append("* Circuit")
    for comp in circuit.components:
        lines.append(comp.to_spice())

    if load_subckt:
        ports = (case.data.get("load", {}) or {}).get("ports", {}) or {}
        p = str(ports.get("p", output_node))
        n = str(ports.get("n", "0"))
        lines.append("")
        lines.append("* Optional load")
        lines.append(load_subckt)
        lines.append(f"Xload {p} {n} load_model")

    lines.append("")
    lines.append(f".save v({output_node}) i({source_name})")
    lines.append(".control")
    lines.append(f"tran {step} {stop}")
    lines.append(f"wrdata {waveform_file} time v({output_node}) i({source_name})")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Simulation execution
# -----------------------------------------------------------------------------


def _build_provenance(case: Case, params: dict[str, Any], solver_name: str) -> dict[str, Any]:
    solver_cfg = case.data.get("solver", {}) or {}
    if "executable" in solver_cfg:
        executable = solver_cfg.get("executable")
    elif solver_name == "ngspice_cli":
        executable = _default_ngspice_executable()
    else:
        executable = None
    timeout_s = _solver_timeout_s(case)
    solver_info: dict[str, Any] = {
        "name": solver_name,
        "executable": executable,
        "timeout_s": timeout_s,
    }
    if executable:
        solver_info["resolved_executable"] = shutil.which(str(executable))
        solver_info["version"] = _solver_version(str(executable))
    return {
        "platform_version": __version__,
        "python_version": sys.version.split()[0],
        "case_path": str(case.path),
        "case_data_sha256": _sha256_obj(case.data),
        "params_sha256": _sha256_obj(params),
        "plugins": _plugin_provenance(case),
        "solver": solver_info,
    }


def _sha256_obj(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_digest_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plugin_provenance(case: Case) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in case.data.get("plugins") or []:
        path = Path(raw)
        if not path.is_absolute():
            path = case.base_dir / path
        path = path.resolve()
        out.append({"path": str(path), "sha256": _sha256_file(path), "exists": path.exists()})
    return out


def _solver_timeout_s(case: Case) -> float:
    raw = (case.data.get("solver", {}) or {}).get("timeout_s", 300)
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return 300.0
    return timeout if timeout > 0 else 300.0


def _solver_version(executable: str) -> str | None:
    if shutil.which(executable) is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0] if text else None


def diagnose_solver(solver_name: str = "ngspice_cli", executable: str | None = None, timeout_s: float = 300.0) -> dict[str, Any]:
    """Return a small, generic solver environment diagnostic."""

    solver_name = str(solver_name)
    if solver_name == "dummy":
        return {
            "schema": "solver_diagnostic.v1",
            "solver": "dummy",
            "executable": None,
            "resolved_executable": None,
            "version": None,
            "timeout_s": float(timeout_s),
            "batch_runnable": True,
            "windows_prefers_console_binary": False,
            "notes": ["dummy solver does not use an external executable"],
        }
    if solver_name != "ngspice_cli":
        return {
            "schema": "solver_diagnostic.v1",
            "solver": solver_name,
            "executable": executable,
            "resolved_executable": shutil.which(str(executable)) if executable else None,
            "version": None,
            "timeout_s": float(timeout_s),
            "batch_runnable": False,
            "windows_prefers_console_binary": False,
            "notes": [f"no built-in diagnostic for solver '{solver_name}'"],
        }

    default_exe = _default_ngspice_executable()
    exe = str(executable or default_exe)
    resolved = shutil.which(exe)
    version = _solver_version(exe) if resolved else None
    notes: list[str] = []
    if not resolved:
        notes.append(f"executable not found on PATH: {exe}")
    if sys.platform == "win32" and not executable and exe == "ngspice":
        notes.append("ngspice_con.exe was not found on PATH; a GUI-capable ngspice.exe may open a window if used explicitly")
    return {
        "schema": "solver_diagnostic.v1",
        "solver": "ngspice_cli",
        "executable": exe,
        "resolved_executable": resolved,
        "version": version,
        "timeout_s": float(timeout_s),
        "batch_runnable": bool(resolved),
        "windows_prefers_console_binary": bool(sys.platform == "win32" and default_exe == "ngspice_con.exe"),
        "batch_command": [exe, "-b", "-o", "solver.log", "netlist.cir"],
        "notes": notes,
    }


def prepare_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    solver_name: str | None = None,
) -> SimRecord:
    full_params = fill_default_params(case, params)
    root = Path(run_root or case.data.get("run", {}).get("root", "runs")).resolve()
    run_dir = _make_run_dir(root, run_id, full_params)
    run_dir.mkdir(parents=True, exist_ok=True)

    circuit_name, circuit = build_circuit(case, full_params)
    load_name, load_subckt = build_load_subckt(case, full_params)
    netlist_text = render_ngspice_netlist(case, circuit, load_subckt, full_params, waveform_file="waveform.csv")

    (run_dir / "case.yaml").write_text(yaml_dump(case.data), encoding="utf-8")
    write_json(run_dir / "params.json", full_params)
    (run_dir / "netlist.cir").write_text(netlist_text, encoding="utf-8")
    (run_dir / "solver.log").write_text("prepared only; solver was not executed\n", encoding="utf-8")

    meas = case.data.get("measurement", {}) or {}
    record = SimRecord(
        run_dir=run_dir,
        case_id=case.case_id,
        status="prepared",
        params=full_params,
        circuit=circuit_name,
        load=load_name,
        solver=str(solver_name or case.data.get("solver", {}).get("name", "not_run")),
        measurement={"voltage_node": meas.get("voltage_node", circuit.output_node), "current_source": meas.get("current_source", "Vsrc")},
        warnings=case_warnings(case) + circuit.warnings(),
        provenance=_build_provenance(case, full_params, str(solver_name or case.data.get("solver", {}).get("name", "not_run"))),
    )
    write_json(run_dir / "sim_manifest.json", record.manifest())
    _export_circuit_to_plasma_if_needed(case, record)
    return record


def simulate_case(
    case: Case,
    params: dict[str, Any] | None = None,
    run_root: str | Path | None = None,
    solver_override: str | None = None,
    run_id: str | None = None,
) -> SimRecord:
    start = time.perf_counter()
    solver_name = str(solver_override or case.data.get("solver", {}).get("name", "dummy"))
    record: SimRecord | None = None
    try:
        record = prepare_case(case, params=params, run_root=run_root, run_id=run_id, solver_name=solver_name)
        solver = get_sim_method("solver", solver_name)
        result = solver(record.run_dir / "netlist.cir", record.run_dir, case, record.params)
        if not isinstance(result, SimulationResult):
            raise TypeError(f"solver '{solver_name}' must return SimulationResult")
        result.as_frame().to_csv(record.run_dir / "waveform.csv", index=False)
        (record.run_dir / "solver.log").write_text(result.log or "", encoding="utf-8")
        warnings = list(record.warnings)
        if result.status != "ok":
            warnings.append(f"solver status: {result.status}")
        final = SimRecord(
            run_dir=record.run_dir,
            case_id=record.case_id,
            status=result.status,
            params=record.params,
            circuit=record.circuit,
            load=record.load,
            solver=solver_name,
            measurement=record.measurement,
            warnings=warnings,
            run_seconds=time.perf_counter() - start,
            diagnostics=result.diagnostics,
            provenance=record.provenance,
        )
    except Exception as exc:
        full_params = fill_default_params(case, params)
        if record is not None:
            run_dir = record.run_dir
            circuit_name = record.circuit
            load_name = record.load
            measurement = record.measurement
            provenance = record.provenance
            warnings = list(record.warnings)
        else:
            root = Path(run_root or case.data.get("run", {}).get("root", "runs")).resolve()
            root.mkdir(parents=True, exist_ok=True)
            run_dir = _make_run_dir(root, run_id, full_params)
            run_dir.mkdir(parents=True, exist_ok=True)
            circuit_name = "unknown"
            load_name = "unknown"
            measurement = {}
            provenance = _build_provenance(case, full_params, solver_name)
            warnings = case_warnings(case)
        warnings.append(f"simulation exception: {type(exc).__name__}")
        (run_dir / "solver.log").write_text(traceback.format_exc(), encoding="utf-8")
        pd.DataFrame(columns=["time_s", "voltage_V", "current_A"]).to_csv(run_dir / "waveform.csv", index=False)
        write_json(run_dir / "params.json", full_params)
        (run_dir / "case.yaml").write_text(yaml_dump(case.data), encoding="utf-8")
        final = SimRecord(
            run_dir=run_dir,
            case_id=case.case_id,
            status="failed",
            params=full_params,
            circuit=circuit_name,
            load=load_name,
            solver=solver_name,
            measurement=measurement,
            warnings=warnings,
            run_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics={"exception_type": type(exc).__name__, "exception": str(exc)},
            provenance=provenance,
        )
    write_json(final.run_dir / "sim_manifest.json", final.manifest())
    _export_circuit_to_plasma_if_needed(case, final)
    return final


def prepare_candidate_table(case: Case, candidates: pd.DataFrame, run_root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, row in candidates.iterrows():
        params = _row_to_params(row)
        rec = prepare_case(case, params=params, run_root=run_root, run_id=f"sim_{int(i):04d}")
        records.append(rec.manifest())
    return records


def simulate_candidate_table(case: Case, candidates: pd.DataFrame, run_root: str | Path, solver_override: str | None = None) -> list[dict[str, Any]]:
    """Simulation-only batch execution for candidate rows; never scores metrics."""
    records: list[dict[str, Any]] = []
    for i, row in candidates.iterrows():
        params = _row_to_params(row)
        rec = simulate_case(case, params=params, run_root=run_root, solver_override=solver_override, run_id=f"sim_{int(i):04d}")
        records.append(rec.manifest())
    return records


def _row_to_params(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ignore = {"run_dir", "status", "case_id", "circuit", "load", "solver", "schema", "loss"}
    for key, val in row.dropna().to_dict().items():
        key = str(key)
        if key.startswith("metric.") or key in ignore:
            continue
        if key.startswith("param."):
            key = key[len("param."):]
        out[key] = val.item() if hasattr(val, "item") else val
    return out


def _make_run_dir(root: Path, run_id: str | None, params: dict[str, Any]) -> Path:
    if run_id:
        return _ensure_unique_dir(root / run_id)
    digest = hashlib.sha1(repr(sorted(params.items())).encode("utf-8")).hexdigest()[:8]
    return _ensure_unique_dir(root / f"sim_{time.strftime('%Y%m%d_%H%M%S')}_{digest}")


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{i:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot create unique run directory near {path}")


def _export_circuit_to_plasma_if_needed(case: Case, record: SimRecord) -> None:
    cfg = case.data.get("plasma_io", {}) or {}
    if not cfg.get("export", False):
        return
    payload = {
        "schema": "circuit_to_plasma.v3",
        "created_at": utc_now(),
        "case_id": record.case_id,
        "simulation_record": "sim_manifest.json",
        "waveform_file": record.waveform_file,
        "waveform_columns": ["time_s", "voltage_V", "current_A"],
        "voltage_node": record.measurement.get("voltage_node", case.data.get("circuit", {}).get("output_node", "out")),
        "current_source": record.measurement.get("current_source", "Vsrc"),
        "source": case.data.get("source") or case.data.get("sources"),
        "circuit": record.circuit,
        "load": record.load,
        "params": record.params,
        "process_condition": cfg.get("process_condition", {}),
        "geometry": cfg.get("geometry", {}),
        "requested_outputs": cfg.get("requested_outputs", ["Rp_ohm", "Lp_H", "Csh_F"]),
    }
    write_json(record.run_dir / "circuit_to_plasma.json", payload)


# -----------------------------------------------------------------------------
# Built-in solver helpers used by sim_methods
# -----------------------------------------------------------------------------


def dummy_waveform(case: Case, params: dict[str, Any]) -> pd.DataFrame:
    tran = case.data.get("solver", {}).get("tran", {}) or {}
    stop = float(tran.get("stop_s", 2e-6))
    step = float(tran.get("step_s", stop / 500))
    n = max(16, min(50000, int(np.ceil(stop / step)) + 1))
    t = np.linspace(0.0, stop, n)
    src = case.data.get("source", {}) or {}
    typ = str(src.get("type", "sine_voltage"))
    if typ in {"voltage_pulse", "pulse"}:
        v1 = float(param_ref_or_value(src.get("v1_V", 0.0), params, 0.0))
        v2 = float(param_ref_or_value(src.get("v2_V", 1.0), params, 1.0))
        delay = float(param_ref_or_value(src.get("delay_s", 0.0), params, 0.0))
        width = float(param_ref_or_value(src.get("width_s", stop / 4), params, stop / 4))
        period = float(param_ref_or_value(src.get("period_s", max(stop / 2, step)), params, max(stop / 2, step)))
        phase_t = np.mod(np.maximum(t - delay, 0.0), period)
        vin = np.where((t >= delay) & (phase_t < width), v2, v1)
    elif typ in {"dc_voltage", "voltage_dc", "dc"}:
        vin = np.full_like(t, float(param_ref_or_value(src.get("voltage_V", src.get("value", 0.0)), params, 0.0)))
    else:
        amp = float(param_ref_or_value(src.get("amplitude_V", src.get("amplitude", 1.0)), params, 1.0))
        freq = float(param_ref_or_value(src.get("frequency_Hz", src.get("frequency", 1e6)), params, 1e6))
        phase = np.deg2rad(float(param_ref_or_value(src.get("phase_deg", 0.0), params, 0.0)))
        dc = float(param_ref_or_value(src.get("dc_V", 0.0), params, 0.0))
        vin = dc + amp * np.sin(2 * np.pi * freq * t + phase)
    numeric = [float(v) for v in params.values() if isinstance(v, (int, float, np.integer, np.floating)) and abs(float(v)) > 0]
    log_sum = sum(np.tanh(np.log10(abs(v) + 1e-300) / 12.0) for v in numeric) if numeric else 0.0
    gain = 0.65 + 0.12 * np.tanh(log_sum)
    vout = gain * vin
    current = np.gradient(vout, t, edge_order=1) * 1e-10 if len(t) > 2 else np.zeros_like(t)
    return pd.DataFrame({"time_s": t, "voltage_V": vout, "current_A": current})


def parse_wrdata(path: str | Path) -> SimulationResult:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 6:
        # ngspice 46 writes pairs of (scale, vector).  For
        # ``wrdata file time v(node) i(src)`` this becomes
        # time,time,time,V,time,I.
        t, v, i = arr[:, 0], arr[:, 3], arr[:, 5]
    elif arr.shape[1] == 5:
        t, v, i = arr[:, 0], arr[:, 2], arr[:, 4]
    elif arr.shape[1] == 4:
        t, v, i = arr[:, 0], arr[:, 1], arr[:, 3]
    elif arr.shape[1] >= 3:
        t, v, i = arr[:, 0], arr[:, 1], arr[:, 2]
    elif arr.shape[1] == 2:
        t, v, i = arr[:, 0], arr[:, 1], None
    else:
        raise ValueError(f"cannot parse wrdata output: {path}")
    return SimulationResult(time_s=t, voltage_V=v, current_A=i, status="ok", log="")


def _default_ngspice_executable() -> str:
    if sys.platform == "win32" and shutil.which("ngspice_con.exe"):
        return "ngspice_con.exe"
    return "ngspice"


def ngspice_cli(netlist_path: str | Path, run_dir: str | Path, case: Case, params: dict[str, Any]) -> SimulationResult:
    run_dir = Path(run_dir)
    exe = str(case.data.get("solver", {}).get("executable") or _default_ngspice_executable())
    timeout_s = _solver_timeout_s(case)
    diagnostics: dict[str, Any] = {"executable": exe, "timeout_s": timeout_s}
    if shutil.which(exe) is None:
        diagnostics["missing_executable"] = True
        return SimulationResult(
            np.array([0.0]),
            np.array([np.nan]),
            np.array([np.nan]),
            status="failed",
            log=f"ngspice executable not found: {exe}",
            diagnostics=diagnostics,
        )
    log_path = run_dir / "solver.log"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            [exe, "-b", "-o", str(log_path), str(netlist_path)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        diagnostics["timed_out"] = True
        diagnostics["timeout_command"] = list(exc.cmd) if exc.cmd else [exe]
        log = f"ngspice timed out after {timeout_s:g}s\n"
        log += "\nSTDOUT:\n" + (exc.stdout or "") + "\nSTDERR:\n" + (exc.stderr or "")
        return SimulationResult(
            np.array([0.0]),
            np.array([np.nan]),
            np.array([np.nan]),
            status="failed",
            log=log,
            diagnostics=diagnostics,
        )
    log = ""
    if log_path.exists():
        log += log_path.read_text(encoding="utf-8", errors="replace")
    log += "\nSTDOUT:\n" + (completed.stdout or "") + "\nSTDERR:\n" + (completed.stderr or "")
    diagnostics["returncode"] = completed.returncode
    out = run_dir / "waveform.csv"
    if completed.returncode != 0 or not out.exists():
        diagnostics["missing_waveform"] = not out.exists()
        return SimulationResult(
            np.array([0.0]),
            np.array([np.nan]),
            np.array([np.nan]),
            status="failed",
            log=log,
            diagnostics=diagnostics,
        )
    try:
        result = parse_wrdata(out)
    except Exception as exc:
        diagnostics["parse_error"] = f"{type(exc).__name__}: {exc}"
        return SimulationResult(
            np.array([0.0]),
            np.array([np.nan]),
            np.array([np.nan]),
            status="failed",
            log=log,
            diagnostics=diagnostics,
        )
    result.log = log
    result.diagnostics.update(diagnostics)
    return result
