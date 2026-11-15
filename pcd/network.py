"""Measure one frozen SPICE DUT through explicitly declared electrical ports."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis.network import PortExcitation, PortReference, ScatteringResult, solve_scattering
from .artifacts import capture_evidence, file_sha256, write_json, yaml_dump
from .case import load_case
from .inputs import normalize_quantities, read_input_mapping, reject_unknown
from .netlist import reject_generated_name_collisions, spice_names
from .netlist_import import (
    executable_netlist_lines,
    flatten_netlist_file,
    has_implicit_node_shunts,
    stochastic_netlist_function,
    top_level_spice_nodes,
)
from .sim_core import DEBUG_MANIFEST_FILE, execute_case
from .simulation import AcSweep
from .simulation_input import SolverSettings
from .solver import bind_solver_settings
from .sources import rendered_source_scope
from .spice import spice_value
from .validation import validate_case

_NODE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class _FixturePort:
    name: str
    p: str
    n: str
    reference_impedance_ohm: float
    termination_ohm: float
    dc_bias_V: float
    reference_plane: str


def _ports(values: Any) -> tuple[_FixturePort, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("network.ports must be a nonempty list")
    ports = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each network port must be a mapping")
        reject_unknown(
            value,
            {"name", "p", "n", "reference_impedance_ohm", "termination_ohm", "dc_bias_V", "reference_plane"},
            "network port",
        )
        name, p, n = value["name"], str(value["p"]), str(value.get("n", "0"))
        if not isinstance(name, str) or not name.strip() or not _NODE.fullmatch(p) or not _NODE.fullmatch(n):
            raise ValueError("network ports require a name and literal SPICE terminal identifiers")
        if "gnd" in {p.lower(), n.lower()}:
            raise ValueError("network ports must use 0 for the ground terminal")
        if p.lower() == n.lower():
            raise ValueError(f"network port {name!r} must have distinct terminals")
        z0 = _number(value["reference_impedance_ohm"], "reference_impedance_ohm", positive=True)
        termination = _number(value.get("termination_ohm", z0), "termination_ohm", positive=True)
        bias = _number(value.get("dc_bias_V", 0), "dc_bias_V")
        plane = value.get("reference_plane", name)
        if not isinstance(plane, str) or not plane.strip():
            raise ValueError(f"network port {name!r} requires a reference plane label")
        ports.append(_FixturePort(name, p, n, z0, termination, bias, plane))
    if len({port.name for port in ports}) != len(ports):
        raise ValueError("network port names must be unique")
    pairs = {frozenset((port.p.lower(), port.n.lower())) for port in ports}
    if len(pairs) != len(ports):
        raise ValueError("network ports cannot repeat the same terminal pair")
    return tuple(ports)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{field} must be finite" + (" and positive" if positive else ""))
    return number


def _fixture(dut: str, ports: tuple[_FixturePort, ...]) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    names = spice_names(dut)
    nodes = top_level_spice_nodes(dut)
    sources, probes, resistors = [], {}, []
    for index, port in enumerate(ports, start=1):
        if {port.p.lower(), port.n.lower()} - nodes:
            raise ValueError(f"network port {port.name!r} is not established by a top-level DUT connection")
        source, resistor, generator = f"Vpcd_net_{index}", f"Rpcd_net_{index}", f"pcd_net_gen_{index}"
        reject_generated_name_collisions((source, resistor, generator), names)
        sources.append(
            {
                "type": "dc_voltage",
                "name": source,
                "p": generator,
                "n": port.n,
                "voltage_V": port.dc_bias_V,
                "ac_magnitude_V": 2.0,
            }
        )
        resistors.append(f"{resistor} {generator} {port.p} {spice_value(port.termination_ohm)}")
        probes[f"port_{index}_v"] = f"v({port.p},{port.n})" if port.n != "0" else f"v({port.p})"
        probes[f"port_{index}_i"] = f"i({source})"
    return "\n".join([dut.rstrip(), *resistors]) + "\n", sources, probes


def run_network(specification: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Measure the same linearized DUT with one AC excitation per declared port."""
    path = Path(specification).resolve()
    data = normalize_quantities(read_input_mapping(path))
    reject_unknown(data, {"schema", "dut", "ports", "ac", "condition_limit"}, "network")
    if data.get("schema") != "pcd.network.v1":
        raise ValueError("network schema must be pcd.network.v1")
    dut = data["dut"]
    reject_unknown(dut, {"file", "format"}, "network.dut")
    mode = dut.get("format", "fragment")
    ports = _ports(data["ports"])
    ac = data["ac"]
    sweep = AcSweep.from_config(ac)
    if "source" in ac:
        raise ValueError("network.ac.source is selected automatically from the declared ports")
    condition_limit = _number(data.get("condition_limit", 1e8), "condition_limit", positive=True)
    if condition_limit < 1:
        raise ValueError("condition_limit must be at least 1")

    dut_path = (path.parent / dut["file"]).resolve()
    captured: dict[Path, bytes] = {}
    flattened, _ = flatten_netlist_file(dut_path, mode=mode, captured=captured)
    if function := stochastic_netlist_function(flattened, mode=mode):
        raise ValueError(f"network DUT contains stochastic SPICE function {function!r}")
    body = "\n".join(line.text for line in executable_netlist_lines(flattened, mode=mode)) + "\n"
    if has_implicit_node_shunts(body):
        raise ValueError("network DUT cannot use rshunt or cshunt because they also attach to fixture nodes")
    dut_scope = rendered_source_scope(body, "", mode="fragment")
    if dut_scope["ac_excitations"] or dut_scope["ac_unavailable_reason"]:
        raise ValueError("network DUT must have no independent AC excitation or unresolved source include")
    fixture, sources, probes = _fixture(body, ports)

    root = Path(out_dir).resolve()
    root.mkdir(parents=True)
    frozen_input = {**data, "dut": {"file": "dut.cir", "format": "fragment"}}
    (root / "network.yaml").write_text(yaml_dump(frozen_input), encoding="utf-8")
    (root / "dut.cir").write_bytes(body.encode("utf-8"))
    (root / "fixture.cir").write_bytes(fixture.encode("utf-8"))
    settings = bind_solver_settings(SolverSettings())
    states, records = _run_excitations(root, ports, sources, probes, ac, sweep, settings)
    references = [PortReference(port.name, port.reference_impedance_ohm, "out_of_dut") for port in ports]
    result = solve_scattering(references, states, condition_limit=condition_limit)
    artifacts = _write_network_tables(root, result, states)
    summary = {
        "schema": "network_result.v1",
        "status": "complete" if result.complete else "incomplete_evidence",
        "run_root": str(root),
        "scope": "small_signal_scattering_at_declared_electrical_planes",
        "wave_convention": "peak_phasors_into_dut_normalized_to_sqrt_W",
        "ports": [asdict(port) for port in ports],
        "condition_limit": condition_limit,
        "frequency_points": len(result.frequency_hz),
        "solved_points": result.status.count("ok"),
        "excitations": records,
        "dut_source_sha256": {str(source): hashlib.sha256(content).hexdigest() for source, content in captured.items()},
        "artifacts": {"specification": "network.yaml", "dut": "dut.cir", "fixture": "fixture.cir", **artifacts},
    }
    summary["artifact_sha256"] = capture_evidence(root, summary["artifacts"])
    write_json(root / "network_result.json", summary)
    return summary


def _run_excitations(
    root: Path,
    ports: tuple[_FixturePort, ...],
    sources: list[dict[str, Any]],
    probes: dict[str, str],
    ac: dict[str, Any],
    sweep: AcSweep,
    settings: SolverSettings,
) -> tuple[list[PortExcitation], list[dict[str, Any]]]:
    observations, records = [], []
    fixture_digest = file_sha256(root / "fixture.cir")
    for index, (port, source) in enumerate(zip(ports, sources, strict=True), start=1):
        name = f"excitation_{index:02d}"
        path = root / f"{name}.yaml"
        case = {
            "schema": "case_yaml.v1",
            "case_id": name,
            "metadata": {"workflow": "simulation"},
            "sources": sources,
            "circuit": {
                "builder": "from_netlist",
                "netlist_file": "fixture.cir",
                "netlist_mode": "fragment",
                "source_policy": "preserve",
                "output_node": ports[0].p,
            },
            "measurement": {"current_source": source["name"], "probes": probes},
            "solver": {"ac": {**ac, "source": source["name"]}},
        }
        path.write_text(yaml_dump(case), encoding="utf-8")
        loaded = load_case(path)
        validation = validate_case(loaded)
        if not validation.ok:
            raise ValueError(validation.format_text())
        run = execute_case(loaded, run_root=root, run_id=name, solver_settings=settings)
        record = run.record
        archived_fixture = record.run_dir / record.case_files["imported_netlist"]
        if file_sha256(archived_fixture) != fixture_digest:
            raise ValueError("excitation did not use the frozen network fixture")
        scope = record.diagnostics["source_scope"]
        if scope["ac_unavailable_reason"] or scope["ac_excitations"] != [source["name"]]:
            raise ValueError(f"rendered excitation {port.name!r} does not isolate its declared drive")
        frame = run.response.frequency_response if record.status == "ok" else None
        observations.append(
            None
            if frame is None
            else PortExcitation(
                port.name,
                frame["frequency_Hz"].to_numpy(float),
                _phasors(frame, "v", len(ports)),
                _phasors(frame, "i", len(ports)),
            )
        )
        records.append(
            {
                "port": port.name,
                "status": record.status,
                "manifest": (record.run_dir / DEBUG_MANIFEST_FILE).relative_to(root).as_posix(),
                "error": record.error,
            }
        )
    if all(item is not None for item in observations):
        return [item for item in observations if item is not None], records
    grid = next((item.frequency_hz for item in observations if item is not None), sweep.frequencies_hz())
    missing = np.full((len(grid), len(ports)), np.nan + 1j * np.nan)
    return [
        item or PortExcitation(port.name, grid, missing, missing)
        for port, item in zip(ports, observations, strict=True)
    ], records


def _phasors(frame: pd.DataFrame, kind: str, port_count: int) -> np.ndarray:
    columns = [f"port_{index}_{kind}" for index in range(1, port_count + 1)]
    real = frame[[f"{name}_re" for name in columns]].to_numpy(float)
    imag = frame[[f"{name}_im" for name in columns]].to_numpy(float)
    return real + 1j * imag


def _write_network_tables(root: Path, result: ScatteringResult, states: list[PortExcitation]) -> dict[str, str]:
    ports = [port.name for port in result.ports]
    basis = pd.DataFrame(
        {
            "frequency_Hz": result.frequency_hz,
            "status": result.status,
            "incident_rank": result.incident_rank,
            "incident_condition": result.incident_condition,
            "relative_residual": result.relative_residual,
        }
    )
    basis.replace([np.inf, -np.inf], np.nan).to_csv(root / "basis.csv", index=False)
    scattering, waves = [], []
    for i, frequency in enumerate(result.frequency_hz):
        for j, outgoing in enumerate(ports):
            for k, incident in enumerate(ports):
                s = result.scattering[i, j, k]
                scattering.append((frequency, outgoing, incident, s.real, s.imag))
                state = states[k]
                v, current = state.voltage_v[i, j], -state.current_a[i, j]
                a, b = result.incident_waves[i, j, k], result.reflected_waves[i, j, k]
                waves.append(
                    (
                        frequency,
                        outgoing,
                        state.name,
                        v.real,
                        v.imag,
                        current.real,
                        current.imag,
                        a.real,
                        a.imag,
                        b.real,
                        b.imag,
                    )
                )
    pd.DataFrame(scattering, columns=["frequency_Hz", "outgoing_port", "incident_port", "s_re", "s_im"]).to_csv(
        root / "scattering.csv", index=False
    )
    pd.DataFrame(
        waves,
        columns=[
            "frequency_Hz",
            "observation_port",
            "excitation_port",
            "voltage_re_V",
            "voltage_im_V",
            "current_into_dut_re_A",
            "current_into_dut_im_A",
            "incident_re_sqrt_W",
            "incident_im_sqrt_W",
            "reflected_re_sqrt_W",
            "reflected_im_sqrt_W",
        ],
    ).to_csv(root / "waves.csv", index=False)
    return {"basis": "basis.csv", "scattering": "scattering.csv", "waves": "waves.csv"}
