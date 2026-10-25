"""One saved simulation run to a concise electrical summary and figures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd import __version__
from pcd.analysis.ac import DEFAULT_Z0, frequency_sweep_metrics
from pcd.analysis.measurements import ac_operating_metrics, periodic_operating_metrics
from pcd.analysis.traces import frequency_metrics, instantaneous_power, relative_harmonic_levels
from pcd.analysis.transient import rf_measurement_options
from pcd.artifacts import (
    create_run_directory,
    file_sha256,
    package_source_sha256,
    runtime_environment,
    utc_now,
    write_json,
)
from pcd.case import Case, load_case
from pcd.component_models import observed_components
from pcd.core.models import UnsettledMeasurementError
from pcd.output import output_settings
from pcd.probes import ComponentObservation, signal_catalog
from pcd.records import artifact_path, frequency_response_path, load_frequency_response, load_waveform, read_sim_record
from pcd.signals import time_average
from pcd.simulation import LOAD_VOLTAGE_COLUMN
from pcd.simulation_input import resolve_simulation_case
from pcd.spice import ac_operating_frequency_hz, periodic_frequency_hz

from .figures import render_frequency_response, render_operating_point, render_transient_response
from .harmonics import render_harmonic_spectrum
from .recipes import render_recipe

SUMMARY_FILE = "summary.json"


def analyze_run(
    record_or_path: dict[str, Any] | str | Path,
    *,
    out_dir: str | Path | None = None,
    frequency_hz: float | None = None,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate standard electrical outputs from one successful saved run."""

    record = read_sim_record(record_or_path)
    if record.get("status") != "ok":
        raise ValueError(f"cannot analyze simulation record with status {record.get('status')!r}")
    case = _archived_case(record)
    params = dict(record.get("params") or {})
    settings = output_settings(case.data.get("output") if output is None else output)
    simulation = resolve_simulation_case(case, params)
    components = observed_components(case, params)
    z0 = _reference_impedance(case)
    waveform = load_waveform(record)
    ac = _frequency_response(record, simulation.probes.ac_columns)
    if frequency_hz is not None and (ac is None or ac.empty):
        raise ValueError("--frequency-Hz selects an AC operating point; this run has no AC response")
    marker_hz = _measurement_frequency(case, params, frequency_hz) if ac is not None and not ac.empty else 0.0
    periodic_hz = _periodic_frequency(case, params)
    analyses, operating_metrics = _summarize_analyses(
        ac,
        waveform,
        marker_hz,
        periodic_hz,
        z0,
        simulation.probes.load_current_column,
        components,
        case,
    )
    if not analyses:
        raise ValueError("saved run contains neither a frequency response nor a transient waveform")

    provenance = _analysis_provenance(record, params, frequency_hz, settings)
    requested = (Path(out_dir) if out_dir is not None else Path(record["run_dir"]) / "analysis").resolve()
    destination = create_run_directory(requested.parent, requested.name)
    title = str(record.get("case_id", case.case_id))
    measured = dict(record.get("measurement") or {})
    p = simulation.measurement.load_positive or str((case.data.get("circuit") or {}).get("output_node", "out"))
    n = simulation.measurement.load_negative
    load_vector = f"v({p},{n})" if n != "0" else f"v({p})"
    measured.setdefault("load_ports", {"p": p, "n": n})
    measured.setdefault(
        "output_voltage_vector",
        f"v({simulation.measurement.voltage_node})" if simulation.measurement.voltage_node else load_vector,
    )
    measured.setdefault("drive_basis", (case.data.get("metadata") or {}).get("drive_basis", "declared_or_imported"))
    catalog = signal_catalog(
        simulation.probes,
        measured,
        transient_columns=tuple(waveform.columns) if not waveform.empty else (),
        ac_columns=tuple(ac.columns) if ac is not None else (),
    )
    power = (
        instantaneous_power(waveform, simulation.probes.load_current_column) if not waveform.empty else pd.DataFrame()
    )
    for name in map(str, power.columns):
        if name != "time_s":
            waveform[name] = power[name]
            expression = {
                "source_delivered_current_A": "-current_A",
                "source_power_W": "-source_voltage_V * current_A",
                "load_power_W": f"load_voltage_V * {simulation.probes.load_current_column}",
            }[name]
            catalog["signals"][name] = {
                "vector": f"derived: {expression}",
                "formula": expression,
                "unit": "A" if name.endswith("_A") else "W",
                "label": name,
                "positive_direction": "delivered_by_source" if name.startswith("source") else "into_load",
                "columns": {"transient": [name]},
            }
    context = title + (
        "\n1 V peak reference excitation; power and stress at this reference"
        if measured["drive_basis"] == "normalized_1V_peak"
        else ""
    )
    write_json(destination / "signals.json", catalog)
    artifacts = {"summary": SUMMARY_FILE, "signals": "signals.json"}
    if ac is not None and not ac.empty:
        derived = frequency_metrics(ac, z0, simulation.probes.load_current_column, components)
        derived.to_csv(destination / "frequency_metrics.csv", index=False)
        artifacts["frequency_metrics"] = "frequency_metrics.csv"
    harmonic_rows = []
    periodic = _periodic_metrics(analyses)
    for key, unit in (("voltage_harmonic_amplitude_V", "V"), ("current_harmonic_amplitude_A", "A")):
        values = periodic.get(key) or {}
        if values:
            orders = sorted(int(name[1:]) for name in values)
            amplitude = np.asarray([values[f"h{order}"] for order in orders])
            for order, value, level in zip(orders, amplitude, relative_harmonic_levels(amplitude), strict=True):
                harmonic_rows.append(
                    {
                        "signal": "load_voltage_V" if unit == "V" else simulation.probes.load_current_column,
                        "unit": unit,
                        "harmonic": order,
                        "frequency_Hz": order * periodic["fundamental_Hz"],
                        "amplitude": value,
                        "relative_dBc": level,
                        "window_start_s": periodic["measurement_start_s"],
                        "window_end_s": periodic["measurement_end_s"],
                        "measurement_cycles": periodic["measurement_cycles"],
                    }
                )
    if harmonic_rows:
        pd.DataFrame(harmonic_rows).to_csv(destination / "harmonic_metrics.csv", index=False)
        artifacts["harmonic_metrics"] = "harmonic_metrics.csv"
    if not power.empty:
        power.to_csv(destination / "instantaneous_power.csv", index=False)
        artifacts["instantaneous_power"] = "instantaneous_power.csv"
    pd.DataFrame(
        [{key: value for key, value in operating_metrics.items() if not isinstance(value, (dict, list, tuple))}]
    ).to_csv(destination / "operating_metrics.csv", index=False)
    artifacts["operating_metrics"] = "operating_metrics.csv"
    outputs = _render_artifacts(
        destination,
        context,
        ac,
        waveform,
        marker_hz,
        z0,
        simulation.probes.load_current_column,
        operating_metrics,
        components,
        analyses,
        settings,
        catalog,
        periodic_hz,
        case.data.get("measurement"),
    )
    artifacts.update(outputs.pop("artifacts"))
    schematic: dict[str, Any] = {"status": "not_requested"}
    if settings["schematic"]["enabled"]:
        netlist = artifact_path(record, "netlist")
        if netlist is None or not netlist.is_file():
            schematic = {"status": "not_available", "issues": ["saved record has no netlist"]}
        else:
            from pcd.netlist_viz import render_netlist_schematic

            for extension in settings["formats"]:
                schematic = render_netlist_schematic(
                    netlist,
                    destination / f"circuit.{extension}",
                    title=context,
                    view=settings["schematic"]["view"],
                    netlist_mode="deck",
                    signals=catalog,
                    params=params,
                )
            schematic["figures"] = [f"circuit.{extension}" for extension in settings["formats"]]
            write_json(destination / "circuit.json", schematic)
            artifacts["circuit"] = schematic["figures"][0]
            artifacts["circuit_connectivity"] = "circuit.json"
    outputs["schematic"] = schematic
    required_schematic = "output" in case.data or output is not None
    incomplete = any(item["status"] != "complete" for item in outputs["plots"])
    incomplete |= schematic["status"] == "partial" or (required_schematic and schematic["status"] == "not_available")

    summary = {
        "schema": "run_analysis.v1",
        "status": "partial" if incomplete else "complete",
        "output": {"settings": settings, **outputs},
        "case_id": title,
        "simulation": {
            "status": "ok",
            "solver": record.get("solver"),
            "solver_version": ((record.get("provenance") or {}).get("solver") or {}).get("version"),
            "run_dir": str(Path(record["run_dir"]).resolve()),
            "provenance": record.get("provenance") or {},
        },
        "measurement": {
            "load_reference_plane": simulation.measurement.reference_plane,
            "impedance_reference_plane": "source_port",
            "drive_basis": measured["drive_basis"],
            "reference_impedance_ohm": z0,
            "ac_frequency_Hz": marker_hz if ac is not None and not ac.empty else None,
            "periodic_fundamental_Hz": periodic_hz,
            "observed_voltage_node": record.get("measurement", {}).get("voltage_node"),
            "load_ports": record.get("measurement", {}).get("load_ports"),
        },
        "analyses": analyses,
        "provenance": provenance,
        "scope": "electrical measurements only; engineering acceptance is evaluated by a study",
        "output_dir": str(destination),
        "artifacts": artifacts,
    }
    write_json(destination / SUMMARY_FILE, summary)
    return summary


def _analysis_provenance(
    record: dict[str, Any], params: dict[str, Any], frequency_hz: float | None, settings: dict[str, Any]
) -> dict[str, Any]:
    from importlib.metadata import version

    inputs = {}
    for name in ("case", "waveform", "frequency_response"):
        path = artifact_path(record, name)
        if path is not None:
            inputs[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    identity = {
        "platform_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "environment": runtime_environment(),
        "inputs": inputs,
        "simulation_record_sha256": hashlib.sha256(
            json.dumps(
                {key: value for key, value in record.items() if key != "output"}, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest(),
        "params": params,
        "options": {"frequency_Hz": frequency_hz, "output": settings},
        "rendering_environment": {name: version(name) for name in ("matplotlib", "schemdraw")},
    }
    return {
        **identity,
        "analysis_id": hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "created_at": utc_now(),
    }


def _summarize_analyses(
    ac: pd.DataFrame | None,
    waveform: pd.DataFrame,
    marker_hz: float,
    periodic_hz: float | None,
    z0: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    case: Case,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analyses: dict[str, Any] = {}
    operating_metrics: dict[str, Any] = {}
    if ac is not None and not ac.empty:
        ac_metrics = _ac_summary(ac, marker_hz, z0, load_current_column, components)
        analyses["ac"] = ac_metrics
        operating_metrics.update(ac_metrics)
    if not waveform.empty:
        transient, periodic_metrics = _transient_summary(
            waveform,
            periodic_hz,
            load_current_column,
            components,
            case,
        )
        analyses["transient"] = transient
        if not operating_metrics:
            operating_metrics.update(periodic_metrics)
    return analyses, operating_metrics


def _render_artifacts(
    destination: Path,
    title: str,
    ac: pd.DataFrame | None,
    waveform: pd.DataFrame,
    marker_hz: float,
    z0: float,
    load_current_column: str | None,
    operating_metrics: dict[str, Any],
    components: tuple[ComponentObservation, ...],
    analyses: dict[str, Any],
    settings: dict[str, Any],
    catalog: dict[str, Any],
    periodic_hz: float | None,
    measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    plots: list[dict[str, Any]] = []
    recipes = settings["plots"]
    if recipes == "auto":
        for extension in settings["formats"]:
            if ac is not None and not ac.empty:
                filename = f"frequency_response.{extension}"
                render_frequency_response(
                    ac,
                    destination / filename,
                    title=title,
                    z0=z0,
                    marker_hz=marker_hz,
                    sweep_metrics=operating_metrics.get("sweep"),
                )
                artifacts.setdefault("frequency_response", filename)
            if not waveform.empty:
                filename = f"transient_response.{extension}"
                render_transient_response(
                    waveform, destination / filename, title=title, load_current_column=load_current_column
                )
                artifacts.setdefault("transient_response", filename)
            filename = f"operating_point.{extension}"
            if render_operating_point(operating_metrics, components, destination / filename, title=title):
                artifacts.setdefault("operating_point", filename)
            filename = f"harmonic_spectrum.{extension}"
            if render_harmonic_spectrum(_periodic_metrics(analyses), destination / filename, title=title):
                artifacts.setdefault("harmonic_spectrum", filename)
        recipes = []
        for domain, kind in (("transient", "waveform"), ("ac", "ac")):
            # Prefer explicit aliases over duplicate automatic voltage/current vectors.
            unique: dict[tuple[str, str | None], str] = {}
            for name, item in catalog["signals"].items():
                if domain in item["columns"]:
                    unique[(item["vector"], item.get("unit"))] = name
            if unique:
                recipes.append({"type": kind, "name": f"signals_{domain}", "signals": list(unique.values())})
    for recipe in recipes:
        try:
            plots.append(
                render_recipe(
                    recipe,
                    destination,
                    settings["formats"],
                    catalog,
                    waveform,
                    ac,
                    title=title,
                    fundamental_hz=periodic_hz,
                    measurement=measurement,
                )
            )
        except ValueError as exc:
            plots.append({"status": "not_available", "recipe": recipe, "reason": str(exc)})
    return {"artifacts": artifacts, "plots": plots}


def _periodic_metrics(analyses: dict[str, Any]) -> dict[str, Any]:
    transient = analyses.get("transient")
    if not isinstance(transient, dict):
        return {}
    periodic = transient.get("periodic")
    return periodic if isinstance(periodic, dict) and periodic.get("status") == "ok" else {}


def _archived_case(record: dict[str, Any]) -> Case:
    path = artifact_path(record, "case")
    if path is None or not path.is_file():
        raise ValueError("simulation record does not contain its archived executable case")
    return load_case(path)


def _measurement_frequency(case: Case, params: dict[str, Any], requested: float | None) -> float:
    frequency = ac_operating_frequency_hz(case, params) if requested is None else requested
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("analysis frequency must be positive and finite")
    return frequency


def _periodic_frequency(case: Case, params: dict[str, Any]) -> float | None:
    """Resolve a declared periodic fundamental without inventing one for DC transients."""

    return periodic_frequency_hz(case, params)


def _reference_impedance(case: Case) -> float:
    measurement = case.data.get("measurement", {}) or {}
    value = float(measurement.get("reference_impedance_ohm", DEFAULT_Z0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("measurement.reference_impedance_ohm must be positive and finite")
    return value


def _frequency_response(record: dict[str, Any], probe_columns: tuple[str, ...]) -> pd.DataFrame | None:
    if frequency_response_path(record) is None:
        return None
    extra_columns = (LOAD_VOLTAGE_COLUMN, *probe_columns)
    return load_frequency_response(record, extra_columns)


def _ac_summary(
    ac: pd.DataFrame,
    frequency_hz: float,
    z0: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
) -> dict[str, Any]:
    metrics = ac_operating_metrics(ac, frequency_hz, z0, load_current_column, components)
    reflection_db = metrics.pop("reflection_db")
    metrics["return_loss_db"] = None if reflection_db is None else -reflection_db
    sweep = frequency_sweep_metrics(ac, z0, load_current_column)
    if sweep is not None:
        metrics["sweep"] = sweep
    return metrics


def _transient_summary(
    waveform: pd.DataFrame,
    frequency_hz: float | None,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    case: Case,
) -> tuple[dict[str, Any], dict[str, Any]]:
    time_s = waveform["time_s"].to_numpy(float)
    voltage = waveform["voltage_V"].to_numpy(float)
    summary: dict[str, Any] = {
        "samples": len(waveform),
        "start_s": float(time_s[0]),
        "end_s": float(time_s[-1]),
        "duration_s": float(time_s[-1] - time_s[0]),
        "waveform_voltage_peak_V": float(np.nanmax(np.abs(voltage))),
        "waveform_voltage_rms_V": float(np.sqrt(max(time_average(voltage**2, time_s), 0.0))),
        "waveform_voltage_dc_V": time_average(voltage, time_s),
    }
    if frequency_hz is None:
        summary["periodic"] = {
            "status": "not_applicable",
            "reason": "no periodic fundamental was declared; basic transient statistics remain available",
        }
        return summary, {}
    options = rf_measurement_options(case.data.get("measurement"))
    try:
        periodic = periodic_operating_metrics(waveform, frequency_hz, load_current_column, components, options)
    except UnsettledMeasurementError as exc:
        summary["periodic"] = {"status": "unsettled", "reason": str(exc)}
        return summary, {}

    if periodic:
        summary["periodic"] = {"status": "ok", **periodic}
    else:
        summary["periodic"] = {
            "status": "not_available",
            "reason": "no load-current or observed-component probes were declared",
        }
    return summary, periodic
