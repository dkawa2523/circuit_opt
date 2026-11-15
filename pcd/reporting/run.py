"""One saved simulation run to a concise electrical summary and figures."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd import __version__
from pcd.analysis.ac import ac_frequency_sample, input_impedance, reference_impedance_ohm, summarize_impedance_sweep
from pcd.analysis.measurements import ac_sample_metrics, periodic_operating_metrics
from pcd.analysis.traces import frequency_metrics, relative_harmonic_levels
from pcd.analysis.transient import rf_measurement_options, transient_power_trace
from pcd.artifacts import (
    create_run_directory,
    data_file_references,
    evidence_digest,
    file_sha256,
    package_source_sha256,
    read_json,
    resolve_artifact_path,
    runtime_environment,
    utc_now,
    write_json,
)
from pcd.case import Case, load_case, resolve_path
from pcd.component_models import observed_components
from pcd.core.models import MeasurementUnavailableError, UnsettledMeasurementError
from pcd.netlist_import import flatten_netlist_file
from pcd.output import output_settings
from pcd.probes import ComponentObservation, signal_catalog
from pcd.records import artifact_path, load_simulation_result, read_sim_record
from pcd.signals import time_average
from pcd.simulation import LOAD_VOLTAGE_COLUMN
from pcd.simulation_input import resolve_simulation_case
from pcd.sources import ac_port_issue, ac_power_balance_issue, rendered_source_scope, transient_power_balance_issue
from pcd.spice import ac_reference_frequency_hz, periodic_frequency_hz

from .figures import render_frequency_response, render_operating_point, render_transient_response
from .harmonics import render_harmonic_spectrum
from .recipes import render_recipe

SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class _AnalysisTables:
    waveform: pd.DataFrame
    ac: pd.DataFrame | None
    impedance: pd.DataFrame | None
    power: pd.DataFrame
    harmonics: pd.DataFrame


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
    provenance = _analysis_provenance(record, case, params, frequency_hz, settings)
    simulation = resolve_simulation_case(case, params, allow_legacy_energy_probes=True)
    components = observed_components(case, params)
    z0 = reference_impedance_ohm(case.data.get("measurement"))
    response = load_simulation_result(record, ac_columns=(LOAD_VOLTAGE_COLUMN, *simulation.probes.ac_columns))
    response.validate_for(simulation.analysis)
    waveform = response.as_frame()
    ac = response.frequency_response
    if frequency_hz is not None and (ac is None or ac.empty):
        raise ValueError("--frequency-Hz selects an AC operating point; this run has no AC response")
    periodic_hz = periodic_frequency_hz(case, params)
    measured = dict(record.get("measurement") or {})
    saved_load_current = measured.get("load_current", simulation.probes.load_current_column)
    load_current = response.diagnostics.get("load_current_column", saved_load_current)
    load_power_issue = response.diagnostics.get("load_power_unavailable_reason")
    source_name = measured.get("current_source", simulation.probes.source_name)
    analyses: dict[str, Any] = {}
    operating_points: dict[str, dict[str, Any]] = {}
    marker_hz = None
    ac_sample = None
    port_issue = None
    frequency_table = None
    periodic: dict[str, Any] = {}
    source_scope = response.diagnostics.get("source_scope")
    if record.get("solver") == "ngspice_cli":
        netlist = artifact_path(record, "netlist")
        if netlist is not None and netlist.is_file():
            source_scope = rendered_source_scope(
                netlist.read_text(encoding="utf-8"),
                source_name,
                observed_voltage=measured.get("source_voltage_vector", simulation.probes.source_voltage_vector),
                observed_current=measured.get("source_current_vector", simulation.probes.source_current_vector),
            )
    transient_port_issue = (source_scope or {}).get("transient_port_unavailable_reason")
    if ac is not None and not ac.empty:
        requested_drive = simulation.analysis.ac.source if simulation.analysis.ac is not None else None
        port_issue = ac_port_issue(case, source_name, requested_drive, source_scope=source_scope)
        ac_sample, frequency_evidence = _measurement_sample(case, params, ac, frequency_hz)
        driven_port = port_issue is None
        ac_balance_issue = ac_power_balance_issue(source_scope)
        frequency_table = frequency_metrics(
            ac, z0, load_current, components, driven_port=driven_port, power_balance_unavailable_reason=ac_balance_issue
        )
        ac_summary = _ac_summary(
            ac_sample,
            z0,
            load_current,
            components,
            driven_port=driven_port,
            power_balance_unavailable_reason=ac_balance_issue,
        )
        ac_summary.update(frequency_evidence)
        if driven_port and (sweep := summarize_impedance_sweep(frequency_table, load_current)) is not None:
            ac_summary["sweep"] = sweep
        if port_issue is not None:
            ac_summary["source_port_unavailable_reason"] = port_issue
        marker_hz = ac_summary.get("frequency_Hz")
        if marker_hz is not None:
            operating_points["ac"] = {key: value for key, value in ac_summary.items() if key != "sweep"}
        analyses["ac"] = ac_summary
    if not waveform.empty:
        balance_issue = (
            transient_power_balance_issue(case, source_name, source_scope=source_scope)
            if source_scope is not None or load_current
            else None
        )
        transient = _transient_summary(
            waveform, periodic_hz, load_current, components, case, balance_issue, transient_port_issue
        )
        analyses["transient"] = transient
        if transient["periodic"]["status"] == "ok":
            periodic = transient["periodic"]
            operating_points["transient_periodic"] = periodic
    if not analyses:
        raise ValueError("saved run contains neither a frequency response nor a transient waveform")

    requested = (Path(out_dir) if out_dir is not None else Path(record["run_dir"]) / "analysis").resolve()
    destination = create_run_directory(requested.parent, requested.name)
    title = str(record.get("case_id", case.case_id))
    p = simulation.measurement.load_positive or str((case.data.get("circuit") or {}).get("output_node", "out"))
    n = simulation.measurement.load_negative
    load_vector = f"v({p},{n})" if n != "0" else f"v({p})"
    measured.setdefault("load_ports", {"p": p, "n": n})
    measured["load_reference_plane"] = simulation.measurement.reference_plane
    measured.setdefault(
        "output_voltage_vector",
        f"v({simulation.measurement.voltage_node})" if simulation.measurement.voltage_node else load_vector,
    )
    measured.setdefault("drive_basis", (case.data.get("metadata") or {}).get("drive_basis", "declared_or_imported"))
    signals_path = artifact_path(record, "signals")
    if signals_path is not None and signals_path.is_file():
        catalog = read_json(signals_path)
        if load_power_issue and (probe := catalog["signals"].get(saved_load_current)):
            probe["positive_direction"] = "as_declared"
    else:
        catalog = signal_catalog(
            replace(
                simulation.probes,
                source_name=source_name,
                source_voltage_vector=measured.get("source_voltage_vector", simulation.probes.source_voltage_vector),
                source_current_vector=measured.get("source_current_vector", simulation.probes.source_current_vector),
                load_current_column=load_current,
            ),
            measured,
            transient_columns=tuple(waveform.columns) if not waveform.empty else (),
            ac_columns=tuple(ac.columns) if ac is not None else (),
        )
    artifacts = {"summary": SUMMARY_FILE, "signals": "signals.json"}
    power = pd.DataFrame()
    if not waveform.empty:
        power = _add_transient_power(waveform, load_current, catalog["signals"], source_port_issue=transient_port_issue)
        if len(power.columns) > 1:
            analyses["transient"]["energy"] = _energy_summary(power)
            power.to_csv(destination / "instantaneous_power.csv", index=False)
            artifacts["instantaneous_power"] = "instantaneous_power.csv"
    context = title + (
        "\n1 V peak reference excitation; power and stress at this reference"
        if measured["drive_basis"] == "normalized_1V_peak"
        else ""
    )
    write_json(destination / "signals.json", catalog)
    if frequency_table is not None:
        frequency_table.replace([np.inf, -np.inf], np.nan).to_csv(destination / "frequency_metrics.csv", index=False)
        artifacts["frequency_metrics"] = "frequency_metrics.csv"
    harmonics = _periodic_harmonic_table(periodic, load_current)
    if not harmonics.empty:
        harmonics.to_csv(destination / "harmonic_metrics.csv", index=False)
        artifacts["harmonic_metrics"] = "harmonic_metrics.csv"
    if operating_points:
        pd.DataFrame(
            [
                {
                    "analysis": domain,
                    "drive_basis": measured["drive_basis"],
                    "observed_source": source_name,
                    "load_reference_plane": measured["load_reference_plane"],
                    **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list, tuple))},
                    **{
                        f"frequency_{side}_Hz": value
                        for side, value in zip(
                            ("lower", "upper"), metrics.get("frequency_bracket_Hz", ()), strict=False
                        )
                    },
                }
                for domain, metrics in operating_points.items()
            ]
        ).to_csv(destination / "operating_metrics.csv", index=False)
        artifacts["operating_metrics"] = "operating_metrics.csv"
    outputs = _render_artifacts(
        destination,
        context,
        _AnalysisTables(waveform, ac, frequency_table if port_issue is None else None, power, harmonics),
        ac_sample,
        z0,
        load_current,
        operating_points,
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
            "load_reference_plane": measured["load_reference_plane"],
            "impedance_reference_plane": measured.get(
                "impedance_reference_plane", "source_port" if source_name is not None else None
            ),
            "drive_basis": measured["drive_basis"],
            "observed_source": source_name,
            "reference_impedance_ohm": z0,
            **({"source_scope": source_scope} if source_scope is not None else {}),
            "ac_frequency_Hz": marker_hz,
            "periodic_fundamental_Hz": periodic_hz,
            "observed_voltage_node": measured.get("voltage_node"),
            "load_ports": measured["load_ports"],
            "load_current": load_current,
            **({"load_power_unavailable_reason": load_power_issue} if load_power_issue else {}),
        },
        "analyses": analyses,
        "provenance": provenance,
        "scope": "electrical measurements only; engineering acceptance is evaluated by a study",
        "output_dir": str(destination),
        "artifacts": artifacts,
    }
    write_json(destination / SUMMARY_FILE, summary)
    return summary


def _add_transient_power(
    waveform: pd.DataFrame, load_current: str | None, signals: dict[str, Any], *, source_port_issue: str | None
) -> pd.DataFrame:
    power_input = waveform.drop(columns="source_voltage_V", errors="ignore") if source_port_issue else waveform
    trace = transient_power_trace(power_input, load_current)
    load_voltage = LOAD_VOLTAGE_COLUMN if LOAD_VOLTAGE_COLUMN in waveform else "voltage_V"
    formulas = {
        "source_delivered_current_A": ("A", "-current_A"),
        "source_power_W": ("W", "-source_voltage_V * current_A"),
        "load_power_W": ("W", f"{load_voltage} * {load_current}"),
        "source_net_energy_J": ("J", "time integral of source_power_W from the first saved sample"),
        "load_net_energy_J": ("J", "time integral of load_power_W from the first saved sample"),
    }
    for name in map(str, trace.columns.drop("time_s")):
        unit, expression = formulas[name]
        alias = name
        while alias in waveform or alias in signals:
            alias = f"derived.{alias}"
        waveform[alias] = trace[name]
        signals[alias] = {
            "vector": f"derived: {expression}",
            "formula": expression,
            "unit": unit,
            "label": alias,
            "positive_direction": "delivered_by_source" if name.startswith("source") else "into_load",
            "columns": {"transient": [alias]},
        }
    return trace


def _energy_summary(trace: pd.DataFrame) -> dict[str, Any]:
    start, end = (float(trace["time_s"].iloc[i]) for i in (0, -1))
    summary: dict[str, Any] = {"start_s": start, "end_s": end}
    for name in ("source_net_energy_J", "load_net_energy_J"):
        if name in trace:
            value = float(trace[name].iloc[-1])
            summary[name] = value if end > start and math.isfinite(value) else None
            if summary[name] is None:
                summary[f"{name}_unavailable_reason"] = "integration requires a finite power trace over a nonzero span"
    return summary


def _periodic_harmonic_table(periodic: dict[str, Any], load_current: str | None) -> pd.DataFrame:
    rows = []
    for key, signal, unit in (
        ("voltage_harmonic_amplitude_V", LOAD_VOLTAGE_COLUMN, "V"),
        ("current_harmonic_amplitude_A", load_current, "A"),
    ):
        if values := periodic.get(key):
            orders = sorted(int(name[1:]) for name in values)
            amplitude = np.asarray([values[f"h{order}"] for order in orders])
            rows.extend(
                {
                    "signal": signal,
                    "unit": unit,
                    "harmonic": order,
                    "frequency_Hz": order * periodic["fundamental_Hz"],
                    "amplitude": value,
                    "relative_dBc": level,
                    "window_start_s": periodic["measurement_start_s"],
                    "window_end_s": periodic["measurement_end_s"],
                    "measurement_cycles": periodic["measurement_cycles"],
                }
                for order, value, level in zip(orders, amplitude, relative_harmonic_levels(amplitude), strict=True)
            )
    return pd.DataFrame(rows)


def _analysis_inputs(record: dict[str, Any], case: Case, *, schematic: bool) -> tuple[dict[str, dict[str, Any]], str]:
    names = ["case", "waveform", "frequency_response", "signals", "netlist"]
    saved = record.get("artifact_sha256") or {}
    names.extend(name for name in ("input_manifest", "imported_netlist") if name in saved)
    inputs = {}
    expected: dict[Path, str] = {}
    for name in names:
        path = artifact_path(record, name)
        if path is None:
            if name in saved:
                raise ValueError(f"saved evidence does not declare its recorded {name} path")
            continue
        path = path.resolve()
        digest = evidence_digest(record, name, path)
        inputs[name] = {"path": str(path), "sha256": digest}
        if digest is not None and name in saved:
            expected[path] = digest
        if name == "input_manifest":
            for item in read_json(path)["inputs"]:
                if item["artifact"] is not None:
                    dependency = resolve_artifact_path(path.parent, item["artifact"]).resolve()
                    expected[dependency] = item["sha256"]

    sources = {f"case:{field}": resolve_path(case, path) for field, path in data_file_references(case.data)}
    if schematic and (netlist := artifact_path(record, "netlist")) is not None and netlist.is_file():
        _, includes = flatten_netlist_file(netlist)
        sources.update({f"netlist:include[{index}]": path for index, path in enumerate(includes)})
    unverified = []
    for name, path in sources.items():
        path = path.resolve()
        digest = file_sha256(path)
        if path not in expected:
            unverified.append(name)
        elif expected[path] != digest:
            raise ValueError(f"saved evidence differs from its recorded {name}: {path}")
        inputs[name] = {"path": str(path), "sha256": digest}

    if "artifact_sha256" not in record:
        return inputs, "unverified_legacy"
    if not record.get("manifest_sha256_verified"):
        return inputs, "unverified_manifest"
    return inputs, "unverified_dependencies" if unverified else "verified"


def _analysis_provenance(
    record: dict[str, Any], case: Case, params: dict[str, Any], frequency_hz: float | None, settings: dict[str, Any]
) -> dict[str, Any]:
    from importlib.metadata import version

    inputs, integrity = _analysis_inputs(record, case, schematic=settings["schematic"]["enabled"])
    identity = {
        "platform_version": __version__,
        "implementation_sha256": package_source_sha256(),
        "environment": runtime_environment(),
        "inputs": inputs,
        "source_integrity": integrity,
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


def _render_artifacts(
    destination: Path,
    title: str,
    tables: _AnalysisTables,
    ac_sample: pd.Series | None,
    z0: float,
    load_current_column: str | None,
    operating_points: dict[str, dict[str, Any]],
    components: tuple[ComponentObservation, ...],
    analyses: dict[str, Any],
    settings: dict[str, Any],
    catalog: dict[str, Any],
    periodic_hz: float | None,
    measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    ac, waveform = tables.ac, tables.waveform
    artifacts: dict[str, str] = {}
    plots: list[dict[str, Any]] = []
    recipes = settings["plots"]
    if recipes == "auto":

        def paths(name: str) -> list[Path]:
            return [destination / f"{name}.{extension}" for extension in settings["formats"]]

        impedance = tables.impedance
        if ac is not None and impedance is not None:
            marker_hz = float(ac_sample["frequency_Hz"]) if ac_sample is not None else None
            marker_label = (
                "\nOperating point interpolated" if analyses["ac"].get("frequency_basis") == "interpolated" else ""
            )
            if ac_sample is not None and marker_hz is not None and not impedance["frequency_Hz"].eq(marker_hz).any():
                marker = input_impedance(pd.DataFrame([ac_sample]), z0)
                impedance = pd.concat([impedance, marker], ignore_index=True).sort_values("frequency_Hz")
            artifacts["frequency_response"] = render_frequency_response(
                impedance,
                paths("frequency_response"),
                title=title + marker_label,
                z0=z0,
                marker_hz=marker_hz,
                sweep_metrics=analyses["ac"].get("sweep"),
            ).name
        if not waveform.empty:
            artifacts["transient_response"] = render_transient_response(
                waveform,
                tables.power,
                paths("transient_response"),
                title=title,
                load_current_column=load_current_column,
            ).name
        for domain, metrics in operating_points.items():
            label = (
                f"AC at {metrics['frequency_Hz']:g} Hz ({metrics['frequency_basis']})"
                if domain == "ac"
                else f"Periodic transient at {periodic_hz:g} Hz"
            )
            name = "operating_point" if domain == "ac" or len(operating_points) == 1 else "periodic_operating_point"
            if figure := render_operating_point(metrics, components, paths(name), title=f"{title}\n{label}"):
                artifacts[name] = figure.name
        if figure := render_harmonic_spectrum(tables.harmonics, paths("harmonic_spectrum"), title=title):
            artifacts["harmonic_spectrum"] = figure.name
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


def _archived_case(record: dict[str, Any]) -> Case:
    path = artifact_path(record, "case")
    if path is None or not (path := path.resolve()).is_file():
        raise ValueError("simulation record does not contain its archived executable case")
    return load_case(path)


def _measurement_sample(
    case: Case, params: dict[str, Any], ac: pd.DataFrame, requested: float | None
) -> tuple[pd.Series | None, dict[str, Any]]:
    if requested is not None:
        return ac_frequency_sample(ac, requested)
    frequency = float(ac["frequency_Hz"].iloc[0]) if len(ac) == 1 else ac_reference_frequency_hz(case, params)
    if frequency is None:
        return None, {
            "operating_point_unavailable_reason": "no AC operating point was declared; the full sweep is available"
        }
    try:
        return ac_frequency_sample(ac, frequency)
    except ValueError as exc:
        return None, {"operating_point_unavailable_reason": str(exc)}


def _ac_summary(
    sample: pd.Series | None,
    z0: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    *,
    driven_port: bool,
    power_balance_unavailable_reason: str | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if sample is not None:
        metrics = ac_sample_metrics(
            sample,
            z0,
            load_current_column,
            components,
            driven_port=driven_port,
            power_balance_unavailable_reason=power_balance_unavailable_reason,
        )
        if driven_port:
            reflection_db = metrics.pop("reflection_db")
            metrics["return_loss_db"] = None if reflection_db is None else -reflection_db
    return metrics


def _transient_summary(
    waveform: pd.DataFrame,
    frequency_hz: float | None,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
    case: Case,
    power_balance_issue: str | None,
    source_port_issue: str | None,
) -> dict[str, Any]:
    time_s = waveform["time_s"].to_numpy(float)
    voltage = waveform["voltage_V"].to_numpy(float)
    finite_voltage = voltage[np.isfinite(voltage)]
    mean_square = time_average(voltage**2, time_s)
    dc = time_average(voltage, time_s)
    summary: dict[str, Any] = {
        "samples": len(waveform),
        "start_s": float(time_s[0]),
        "end_s": float(time_s[-1]),
        "duration_s": float(time_s[-1] - time_s[0]),
        "waveform_voltage_peak_V": float(np.max(np.abs(finite_voltage))) if len(finite_voltage) else None,
        "waveform_voltage_rms_V": float(np.sqrt(mean_square)) if np.isfinite(mean_square) else None,
        "waveform_voltage_dc_V": dc if np.isfinite(dc) else None,
    }
    if summary["waveform_voltage_dc_V"] is None:
        summary["waveform_voltage_average_unavailable_reason"] = "voltage is missing on a positive-width interval"
    if source_port_issue is not None:
        summary["source_port_unavailable_reason"] = source_port_issue
    if frequency_hz is None:
        summary["periodic"] = {
            "status": "not_applicable",
            "reason": "no periodic fundamental was declared; basic transient statistics remain available",
        }
        return summary
    options = rf_measurement_options(case.data.get("measurement"))
    try:
        periodic = periodic_operating_metrics(
            waveform,
            frequency_hz,
            load_current_column,
            components,
            options,
            power_balance_unavailable_reason=power_balance_issue,
            source_port_unavailable_reason=source_port_issue,
        )
    except UnsettledMeasurementError as exc:
        summary["periodic"] = {"status": "unsettled", "reason": str(exc)}
        return summary
    except MeasurementUnavailableError as exc:
        summary["periodic"] = {"status": "not_available", "reason": str(exc)}
        return summary

    if periodic:
        summary["periodic"] = {"status": "ok", **periodic}
    else:
        summary["periodic"] = {
            "status": "not_available",
            "reason": "no load-current or observed-component probes were declared",
        }
    return summary
