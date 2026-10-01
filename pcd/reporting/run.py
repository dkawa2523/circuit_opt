"""One saved simulation run to a concise electrical summary and figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd.analysis.ac import DEFAULT_Z0, ac_power_flow, at_frequency, frequency_sweep_metrics, input_impedance
from pcd.analysis.stress import ac_component_metrics, component_loss_balance, transient_component_metrics
from pcd.analysis.transient import rf_measurement_options, rf_port_metrics
from pcd.artifacts import write_json
from pcd.case import Case, load_case
from pcd.component_models import observed_components
from pcd.core.models import UnsettledMeasurementError
from pcd.probes import ComponentObservation
from pcd.records import artifact_path, frequency_response_path, load_frequency_response, load_waveform, read_sim_record
from pcd.signals import time_average
from pcd.simulation import AC_LOAD_VOLTAGE_COLUMN
from pcd.simulation_input import resolve_simulation_case
from pcd.spice import fundamental_hz

from .figures import render_frequency_response, render_operating_point, render_transient_response
from .harmonics import render_harmonic_spectrum

SUMMARY_FILE = "summary.json"
FREQUENCY_FIGURE = "frequency_response.png"
TRANSIENT_FIGURE = "transient_response.png"
OPERATING_FIGURE = "operating_point.png"
HARMONIC_FIGURE = "harmonic_spectrum.png"


def analyze_run(
    record_or_path: dict[str, Any] | str | Path,
    *,
    out_dir: str | Path | None = None,
    frequency_hz: float | None = None,
) -> dict[str, Any]:
    """Regenerate standard electrical outputs from one successful saved run."""

    record = read_sim_record(record_or_path)
    if record.get("status") != "ok":
        raise ValueError(f"cannot analyze simulation record with status {record.get('status')!r}")
    case = _archived_case(record)
    params = dict(record.get("params") or {})
    simulation = resolve_simulation_case(case, params)
    components = observed_components(case, params)
    marker_hz = _measurement_frequency(case, params, frequency_hz)
    periodic_hz = _periodic_frequency(case, params, frequency_hz)
    z0 = _reference_impedance(case)
    waveform = load_waveform(record)
    ac = _frequency_response(record, simulation.probes.ac_columns)
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

    destination = Path(out_dir) if out_dir is not None else Path(record["run_dir"]) / "analysis"
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    title = str(record.get("case_id", case.case_id))
    artifacts = _render_artifacts(
        destination,
        title,
        ac,
        waveform,
        marker_hz,
        z0,
        simulation.probes.load_current_column,
        operating_metrics,
        components,
        analyses,
    )

    summary = {
        "schema": "run_analysis.v1",
        "case_id": title,
        "simulation": {
            "status": "ok",
            "solver": record.get("solver"),
            "solver_version": ((record.get("provenance") or {}).get("solver") or {}).get("version"),
            "run_dir": str(Path(record["run_dir"]).resolve()),
        },
        "measurement": {
            "reference_plane": simulation.measurement.reference_plane,
            "reference_impedance_ohm": z0,
            "frequency_Hz": marker_hz if ac is not None and not ac.empty else periodic_hz,
        },
        "analyses": analyses,
        "scope": "electrical measurements only; engineering acceptance is evaluated by a study",
        "output_dir": str(destination),
        "artifacts": artifacts,
    }
    write_json(destination / SUMMARY_FILE, summary)
    return summary


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
) -> dict[str, str]:
    artifacts: dict[str, str] = {"summary": SUMMARY_FILE}
    if ac is not None and not ac.empty:
        sweep = operating_metrics.get("sweep")
        render_frequency_response(
            ac,
            destination / FREQUENCY_FIGURE,
            title=title,
            z0=z0,
            marker_hz=marker_hz,
            sweep_metrics=sweep if isinstance(sweep, dict) else None,
        )
        artifacts["frequency_response"] = FREQUENCY_FIGURE
    if not waveform.empty:
        render_transient_response(
            waveform,
            destination / TRANSIENT_FIGURE,
            title=title,
            load_current_column=load_current_column,
        )
        artifacts["transient_response"] = TRANSIENT_FIGURE
    operating_path = render_operating_point(
        operating_metrics,
        components,
        destination / OPERATING_FIGURE,
        title=title,
    )
    if operating_path is not None:
        artifacts["operating_point"] = OPERATING_FIGURE
    harmonic_path = render_harmonic_spectrum(_periodic_metrics(analyses), destination / HARMONIC_FIGURE, title=title)
    if harmonic_path is not None:
        artifacts["harmonic_spectrum"] = HARMONIC_FIGURE
    return artifacts


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
    frequency = fundamental_hz(case, params) if requested is None else requested
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("analysis frequency must be positive and finite")
    return frequency


def _periodic_frequency(case: Case, params: dict[str, Any], requested: float | None) -> float | None:
    """Resolve a declared periodic fundamental without inventing one for DC transients."""

    if requested is not None:
        return _measurement_frequency(case, params, requested)
    target = case.data.get("target") or {}
    source = case.data.get("source") or {}
    if not isinstance(target, dict) or not isinstance(source, dict):
        return None
    if "fundamental_Hz" not in target and "frequency_Hz" not in source:
        return None
    return _measurement_frequency(case, params, None)


def _reference_impedance(case: Case) -> float:
    measurement = case.data.get("measurement", {}) or {}
    value = float(measurement.get("reference_impedance_ohm", DEFAULT_Z0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("measurement.reference_impedance_ohm must be positive and finite")
    return value


def _frequency_response(record: dict[str, Any], probe_columns: tuple[str, ...]) -> pd.DataFrame | None:
    if frequency_response_path(record) is None:
        return None
    extra_columns = (AC_LOAD_VOLTAGE_COLUMN, *probe_columns)
    return load_frequency_response(record, extra_columns)


def _ac_summary(
    ac: pd.DataFrame,
    frequency_hz: float,
    z0: float,
    load_current_column: str | None,
    components: tuple[ComponentObservation, ...],
) -> dict[str, Any]:
    impedance = input_impedance(ac, z0)
    row = at_frequency(impedance, frequency_hz)
    resistance = float(row["resistance_ohm"])
    reactance = float(row["reactance_ohm"])
    reflection_db = _finite_number(row["reflection_db"])
    metrics: dict[str, Any] = {
        "frequency_Hz": float(row["frequency_Hz"]),
        "resistance_ohm": resistance,
        "reactance_ohm": reactance,
        "magnitude_ohm": float(row["magnitude_ohm"]),
        "phase_deg": float(np.degrees(np.angle(complex(resistance, reactance)))),
        "reflection_magnitude": float(row["reflection_magnitude"]),
        "reflected_power_fraction": float(row["reflection_magnitude"]) ** 2,
        "return_loss_db": None if reflection_db is None else -reflection_db,
        "vswr": _finite_number(row["vswr"]),
    }
    metrics.update(ac_power_flow(row, load_current_column))
    metrics.update(ac_component_metrics(row, components))
    metrics.update(component_loss_balance(metrics))
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
    periodic: dict[str, Any] = {}
    try:
        if load_current_column:
            periodic.update(
                rf_port_metrics(
                    waveform,
                    frequency_hz,
                    load_current_column,
                    periodic_cycles=int(options["periodic_cycles"]),
                    settling_comparisons=int(options["settling_comparisons"]),
                    settling_tolerance=float(options["settling_tolerance"]),
                    harmonic_count=int(options["harmonic_count"]),
                )
            )
        if components:
            periodic.update(
                transient_component_metrics(
                    waveform,
                    frequency_hz,
                    components,
                    periodic_cycles=int(options["periodic_cycles"]),
                    settling_comparisons=int(options["settling_comparisons"]),
                    settling_tolerance=float(options["settling_tolerance"]),
                )
            )
        periodic.update(component_loss_balance(periodic))
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


def _finite_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None
