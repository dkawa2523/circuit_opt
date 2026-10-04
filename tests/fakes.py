"""Test-only solver used to exercise orchestration without claiming physics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pcd.case import Case, load_case
from pcd.records import artifact_path, read_sim_record
from pcd.sim_registry import register_solver
from pcd.simulation import SimulationResult
from pcd.simulation_input import SolverRunRequest
from pcd.spice import param_ref_or_value


def fake_waveform(case: Case, params: dict[str, Any]) -> pd.DataFrame:
    """Return deterministic synthetic data for software-path tests only."""

    tran = case.data.get("solver", {}).get("tran", {}) or {}
    stop = float(tran.get("stop_s", 2e-6))
    step = float(tran.get("step_s", stop / 500))
    sample_count = max(16, min(50000, int(np.ceil(stop / step)) + 1))
    time_s = np.linspace(0.0, stop, sample_count)
    source = case.data.get("source", {}) or {}
    source_voltage = _source_waveform(str(source.get("type", "sine_voltage")), source, params, time_s, stop, step)

    numeric = [
        float(value)
        for value in params.values()
        if isinstance(value, (int, float, np.number)) and abs(float(value)) > 0
    ]
    log_sum = sum(np.tanh(np.log10(abs(value) + 1e-300) / 12.0) for value in numeric) if numeric else 0.0
    voltage = (0.65 + 0.12 * np.tanh(log_sum)) * source_voltage
    current = np.gradient(voltage, time_s, edge_order=1) * 1e-10 if len(time_s) > 2 else np.zeros_like(time_s)
    return pd.DataFrame({"time_s": time_s, "voltage_V": voltage, "current_A": current})


def _source_waveform(
    source_type: str,
    source: dict[str, Any],
    params: dict[str, Any],
    time_s: np.ndarray,
    stop: float,
    step: float,
) -> np.ndarray:
    def value(*keys: str, default: Any) -> float:
        for key in keys:
            if key in source:
                return float(param_ref_or_value(source[key], params, default))
        return float(default)

    if source_type in {"voltage_pulse", "pulse"}:
        low = value("v1_V", default=0.0)
        high = value("v2_V", default=1.0)
        delay = value("delay_s", default=0.0)
        width = value("width_s", default=stop / 4)
        period = value("period_s", default=max(stop / 2, step))
        phase_time = np.mod(np.maximum(time_s - delay, 0.0), period)
        return np.where((time_s >= delay) & (phase_time < width), high, low)
    if source_type in {"dc_voltage", "voltage_dc", "dc"}:
        return np.full_like(time_s, value("voltage_V", "value", default=0.0))
    amplitude = value("amplitude_V", "amplitude", default=1.0)
    frequency = value("frequency_Hz", "frequency", default=1e6)
    phase = np.deg2rad(value("phase_deg", default=0.0))
    return value("dc_V", default=0.0) + amplitude * np.sin(2 * np.pi * frequency * time_s + phase)


@register_solver("test_fake")
def fake_solver(request: SolverRunRequest) -> SimulationResult:
    """Registry adapter for the deterministic test waveform."""

    record = read_sim_record(request.run_dir)
    case_path = artifact_path(record, "case")
    if case_path is None:
        raise ValueError("prepared simulation does not declare its archived case")
    case = load_case(case_path)
    params = dict(record.get("params") or {})
    waveform = fake_waveform(case, params)
    status = "failed" if request.simulation.analysis.ac is not None else "ok"
    return SimulationResult(
        time_s=waveform["time_s"].to_numpy(float),
        voltage_V=waveform["voltage_V"].to_numpy(float),
        current_A=waveform["current_A"].to_numpy(float),
        status=status,
        log="test-only fake solver",
        diagnostics={"unsupported_analysis": "ac"} if status == "failed" else {},
    )
