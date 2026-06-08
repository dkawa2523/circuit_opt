from __future__ import annotations

import numpy as np
import pandas as pd

from pcd.common import resolve_path
from pcd.ml_core import interpolate_to_target
from pcd.ml_registry import register as ml_register


@ml_register("objective", "ccp_waveform_power_proxy")
def ccp_waveform_power_proxy(case, record, waveform):
    """Compact CCP benchmark objective.

    It intentionally uses only saved waveform data, so it remains in the ML layer.
    The metrics are proxies: they are useful for circuit screening, not substitutes
    for a kinetic/fluid plasma model.
    """
    cfg = case.data.get("target", {}) or {}
    target = pd.read_csv(resolve_path(case, cfg["waveform_file"]))
    t, vt, v = interpolate_to_target(target, waveform)
    if len(t) == 0 or np.isnan(v).all():
        return {"loss": 1e30, "status": "failed", "reason": "empty waveform"}

    denom = float(np.sqrt(np.mean(vt ** 2)) + 1e-12)
    rmse = float(np.sqrt(np.mean((v - vt) ** 2)))
    normalized_rmse = rmse / denom

    dt = float(np.median(np.diff(t))) if len(t) > 2 else 1.0
    freq = np.fft.rfftfreq(len(t), d=dt)
    V = np.fft.rfft(v)
    T = np.fft.rfft(vt)
    f0 = float(cfg.get("fundamental_Hz", 13.56e6))
    harmonic_error = 0.0
    for h in cfg.get("harmonics", [1]):
        idx = int(np.argmin(np.abs(freq - float(h) * f0)))
        harmonic_error += float(abs(V[idx] - T[idx]) / (abs(T[idx]) + 1e-12))
    harmonic_error /= max(1, len(cfg.get("harmonics", [1])))

    current = waveform.get("current_A")
    if current is not None:
        ti = waveform["time_s"].to_numpy(float)
        ii = np.interp(t, ti, current.to_numpy(float))
    else:
        ii = np.zeros_like(v)
    p_avg = float(np.mean(v * ii))
    p_abs = abs(p_avg)
    p_target = float(cfg.get("power_target_W", p_abs if p_abs > 0 else 1.0))
    power_error = abs(p_abs - p_target) / max(abs(p_target), 1e-12)

    constraints = cfg.get("constraints", {}) or {}
    penalty = 0.0
    vmax = constraints.get("max_abs_voltage_V")
    if vmax is not None:
        penalty += max(0.0, float(np.nanmax(np.abs(v))) / float(vmax) - 1.0) ** 2 * 10.0
    imax = constraints.get("max_abs_current_A")
    if imax is not None and len(ii):
        penalty += max(0.0, float(np.nanmax(np.abs(ii))) / float(imax) - 1.0) ** 2 * 10.0

    loss = normalized_rmse
    loss += float(cfg.get("harmonic_weight", 0.2)) * harmonic_error
    loss += float(cfg.get("power_weight", 0.0)) * power_error
    loss += penalty

    return {
        "loss": float(loss),
        "objective": "ccp_waveform_power_proxy",
        "normalized_rmse": float(normalized_rmse),
        "rmse_V": float(rmse),
        "harmonic_error": float(harmonic_error),
        "avg_power_proxy_W": float(p_avg),
        "power_error": float(power_error),
        "dc_bias_proxy_V": float(np.mean(v)),
        "v_rms_V": float(np.sqrt(np.mean(v ** 2))),
        "v_peak_abs_V": float(np.nanmax(np.abs(v))),
        "i_rms_A": float(np.sqrt(np.mean(ii ** 2))) if len(ii) else 0.0,
        "constraint_penalty": float(penalty),
    }
