from __future__ import annotations

from typing import Any

import numpy as np

from .common import Case, default_params, sample_param, variable_specs
from .ml_core import BaseOptimizer, constraint_penalty, interpolate_to_target, load_target_waveform
from .ml_registry import register


@register("objective", "waveform_l2")
def objective_waveform_l2(case: Case, record: dict[str, Any], waveform) -> dict[str, Any]:
    target = load_target_waveform(case)
    _t, vt, v = interpolate_to_target(target, waveform)
    if np.isnan(v).all():
        return {"loss": 1e30, "status": "failed", "reason": "empty waveform"}
    denom = float(np.sqrt(np.mean(vt**2)) + 1e-12)
    rmse = float(np.sqrt(np.mean((v - vt) ** 2)))
    norm = rmse / denom
    penalty, info = constraint_penalty(case, v)
    return {
        "loss": float(norm + penalty),
        "normalized_rmse": float(norm),
        "rmse_V": rmse,
        "constraint_penalty": float(penalty),
        "objective": "waveform_l2",
        **info,
    }


@register("objective", "waveform_l2_harmonics")
def objective_waveform_l2_harmonics(case: Case, record: dict[str, Any], waveform) -> dict[str, Any]:
    base = objective_waveform_l2(case, record, waveform)
    if base.get("status") == "failed":
        return base
    target = load_target_waveform(case)
    t, vt, v = interpolate_to_target(target, waveform)
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 1.0
    freq = np.fft.rfftfreq(len(t), d=dt)
    V = np.fft.rfft(v)
    T = np.fft.rfft(vt)
    cfg = case.data.get("target", {}) or {}
    f0 = float(cfg.get("fundamental_Hz", (case.data.get("source", {}) or {}).get("frequency_Hz", 1.0)))
    harmonics = cfg.get("harmonics", [1]) or []
    err = 0.0
    for h in harmonics:
        idx = int(np.argmin(np.abs(freq - float(h) * f0)))
        err += float(abs(V[idx] - T[idx]) / (abs(T[idx]) + 1e-12))
    harmonic_error = err / max(len(harmonics), 1)
    weight = float(cfg.get("harmonic_weight", 0.2))
    base["harmonic_error"] = float(harmonic_error)
    base["loss"] = float(base["loss"] + weight * harmonic_error)
    base["objective"] = "waveform_l2_harmonics"
    return base


class RandomOptimizer(BaseOptimizer):
    def __init__(self, case: Case, seed: int | None = None):
        super().__init__(case=case, seed=seed)
        cfg_seed = (case.data.get("optimizer", {}) or {}).get("seed", 0)
        self.rng = np.random.default_rng(seed if seed is not None else cfg_seed)
        self.specs = variable_specs(case)

    def ask(self) -> dict[str, Any]:
        params = default_params(self.case)
        for name, spec in self.specs.items():
            params[name] = sample_param(self.rng, spec)
        return params


@register("optimizer", "random")
def optimizer_random(case: Case, seed: int | None = None) -> BaseOptimizer:
    return RandomOptimizer(case, seed=seed)


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, case: Case, seed: int | None = None):
        super().__init__(case=case, seed=seed)
        try:
            import optuna
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("optuna is optional. install with: pip install -e '.[optuna]'") from exc
        cfg_seed = (case.data.get("optimizer", {}) or {}).get("seed", 0)
        sampler = optuna.samplers.TPESampler(seed=seed if seed is not None else cfg_seed)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.specs = variable_specs(case)
        self._pending: dict[int, Any] = {}

    def ask(self) -> dict[str, Any]:
        trial = self.study.ask()
        params = default_params(self.case)
        for name, spec in self.specs.items():
            if "choices" in spec:
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            else:
                lo, hi = spec.get("bounds", [0.0, 1.0])
                if spec.get("type") == "int":
                    params[name] = trial.suggest_int(name, int(lo), int(hi), log=bool(spec.get("scale") == "log"))
                else:
                    params[name] = trial.suggest_float(name, float(lo), float(hi), log=bool(spec.get("scale") == "log"))
        self._pending[id(params)] = trial
        return params

    def tell(self, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        trial = self._pending.pop(id(params), None)
        value = float(metrics.get("loss", 1e30))
        if trial is not None:
            self.study.tell(trial, value)
        super().tell(params, metrics)

    def state(self) -> dict[str, Any]:
        best = None
        try:
            best = {"value": self.study.best_value, "params": self.study.best_params}
        except Exception:
            best = None
        return {"type": "OptunaOptimizer", "n_observations": len(self.history), "best": best}


@register("optimizer", "optuna")
def optimizer_optuna(case: Case, seed: int | None = None) -> BaseOptimizer:
    return OptunaOptimizer(case, seed=seed)
