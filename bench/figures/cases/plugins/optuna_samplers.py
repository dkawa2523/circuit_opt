"""Benchmark-local Optuna adapters for the continuous etch-CCP inverse case.

The core PCD optimizer registry remains dependency-free. This plugin translates
PCD's ask/tell boundary to a small, pinned set of Optuna and OptunaHub samplers
used only by the reproducible benchmark.
"""

from __future__ import annotations

from typing import Any, ClassVar

import optuna
import optunahub

from pcd.api import BaseOptimizer, register_optimizer
from pcd.case import default_params, variable_specs

OPTUNAHUB_REGISTRY_REF = "61da9ce3093a6c92da998a127b55720f9d3b8fd7"


class _OptunaOptimizer(BaseOptimizer):
    sampler_label = "optuna"
    sampler_settings: ClassVar[dict[str, Any]] = {}

    def __init__(self, case: Any, seed: int | None = None) -> None:
        super().__init__(case=case, seed=seed)
        configured_seed = (case.data.get("optimizer") or {}).get("seed", 0)
        self.effective_seed = int(seed if seed is not None else configured_seed)
        self.specs = variable_specs(case)
        self.fixed = default_params(case)
        self.study = optuna.create_study(direction="minimize", sampler=self._make_sampler())
        self._pending: optuna.Trial | None = None
        self._last_metadata: dict[str, Any] = {}

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        raise NotImplementedError

    def ask(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Optuna optimizer requires tell() before the next ask()")
        trial = self.study.ask()
        values = dict(self.fixed)
        for name, spec in self.specs.items():
            values[name] = self._suggest(trial, name, spec)
        self._pending = trial
        self._last_metadata = {
            "framework": "optuna",
            "sampler": self.sampler_label,
            "sampler_settings": dict(self.sampler_settings),
            "optuna_trial": trial.number,
        }
        return values

    @staticmethod
    def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
        if "choices" in spec:
            return trial.suggest_categorical(name, list(spec["choices"]))
        if spec.get("type") == "bool":
            return trial.suggest_categorical(name, [False, True])
        low, high = spec["bounds"]
        if spec.get("type") == "int":
            return trial.suggest_int(name, int(low), int(high), log=spec.get("scale") == "log")
        return trial.suggest_float(name, float(low), float(high), log=spec.get("scale") == "log")

    def tell(self, params: dict[str, Any], feedback: dict[str, Any]) -> None:
        if self._pending is None:
            raise RuntimeError("Optuna optimizer received tell() without a pending trial")
        loss = float(feedback.get("loss", 1e30))
        self.study.tell(self._pending, loss)
        frozen = self.study.trials[self._pending.number]
        self._last_metadata["backend_sampler"] = str(
            frozen.system_attrs.get("auto:sampler", type(self.study.sampler).__name__)
        )
        self._pending = None
        super().tell(params, feedback)

    def proposal_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "framework": "optuna",
            "sampler": self.sampler_label,
            "seed": self.effective_seed,
            "n_trials": len(self.study.trials),
        }


class OptunaAutoOptimizer(_OptunaOptimizer):
    sampler_label = "OptunaHub AutoSampler"

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        module = optunahub.load_module(
            package="samplers/auto_sampler",
            repo_owner="optuna",
            ref=OPTUNAHUB_REGISTRY_REF,
        )
        return module.AutoSampler(seed=self.effective_seed)


class OptunaTPEOptimizer(_OptunaOptimizer):
    sampler_label = "Optuna TPESampler"

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        return optuna.samplers.TPESampler(seed=self.effective_seed)


class OptunaGPUCBOptimizer(_OptunaOptimizer):
    """Deterministic GP surrogate with an explicit exploration bonus."""

    sampler_label = "OptunaHub GP-UCB"
    sampler_settings: ClassVar[dict[str, Any]] = {
        "beta": 2.0,
        "deterministic_objective": True,
        "n_startup_trials": 10,
    }

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        module = optunahub.load_module(
            package="samplers/gp_acqf_samplers",
            repo_owner="optuna",
            ref=OPTUNAHUB_REGISTRY_REF,
        )
        return module.GPUCBSampler(
            beta=2.0,
            seed=self.effective_seed,
            n_startup_trials=10,
            deterministic_objective=True,
        )


class OptunaMESOptimizer(_OptunaOptimizer):
    """GP maximum-value entropy search using posterior maximum samples."""

    sampler_label = "OptunaHub MES"
    sampler_settings: ClassVar[dict[str, Any]] = {
        "max_value_sampler": "posterior",
        "n_max_value_samples": 32,
        "n_representer_points": 512,
        "deterministic_objective": True,
        "n_startup_trials": 10,
    }

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        module = optunahub.load_module(
            package="samplers/gp_mes",
            repo_owner="optuna",
            ref=OPTUNAHUB_REGISTRY_REF,
        )
        return module.MESSampler(
            max_value_sampler="posterior",
            n_max_value_samples=32,
            n_representer_points=512,
            seed=self.effective_seed,
            n_startup_trials=10,
            deterministic_objective=True,
        )


class OptunaJanusOptimizer(_OptunaOptimizer):
    """JANUS with rank-scaled scalar feedback for this waveform inverse case."""

    sampler_label = "OptunaHub JANUS"
    sampler_settings: ClassVar[dict[str, Any]] = {"n_startup_trials": 16, "pseudo_target": "rank"}

    def _make_sampler(self) -> optuna.samplers.BaseSampler:
        module = optunahub.load_module(
            package="samplers/janus",
            repo_owner="optuna",
            ref=OPTUNAHUB_REGISTRY_REF,
        )
        return module.JanusSampler(
            seed=self.effective_seed,
            n_startup_trials=16,
            pseudo_target="rank",
        )


@register_optimizer("optuna_auto")
def optuna_auto_optimizer(case: Any, seed: int | None = None) -> BaseOptimizer:
    return OptunaAutoOptimizer(case, seed=seed)


@register_optimizer("optuna_tpe")
def optuna_tpe_optimizer(case: Any, seed: int | None = None) -> BaseOptimizer:
    return OptunaTPEOptimizer(case, seed=seed)


@register_optimizer("optuna_gp_ucb")
def optuna_gp_ucb_optimizer(case: Any, seed: int | None = None) -> BaseOptimizer:
    return OptunaGPUCBOptimizer(case, seed=seed)


@register_optimizer("optuna_gp_mes")
def optuna_gp_mes_optimizer(case: Any, seed: int | None = None) -> BaseOptimizer:
    return OptunaMESOptimizer(case, seed=seed)


@register_optimizer("optuna_janus")
def optuna_janus_optimizer(case: Any, seed: int | None = None) -> BaseOptimizer:
    return OptunaJanusOptimizer(case, seed=seed)
