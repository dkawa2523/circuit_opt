"""Immutable value objects used by every design study."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    """Copy nested containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def to_plain(value: Any) -> Any:
    """Return JSON-ready copies of frozen study values."""

    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_plain(item) for item in value]
    if isinstance(value, frozenset | set):
        return [to_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze(dict(value))


def _require_finite_numbers(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            _require_finite_numbers(item, f"{path}.{name}")
        return
    if isinstance(value, list | tuple | set | frozenset):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite or null")


def _require_id(value: str, kind: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{kind} id must not be empty")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A manufactured or otherwise fixed design choice."""

    candidate_id: str
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, "candidate")
        object.__setattr__(self, "values", _frozen_mapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "values": to_plain(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Candidate:
        return cls(str(data["candidate_id"]), data.get("values", {}) or {})


@dataclass(frozen=True, slots=True)
class Scenario:
    """An exogenous operating condition that the design cannot choose."""

    scenario_id: str
    values: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_id(self.scenario_id, "scenario")
        if self.weight <= 0:
            raise ValueError("scenario weight must be positive")
        object.__setattr__(self, "values", _frozen_mapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, "values": to_plain(self.values), "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        return cls(str(data["scenario_id"]), data.get("values", {}) or {}, float(data.get("weight", 1.0)))


@dataclass(frozen=True, slots=True)
class ControlState:
    """Values an operator or controller may tune for one scenario."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_mapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"values": to_plain(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlState:
        return cls(data.get("values", {}) or {})


@dataclass(frozen=True, slots=True)
class Objective:
    """How one objective metric is compared across scenarios."""

    metric: str
    direction: str = "minimize"
    aggregation: str = "worst"
    cvar_alpha: float = 0.1

    def __post_init__(self) -> None:
        _require_id(self.metric, "objective metric")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("objective direction must be 'minimize' or 'maximize'")
        if self.aggregation not in {"mean", "worst", "cvar"}:
            raise ValueError("objective aggregation must be mean, worst, or cvar")
        if not 0 < self.cvar_alpha <= 1:
            raise ValueError("objective cvar_alpha must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "cvar_alpha": self.cvar_alpha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Objective:
        return cls(
            metric=str(data["metric"]),
            direction=str(data.get("direction", "minimize")),
            aggregation=str(data.get("aggregation", "worst")),
            cvar_alpha=float(data.get("cvar_alpha", 0.1)),
        )


@dataclass(frozen=True, slots=True)
class StudySpec:
    """The invariant plan for a design study."""

    study_id: str
    scenarios: tuple[Scenario, ...] = (Scenario("nominal"),)
    objectives: tuple[Objective, ...] = (Objective("loss"),)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    control_margin_min: float | None = None

    def __post_init__(self) -> None:
        _require_id(self.study_id, "study")
        if not self.scenarios:
            raise ValueError("a study needs at least one scenario")
        if not self.objectives:
            raise ValueError("a study needs at least one objective")
        if len({item.scenario_id for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario ids must be unique")
        if self.control_margin_min is not None:
            margin = float(self.control_margin_min)
            if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
                raise ValueError("control_margin_min must be between 0 and 1")
            object.__setattr__(self, "control_margin_min", margin)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "objectives": [item.to_dict() for item in self.objectives],
            "metadata": to_plain(self.metadata),
            "control_margin_min": self.control_margin_min,
        }


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    candidate: Candidate
    scenario: Scenario
    control: ControlState

    def merged_inputs(self) -> dict[str, Any]:
        """Merge role-separated inputs, rejecting ambiguous duplicate names."""

        result: dict[str, Any] = {}
        owners: dict[str, str] = {}
        for owner, values in (
            ("candidate", self.candidate.values),
            ("scenario", self.scenario.values),
            ("control", self.control.values),
        ):
            for name, value in values.items():
                if name in result:
                    raise ValueError(f"input {name!r} belongs to both {owners[name]} and {owner}")
                result[name] = to_plain(value)
                owners[name] = owner
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "scenario": self.scenario.to_dict(),
            "control": self.control.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRequest:
        return cls(
            Candidate.from_dict(data["candidate"]),
            Scenario.from_dict(data["scenario"]),
            ControlState.from_dict(data["control"]),
        )


@dataclass(frozen=True, slots=True)
class RawResult:
    status: str
    observations: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "failed", "not_settled", "invalid"}:
            raise ValueError(f"unsupported raw result status: {self.status}")
        object.__setattr__(self, "observations", _frozen_mapping(self.observations))
        object.__setattr__(self, "artifacts", _frozen_mapping(self.artifacts))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observations": to_plain(self.observations),
            "artifacts": to_plain(self.artifacts),
            "diagnostics": to_plain(self.diagnostics),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RawResult:
        return cls(
            str(data["status"]),
            data.get("observations", {}) or {},
            data.get("artifacts", {}) or {},
            data.get("diagnostics", {}) or {},
            str(data["error"]) if data.get("error") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class MetricSet:
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plain = to_plain(dict(self.values))
        _require_finite_numbers(plain, "metric")
        object.__setattr__(self, "values", _frozen_mapping(plain))

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self.values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricSet:
        return cls(data)


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    name: str
    satisfied: bool
    violation: float = 0.0
    value: float | None = None
    limit: float | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.name, "constraint")
        if self.violation < 0:
            raise ValueError("constraint violation must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "violation": self.violation,
            "value": self.value,
            "limit": self.limit,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstraintResult:
        return cls(
            name=str(data["name"]),
            satisfied=bool(data["satisfied"]),
            violation=float(data.get("violation", 0.0)),
            value=float(data["value"]) if data.get("value") is not None else None,
            limit=float(data["limit"]) if data.get("limit") is not None else None,
            detail=str(data["detail"]) if data.get("detail") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    request: EvaluationRequest
    raw: RawResult
    metrics: MetricSet = MetricSet()
    constraints: tuple[ConstraintResult, ...] = ()
    cache_key: str = ""
    raw_cache_key: str = ""
    duration_s: float = 0.0
    from_cache: bool = False

    @property
    def feasible(self) -> bool:
        return self.raw.ok and all(item.satisfied for item in self.constraints)

    @property
    def total_violation(self) -> float:
        return sum(item.violation for item in self.constraints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evaluation_result.v1",
            "request": self.request.to_dict(),
            "raw": self.raw.to_dict(),
            "metrics": self.metrics.to_dict(),
            "constraints": [item.to_dict() for item in self.constraints],
            "cache_key": self.cache_key,
            "raw_cache_key": self.raw_cache_key,
            "duration_s": self.duration_s,
            "from_cache": self.from_cache,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationResult:
        return cls(
            request=EvaluationRequest.from_dict(data["request"]),
            raw=RawResult.from_dict(data["raw"]),
            metrics=MetricSet.from_dict(data.get("metrics", {}) or {}),
            constraints=tuple(ConstraintResult.from_dict(item) for item in data.get("constraints", []) or []),
            cache_key=str(data.get("cache_key", "")),
            raw_cache_key=str(data.get("raw_cache_key", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            from_cache=bool(data.get("from_cache", False)),
        )


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """All tried controls and the selected operating point for one scenario."""

    scenario: Scenario
    selected: EvaluationResult
    trials: tuple[EvaluationResult, ...]
    control_margin: float | None = None

    def __post_init__(self) -> None:
        if not self.trials:
            raise ValueError("a scenario result needs at least one control trial")
        if self.selected not in self.trials:
            raise ValueError("selected evaluation must be one of the scenario trials")

    @property
    def edge_limited(self) -> bool:
        """Whether tuning headroom is the selected point's only failed limit."""

        failed = [item.name for item in self.selected.constraints if not item.satisfied]
        return (
            self.control_margin is not None
            and self.selected.raw.ok
            and bool(failed)
            and set(failed) == {"min_control_margin"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "selected_cache_key": self.selected.cache_key,
            "control_margin": self.control_margin,
            "edge_limited": self.edge_limited,
            "selected": self.selected.to_dict(),
            "trials": [item.to_dict() for item in self.trials],
        }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate: Candidate
    scenarios: tuple[ScenarioResult, ...]
    aggregates: Mapping[str, float | None]
    feasible_fraction: float
    success_fraction: float
    total_violation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregates", _frozen_mapping(self.aggregates))

    @property
    def evaluations(self) -> tuple[EvaluationResult, ...]:
        """Selected operating point for each scenario."""

        return tuple(item.selected for item in self.scenarios)

    @property
    def control_evaluations(self) -> tuple[EvaluationResult, ...]:
        return tuple(evaluation for scenario in self.scenarios for evaluation in scenario.trials)

    @property
    def control_margin(self) -> float | None:
        """Worst selected control margin across scenarios, or null when not numeric."""

        margins = [item.control_margin for item in self.scenarios]
        numeric = [item for item in margins if item is not None]
        return min(numeric) if numeric else None

    @property
    def edge_limited(self) -> bool:
        """Whether margin-only scenarios are the sole barrier to full feasibility."""

        return any(item.edge_limited for item in self.scenarios) and all(
            item.selected.feasible or item.edge_limited for item in self.scenarios
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "candidate_result.v1",
            "candidate": self.candidate.to_dict(),
            "aggregates": to_plain(self.aggregates),
            "feasible_fraction": self.feasible_fraction,
            "success_fraction": self.success_fraction,
            "total_violation": self.total_violation,
            "control_margin": self.control_margin,
            "edge_limited": self.edge_limited,
            "scenarios": [item.to_dict() for item in self.scenarios],
        }
