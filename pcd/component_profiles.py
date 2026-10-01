"""External time profiles for explicitly declared circuit components.

Only prescribed resistance is supported.  A time-varying capacitor or
inductor needs a charge/flux constitutive law and an energy convention; it is
not equivalent to replacing a scalar C or L value at every time step.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from .case import Case, resolve_path
from .simulation_input import resolve_analysis_request

_PROFILE_FIELDS = {
    "profile",
    "time_column",
    "value_column",
    "interpolation",
    "repeat_period_s",
}


@dataclass(frozen=True)
class ResistanceProfile:
    """Validated, prescribed resistance samples in SI units."""

    time_s: tuple[float, ...]
    resistance_ohm: tuple[float, ...]
    interpolation: Literal["linear", "hold"]
    repeat_period_s: float | None = None

    @property
    def minimum_interval_s(self) -> float:
        return min(current - previous for previous, current in pairwise(self.time_s))

    def spice_expression(self) -> str:
        """Return a bounded ngspice behavioral-resistor expression."""

        phase = "time"
        if self.repeat_period_s is not None:
            period = _number(self.repeat_period_s)
            phase = f"(time-floor(time/{period})*{period})"

        if self.interpolation == "hold":
            expression = _number(self.resistance_ohm[-1])
            for threshold, previous in reversed(tuple(zip(self.time_s[1:], self.resistance_ohm[:-1], strict=True))):
                expression = f"{phase} < {_number(threshold)} ? {_number(previous)} : {expression}"
            return expression

        pairs = ", ".join(
            f"{_number(time_s)}, {_number(resistance)}"
            for time_s, resistance in zip(self.time_s, self.resistance_ohm, strict=True)
        )
        interpolated = f"pwl({phase}, {pairs})"
        if self.repeat_period_s is not None:
            return interpolated
        # ngspice's B-source PWL extrapolates the final slope.  Clamp it so a
        # completed profile cannot drift through zero after its final sample.
        return f"time < {_number(self.time_s[-1])} ? {interpolated} : {_number(self.resistance_ohm[-1])}"


def profiled_resistor_line(
    case: Case,
    params: Mapping[str, Any],
    reference: str,
    n1: str,
    n2: str,
    value: Any,
) -> str | None:
    """Render a profiled resistor, or return ``None`` for a scalar value."""

    if not isinstance(value, Mapping):
        return None
    if "profile" not in value:
        raise ValueError(f"component {reference} value mapping must contain profile")
    if not reference.upper().startswith("R"):
        raise ValueError(
            f"component {reference} uses a time profile, but only resistance R(t) is supported; "
            "C(t) and L(t) require explicit charge/flux and energy semantics"
        )

    profile = load_resistance_profile(case, value, reference)
    analysis = resolve_analysis_request(case, params)
    if analysis.transient is None:
        raise ValueError(f"component {reference} resistance profile requires solver.tran")
    if analysis.ac is not None:
        raise ValueError(
            f"component {reference} resistance profile is transient-only; use a scalar resistance for AC analysis"
        )
    if analysis.transient.step_s > profile.minimum_interval_s:
        raise ValueError(
            f"solver.tran.step_s ({analysis.transient.step_s:g} s) must not exceed component {reference} "
            f"profile minimum interval ({profile.minimum_interval_s:g} s)"
        )
    return f"{reference} {n1} {n2} R = '{profile.spice_expression()}'"


def load_resistance_profile(
    case: Case,
    config: Mapping[str, Any],
    reference: str = "resistor",
) -> ResistanceProfile:
    """Load and validate one CSV-backed resistance profile."""

    unknown = sorted(set(config) - _PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"component {reference} profile has unknown fields: {unknown}")

    declared = config.get("profile")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"component {reference} profile must be a non-empty path string")
    time_column = _column_name(config.get("time_column", "time_s"), reference, "time_column")
    value_column = _column_name(config.get("value_column", "resistance_ohm"), reference, "value_column")
    if time_column == value_column:
        raise ValueError(f"component {reference} profile time_column and value_column must differ")

    interpolation = str(config.get("interpolation", "linear")).strip().lower()
    if interpolation not in {"linear", "hold"}:
        raise ValueError(f"component {reference} profile interpolation must be linear or hold")
    interpolation_value: Literal["linear", "hold"] = "linear" if interpolation == "linear" else "hold"

    path = resolve_path(case, declared)
    times, values = _read_profile_csv(path, time_column, value_column, reference)
    repeat_period = _repeat_period(config.get("repeat_period_s"), reference)
    _validate_profile_values(times, values, repeat_period, reference)
    return ResistanceProfile(
        time_s=tuple(times),
        resistance_ohm=tuple(values),
        interpolation=interpolation_value,
        repeat_period_s=repeat_period,
    )


def _column_name(value: Any, reference: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"component {reference} profile {field} must be a non-empty string")
    return value.strip()


def _read_profile_csv(
    path: Path,
    time_column: str,
    value_column: str,
    reference: str,
) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise ValueError(f"component {reference} profile file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"component {reference} profile CSV has no header: {path}")
        if len(fields) != len(set(fields)):
            raise ValueError(f"component {reference} profile CSV has duplicate column names: {path}")
        missing = [name for name in (time_column, value_column) if name not in fields]
        if missing:
            raise ValueError(f"component {reference} profile CSV is missing columns {missing}: {path}")

        times: list[float] = []
        values: list[float] = []
        for row in reader:
            times.append(_finite_csv_number(row.get(time_column), reference, time_column, reader.line_num))
            values.append(_finite_csv_number(row.get(value_column), reference, value_column, reader.line_num))
    return times, values


def _finite_csv_number(value: Any, reference: str, column: str, line: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"component {reference} profile {column} must be numeric at CSV line {line}") from exc
    if not math.isfinite(number):
        raise ValueError(f"component {reference} profile {column} must be finite at CSV line {line}")
    return number


def _repeat_period(value: Any, reference: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"component {reference} profile repeat_period_s must be positive and finite")
    try:
        period = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"component {reference} profile repeat_period_s must be positive and finite") from exc
    if not math.isfinite(period) or period <= 0:
        raise ValueError(f"component {reference} profile repeat_period_s must be positive and finite")
    return period


def _validate_profile_values(
    times: list[float],
    values: list[float],
    repeat_period_s: float | None,
    reference: str,
) -> None:
    if len(times) < 2:
        raise ValueError(f"component {reference} profile needs at least two rows; use a scalar value for a constant")
    if times[0] != 0.0:
        raise ValueError(f"component {reference} profile must start at time 0")
    if any(current <= previous for previous, current in pairwise(times)):
        raise ValueError(f"component {reference} profile time values must be strictly increasing")
    if any(value <= 0 for value in values):
        raise ValueError(f"component {reference} profile resistance values must be positive")
    if repeat_period_s is None:
        return
    if not math.isclose(times[-1], repeat_period_s, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(
            f"component {reference} profile final time must equal repeat_period_s for an explicit cycle boundary"
        )
    if values[-1] != values[0]:
        raise ValueError(
            f"component {reference} profile final resistance must equal its first resistance when repeating"
        )


def _number(value: float) -> str:
    """Shortest decimal that round-trips to the validated binary value."""

    return repr(value)
