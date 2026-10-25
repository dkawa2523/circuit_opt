"""Value resolution and formatting shared by SPICE-producing modules."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .case import Case

_PLAIN_NAME = re.compile(r"^[A-Za-z_]\w*$")
_SPICE_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*$")
_SPECIAL_PARAM_NAMES = {
    "topology",
    "circuit_builder",
    "builder",
    "load_model",
    "load_name",
    "name_variable",
    "solver",
    "objective",
    "optimizer",
}


def is_special_param(name: str) -> bool:
    return name in _SPECIAL_PARAM_NAMES or name.endswith("_choice") or name.endswith("_name")


def is_spice_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        text = value.strip()
        return not text.startswith("$") and bool(_SPICE_NUMBER.match(text))
    return False


def should_emit_spice_param(name: str, value: Any) -> bool:
    return not is_special_param(str(name)) and is_spice_literal(value)


def resolve_value(value: Any, params: dict[str, Any], default: Any | None = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        key = value[1:] if value.startswith("$") else value
        if key in params:
            return params[key]
    return value


param_ref_or_value = resolve_value


def periodic_frequency_hz(case: Case, params: dict[str, Any] | None = None) -> float | None:
    """Declared periodic fundamental; never invent 1 Hz for an unknown source."""

    from .sources import selected_source

    target = case.data.get("target") or {}
    _name, source = selected_source(case)
    source = source or {}
    raw = target.get("fundamental_Hz", source.get("frequency_Hz", source.get("frequency")))
    if raw is None and source.get("type") in {"pulse", "voltage_pulse"}:
        period = resolve_value(source.get("period_s", 2e-6), params or {})
        if isinstance(period, bool) or not math.isfinite(float(period)) or float(period) <= 0:
            raise ValueError("source.period_s must be positive and finite")
        return 1.0 / float(period)
    if raw is None:
        return None
    value = resolve_value(raw, params or {})
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target.fundamental_Hz / source.frequency_Hz must resolve to a number, got {value!r}"
        ) from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError("target.fundamental_Hz / source.frequency_Hz must be positive and finite")
    return number


def fundamental_hz(case: Case, params: dict[str, Any] | None = None) -> float:
    frequency = periodic_frequency_hz(case, params)
    if frequency is None:
        raise ValueError("no periodic fundamental declared; set target.fundamental_Hz or the observed source frequency")
    return frequency


def ac_operating_frequency_hz(case: Case, params: dict[str, Any]) -> float:
    """The declared target point, or the explicit single-point AC request."""

    if "fundamental_Hz" in (case.data.get("target") or {}):
        return fundamental_hz(case, params)
    ac = (case.data.get("solver") or {}).get("ac") or {}
    if "frequency_Hz" in ac:
        value = float(resolve_value(ac["frequency_Hz"], params))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("solver.ac.frequency_Hz must be positive and finite")
        return value
    if ac.get("start_Hz") == ac.get("stop_Hz") and "start_Hz" in ac:
        return float(resolve_value(ac["start_Hz"], params))
    return fundamental_hz(case, params)


def pick_value(cfg: dict[str, Any], key: str, params: dict[str, Any], default: Any) -> Any:
    return resolve_value(cfg.get(key, default), params, default)


def spice_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("SPICE numeric values must be finite")
        text = repr(number)
        return text[:-2] if text.endswith(".0") else text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        raise ValueError("SPICE value cannot be None")
    return _render_text_value(str(value).strip())


def _render_text_value(text: str) -> str:
    if not text:
        return text
    if text.startswith("$") and _PLAIN_NAME.match(text[1:]):
        return "{" + text[1:] + "}"
    if text.startswith("{") and text.endswith("}"):
        return text
    if _SPICE_NUMBER.match(text):
        return text
    if _PLAIN_NAME.match(text):
        return "{" + text + "}"
    return text
