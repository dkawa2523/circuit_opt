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


def fundamental_hz(case: Case, params: dict[str, Any] | None = None) -> float:
    raw = (case.data.get("target", {}) or {}).get(
        "fundamental_Hz", (case.data.get("source", {}) or {}).get("frequency_Hz", 1.0)
    )
    value = resolve_value(raw, params or {}, 1.0)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target.fundamental_Hz / source.frequency_Hz must resolve to a number, got {value!r}"
        ) from exc


def pick_value(cfg: dict[str, Any], key: str, params: dict[str, Any], default: Any) -> Any:
    return resolve_value(cfg.get(key, default), params, default)


def spice_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
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
