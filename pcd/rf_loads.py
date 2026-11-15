"""Small, explicit RF-load models for equipment-level circuit studies.

These functions describe what the circuit port sees.  They deliberately do
not infer plasma state, sheath area, density, or species-dependent power.
Supply independent electrical parameters and identify the load reference plane.
A point impedance also requires its measurement frequency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_ICP_PARALLEL_CAPACITANCE_F = 0.0
# None marks a required parameter; any numeric value is its default.
EFFECTIVE_LOAD_PARAMETERS = {
    "ccp_lumped": {"R_eff_ohm": None, "L_eff_H": None, "C_sheath_eq_F": None},
    "icp_transformer": {
        "R_coil_ohm": None,
        "L_coil_H": None,
        "reflected_inductance_H": None,
        "secondary_damping_rate_rad_s": None,
        "C_parallel_F": _ICP_PARALLEL_CAPACITANCE_F,
    },
}


def _finite(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def effective_load_parameters(model: str, values: Mapping[str, Any], *, path: str = "load") -> dict[str, float]:
    """Normalize already resolved electrical values for one built-in load."""
    defaults = EFFECTIVE_LOAD_PARAMETERS[model]
    if missing := sorted(name for name, default in defaults.items() if default is None and name not in values):
        raise ValueError(f"{path} is missing {missing} for {model}")
    parameters = {name: _finite(f"{path}.{name}", values.get(name, default)) for name, default in defaults.items()}
    if model == "ccp_lumped":
        _check_ccp_domain(parameters["R_eff_ohm"], parameters["L_eff_H"], parameters["C_sheath_eq_F"])
    else:
        _check_icp_domain(
            parameters["R_coil_ohm"],
            parameters["L_coil_H"],
            parameters["reflected_inductance_H"],
            parameters["secondary_damping_rate_rad_s"],
            parameters["C_parallel_F"],
        )
    return parameters


def impedance_point(resistance_ohm: float, reactance_ohm: float) -> complex:
    """A passive single-frequency one-port observation, ``R + jX``."""

    resistance = _finite("resistance_ohm", resistance_ohm)
    reactance = _finite("reactance_ohm", reactance_ohm)
    if resistance < 0:
        raise ValueError("resistance_ohm must be non-negative for a passive load")
    return complex(resistance, reactance)


def impedance_point_reactive_element(reactance_ohm: float, frequency_hz: float) -> tuple[str, float] | None:
    """Convert an impedance-point reactance to its simplest series L or C.

    The conversion is exact at ``frequency_hz`` only.  It is intentionally not
    advertised as a broadband material or plasma model.
    """

    reactance = _finite("reactance_ohm", reactance_ohm)
    frequency = _finite("model_frequency_Hz", frequency_hz)
    if frequency <= 0:
        raise ValueError("model_frequency_Hz must be positive")
    if abs(reactance) <= 1e-15:
        return None
    omega = 2.0 * math.pi * frequency
    if reactance > 0:
        return "L", reactance / omega
    return "C", -1.0 / (omega * reactance)


def ccp_lumped_impedance(
    frequency_hz: float,
    resistance_ohm: float,
    inductance_h: float,
    sheath_capacitance_f: float,
) -> complex:
    """Effective CCP port impedance: ``R + jwL + 1/(jwC)``."""

    frequency = _finite("frequency_hz", frequency_hz)
    resistance = _finite("R_eff_ohm", resistance_ohm)
    inductance = _finite("L_eff_H", inductance_h)
    capacitance = _finite("C_sheath_eq_F", sheath_capacitance_f)
    if frequency <= 0:
        raise ValueError("frequency_hz must be positive")
    _check_ccp_domain(resistance, inductance, capacitance)
    omega = 2.0 * math.pi * frequency
    return complex(resistance, omega * inductance - 1.0 / (omega * capacitance))


def _check_ccp_domain(resistance: float, inductance: float, capacitance: float) -> None:
    if inductance <= 0 or capacitance <= 0:
        raise ValueError("L_eff_H and C_sheath_eq_F must be positive")
    if resistance < 0:
        raise ValueError("R_eff_ohm must be non-negative for a passive load")


def icp_effective_impedance(
    frequency_hz: float,
    coil_resistance_ohm: float,
    coil_inductance_h: float,
    reflected_inductance_h: float,
    secondary_damping_rate_rad_s: float,
    parallel_capacitance_f: float = _ICP_PARALLEL_CAPACITANCE_F,
) -> complex:
    """Identifiable effective ICP coil-port impedance.

    ``reflected_inductance_H`` and ``secondary_damping_rate_rad_s`` are
    terminal-model parameters.  The damping rate must not be interpreted as a
    collision frequency without independent plasma evidence.  The optional
    ideal shunt capacitance contributes susceptance, not plasma heating.
    """

    frequency = _finite("frequency_hz", frequency_hz)
    rc = _finite("R_coil_ohm", coil_resistance_ohm)
    lc = _finite("L_coil_H", coil_inductance_h)
    reflected = _finite("reflected_inductance_H", reflected_inductance_h)
    damping = _finite("secondary_damping_rate_rad_s", secondary_damping_rate_rad_s)
    cp = _finite("C_parallel_F", parallel_capacitance_f)
    if frequency <= 0:
        raise ValueError("frequency_hz must be positive")
    _check_icp_domain(rc, lc, reflected, damping, cp)

    omega = 2.0 * math.pi * frequency
    series = complex(rc, omega * lc) + omega**2 * reflected / complex(damping, omega)
    if cp == 0:
        return series
    return 1.0 / (1.0 / series + 1j * omega * cp)


def _check_icp_domain(rc: float, lc: float, reflected: float, damping: float, cp: float) -> None:
    if lc <= 0 or damping <= 0:
        raise ValueError("L_coil_H and secondary_damping_rate_rad_s must be positive")
    if rc < 0 or reflected < 0 or cp < 0:
        raise ValueError("R_coil_ohm, reflected_inductance_H, and C_parallel_F must be non-negative")
    if reflected > lc:
        raise ValueError("reflected_inductance_H must be less than or equal to L_coil_H")
