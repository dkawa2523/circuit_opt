"""Recover a linear multi-port response from observed terminal phasors.

The caller defines the device boundary and supplies independent excitations of
the same linearized circuit at common frequencies. This module checks whether
the recorded waves identify a scattering matrix at each frequency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

CurrentDirection = Literal["into_dut", "out_of_dut"]
ScatteringStatus = Literal[
    "ok",
    "unavailable_observation",
    "rank_deficient",
    "ill_conditioned",
    "inconsistent_excitations",
    "numerical_failure",
]


@dataclass(frozen=True, slots=True)
class PortReference:
    """A named observation plane with a positive real wave reference impedance."""

    name: str
    reference_impedance_ohm: float
    current_direction: CurrentDirection = "into_dut"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("port name must be nonempty")
        if not math.isfinite(self.reference_impedance_ohm) or self.reference_impedance_ohm <= 0:
            raise ValueError(f"port {self.name!r} needs a positive finite real reference impedance")
        if self.current_direction not in {"into_dut", "out_of_dut"}:
            raise ValueError(f"port {self.name!r} has an invalid current direction")


@dataclass(frozen=True, slots=True)
class PortExcitation:
    """One named state with peak V/I phasors in ``(frequency, port)`` order."""

    name: str
    frequency_hz: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray


@dataclass(frozen=True, slots=True)
class ScatteringResult:
    """A frequency-wise result, with undefined scattering entries left NaN.

    ``scattering`` uses ``(frequency, outgoing port, incident port)`` order.
    Wave arrays use ``(frequency, port, excitation)`` order and sqrt(W) units.
    """

    frequency_hz: np.ndarray
    ports: tuple[PortReference, ...]
    excitation_names: tuple[str, ...]
    scattering: np.ndarray
    incident_waves: np.ndarray
    reflected_waves: np.ndarray
    incident_rank: np.ndarray
    incident_condition: np.ndarray
    relative_residual: np.ndarray
    status: tuple[ScatteringStatus, ...]

    @property
    def complete(self) -> bool:
        return all(status == "ok" for status in self.status)


def solve_scattering(
    ports: Sequence[PortReference],
    excitations: Sequence[PortExcitation],
    *,
    condition_limit: float = 1e8,
    residual_limit: float = 1e-6,
) -> ScatteringResult:
    """Solve ``B = S A`` on the shared frequency grid, without interpolation.

    Columns of ``A`` and ``B`` correspond to the declared excitation states.
    More states than ports allow a consistency check. With exactly as many
    states as ports, the residual only measures the numerical solve.
    """
    ports = tuple(ports)
    excitations = tuple(excitations)
    port_names = tuple(port.name for port in ports)
    excitation_names = tuple(excitation.name for excitation in excitations)
    if not ports or len(set(port_names)) != len(ports):
        raise ValueError("port names must be nonempty and unique")
    if (
        not excitations
        or not all(name.strip() for name in excitation_names)
        or len(set(excitation_names)) != len(excitations)
    ):
        raise ValueError("excitation names must be nonempty and unique")
    if not math.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("condition_limit must be finite and at least 1")
    if not math.isfinite(residual_limit) or residual_limit <= 0:
        raise ValueError("residual_limit must be positive and finite")

    frequency, voltage, current = _observations(excitations, len(ports))
    root_impedance = np.sqrt([port.reference_impedance_ohm for port in ports])[None, :, None]
    polarity = np.array([1.0 if port.current_direction == "into_dut" else -1.0 for port in ports])[None, :, None]
    # Peak phasors carry average real power Re(V I*)/2. Thus the square of
    # each wave amplitude is a power and unequal port impedances are allowed.
    root_eight = math.sqrt(8)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_voltage = voltage / (root_impedance * root_eight)
        scaled_current = current * polarity * (root_impedance / root_eight)
        incident = scaled_voltage + scaled_current
        reflected = scaled_voltage - scaled_current

    count = len(frequency)
    scattering = np.full((count, len(ports), len(ports)), np.nan + 1j * np.nan)
    rank = np.full(count, np.nan)
    condition = np.full(count, np.nan)
    residual = np.full(count, np.nan)
    status: list[ScatteringStatus] = []
    for index, (a, b) in enumerate(zip(incident, reflected, strict=True)):
        matrix, rank[index], condition[index], residual[index], verdict = _frequency_solution(
            a, b, condition_limit, residual_limit
        )
        if matrix is not None:
            scattering[index] = matrix
        status.append(verdict)

    return ScatteringResult(
        frequency, ports, excitation_names, scattering, incident, reflected, rank, condition, residual, tuple(status)
    )


def _observations(excitations: tuple[PortExcitation, ...], ports: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.asarray(excitations[0].frequency_hz, dtype=float).copy()
    voltage, current = [], []
    for excitation in excitations:
        grid = np.asarray(excitation.frequency_hz, dtype=float)
        if (
            grid.ndim != 1
            or not len(grid)
            or not np.isfinite(grid).all()
            or np.any(grid <= 0)
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError(f"excitation {excitation.name!r} needs increasing positive finite frequencies")
        # Tolerate only floating-point representation roundoff, never
        # resampling or joining measurements on a neighboring frequency.
        if grid.shape != frequency.shape or not np.allclose(grid, frequency, rtol=8 * np.finfo(float).eps, atol=0):
            raise ValueError(f"excitation {excitation.name!r} has a different frequency grid")
        voltage_v = np.asarray(excitation.voltage_v, dtype=complex)
        current_a = np.asarray(excitation.current_a, dtype=complex)
        if voltage_v.shape != (len(frequency), ports) or current_a.shape != voltage_v.shape:
            raise ValueError(f"excitation {excitation.name!r} needs V/I arrays in (frequency, port) order")
        voltage.append(voltage_v)
        current.append(current_a)
    return frequency, np.stack(voltage, axis=-1), np.stack(current, axis=-1)


def _frequency_solution(
    incident: np.ndarray, reflected: np.ndarray, condition_limit: float, residual_limit: float
) -> tuple[np.ndarray | None, float, float, float, ScatteringStatus]:
    if not np.isfinite(incident).all() or not np.isfinite(reflected).all():
        return None, math.nan, math.nan, math.nan, "unavailable_observation"

    # One common scaling leaves S and rank unchanged while keeping norms and
    # the least-squares solve in range for small or large phasors.
    scale = max(np.abs(part).max() for part in (incident.real, incident.imag, reflected.real, reflected.imag))
    if scale == 0:
        return None, 0, math.inf, math.nan, "rank_deficient"
    a = incident.real / scale + 1j * (incident.imag / scale)
    b = reflected.real / scale + 1j * (reflected.imag / scale)
    try:
        operator, _, rank, singular = np.linalg.lstsq(a.T, b.T, rcond=None)
    except np.linalg.LinAlgError:
        return None, math.nan, math.nan, math.nan, "numerical_failure"
    if rank < incident.shape[0]:
        return None, float(rank), math.inf, math.nan, "rank_deficient"
    condition = float(singular[0] / singular[-1])
    if condition > condition_limit:
        return None, float(rank), condition, math.nan, "ill_conditioned"

    matrix = operator.T
    error = float(np.linalg.norm(b - matrix @ a) / max(np.linalg.norm(a), np.linalg.norm(b)))
    if not np.isfinite(matrix).all() or not math.isfinite(error):
        return None, float(rank), condition, math.nan, "numerical_failure"
    if error > residual_limit:
        return None, float(rank), condition, error, "inconsistent_excitations"
    return matrix, float(rank), condition, error, "ok"
