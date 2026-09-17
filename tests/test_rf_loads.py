from __future__ import annotations

import math

import pytest

from pcd.rf_loads import (
    ccp_lumped_impedance,
    icp_effective_impedance,
    impedance_point,
    impedance_point_reactive_element,
)


def test_impedance_point_conversion_is_exact_at_its_anchor_frequency():
    frequency = 13.56e6
    for reactance in (-80.0, 35.0, 0.0):
        element = impedance_point_reactive_element(reactance, frequency)
        if element is None:
            recovered = 0.0
        elif element[0] == "L":
            recovered = 2 * math.pi * frequency * element[1]
        else:
            recovered = -1 / (2 * math.pi * frequency * element[1])
        assert recovered == pytest.approx(reactance)


def test_ccp_lumped_matches_the_declared_series_equation():
    frequency, resistance, inductance, capacitance = 13.56e6, 25.0, 2e-7, 1.2e-10
    expected_x = 2 * math.pi * frequency * inductance - 1 / (2 * math.pi * frequency * capacitance)
    assert ccp_lumped_impedance(frequency, resistance, inductance, capacitance) == pytest.approx(
        complex(resistance, expected_x)
    )


def test_icp_zero_coupling_reduces_to_the_coil_and_remains_passive():
    frequency = 13.56e6
    uncoupled = icp_effective_impedance(frequency, 0.4, 2e-6, 0.0, 2.0 / 3e-7)
    coupled = icp_effective_impedance(frequency, 0.4, 2e-6, 0.35**2 * 2e-6, 2.0 / 3e-7)
    assert uncoupled == pytest.approx(complex(0.4, 2 * math.pi * frequency * 2e-6))
    assert coupled.real > uncoupled.real >= 0


def test_icp_parallel_capacitance_is_applied_at_the_port():
    frequency = 13.56e6
    reflected = 0.35**2 * 2e-6
    damping = 2.0 / 3e-7
    series = icp_effective_impedance(frequency, 0.4, 2e-6, reflected, damping)
    capacitance = 2e-11
    expected = 1 / (1 / series + 1j * 2 * math.pi * frequency * capacitance)
    assert icp_effective_impedance(frequency, 0.4, 2e-6, reflected, damping, capacitance) == pytest.approx(expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda: impedance_point(-1, 0),
        lambda: impedance_point_reactive_element(-10, 0),
        lambda: ccp_lumped_impedance(1e6, 1, 0, 1e-9),
        lambda: ccp_lumped_impedance(1e6, -1, 1e-6, 1e-9),
        lambda: icp_effective_impedance(1e6, -1, 1e-6, 0.5e-6, 1e6),
        lambda: icp_effective_impedance(1e6, 1, 1e-6, 2e-6, 1e6),
        lambda: icp_effective_impedance(1e6, 1, 1e-6, 0.5e-6, 0),
        lambda: impedance_point(float("nan"), 0),
    ],
)
def test_non_passive_or_undefined_rf_parameters_are_rejected(call):
    with pytest.raises(ValueError, match="must be"):
        call()
