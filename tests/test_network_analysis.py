"""Numerical evidence for terminal-wave recovery independently of any solver."""

import numpy as np
import pytest

from pcd.analysis.network import PortExcitation, PortReference, solve_scattering


def _terminal_states(frequency, ports, incident, reflected):
    impedance = np.array([port.reference_impedance_ohm for port in ports])[None, :, None]
    sign = np.array([1 if port.current_direction == "into_dut" else -1 for port in ports])[None, :, None]
    voltage = (incident + reflected) * np.sqrt(2 * impedance)
    recorded_current = (incident - reflected) * np.sqrt(2 / impedance) * sign
    return tuple(
        PortExcitation(f"state-{i}", frequency, voltage[:, :, i], recorded_current[:, :, i])
        for i in range(incident.shape[2])
    )


def test_mixed_excitation_recovers_nonreciprocal_three_port_with_unequal_references():
    ports = (PortReference("in", 25), PortReference("coupled", 50, "out_of_dut"), PortReference("out", 75))
    frequency = np.array([2e6, 4e6])
    incident = np.array(
        [
            [[1, 0.2j, 0.1, 0.6], [0.3, 2, -0.1j, -0.4], [0.1, 0.2, 0.9, 1.1j]],
            [[0.4j, 0.3, 0.2, 0.3], [0.2, 1.2, 0.2, 0.5j], [0.4, 0.2, 2j, 1]],
        ]
    )
    expected = np.array(
        [
            [[0.03j, 0.1, 0.2], [0.3j, 0.2, 0.2j], [0.6, 0.3, -0.3j]],
            [[0.01, -0.2j, 0.1], [-0.3j, 0.1, -0.2j], [0.7, 0.2, 0.4j]],
        ]
    )
    states = _terminal_states(frequency, ports, incident, expected @ incident)

    result = solve_scattering(ports, states)

    assert result.complete
    assert result.status == ("ok", "ok")
    np.testing.assert_array_equal(result.frequency_hz, frequency)
    np.testing.assert_array_equal(result.incident_rank, [3, 3])
    np.testing.assert_allclose(result.scattering, expected, atol=1e-14)
    assert result.relative_residual.max() < 1e-14
    # Every wave column must also give the measured average real power into the DUT.
    wave_power = np.abs(result.incident_waves) ** 2 - np.abs(result.reflected_waves) ** 2
    terminal_power = np.stack(
        [0.5 * np.real(state.voltage_v * state.current_a.conj() * np.array([1, -1, 1])) for state in states], axis=-1
    )
    np.testing.assert_allclose(wave_power, terminal_power, atol=1e-14)


def test_frequency_wise_failure_cannot_publish_a_synthetic_matrix():
    ports = (PortReference("one", 50), PortReference("two", 50))
    frequency = np.arange(1, 6) * 1e6
    incident = np.tile(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), (5, 1, 1))
    incident[2, 1] = incident[2, 0]
    incident[3, 1] *= 1e-5
    expected = np.array([[0.1, 0.5], [-0.2j, 0.3]])
    reflected = expected @ incident
    reflected[4, 0, 2] += 0.5
    states = _terminal_states(frequency, ports, incident, reflected)
    states[0].current_a[1, 0] = np.nan

    result = solve_scattering(ports, states, condition_limit=100)

    assert not result.complete
    assert result.status == (
        "ok",
        "unavailable_observation",
        "rank_deficient",
        "ill_conditioned",
        "inconsistent_excitations",
    )
    np.testing.assert_allclose(result.scattering[0], expected, atol=1e-14)
    assert np.isnan(result.scattering[1:]).all()
    assert result.incident_rank[2] == 1
    assert result.incident_condition[3] > 100
    assert result.relative_residual[4] > 0.01


def test_drive_states_must_share_grid_and_port_shape():
    ports = (PortReference("one", 50),)
    grid = np.array([1e6, 2e6])
    voltage = np.ones((2, 1))
    current = voltage / 50
    base = PortExcitation("base", grid, voltage, current)
    rounded = PortExcitation("rounded", np.nextafter(grid, np.inf), voltage, current)
    shifted = PortExcitation("shifted", grid + 1, voltage, current)
    wrong_shape = PortExcitation("wrong", grid, voltage[:, 0], current)

    assert solve_scattering(ports, (base, rounded)).complete
    with pytest.raises(ValueError, match="different frequency grid"):
        solve_scattering(ports, (base, shifted))
    with pytest.raises(ValueError, match="V/I arrays"):
        solve_scattering(ports, (base, wrong_shape))
