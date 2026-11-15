from __future__ import annotations

import numpy as np
import pytest

from pcd.signals import harmonic_phasors, periodic_window, real_power, time_average
from pcd.signals.power import cumulative_energy


def test_exact_tone_fit_recovers_harmonics_on_an_irregular_grid():
    f0 = 2.3e6
    end = 7.4 / f0
    time = np.linspace(0.0, 1.0, 5003) ** 1.8 * end
    signal = 7.0 + 100.0 * np.sin(2 * np.pi * f0 * time + 0.2) + 17.0 * np.cos(4 * np.pi * f0 * time - 0.4)
    result = harmonic_phasors(time, signal, f0, (1, 2))
    assert abs(result[1]) == pytest.approx(100.0, rel=2e-3)
    assert abs(result[2]) == pytest.approx(17.0, rel=2e-3)


def test_periodic_window_detects_a_settled_tail():
    f0 = 1e6
    time = np.linspace(0.0, 10 / f0, 5001)
    amplitude = np.minimum(time * f0 / 5.0, 1.0)
    signal = amplitude * np.sin(2 * np.pi * f0 * time)
    window = periodic_window(time, signal, f0, measure_cycles=3, consecutive=2, tolerance=1e-3)
    assert window is not None
    assert window.settled
    assert window.cycles == 3
    assert window.residual == pytest.approx(0.0, abs=1e-12)


def test_periodic_window_rejects_a_continuously_changing_waveform():
    f0 = 1e6
    time = np.linspace(0.0, 6 / f0, 3001)
    signal = np.exp(0.1 * time * f0) * np.sin(2 * np.pi * f0 * time)
    window = periodic_window(time, signal, f0, consecutive=2, tolerance=1e-3)
    assert window is not None
    assert not window.settled
    assert window.residual is not None
    assert window.residual > 0.02


def test_periodic_window_does_not_shorten_a_requested_measurement():
    f0 = 1e6
    time = np.linspace(0.0, 2 / f0, 1001)
    signal = np.sin(2 * np.pi * f0 * time)

    window = periodic_window(time, signal, f0, measure_cycles=3, consecutive=1)

    assert window is not None
    assert not window.settled
    assert window.cycles == 2
    assert window.available_cycles == 2
    assert window.required_cycles == 3


def test_settling_covers_all_measured_cycles():
    time = np.linspace(0.0, 8.0, 801)
    tone = np.sin(2 * np.pi * time)
    signal = np.where(time < 5.0, (0.25 + 0.1 * time) * tone, tone)
    tail = periodic_window(time, signal, 1.0, measure_cycles=3, consecutive=2)
    extended = periodic_window(time, signal, 1.0, measure_cycles=6, consecutive=2)
    assert tail is not None
    assert tail.settled
    assert extended is not None
    assert not extended.settled
    assert extended.observed
    assert extended.compared_cycles == 5
    assert extended.start_s == pytest.approx(2.0)


def test_sample_spacing_must_resolve_the_fundamental_and_requested_harmonics():
    sparse = np.arange(5, dtype=float)
    ambiguous_zero = periodic_window(sparse, np.zeros_like(sparse), 1.0)
    assert ambiguous_zero is not None
    assert not ambiguous_zero.observed
    assert not ambiguous_zero.settled

    time = np.linspace(0.0, 4.0, 21)
    signal = np.cos(2 * np.pi * time)
    window = periodic_window(time, signal, 1.0)
    assert window is not None
    assert window.settled
    assert harmonic_phasors(time, signal, 1.0, (1,), window=window)[1] == pytest.approx(1)
    assert harmonic_phasors(time, signal, 1.0, (1, 2, 3), window=window) == {}


def test_real_power_is_not_biased_by_dense_sampling():
    f0 = 1e6
    dense = np.linspace(0.0, 0.5 / f0, 2000, endpoint=False)
    sparse = np.linspace(0.5 / f0, 1.0 / f0, 50)
    time = np.concatenate((dense, sparse))
    voltage = 10 * np.sin(2 * np.pi * f0 * time)
    current = 2 * np.sin(2 * np.pi * f0 * time)
    assert real_power(time, voltage, current) == pytest.approx(10.0, rel=2e-3)


def test_signed_energy_keeps_event_samples_and_does_not_bridge_gaps():
    time = np.array([0.0, 1.0, 1.0, 3.0, 4.0, 5.0])
    power = np.array([2.0, 2.0, -1.0, -1.0, np.nan, 2.0])
    np.testing.assert_allclose(cumulative_energy(time, power), [0.0, 2.0, 2.0, 0.0, np.nan, np.nan])
    assert time_average(power[:4], time[:4]) == pytest.approx(0.0)
    assert np.isnan(time_average(power, time))

    grid = np.linspace(0, 5, 501)
    waveform = np.sin(2 * np.pi * grid)
    waveform[-40] = np.nan
    window = periodic_window(grid, waveform, 1.0)
    assert window is not None
    assert not window.settled
    assert harmonic_phasors(grid, waveform, 1.0, window=window) == {}


def test_settling_compares_narrow_events_at_native_solver_knots():
    period, phase, width = 1e-6, 0.35713e-6, 1e-11
    time = np.unique(
        np.r_[
            np.linspace(0, 8 * period, 801),
            [cycle * period + phase + delta for cycle in range(8) for delta in (-width, 0, width)],
        ]
    )
    signal = np.zeros_like(time)
    for cycle in range(8):
        signal[np.abs(time - (cycle * period + phase)) < width / 10] = 100.0 if cycle < 7 else 200.0
    window = periodic_window(time, signal, 1 / period, tolerance=1e-4)
    assert window is not None
    assert not window.settled


def test_settling_compares_both_sides_of_equal_time_events():
    time = np.array(
        [sample for cycle in range(4) for sample in (cycle, cycle + 0.25, cycle + 0.25, cycle + 0.625)] + [4]
    )
    signal = np.array([0, 0, 2, 1, 0, 2, 0, 0] * 2 + [0])
    window = periodic_window(time, signal, 1.0)
    assert window is not None
    assert not window.settled
    assert window.residual is not None
    assert window.residual > 0.5
