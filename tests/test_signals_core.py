from __future__ import annotations

import numpy as np
import pytest

from pcd.signals import harmonic_phasors, periodic_window, real_power


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


def test_real_power_is_not_biased_by_dense_sampling():
    f0 = 1e6
    dense = np.linspace(0.0, 0.5 / f0, 2000, endpoint=False)
    sparse = np.linspace(0.5 / f0, 1.0 / f0, 50)
    time = np.concatenate((dense, sparse))
    voltage = 10 * np.sin(2 * np.pi * f0 * time)
    current = 2 * np.sin(2 * np.pi * f0 * time)
    assert real_power(time, voltage, current) == pytest.approx(10.0, rel=2e-3)
