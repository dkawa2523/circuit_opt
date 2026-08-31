"""pcd.response_plot — drawing what a run produced.

A plot cannot be checked pixel by pixel, so these tests pin the two things that
actually break: which panels get drawn for which data, and that the Smith chart
maps a known impedance to the right point on the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcd.response_plot import render_response


def _waveform(n=64):
    t = np.linspace(0.0, 1e-6, n)
    return pd.DataFrame(
        {
            "time_s": t,
            "voltage_V": np.sin(2 * np.pi * 1e6 * t),
            "current_A": np.zeros(n),
            "source_voltage_V": np.cos(2 * np.pi * 1e6 * t),
        }
    )


def _ac(resistance, reactance, freqs=(1e6, 1e7, 1e8)):
    """A sweep whose input impedance is exactly R + jX at every point."""

    z = complex(resistance, reactance)
    current = -(1.0 / z)  # ngspice reports current into the source's + terminal
    return pd.DataFrame(
        {
            "frequency_Hz": list(freqs),
            "voltage_re": [1.0] * len(freqs),
            "voltage_im": [0.0] * len(freqs),
            "current_re": [current.real] * len(freqs),
            "current_im": [current.imag] * len(freqs),
        }
    )


def test_a_transient_only_run_draws_one_panel(tmp_path):
    out = render_response(_waveform(), None, tmp_path / "a.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_a_run_with_a_sweep_draws_the_smith_chart_too(tmp_path):
    small = render_response(_waveform(), None, tmp_path / "b.png")
    full = render_response(_waveform(), _ac(50.0, 0.0), tmp_path / "c.png")
    # Three panels rather than one: a wider figure, hence a larger file.
    assert full.stat().st_size > small.stat().st_size


def test_a_sweep_without_a_transient_still_plots(tmp_path):
    assert render_response(None, _ac(50.0, 0.0), tmp_path / "d.png").exists()


def test_plotting_nothing_is_an_error_rather_than_an_empty_figure(tmp_path):
    with pytest.raises(ValueError, match="nothing to plot"):
        render_response(None, None, tmp_path / "e.png")
    with pytest.raises(ValueError, match="nothing to plot"):
        render_response(pd.DataFrame(), pd.DataFrame(), tmp_path / "f.png")


def test_the_marked_point_sits_where_the_impedance_says(tmp_path):
    """A perfect match is the centre of the chart; a pure open is its right edge."""

    from pcd.analysis import input_impedance

    matched = input_impedance(_ac(50.0, 0.0), z0=50.0)
    assert matched["reflection_magnitude"].to_numpy() == pytest.approx(0.0, abs=1e-12)

    # 50 - 50j: |gamma| = |(-50j)/(100 - 50j)| = 50/111.8 = 0.447
    reactive = input_impedance(_ac(50.0, -50.0), z0=50.0)
    assert reactive["reflection_magnitude"].iloc[0] == pytest.approx(0.4472136, rel=1e-6)

    render_response(None, _ac(50.0, -50.0), tmp_path / "g.png", marker_hz=1e7)


def test_the_plot_survives_a_totally_reflecting_sweep(tmp_path):
    """An open circuit is a legitimate design candidate, not a crash."""

    open_circuit = _ac(50.0, 0.0)
    open_circuit[["current_re", "current_im"]] = 0.0
    assert render_response(_waveform(), open_circuit, tmp_path / "h.png", marker_hz=1e7).exists()


def test_a_missing_output_directory_is_created(tmp_path):
    out = render_response(_waveform(), None, tmp_path / "nested" / "deep" / "i.png")
    assert out.exists()
