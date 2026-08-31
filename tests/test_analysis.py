"""pcd.analysis — asking for an analysis, and reading the answer back.

The impedance tests check against closed-form values rather than a golden file:
a series R-C has an exactly known Z(f), so a sign or scaling error cannot hide.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcd.analysis import (
    AC_LOAD_VOLTAGE,
    DEFAULT_Z0,
    LOAD_CURRENT,
    AcSweep,
    ac_component_metrics,
    ac_power_flow,
    ac_sweep,
    at_frequency,
    component_loss_balance,
    control_lines,
    harmonic_spectrum,
    input_impedance,
    mean_over_time,
    power_flow,
    read_ac,
    rf_port_metrics,
    transient_component_metrics,
    transient_requested,
)
from pcd.component_models import ComponentObservation

# --- requesting an analysis -------------------------------------------------


def test_a_transient_only_case_asks_for_no_sweep():
    assert ac_sweep({"tran": {"step_s": 1e-9, "stop_s": 1e-6}}) is None


def test_an_ac_sweep_is_read_from_the_case():
    sweep = ac_sweep({"ac": {"sweep": "lin", "points": 50, "start_Hz": 1e6, "stop_Hz": 2e7}})
    assert sweep is not None
    assert sweep == AcSweep(sweep="lin", points=50, start_hz=1e6, stop_hz=2e7)
    assert sweep.command() == "ac lin 50 1e+06 2e+07"


@pytest.mark.parametrize("reference", ["rf_frequency_Hz", "$rf_frequency_Hz"])
def test_a_parameterized_ac_point_is_resolved_for_each_scenario(reference):
    sweep = ac_sweep({"ac": {"frequency_Hz": reference}}, {"rf_frequency_Hz": 27.12e6})
    assert sweep is not None
    assert sweep == AcSweep(sweep="lin", points=1, start_hz=27.12e6, stop_hz=27.12e6)
    assert sweep.command() == "ac lin 1 2.712e+07 2.712e+07"


def test_the_control_block_runs_only_a_transient_by_default():
    lines = control_lines({"tran": {"step_s": 1e-9, "stop_s": 1e-6}}, "out", "Vsrc", "src", [])
    assert "tran 1e-09 1e-06" in lines
    assert "wrdata waveform.csv time v(out) i(Vsrc)" in lines
    assert not any(line.startswith("ac ") for line in lines)


def test_requesting_a_sweep_adds_it_to_the_same_run():
    """One solver invocation produces both files; there is no second run."""

    solver = {"tran": {"step_s": 1e-9, "stop_s": 1e-6}, "ac": {"points": 10}}
    lines = control_lines(solver, "out", "Vsrc", "src", [])
    assert any(line.startswith("tran ") for line in lines)
    assert any(line.startswith("ac dec 10") for line in lines)
    assert "set numdgt=15" in lines
    # Impedance needs the source node voltage and the source current together.
    assert "wrdata ac.csv v(src) i(Vsrc) v(out)" in lines


def test_an_ac_only_case_does_not_run_or_write_a_transient():
    solver = {"ac": {"sweep": "lin", "points": 3, "start_Hz": 1e6, "stop_Hz": 2e6}}
    lines = control_lines(solver, "out", "Vsrc", "src", [])
    assert transient_requested(solver) is False
    assert not any(line.startswith("tran ") for line in lines)
    assert not any("waveform.csv" in line for line in lines)
    assert "wrdata ac.csv v(src) i(Vsrc) v(out)" in lines


def test_a_scenario_frequency_generates_one_exact_ac_point():
    lines = control_lines(
        {"ac": {"frequency_Hz": "rf_frequency_Hz"}},
        "out",
        "Vsrc",
        "src",
        [],
        {"rf_frequency_Hz": 6.78e6},
    )
    assert "ac lin 1 6.78e+06 6.78e+06" in lines


def test_extra_probes_are_saved_alongside_the_standard_vectors():
    lines = control_lines({}, "out", "Vsrc", "src", ["i(L1)", "v(mid)"])
    assert "wrdata waveform.csv time v(out) i(Vsrc) i(L1) v(mid)" in lines
    assert lines[0].startswith(".save v(out) i(Vsrc) i(L1) v(mid)")


def test_ac_probes_are_written_after_the_standard_source_and_load_vectors():
    lines = control_lines(
        {"ac": {"frequency_Hz": 1e6}},
        "v(load)",
        "Vsrc",
        "src",
        [],
        ac_probes=["i(Vobserve_L1)"],
    )
    assert "wrdata ac.csv v(src) i(Vsrc) v(load) i(Vobserve_L1)" in lines


# --- reading the answer back ------------------------------------------------


def _write_ac(tmp_path, frequencies, voltage, current):
    """Write an ngspice-style AC file: (scale, re, im) per vector."""

    rows = [[f, v.real, v.imag, f, i.real, i.imag] for f, v, i in zip(frequencies, voltage, current, strict=True)]
    path = tmp_path / "ac.csv"
    np.savetxt(path, np.array(rows))
    return path


def test_an_ac_file_parses_into_real_and_imaginary_columns(tmp_path):
    path = _write_ac(tmp_path, [1e6, 2e6], [1 + 0j, 0.5 - 0.5j], [-0.01 + 0j, -0.02 + 0.01j])
    ac = read_ac(path)
    assert list(ac.columns) == ["frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"]
    assert ac["frequency_Hz"].to_list() == [1e6, 2e6]
    # Real columns stay real: a complex dtype would upcast a whole row on .iloc.
    assert all(ac[c].dtype == np.float64 for c in ac.columns)


def test_a_short_ac_file_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    np.savetxt(path, np.array([[1e6, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="needs 6 columns"):
        read_ac(path)


def test_named_ac_probes_support_power_stress_and_loss_closure(tmp_path):
    frequency, source_peak, series_r, load_r = 1e6, 100.0, 5.0, 50.0
    current_peak = source_peak / (series_r + load_r)
    vectors = [
        complex(source_peak),
        complex(-current_peak),  # ngspice source-current sign
        complex(load_r * current_peak),
        complex(series_r * current_peak),
        complex(current_peak),
        complex(current_peak),
    ]
    row = []
    for value in vectors:
        row.extend([frequency, value.real, value.imag])
    path = tmp_path / "ac_probes.csv"
    np.savetxt(path, np.asarray([row]))

    component = ComponentObservation("L1", "src", "load", series_r)
    response = read_ac(
        path,
        [
            AC_LOAD_VOLTAGE,
            component.voltage_column,
            component.current_column,
            LOAD_CURRENT,
        ],
    ).iloc[0]
    power = ac_power_flow(response, LOAD_CURRENT)
    stress = ac_component_metrics(response, (component,))
    metrics = {**power, **stress}
    balance = component_loss_balance(metrics)

    expected_loss = (current_peak / np.sqrt(2.0)) ** 2 * series_r
    assert power["network_loss_W"] == pytest.approx(expected_loss)
    assert power["source_current_rms_A"] == pytest.approx(current_peak / np.sqrt(2.0))
    assert power["source_apparent_power_VA"] == pytest.approx(source_peak * current_peak / 2.0)
    assert stress["component_L1_voltage_peak_V"] == pytest.approx(series_r * current_peak)
    assert stress["component_L1_current_rms_A"] == pytest.approx(current_peak / np.sqrt(2.0))
    assert stress["component_L1_loss_W"] == pytest.approx(expected_loss)
    assert balance["component_loss_balance_residual_W"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("frequency", [1e6, 1e7, 1e8])
def test_impedance_matches_the_closed_form_for_a_series_rc(tmp_path, frequency):
    """R + 1/(jwC) with a 1 V drive: current is exactly V/Z."""

    r, c = 50.0, 1e-9
    z_true = r + 1 / (2j * np.pi * frequency * c)
    # ngspice reports current into the source's + terminal, hence the sign.
    path = _write_ac(tmp_path, [frequency], [1 + 0j], [-(1 / z_true)])

    row = input_impedance(read_ac(path)).iloc[0]
    assert row["resistance_ohm"] == pytest.approx(z_true.real, rel=1e-9)
    assert row["reactance_ohm"] == pytest.approx(z_true.imag, rel=1e-9)


def test_a_perfect_match_has_no_reflection(tmp_path):
    path = _write_ac(tmp_path, [1e7], [1 + 0j], [-(1 / DEFAULT_Z0)])
    row = input_impedance(read_ac(path)).iloc[0]
    assert row["resistance_ohm"] == pytest.approx(50.0)
    assert row["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert row["vswr"] == pytest.approx(1.0)


def test_a_near_open_reflects_almost_everything(tmp_path):
    """Vanishing current: |gamma| approaches 1 and VSWR blows up."""

    path = _write_ac(tmp_path, [1e7], [1 + 0j], [-1e-18 + 0j])
    row = input_impedance(read_ac(path)).iloc[0]
    assert row["reflection_magnitude"] == pytest.approx(1.0, abs=1e-9)
    assert row["vswr"] > 1e6


def test_a_non_physical_reflection_reports_infinite_vswr(tmp_path):
    """|gamma| >= 1 has no finite standing-wave ratio; say so rather than divide."""

    path = _write_ac(tmp_path, [1e7], [1 + 0j], [0j])  # exactly zero current
    row = input_impedance(read_ac(path)).iloc[0]
    assert not np.isfinite(row["vswr"])


def test_the_reference_impedance_is_configurable(tmp_path):
    path = _write_ac(tmp_path, [1e7], [1 + 0j], [-(1 / 75.0)])
    assert input_impedance(read_ac(path), z0=75.0).iloc[0]["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert input_impedance(read_ac(path), z0=50.0).iloc[0]["reflection_magnitude"] > 0.1


def test_reflection_in_decibels_is_negative_for_a_good_match(tmp_path):
    path = _write_ac(tmp_path, [1e7], [1 + 0j], [-(1 / 55.0)])
    row = input_impedance(read_ac(path)).iloc[0]
    assert row["reflection_db"] < -20.0


def test_an_exact_fundamental_row_is_selected(tmp_path):
    freqs = [1e6, 1.3e7, 1.356e7, 2e7]
    path = _write_ac(tmp_path, freqs, [1 + 0j] * 4, [-0.02 + 0j] * 4)
    assert at_frequency(read_ac(path), 13.56e6)["frequency_Hz"] == pytest.approx(1.356e7)


def test_a_frequency_between_points_is_interpolated_without_extrapolation(tmp_path):
    path = _write_ac(tmp_path, [1e6, 3e6], [1 + 0j, 3 + 2j], [-0.01 + 0j, -0.03 - 0.02j])
    row = at_frequency(read_ac(path), 2e6)
    assert row["frequency_Hz"] == pytest.approx(2e6)
    assert row["voltage_re"] == pytest.approx(2.0)
    assert row["voltage_im"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the simulated sweep"):
        at_frequency(read_ac(path), 4e6)


def test_selecting_from_an_empty_sweep_is_an_error():
    import pandas as pd

    with pytest.raises(ValueError, match="frequency response is empty"):
        at_frequency(pd.DataFrame(columns=["frequency_Hz"]), 1e6)


# --- where the power goes ---------------------------------------------------


def _sine_waveform(v_amp, i_amp, phase=0.0, n=2001, freq=1e6, load_current=None):
    import pandas as pd

    t = np.linspace(0.0, 4.0 / freq, n)
    w = 2 * np.pi * freq
    frame = pd.DataFrame(
        {
            "time_s": t,
            "voltage_V": v_amp * np.sin(w * t),
            # ngspice reports current into the source's + terminal.
            "current_A": -i_amp * np.sin(w * t + phase),
            "source_voltage_V": v_amp * np.sin(w * t),
        }
    )
    if load_current is not None:
        frame["i(Vload)"] = load_current * np.sin(w * t)
    return frame


def test_source_power_matches_the_closed_form_for_a_resistive_drive():
    """V and I in phase: P = Vpk*Ipk/2 exactly."""

    power = power_flow(_sine_waveform(100.0, 2.0))
    assert power["source_power_W"] == pytest.approx(100.0 * 2.0 / 2, rel=1e-3)
    assert power["source_current_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert power["source_apparent_power_VA"] == pytest.approx(100.0, rel=1e-3)
    assert "load_power_W" not in power  # no load probe was declared


def test_a_purely_reactive_drive_delivers_no_real_power():
    power = power_flow(_sine_waveform(100.0, 2.0, phase=np.pi / 2))
    assert power["source_power_W"] == pytest.approx(0.0, abs=1e-3)
    assert power["source_apparent_power_VA"] == pytest.approx(100.0, rel=1e-3)


def test_a_declared_load_probe_yields_an_efficiency():
    """The match itself draws current, so source and load power differ."""

    frame = _sine_waveform(100.0, 2.0, load_current=1.5)
    power = power_flow(frame, load_current="i(Vload)")
    assert power["load_power_W"] == pytest.approx(100.0 * 1.5 / 2, rel=1e-3)
    assert power["network_loss_W"] == pytest.approx(power["source_power_W"] - power["load_power_W"])
    assert power["transfer_efficiency"] == pytest.approx(0.75, rel=1e-3)


def test_transient_component_stress_uses_the_same_metric_names_as_ac():
    frame = _sine_waveform(100.0, 2.0)
    phase = 2 * np.pi * 1e6 * frame["time_s"].to_numpy(float)
    frame["component_L1_voltage_V"] = 10.0 * np.sin(phase)
    frame["component_L1_current_A"] = 2.0 * np.sin(phase)
    metrics = transient_component_metrics(frame, 1e6, (ComponentObservation("L1", "src", "out", 5.0),))

    assert metrics["component_L1_voltage_peak_V"] == pytest.approx(10.0, rel=1e-4)
    assert metrics["component_L1_current_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["component_L1_loss_W"] == pytest.approx(10.0, rel=1e-3)


def test_an_undeclared_probe_name_is_ignored_rather_than_guessed():
    power = power_flow(_sine_waveform(100.0, 2.0), load_current="i(Vnotthere)")
    assert set(power) == {"source_power_W", "source_current_rms_A", "source_apparent_power_VA"}


def test_load_power_remains_available_without_a_source_voltage_probe():
    """External port data may omit the upstream source measurement."""

    import pandas as pd

    assert power_flow(pd.DataFrame({"time_s": [0.0], "voltage_V": [1.0], "current_A": [1.0]})) == {}
    frame = _sine_waveform(100.0, 2.0, load_current=1.5).drop(columns="source_voltage_V")
    power = power_flow(frame, "i(Vload)")
    assert power["load_power_W"] == pytest.approx(75.0, rel=1e-3)
    assert "source_power_W" not in power
    assert power_flow(pd.DataFrame()) == {}


def test_rf_port_metrics_reject_missing_measurement_inputs():
    import pandas as pd

    with pytest.raises(ValueError, match="waveform is empty"):
        rf_port_metrics(pd.DataFrame(), 1e6, "load_current_A")
    frame = _sine_waveform(10.0, 1.0)
    with pytest.raises(ValueError, match=r"require measurement\.load_current"):
        rf_port_metrics(frame, 1e6, None)
    with pytest.raises(ValueError, match="positive source fundamental"):
        rf_port_metrics(frame.assign(load_current_A=1.0), 0.0, "load_current_A")


def test_rf_port_metrics_preserve_source_terminal_rms_and_apparent_power():
    metrics = rf_port_metrics(_sine_waveform(100.0, 2.0, load_current=1.5), 1e6, "i(Vload)")
    assert metrics["source_current_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["source_apparent_power_VA"] == pytest.approx(100.0, rel=1e-3)
    assert metrics["source_real_power_W"] == pytest.approx(100.0, rel=1e-3)


def test_total_reflection_gives_infinite_vswr_without_a_numpy_warning(tmp_path):
    """|gamma| = 1 is an open circuit, not a numerical accident."""

    import warnings

    path = _write_ac(tmp_path, [1e7], [1 + 0j], [0j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = input_impedance(read_ac(path)).iloc[0]
    assert np.isinf(row["vswr"])
    assert row["reflection_magnitude"] == pytest.approx(1.0)


def test_power_is_averaged_over_time_not_over_samples():
    """ngspice picks its own timesteps; an unweighted mean is not a time average.

    A signal sampled densely where it is large and sparsely where it is small
    fools a plain mean.  Measured on a real CCP case the error was 22%.
    """

    import pandas as pd

    # One full cycle of a sine, sampled 25x more densely over the positive half.
    # The true time average is zero; an unweighted mean mostly sees the peak.
    dense = np.linspace(0.0, 0.5e-6, 1000, endpoint=False)
    sparse = np.linspace(0.5e-6, 1e-6, 40)
    t = np.concatenate([dense, sparse])
    signal = np.sin(2 * np.pi * 1e6 * t)

    assert mean_over_time(signal, t) == pytest.approx(0.0, abs=1e-3)
    assert float(signal.mean()) > 0.5, "an unweighted mean should be badly wrong here"

    # v and i in antiphase is ngspice's convention for a source delivering
    # power: 1/2 W here, and the uneven sampling must not distort it.
    frame = pd.DataFrame(
        {
            "time_s": t,
            "voltage_V": signal,
            "current_A": -signal,
            "source_voltage_V": signal,
        }
    )
    assert power_flow(frame)["source_power_W"] == pytest.approx(0.5, rel=1e-3)


def test_a_single_sample_has_no_time_span_to_average_over():
    assert mean_over_time(np.array([5.0]), np.array([0.0])) == 5.0
    assert mean_over_time(np.array([]), np.array([])) == 0.0
    # A record with no elapsed time falls back to the plain mean.
    assert mean_over_time(np.array([2.0, 4.0]), np.array([1e-9, 1e-9])) == 3.0


# --- harmonic content -------------------------------------------------------


def _three_tone(t, f0=13.56e6):
    return 100 * np.sin(2 * np.pi * f0 * t) + 30 * np.sin(4 * np.pi * f0 * t) + 10 * np.sin(6 * np.pi * f0 * t)


def test_harmonics_are_recovered_from_unevenly_sampled_data():
    """An FFT assumes uniform spacing; a solver does not provide it.

    On a real run the timesteps spanned 1190x, which put the fundamental out by
    3% and the second and third harmonics by about 15%.
    """

    f0 = 13.56e6
    clustered = np.linspace(0.0, 1.0, 4001) ** 1.7 * (8 / f0)
    spectrum = harmonic_spectrum(clustered, _three_tone(clustered), f0, 3)
    assert spectrum is not None
    assert np.abs(spectrum) == pytest.approx([100.0, 30.0, 10.0], rel=5e-3)


def test_uniform_and_clustered_sampling_agree():
    f0 = 13.56e6
    uniform = np.linspace(0.0, 8 / f0, 4001)
    clustered = np.linspace(0.0, 1.0, 4001) ** 1.7 * (8 / f0)
    on_uniform = harmonic_spectrum(uniform, _three_tone(uniform), f0, 3)
    on_clustered = harmonic_spectrum(clustered, _three_tone(clustered), f0, 3)
    assert on_uniform is not None
    assert on_clustered is not None
    assert np.abs(on_uniform) == pytest.approx(np.abs(on_clustered), rel=1e-2)


def test_the_spectrum_keeps_phase_not_just_magnitude():
    """The waveform objective compares complex amplitudes, not magnitudes."""

    f0 = 1e6
    t = np.linspace(0.0, 8 / f0, 2001)
    in_phase = harmonic_spectrum(t, np.sin(2 * np.pi * f0 * t), f0, 1)
    quadrature = harmonic_spectrum(t, np.cos(2 * np.pi * f0 * t), f0, 1)
    assert in_phase is not None
    assert quadrature is not None
    assert abs(in_phase[0]) == pytest.approx(abs(quadrature[0]), rel=1e-2)
    assert abs(in_phase[0] - quadrature[0]) > 1.0  # a phase shift must be visible


@pytest.mark.parametrize(
    ("time_s", "fundamental"),
    [
        (np.linspace(0.0, 1e-6, 3), 1e6),  # too few samples
        (np.linspace(0.0, 1e-6, 100), 0.0),  # no fundamental to refer to
        (np.zeros(100), 1e6),  # no elapsed time
    ],
)
def test_a_spectrum_that_cannot_be_taken_returns_nothing(time_s, fundamental):
    assert harmonic_spectrum(time_s, np.ones_like(time_s), fundamental, 3) is None
