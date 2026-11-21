"""Electrical measurements over canonical AC and transient results.

The impedance tests check against closed-form values rather than a golden file:
a series R-C has an exactly known Z(f), so a sign or scaling error cannot hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcd.analysis import (
    DEFAULT_Z0,
    ac_component_metrics,
    ac_power_flow,
    at_frequency,
    component_loss_balance,
    frequency_sweep_metrics,
    half_power_bandwidth,
    harmonic_spectrum,
    input_impedance,
    power_flow,
    rf_port_metrics,
    transient_component_metrics,
)
from pcd.analysis.measurements import ac_operating_metrics, periodic_operating_metrics
from pcd.analysis.transient import rf_measurement_options
from pcd.core import MeasurementUnavailableError, MetricSet, UnsettledMeasurementError
from pcd.probes import ComponentObservation
from pcd.records import load_waveform
from pcd.signals import time_average
from pcd.simulation import LOAD_CURRENT_COLUMN, LOAD_VOLTAGE_COLUMN, SimulationResult

# --- canonical AC input -----------------------------------------------------


def _ac_frame(frequencies, voltage, current):
    return pd.DataFrame(
        {
            "frequency_Hz": frequencies,
            "voltage_re": np.real(voltage),
            "voltage_im": np.imag(voltage),
            "current_re": np.real(current),
            "current_im": np.imag(current),
        }
    )


def test_named_ac_probes_support_power_stress_and_loss_closure():
    frequency, source_peak, series_r, load_r = 1e6, 100.0, 5.0, 50.0
    current_peak = source_peak / (series_r + load_r)
    component = ComponentObservation("L1", "src", "load", series_r)
    frame = _ac_frame([frequency], [source_peak], [-current_peak])
    for name, peak in (
        (LOAD_VOLTAGE_COLUMN, load_r * current_peak),
        (component.voltage_column, series_r * current_peak),
        (component.current_column, current_peak),
        (LOAD_CURRENT_COLUMN, current_peak),
    ):
        frame[f"{name}_re"] = peak
        frame[f"{name}_im"] = 0.0
    response = frame.iloc[0]
    power = ac_power_flow(response, LOAD_CURRENT_COLUMN)
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


@pytest.mark.parametrize("missing_signal", ["load", "component"])
def test_missing_ac_probe_values_preserve_independent_measurements(missing_signal):
    frequency = 1e6
    component = ComponentObservation("L1", "src", "load", 5.0)
    frame = pd.DataFrame(
        {
            "frequency_Hz": [frequency],
            "voltage_re": [1.0],
            "voltage_im": [0.0],
            "current_re": [-0.02],
            "current_im": [0.0],
            f"{LOAD_VOLTAGE_COLUMN}_re": [0.9],
            f"{LOAD_VOLTAGE_COLUMN}_im": [0.0],
            f"{LOAD_CURRENT_COLUMN}_re": [np.nan if missing_signal == "load" else 0.02],
            f"{LOAD_CURRENT_COLUMN}_im": [0.0],
            f"{component.voltage_column}_re": [0.1],
            f"{component.voltage_column}_im": [0.0],
            f"{component.current_column}_re": [0.02],
            f"{component.current_column}_im": [np.nan if missing_signal == "component" else 0.0],
        }
    )
    metrics = ac_operating_metrics(frame, frequency, load_current_column=LOAD_CURRENT_COLUMN, components=(component,))

    assert metrics["source_real_power_W"] == pytest.approx(0.01)
    assert metrics["reflection_magnitude"] == pytest.approx(0.0)
    assert metrics["component_L1_voltage_peak_V"] == pytest.approx(0.1)
    assert metrics["component_loss_balance_residual_W"] is None
    assert metrics["component_loss_balance_residual_W_unavailable_reason"]
    assert metrics["component_loss_balance_fraction_of_source"] is None
    if missing_signal == "load":
        assert metrics["modeled_component_loss_W"] == pytest.approx(0.001)
        unavailable = ("load_real_power_W", "network_loss_W", "transfer_efficiency")
    else:
        assert metrics["load_real_power_W"] == pytest.approx(0.009)
        assert metrics["network_loss_W"] == pytest.approx(0.001)
        unavailable = ("component_L1_current_peak_A", "component_L1_loss_W", "modeled_component_loss_W")
    for name in unavailable:
        assert metrics[name] is None
        assert metrics[f"{name}_unavailable_reason"]


@pytest.mark.parametrize("frequency", [1e6, 1e7, 1e8])
def test_impedance_matches_the_closed_form_for_a_series_rc(frequency):
    """R + 1/(jwC) with a 1 V drive: current is exactly V/Z."""

    r, c = 50.0, 1e-9
    z_true = r + 1 / (2j * np.pi * frequency * c)
    # ngspice reports current into the source's + terminal, hence the sign.
    row = input_impedance(_ac_frame([frequency], [1 + 0j], [-(1 / z_true)])).iloc[0]
    assert row["resistance_ohm"] == pytest.approx(z_true.real, rel=1e-9)
    assert row["reactance_ohm"] == pytest.approx(z_true.imag, rel=1e-9)


def test_a_perfect_match_has_no_reflection():
    row = input_impedance(_ac_frame([1e7], [1 + 0j], [-(1 / DEFAULT_Z0)])).iloc[0]
    assert row["resistance_ohm"] == pytest.approx(50.0)
    assert row["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert row["vswr"] == pytest.approx(1.0)


def test_a_near_open_reflects_almost_everything():
    """Vanishing current: |gamma| approaches 1 and VSWR blows up."""

    row = input_impedance(_ac_frame([1e7], [1 + 0j], [-1e-18 + 0j])).iloc[0]
    assert row["reflection_magnitude"] == pytest.approx(1.0, abs=1e-9)
    assert row["vswr"] > 1e6


def test_the_reference_impedance_is_configurable():
    frame = _ac_frame([1e7], [1 + 0j], [-(1 / 75.0)])
    assert input_impedance(frame, z0=75.0).iloc[0]["reflection_magnitude"] == pytest.approx(0.0, abs=1e-12)
    mismatched = input_impedance(frame, z0=50.0).iloc[0]
    assert mismatched["reflection_magnitude"] == pytest.approx(0.2)
    assert mismatched["reflection_db"] == pytest.approx(20 * np.log10(0.2))


def test_ac_power_flow_separates_forward_and_reflected_power():
    voltage = 10.0 + 0j
    impedance = 100.0 + 0j
    point = _ac_frame([1e6], [voltage], [-(voltage / impedance)]).iloc[0]

    power = ac_power_flow(point, reference_impedance_ohm=50.0)

    assert power["forward_power_W"] == pytest.approx(0.5625)
    assert power["reflected_power_W"] == pytest.approx(0.0625)
    assert power["source_real_power_W"] == pytest.approx(0.5)
    assert power["forward_power_W"] - power["reflected_power_W"] == pytest.approx(power["source_real_power_W"])
    with pytest.raises(ValueError, match="reference_impedance_ohm"):
        ac_power_flow(point, reference_impedance_ohm=0.0)


def test_an_exact_fundamental_row_is_selected():
    freqs = [1e6, 1.3e7, 1.356e7, 2e7]
    frame = _ac_frame(freqs, [1 + 0j] * 4, [-0.02 + 0j] * 4)
    assert at_frequency(frame, 13.56e6)["frequency_Hz"] == pytest.approx(1.356e7)


def test_a_frequency_between_points_is_interpolated_without_extrapolation():
    frame = _ac_frame([1e6, 3e6], [1 + 0j, 3 + 2j], [-0.01 + 0j, -0.03 - 0.02j])
    row = at_frequency(frame, 2e6)
    assert row["frequency_Hz"] == pytest.approx(2e6)
    assert row["voltage_re"] == pytest.approx(2.0)
    assert row["voltage_im"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the simulated sweep"):
        at_frequency(frame, 4e6)


def test_rounding_at_a_single_frequency_does_not_put_the_point_outside_the_sweep():
    ac = pd.DataFrame({"frequency_Hz": [123456.1234567891], "voltage_re": [1.0]})
    assert at_frequency(ac, 123456.12345678912)["voltage_re"] == 1.0


def test_selecting_from_an_empty_sweep_is_an_error():
    import pandas as pd

    with pytest.raises(ValueError, match="frequency response is empty"):
        at_frequency(pd.DataFrame(columns=["frequency_Hz"]), 1e6)


def test_half_power_bandwidth_recovers_a_known_resonance():
    frequency = np.linspace(8e6, 12e6, 401)
    response = 1.0 / (1.0 + ((frequency - 10e6) / 0.5e6) ** 2)

    result = half_power_bandwidth(frequency, response)

    assert result is not None
    assert result["resonant_frequency_Hz"] == pytest.approx(10e6)
    assert result["lower_frequency_Hz"] == pytest.approx(9.5e6)
    assert result["upper_frequency_Hz"] == pytest.approx(10.5e6)
    assert result["bandwidth_Hz"] == pytest.approx(1e6)
    assert result["fractional_bandwidth"] == pytest.approx(0.1)
    assert result["loaded_quality_factor"] == pytest.approx(10.0)


def test_half_power_bandwidth_is_absent_when_the_sweep_does_not_bracket_it():
    frequency = np.linspace(9.8e6, 10.2e6, 41)
    response = 1.0 / (1.0 + ((frequency - 10e6) / 0.5e6) ** 2)

    assert half_power_bandwidth(np.array([9e6, 10e6]), np.array([0.5, 1.0])) is None
    assert half_power_bandwidth(frequency, response) is None


def test_frequency_sweep_uses_saved_load_power_when_available():
    import pandas as pd

    frequency = np.linspace(8e6, 12e6, 401)
    load_power = 1.0 / (1.0 + ((frequency - 10e6) / 0.5e6) ** 2)
    ac = pd.DataFrame(
        {
            "frequency_Hz": frequency,
            "voltage_re": np.ones_like(frequency),
            "voltage_im": np.zeros_like(frequency),
            "current_re": np.full_like(frequency, -1.0 / DEFAULT_Z0),
            "current_im": np.zeros_like(frequency),
            f"{LOAD_VOLTAGE_COLUMN}_re": np.ones_like(frequency),
            f"{LOAD_VOLTAGE_COLUMN}_im": np.zeros_like(frequency),
            f"{LOAD_CURRENT_COLUMN}_re": 2.0 * load_power,
            f"{LOAD_CURRENT_COLUMN}_im": np.zeros_like(frequency),
        }
    )

    result = frequency_sweep_metrics(ac, load_current_column=LOAD_CURRENT_COLUMN)

    assert result is not None
    assert result["sample_count"] == 401
    assert result["sampled_best_match_frequency_Hz"] == pytest.approx(8e6)
    resonance = result["half_power_resonance"]
    assert resonance is not None
    assert resonance["basis"] == "load_real_power_W"
    assert resonance["resonant_frequency_Hz"] == pytest.approx(10e6)
    assert resonance["loaded_quality_factor"] == pytest.approx(10.0)


def test_frequency_sweep_uses_accepted_power_when_load_probes_are_absent():
    import pandas as pd

    frequency = np.linspace(8e6, 12e6, 401)
    accepted_power = 1.0 / (1.0 + ((frequency - 10e6) / 0.5e6) ** 2)
    reflection = np.sqrt(1.0 - accepted_power)
    impedance = DEFAULT_Z0 * (1.0 + reflection) / (1.0 - reflection)
    ac = pd.DataFrame(
        {
            "frequency_Hz": frequency,
            "voltage_re": np.ones_like(frequency),
            "voltage_im": np.zeros_like(frequency),
            "current_re": -1.0 / impedance,
            "current_im": np.zeros_like(frequency),
        }
    )

    result = frequency_sweep_metrics(ac)

    assert result is not None
    assert result["sampled_best_match_frequency_Hz"] == pytest.approx(10e6)
    resonance = result["half_power_resonance"]
    assert resonance is not None
    assert resonance["basis"] == "accepted_power_fraction"
    assert resonance["bandwidth_Hz"] == pytest.approx(1e6)

    with_missing_load_power = ac.assign(
        load_voltage_V_re=1.0, load_voltage_V_im=0.0, load_current_A_re=2.0, load_current_A_im=0.0
    )
    with_missing_load_power.loc[200, "load_current_A_re"] = np.nan
    load_result = frequency_sweep_metrics(with_missing_load_power, load_current_column=LOAD_CURRENT_COLUMN)
    assert load_result is not None
    assert load_result["sampled_best_match_frequency_Hz"] == pytest.approx(10e6)
    assert load_result["half_power_resonance"] is None
    assert "load_real_power_W" in load_result["half_power_unavailable_reason"]


def test_half_power_bandwidth_rejects_a_non_monotonic_frequency_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        half_power_bandwidth(np.array([1e6, 2e6, 1.5e6]), np.array([0.5, 1.0, 0.5]))


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
    components = (ComponentObservation("L1", "src", "out", 5.0),)
    metrics = transient_component_metrics(frame, 1e6, components)

    assert metrics["component_L1_voltage_peak_V"] == pytest.approx(10.0, rel=1e-4)
    assert metrics["component_L1_current_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["component_L1_loss_W"] == pytest.approx(10.0, rel=1e-3)
    operating = periodic_operating_metrics(frame, 1e6, None, components, rf_measurement_options())
    assert operating["component_L1_loss_W"] == metrics["component_L1_loss_W"]
    assert operating["measurement_end_s"] - operating["measurement_start_s"] == pytest.approx(3e-6)


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
    with pytest.raises(ValueError, match="finite load voltage/current"):
        rf_port_metrics(frame.assign(load_current_A=np.nan), 1e6, "load_current_A")

    sparse_zero = pd.DataFrame({"time_s": np.arange(5), "voltage_V": 0.0, "load_current_A": 0.0})
    with pytest.raises(MeasurementUnavailableError, match="less than half the fundamental period"):
        rf_port_metrics(sparse_zero, 1.0, "load_current_A")


@pytest.mark.parametrize("source_coverage", ["none", "startup_only", "measurement_only", "settling_and_measurement"])
def test_periodic_load_measurements_survive_optional_missing_source_channels(source_coverage):
    time = np.linspace(0, 5e-6, 501)
    tone = np.cos(2 * np.pi * 1e6 * time)
    source_covered = {
        "none": np.zeros_like(time, dtype=bool),
        "startup_only": time < 1e-6,
        "measurement_only": time >= 2.5e-6,
        "settling_and_measurement": time >= 1.5e-6,
    }[source_coverage]
    component = ComponentObservation("R1", "source", "load", 5.0)
    waveform = pd.DataFrame(
        {
            "time_s": time,
            "voltage_V": 10 * tone,
            "load_current_A": 0.2 * tone,
            "current_A": np.where(source_covered, -0.2 * tone, np.nan),
            "source_voltage_V": 11 * tone,
            component.voltage_column: tone,
            component.current_column: 0.2 * tone,
        }
    )
    options = rf_measurement_options({"periodic_cycles": 2})
    combined = periodic_operating_metrics(waveform, 1e6, "load_current_A", (component,), options)
    direct = rf_port_metrics(waveform, 1e6, "load_current_A", periodic_cycles=2)
    assert combined["modeled_component_loss_W"] == pytest.approx(0.1)
    for metrics in (combined, direct):
        assert metrics["periodic_settled"]
        assert metrics["load_v_peak_V"] == pytest.approx(10.0)
        assert metrics["load_real_power_W"] == pytest.approx(1.0)
        if source_coverage == "settling_and_measurement":
            assert metrics["source_real_power_W"] == pytest.approx(1.1)
        else:
            for name in ("source_real_power_W", "network_loss_W", "transfer_efficiency"):
                assert metrics[name] is None
                assert "settling interval" in metrics[f"{name}_unavailable_reason"]


def test_rf_port_metrics_preserve_source_terminal_rms_and_apparent_power():
    metrics = rf_port_metrics(_sine_waveform(100.0, 2.0, load_current=1.5), 1e6, "i(Vload)")
    assert metrics["source_current_rms_A"] == pytest.approx(np.sqrt(2.0), rel=1e-3)
    assert metrics["source_apparent_power_VA"] == pytest.approx(100.0, rel=1e-3)
    assert metrics["source_real_power_W"] == pytest.approx(100.0, rel=1e-3)


def test_open_periodic_load_only_fails_the_affected_impedance_objective():
    from pcd.core import Candidate, Objective, RawResult, StudyRunner, StudySpec

    metric = "load_fundamental_resistance_ohm"

    class Backend:
        def evaluate(self, request):
            return RawResult("ok")

        def compute(self, request, raw):
            current = request.candidate.values["current"]
            waveform = _sine_waveform(10.0, current, load_current=current)
            return MetricSet(rf_port_metrics(waveform, 1e6, "i(Vload)"))

    backend = Backend()
    study = StudySpec("periodic_load", objectives=(Objective(metric),))
    runner = StudyRunner(study, backend, (backend,))
    finite, open_load = runner.run([Candidate("open", {"current": 0.0}), Candidate("finite", {"current": 2.0})])

    assert finite.aggregates[metric] == pytest.approx(5.0)
    assert open_load.evaluations[0].raw.status == "failed"
    assert "nonzero current phasors" in str(open_load.evaluations[0].raw.error)


@pytest.mark.parametrize(("cycles", "slope", "tolerance"), [(2, 0.001, 0.001), (8, 0.001, 0.001), (8, 0.02, 0.1)])
def test_declared_periodic_window_controls_power_harmonics_and_probe_tables(cycles, slope, tolerance):
    from pcd.analysis.traces import harmonic_trace

    time = np.linspace(0, 12, 12001)
    voltage = (1 + slope * time) * np.sin(2 * np.pi * time)
    waveform = pd.DataFrame(
        {"time_s": time, "voltage_V": voltage, "load_voltage_V": voltage, "load_current_A": voltage / 50}
    )
    options = {"periodic_cycles": cycles, "settling_tolerance": tolerance, "harmonic_count": 5}
    metrics = rf_port_metrics(waveform, 1, "load_current_A", **options)
    table = harmonic_trace(time, voltage, 1, measurement=options)
    start, end = 12 - cycles, 12
    mean_amplitude = 1 + slope * (start + end) / 2
    expected_h1 = np.hypot(mean_amplitude, slope / (4 * np.pi))
    expected_v2 = (1 + slope * (start + end) + slope**2 * (start**2 + start * end + end**2) / 3) / 2 - slope**2 / (
        4 * np.pi
    ) ** 2
    assert metrics["measurement_cycles"] == cycles
    assert metrics["measurement_start_s"] == pytest.approx(start)
    assert metrics["measurement_end_s"] == pytest.approx(end)
    assert metrics["load_real_power_W"] == pytest.approx(expected_v2 / 50, rel=1e-7)
    assert metrics["load_v_rms_V"] == pytest.approx(np.sqrt(expected_v2), rel=1e-7)
    assert metrics["voltage_harmonic_amplitude_V"]["h1"] == pytest.approx(expected_h1, rel=1e-7)
    assert table["amplitude"].iloc[0] == pytest.approx(expected_h1, rel=1e-7)
    assert table["measurement_cycles"].tolist() == [cycles] * 5
    assert table["window_start_s"].tolist() == [start] * 5
    assert set(metrics["voltage_harmonic_amplitude_V"]) == {"h1", "h2", "h3", "h4", "h5"}


@pytest.mark.parametrize("source_power", [0.0, 1e-40, 1.0])
def test_efficiency_distinguishes_undefined_from_finite_power_ratios(source_power):
    from pcd.core import Candidate, ControlState, EvaluationRequest, MetricSet, RawResult, Scenario
    from pcd.metrics import MetricLimitConstraint

    load_power = source_power / 2
    ac = pd.Series(
        {
            "voltage_re": 1.0,
            "voltage_im": 0.0,
            "current_re": -2 * source_power,
            "current_im": 0.0,
            "load_voltage_V_re": 1.0,
            "load_voltage_V_im": 0.0,
            "load_current_A_re": 2 * load_power,
            "load_current_A_im": 0.0,
        }
    )
    waveform = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "source_voltage_V": [1.0, 1.0],
            "current_A": [-source_power, -source_power],
            "load_voltage_V": [1.0, 1.0],
            "load_current_A": [load_power, load_power],
        }
    )
    request = EvaluationRequest(Candidate("point"), Scenario("condition"), ControlState())
    for metrics in (ac_power_flow(ac, "load_current_A"), power_flow(waveform, "load_current_A")):
        balance = component_loss_balance(
            {
                "source_real_power_W": source_power,
                "network_loss_W": source_power,
                "modeled_component_loss_W": source_power / 2,
            }
        )
        constraint = MetricLimitConstraint("transfer_efficiency", "min", 0).evaluate(
            request, RawResult("ok"), MetricSet(metrics)
        )
        if source_power == 0:
            assert balance["component_loss_balance_fraction_of_source"] is None
            assert metrics["transfer_efficiency"] is None
            assert "nonzero source real power" in metrics["transfer_efficiency_unavailable_reason"]
            assert not constraint.satisfied
            assert constraint.value is None
            assert constraint.margin is None
        else:
            assert balance["component_loss_balance_fraction_of_source"] == pytest.approx(0.5)
            assert metrics["transfer_efficiency"] == pytest.approx(0.5)
            assert constraint.satisfied


def test_saved_simulation_result_reproduces_rf_metrics(tmp_path):
    """Analysis consumes the canonical artifact, not solver-owned state."""

    frame = _sine_waveform(100.0, 2.0, load_current=1.5)
    result = SimulationResult(
        time_s=frame["time_s"].to_numpy(float),
        voltage_V=frame["voltage_V"].to_numpy(float),
        current_A=frame["current_A"].to_numpy(float),
        probes={
            "source_voltage_V": frame["source_voltage_V"].to_numpy(float),
            "i(Vload)": frame["i(Vload)"].to_numpy(float),
        },
    )
    in_memory = rf_port_metrics(result.as_frame(), 1e6, "i(Vload)")

    result.as_frame().to_csv(tmp_path / "waveform.csv", index=False)
    restored = load_waveform(tmp_path / "waveform.csv")
    persisted = rf_port_metrics(restored, 1e6, "i(Vload)")

    assert persisted.keys() == in_memory.keys()
    for name, expected in in_memory.items():
        if isinstance(expected, bool):
            assert persisted[name] is expected
        else:
            assert persisted[name] == pytest.approx(expected)


@pytest.mark.parametrize("column", ["i(Vload)", "current_A"])
def test_rf_port_metrics_require_every_used_signal_to_be_periodic(column):
    frame = _sine_waveform(100.0, 2.0, load_current=1.5)
    frame[column] *= 1.0 + 0.2 * frame["time_s"] * 1e6

    with pytest.raises(UnsettledMeasurementError) as error:
        rf_port_metrics(frame, 1e6, "i(Vload)")
    assert column in str(error.value)


def test_missing_current_does_not_hide_a_voltage_spike_from_settling():
    time = np.linspace(0.0, 5.0, 501)
    tone = np.sin(2 * np.pi * time)
    waveform = pd.DataFrame({"time_s": time, "voltage_V": tone, "load_current_A": tone})
    waveform.loc[450, ["voltage_V", "load_current_A"]] = [1000, np.nan]

    with pytest.raises(UnsettledMeasurementError, match="voltage_V"):
        periodic_operating_metrics(waveform, 1.0, "load_current_A", (), rf_measurement_options())


def test_rf_port_metrics_reject_a_stable_but_too_short_measurement():
    frame = _sine_waveform(100.0, 2.0, load_current=1.5)
    frame = frame.loc[frame["time_s"] <= 2.0e-6]

    with pytest.raises(UnsettledMeasurementError, match="available=2, required=3"):
        rf_port_metrics(frame, 1e6, "i(Vload)")


def test_total_reflection_gives_infinite_vswr_without_a_numpy_warning():
    """|gamma| = 1 is an open circuit, not a numerical accident."""

    import warnings

    frame = _ac_frame([1e7], [1 + 0j], [0j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = input_impedance(frame).iloc[0]
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

    assert time_average(signal, t) == pytest.approx(0.0, abs=1e-3)
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
    assert time_average(np.array([5.0]), np.array([0.0])) == 5.0
    assert time_average(np.array([]), np.array([])) == 0.0
    # A record with no elapsed time falls back to the plain mean.
    assert time_average(np.array([2.0, 4.0]), np.array([1e-9, 1e-9])) == 3.0


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
