"""Characterization tests for case validation.

``_validate_variables`` was the second-most complex function in the package
(CC 20) and every branch of it emits a distinct, user-visible diagnostic code.
These tests pin one code per branch so the split into per-rule helpers cannot
drop or rename a diagnostic.
"""

from __future__ import annotations

import pytest

from pcd.validation import validate_case


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _variable_case(make_case, spec):
    return make_case({"case_id": "vars", "source": {"type": "sine_voltage"}, "variables": {"x": spec}})


# --- variable rules --------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"choices": [], "default": 1}, "variable.empty_choices"),
        ({"choices": "not-a-list"}, "variable.empty_choices"),
        ({"choices": ["a", "b"], "default": "c"}, "variable.default_not_in_choices"),
        ({"bounds": [1]}, "variable.invalid_bounds"),
        ({"bounds": "nope"}, "variable.invalid_bounds"),
        ({"bounds": ["a", "b"]}, "variable.non_numeric_bounds"),
        ({"bounds": [10, 1]}, "variable.bounds_reversed"),
        ({"bounds": [-1, 10], "scale": "log"}, "variable.log_bounds_non_positive"),
        ({"bounds": [1, 10], "default": 99}, "variable.default_out_of_bounds"),
    ],
)
def test_each_variable_rule_emits_its_own_code(make_case, spec, expected):
    assert expected in codes(validate_case(_variable_case(make_case, spec)))


def test_a_well_formed_variable_produces_no_variable_diagnostics(make_case):
    case = _variable_case(make_case, {"bounds": [1.0, 10.0], "scale": "log", "default": 5.0})
    assert not {c for c in codes(validate_case(case)) if c.startswith("variable.")}


def test_non_numeric_default_with_bounds_is_tolerated(make_case):
    """A categorical default alongside bounds must not crash the validator."""

    case = _variable_case(make_case, {"bounds": [1, 10], "default": "auto"})
    assert "variable.default_out_of_bounds" not in codes(validate_case(case))


def test_scalar_variable_spec_is_normalized_not_rejected(make_case):
    """`variable_specs` wraps a bare scalar as {"default": value}."""

    case = make_case({"case_id": "scalar", "source": {"type": "sine_voltage"}, "variables": {"x": 5}})
    assert "variable.spec_not_mapping" not in codes(validate_case(case))


# --- solver rules ----------------------------------------------------------


@pytest.mark.parametrize(
    ("solver", "expected"),
    [
        ({"name": "ngspice_cli", "tran": "nope"}, "solver.tran_not_mapping"),
        ({"name": "ngspice_cli", "tran": {"step_s": "x", "stop_s": 1}}, "solver.tran_non_numeric"),
        ({"name": "ngspice_cli", "tran": {"step_s": -1, "stop_s": 1}}, "solver.tran_non_positive"),
        ({"name": "ngspice_cli", "tran": {"step_s": 10, "stop_s": 1}}, "solver.step_exceeds_stop"),
        ({"name": "ngspice_cli", "timeout_s": 0}, "solver.timeout_non_positive"),
        ({"name": "ngspice_cli", "timeout_s": "soon"}, "solver.timeout_non_numeric"),
    ],
)
def test_each_solver_rule_emits_its_own_code(make_case, solver, expected):
    case = make_case({"case_id": "solver", "source": {"type": "sine_voltage"}, "solver": solver})
    assert expected in codes(validate_case(case))


@pytest.mark.parametrize("name", ["dummy", "missing_solver"])
def test_unknown_solver_is_rejected_before_execution(make_case, name):
    case = make_case({"case_id": "solver", "source": {"type": "sine_voltage"}, "solver": {"name": name}})

    assert "solver.unknown" in codes(validate_case(case))


def test_removed_dummy_solver_is_rejected_even_when_a_plugin_is_present(make_case):
    case = make_case(
        {
            "case_id": "solver",
            "source": {"type": "sine_voltage"},
            "plugins": [__file__],
            "solver": {"name": "dummy"},
        }
    )

    assert "solver.unknown" in codes(validate_case(case))


@pytest.mark.parametrize(
    ("ac", "expected"),
    [
        ("bad", "solver.ac_not_mapping"),
        ({"sweep": "random"}, "solver.ac_invalid_sweep"),
        ({"points": "many"}, "solver.ac_non_numeric"),
        ({"points": 0, "start_Hz": 2e6, "stop_Hz": 1e6}, "solver.ac_invalid_range"),
        ({"frequency_Hz": 0}, "solver.ac_invalid_range"),
        ({"frequency_Hz": []}, "solver.ac_non_numeric"),
        ({"frequency_Hz": "rf_frequency_Hz", "points": 1}, "solver.ac_point_conflict"),
    ],
)
def test_each_ac_rule_emits_its_own_code(make_case, ac, expected):
    case = make_case({"case_id": "ac", "source": {"type": "sine_voltage"}, "solver": {"name": "ngspice_cli", "ac": ac}})
    assert expected in codes(validate_case(case))


def test_a_parameter_reference_is_valid_for_an_ac_point(make_case):
    case = make_case(
        {
            "case_id": "ac_point",
            "variables": {"rf_frequency_Hz": {"default": 13.56e6}},
            "source": {"type": "sine_voltage", "frequency_Hz": "rf_frequency_Hz"},
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": "rf_frequency_Hz"}},
        }
    )
    assert not {"solver.ac_non_numeric", "solver.ac_invalid_range"} & codes(validate_case(case))


def test_rf_load_models_require_parameters_reference_plane_and_origin(make_case):
    case = make_case(
        {
            "case_id": "load",
            "source": {"type": "sine_voltage"},
            "load": {"name": "impedance_point", "resistance_ohm": 20},
            "solver": {"name": "ngspice_cli", "ac": {}},
        }
    )
    found = codes(validate_case(case))
    assert {
        "load.missing_parameters",
        "load.missing_reference_plane",
        "load.missing_characterization",
    } <= found


def test_impedance_point_rejects_a_broadband_ac_sweep(make_case):
    case = make_case(
        {
            "case_id": "point_sweep",
            "source": {"type": "sine_voltage", "frequency_Hz": 13.56e6},
            "load": {
                "name": "impedance_point",
                "resistance_ohm": 20,
                "reactance_ohm": -80,
                "model_frequency_Hz": 13.56e6,
                "reference_plane": "electrode",
                "characterization": {"origin": "measured"},
            },
            "solver": {"name": "ngspice_cli", "ac": {"sweep": "dec", "points": 10}},
        }
    )

    assert "load.impedance_point_requires_ac_point" in codes(validate_case(case))


def test_rf_load_metric_requires_transient_and_load_current(make_case):
    case = make_case(
        {
            "case_id": "rf_metric",
            "source": {"type": "sine_voltage"},
            "solver": {"name": "ngspice_cli", "ac": {}},
            "target": {"objective": "rf_load"},
        }
    )
    assert {"target.rf_load_without_tran", "target.rf_load_without_current"} <= codes(validate_case(case))


# --- source, plugin and target rules ---------------------------------------


def test_missing_source_is_a_warning(make_case):
    assert "case.no_source" in codes(validate_case(make_case({"case_id": "nosrc"})))


def test_malformed_source_containers_are_errors(make_case):
    assert "case.source_not_mapping" in codes(validate_case(make_case({"case_id": "s", "source": [1, 2]})))
    assert "case.sources_not_list" in codes(validate_case(make_case({"case_id": "s", "sources": {"a": 1}})))


def test_missing_plugin_is_an_error(make_case):
    case = make_case({"case_id": "p", "source": {"type": "sine_voltage"}, "plugins": ["nope.py"]})
    assert "plugin.not_found" in codes(validate_case(case))


def test_missing_target_waveform_is_an_error(make_case):
    case = make_case(
        {
            "case_id": "t",
            "source": {"type": "sine_voltage"},
            "target": {"waveform_file": "absent.csv"},
        }
    )
    assert "target.waveform_not_found" in codes(validate_case(case))


def test_target_without_waveform_file_is_a_warning(make_case):
    case = make_case({"case_id": "t", "source": {"type": "sine_voltage"}, "target": {"objective": "waveform_l2"}})
    assert "target.no_waveform" in codes(validate_case(case))


def test_a_frequency_domain_objective_does_not_require_a_waveform_target(make_case):
    case = make_case({"case_id": "t", "source": {"type": "sine_voltage"}, "target": {"objective": "impedance_match"}})
    assert "target.no_waveform" not in codes(validate_case(case))


# --- strictness ------------------------------------------------------------


def test_strict_mode_promotes_warnings_to_failure(make_case):
    case = make_case({"case_id": "w"})
    assert validate_case(case, strict=False).ok is True
    assert validate_case(case, strict=True).ok is False


def test_report_serialization_round_trips(make_case):
    report = validate_case(make_case({"case_id": "w"}))
    payload = report.to_dict()
    assert payload["ok"] == report.ok
    assert len(payload["issues"]) == len(report.issues)
    assert report.format_text()


def test_auto_metering_without_a_load_is_an_error(make_case):
    """The ammeter goes in series with the load, so there has to be one."""

    case = make_case(
        {
            "case_id": "no_load",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-9, "stop_s": 1e-6}},
        }
    )
    report = validate_case(case)
    assert not report.ok
    assert any(item.code == "measurement.auto_meter_without_load" for item in report.issues)


def test_auto_metering_with_a_load_is_accepted(make_case):
    case = make_case(
        {
            "case_id": "with_load",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-9, "stop_s": 1e-6}},
        }
    )
    assert not [i for i in validate_case(case).issues if i.code == "measurement.auto_meter_without_load"]


def test_a_fractional_stop_is_valid_when_the_measurement_window_is_long_enough(make_case):

    case = make_case(
        {
            "case_id": "partial",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 600, "frequency_Hz": 13.56e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-10, "stop_s": 3.0e-7}},
        }
    )
    assert not [i for i in validate_case(case).issues if i.code == "solver.insufficient_rf_cycles"]


def test_a_record_shorter_than_the_measurement_window_is_reported(make_case):

    case = make_case(
        {
            "case_id": "too_short",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 600, "frequency_Hz": 13.56e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-11, "stop_s": 0.5 / 13.56e6}},
        }
    )
    warnings = [i for i in validate_case(case).issues if i.code == "solver.insufficient_rf_cycles"]
    assert len(warnings) == 1
    assert "at least 3 cycles" in warnings[0].message


def test_the_cycle_check_is_skipped_without_a_power_measurement(make_case):
    """Only cases asking for a power figure care about whole cycles."""

    case = make_case(
        {
            "case_id": "no_power",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 600, "frequency_Hz": 13.56e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-10, "stop_s": 3.0e-7}},
        }
    )
    assert not [i for i in validate_case(case).issues if i.code == "solver.insufficient_rf_cycles"]
