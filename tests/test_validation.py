"""Case-level validation tests for actionable errors and engineering checks."""

from __future__ import annotations

import pytest

from pcd.api import run_case_study, simulate_case
from pcd.case import default_params
from pcd.cli import main
from pcd.inputs import parameter_spec
from pcd.netlist import compile_netlist
from pcd.simulation_input import resolve_simulation_case
from pcd.study_config import study_spec_from_case
from pcd.validation import validate_case


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _variable_case(make_case, spec):
    return make_case({"case_id": "vars", "source": {"type": "sine_voltage"}, "variables": {"x": spec}})


# --- variable rules --------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"choices": []}, "nonempty list"),
        ({"choices": [1, float("inf")]}, "finite"),
        ({"bounds": [1]}, "two-item list"),
        ({"bounds": ["a", "b"]}, "numeric"),
        ({"bounds": [10, 1]}, "finite and ordered"),
        ({"bounds": [float("nan"), 1]}, "finite and ordered"),
        ({"bounds": [1.2, 1.8], "type": "int"}, "contain an integer"),
        ({"bounds": [-1, 10], "scale": "log"}, "positive"),
    ],
)
def test_parameter_input_owns_domain_validation(spec, message):
    with pytest.raises(ValueError, match=message):
        parameter_spec(spec, "variables.x")


def test_preflight_reports_the_parameter_error_from_execution_input(make_case):
    case = _variable_case(make_case, {"bounds": [10, 1]})
    with pytest.raises(ValueError, match="finite and ordered") as error:
        default_params(case)

    report = validate_case(case)
    assert not report.ok
    assert [(issue.code, issue.message) for issue in report.issues] == [("parameter.invalid", str(error.value))]


def test_default_outside_a_valid_domain_is_an_input_warning(make_case):
    case = _variable_case(make_case, {"role": "design", "bounds": [1, 10], "default": 99})
    assert validate_case(case).ok
    assert codes(validate_case(case, strict=True)) == {"variable.default_out_of_bounds"}


@pytest.mark.parametrize(
    ("section", "field", "code"),
    [("target", "objective", "target.invalid_objective"), ("optimizer", "name", "optimizer.invalid")],
)
def test_removed_scoring_methods_fail_preflight_and_execution(rc_case, tmp_path, section, field, code):
    rc_case.data[section][field] = "removed_method"
    report = validate_case(rc_case, include_plan=True)
    assert codes(report) == {code}
    assert report.plan is None
    with pytest.raises(ValueError, match=f"{code}.*removed_method"):
        run_case_study(rc_case, tmp_path / "study", solver_override="test_fake")
    assert not (tmp_path / "study").exists()


def test_grid_plan_checks_the_builtin_domain_and_budget(rc_case):
    rc_case.data["optimizer"]["name"] = "grid"
    report = validate_case(rc_case, include_plan=True)
    assert report.plan is None
    assert "requires finite choices" in report.format_text()

    rc_case.data["circuit"]["variables"]["C1"] = {"choices": [1e-9, 2e-9], "default": 1e-9, "role": "design"}
    rc_case.data["run"]["trials"] = 1
    assert "requires exactly 2 trials" in validate_case(rc_case).format_text()
    rc_case.data["run"]["trials"] = 2
    assert validate_case(rc_case, strict=True).ok


def test_scoring_methods_do_not_gate_a_forward_simulation(make_case, tmp_path):
    case = make_case(
        {
            "metadata": {"workflow": "simulation"},
            "optimizer": {"name": "unused_optimizer"},
            "target": {"objective": "unused_objective"},
            "solver": {"name": "test_fake", "tran": {"stop_s": 2e-6}},
        }
    )
    assert validate_case(case, strict=True, include_plan=True).ok
    assert simulate_case(case, run_root=tmp_path / "run").status == "ok"


# --- solver rules ----------------------------------------------------------


def test_invalid_solver_input_is_reported_at_the_simulation_boundary(make_case):
    case = make_case({"source": {"type": "sine_voltage"}, "solver": {"name": "ngspice_cli", "timeout_s": "soon"}})
    [issue] = validate_case(case).issues
    assert issue.code == "simulation.invalid_input"
    assert "timeout_s" in issue.message


def test_unknown_solver_is_rejected_before_execution(make_case):
    case = make_case({"source": {"type": "sine_voltage"}, "solver": {"name": "missing_solver"}})
    assert "solver.unknown" in codes(validate_case(case))


def test_a_parameter_reference_is_valid_for_an_ac_point(make_case):
    case = make_case(
        {
            "case_id": "ac_point",
            "variables": {"rf_frequency_Hz": {"default": 13.56e6}},
            "source": {"type": "sine_voltage", "frequency_Hz": "rf_frequency_Hz"},
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": "rf_frequency_Hz"}},
        }
    )
    assert "simulation.invalid_input" not in codes(validate_case(case))


def test_invalid_rf_load_parameters_use_the_model_build_boundary(make_case):
    case = make_case(
        {
            "case_id": "load",
            "source": {"type": "sine_voltage"},
            "load": {"name": "impedance_point", "resistance_ohm": 20},
            "solver": {"name": "ngspice_cli", "ac": {}},
        }
    )
    report = validate_case(case)
    issues = [issue for issue in report.issues if issue.code == "model.invalid_input"]
    assert len(issues) == 1
    assert "reactance_ohm" in issues[0].message


@pytest.mark.parametrize(
    ("source_type", "ac", "message"),
    [
        ("unsupported", {"frequency_Hz": 13.56e6}, "unknown source type"),
        ("sine_voltage", {"sweep": "dec", "points": 10}, "requires a single AC frequency"),
        ("sine_voltage", {"frequency_Hz": 27.12e6}, "must equal the requested AC frequency"),
    ],
)
def test_preflight_preview_and_simulation_share_the_compiled_input(
    make_case, tmp_path, capsys, source_type, ac, message
):
    case = make_case(
        {
            "case_id": "point_scope",
            "source": {"type": source_type, "frequency_Hz": 13.56e6},
            "load": {
                "name": "impedance_point",
                "resistance_ohm": 20,
                "reactance_ohm": -80,
                "model_frequency_Hz": 13.56e6,
                "characterization": {"origin": "measured"},
            },
            "solver": {"ac": ac},
        }
    )

    report = validate_case(case, strict=True)
    assert not report.ok
    assert message in report.format_text()
    netlist = tmp_path / "preview.cir"
    with pytest.raises(SystemExit) as exit_status:
        main(["sim-netlist", str(case.path), "--out", str(netlist)])
    assert exit_status.value.code == 2
    assert message in capsys.readouterr().out
    with pytest.raises(ValueError, match=message):
        simulate_case(case, run_root=tmp_path / "runs")
    assert not netlist.exists()
    assert not (tmp_path / "runs").exists()


def test_equivalent_load_with_no_plane_label_keeps_its_explicit_terminals(make_case):
    frequency = 13.56e6
    case = make_case(
        {
            "case_id": "unlabelled_load",
            "source": {"type": "sine_voltage", "frequency_Hz": frequency},
            "load": {
                "name": "impedance_point",
                "ports": {"p": "electrode", "n": "return_node"},
                "resistance_ohm": 20,
                "reactance_ohm": -80,
                "model_frequency_Hz": frequency,
                "reference_plane": None,
                "characterization": {"origin": "test"},
            },
            "solver": {"ac": {"frequency_Hz": frequency}},
        }
    )
    assert validate_case(case, strict=True).ok
    params = default_params(case)
    simulation = resolve_simulation_case(case, params)
    _, simulation, deck = compile_netlist(case, params, simulation)
    assert "Xload electrode return_node load_model" in deck
    assert simulation.measurement.reference_plane == "load_ports"
    assert study_spec_from_case(case).metadata["reference_plane"] == "load_ports"


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


def test_malformed_source_containers_are_errors(make_case):
    assert "simulation.invalid_input" in codes(validate_case(make_case({"case_id": "s", "source": [1, 2]})))
    assert "simulation.invalid_input" in codes(validate_case(make_case({"case_id": "s", "sources": 1})))


def test_duplicate_source_names_use_the_simulation_boundary_diagnostic(make_case):
    case = make_case(
        {
            "case_id": "ambiguous_measurement",
            "sources": [
                {"type": "dc_voltage", "name": "Vsame"},
                {"type": "dc_voltage", "name": "Vsame"},
            ],
            "measurement": {"current_source": "Vsame"},
        }
    )

    assert "simulation.invalid_input" in codes(validate_case(case))


def test_measurement_current_source_must_name_a_structured_source(make_case):
    case = make_case(
        {
            "case_id": "unknown_source",
            "source": {"type": "sine_voltage", "name": "Vrf"},
            "measurement": {"current_source": "Vmissing"},
        }
    )

    assert "simulation.invalid_input" in codes(validate_case(case))


@pytest.mark.parametrize(
    "measurement",
    [
        {"reference_impedance_ohm": 0},
        {"reference_impedance_ohm": float("inf")},
        {"reference_impedance_ohm": "open"},
        {"periodic_cycles": 0},
        {"settling_comparisons": 1.5},
        {"settling_tolerance": -1},
        {"settling_tolerance": "tight"},
        {"harmonic_count": "many"},
    ],
)
def test_invalid_rf_measurement_options_are_rejected(make_case, measurement):
    case = make_case({"case_id": "bad_measurement", "source": {"type": "sine_voltage"}, "measurement": measurement})
    assert not validate_case(case).ok


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("netlist_mode", "automatic"),
        ("source_policy", "replace_by_node"),
    ],
)
def test_external_netlist_policies_are_owned_by_the_model_builder(make_case, field, value):
    case = make_case(
        {
            "circuit": {
                "builder": "from_netlist",
                "netlist_file": "external.cir",
                field: value,
            },
        }
    )

    report = validate_case(case)
    issues = [issue for issue in report.issues if issue.code == "model.invalid_input"]
    assert len(issues) == 1
    assert field in issues[0].message


@pytest.mark.parametrize(
    ("section", "value", "expected"),
    [
        ("circuit", [], "model.invalid_input"),
        ("solver", [], "simulation.invalid_input"),
        ("load", [], "simulation.invalid_input"),
        ("measurement", [], "simulation.invalid_input"),
        ("target", [], "target.not_mapping"),
        ("plugins", {}, "plugin.invalid"),
        ("plugins", [1], "plugin.invalid"),
    ],
)
def test_malformed_nested_sections_are_reported_without_crashing(make_case, section, value, expected):
    case = make_case({"case_id": "bad", "source": {"type": "sine_voltage", "frequency_Hz": 1e6}, section: value})

    assert expected in codes(validate_case(case))


def test_plugin_loading_and_solver_registration_are_validated_together(tmp_path, make_case):
    plugin = tmp_path / "solver_plugin.py"
    plugin.write_text("def broken(:\n", encoding="utf-8")
    case = make_case(
        {
            "case_id": "plugin",
            "source": {"type": "sine_voltage"},
            "plugins": [plugin.name],
            "solver": {"name": "validation_plugin_solver"},
        }
    )

    [issue] = validate_case(case).issues
    assert issue.code == "plugin.invalid"
    assert "SyntaxError" in issue.message
    assert plugin.name in issue.message

    plugin.write_text(
        "from pcd.sim_registry import register\n"
        "@register('solver', 'validation_plugin_solver')\n"
        "def solve(*args, **kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    assert validate_case(case).ok
    case.data.pop("plugins")
    [issue] = validate_case(case).issues
    assert issue.code == "solver.unknown"
    assert "undeclared plugin" in issue.message


def test_preflight_reports_the_waveform_reader_failure(make_case):
    case = make_case(
        {
            "case_id": "t",
            "source": {"type": "sine_voltage"},
            "target": {"waveform_file": "absent.csv"},
        }
    )
    [issue] = validate_case(case).issues
    assert issue.code == "target.invalid_waveform"
    assert "absent.csv" in issue.message


def test_scored_waveform_target_requires_a_file(make_case):
    case = make_case({"case_id": "t", "source": {"type": "sine_voltage"}, "target": {"objective": "waveform_l2"}})
    [issue] = validate_case(case).issues
    assert issue.code == "target.no_waveform"
    assert issue.level == "error"


def test_known_target_duration_mismatch_is_an_input_error(make_case):
    case = make_case(
        {
            "solver": {"tran": {"stop_s": 1e-6}},
            "target": {"objective": "waveform_l2", "waveform_file": "target.csv"},
        }
    )
    (case.base_dir / "target.csv").write_text("time_s,voltage_V\n0,0\n2e-6,1\n", encoding="utf-8")
    assert "target.waveform_outside_transient" in codes(validate_case(case))


def test_a_frequency_domain_objective_does_not_require_a_waveform_target(make_case):
    case = make_case({"case_id": "t", "source": {"type": "sine_voltage"}, "target": {"objective": "impedance_match"}})
    assert "target.no_waveform" not in codes(validate_case(case))


# --- strictness ------------------------------------------------------------


def test_strict_mode_promotes_warnings_to_failure(make_case):
    case = _variable_case(make_case, {"bounds": [1, 10], "default": 1})
    normal = validate_case(case).to_dict()
    strict = validate_case(case, strict=True).to_dict()
    assert normal["ok"] is True
    assert strict["ok"] is False
    assert strict["issues"] == normal["issues"]
    assert [issue["code"] for issue in strict["issues"]] == ["variable.implicit_design"]


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
    assert any("requires an attached load" in item.message for item in report.issues)


def test_auto_metering_with_a_load_is_accepted(make_case):
    case = make_case(
        {
            "case_id": "with_load",
            "variables": {"load_model": {"role": "fixed", "default": "resistor"}},
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 1, "frequency_Hz": 1e6},
            "load": {"name_variable": "load_model", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-9, "stop_s": 1e-6}},
        }
    )
    assert validate_case(case).ok


def test_a_fractional_stop_is_valid_when_the_measurement_window_is_long_enough(make_case):

    case = make_case(
        {
            "case_id": "partial",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 600, "frequency_Hz": 13.56e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-10, "stop_s": 3.0e-7}},
            "target": {"objective": "rf_load"},
        }
    )
    assert not [i for i in validate_case(case).issues if i.code == "solver.insufficient_rf_cycles"]


def test_a_short_record_only_warns_when_scoring_requires_a_settled_window(make_case):

    case = make_case(
        {
            "case_id": "too_short",
            "source": {"type": "sine_voltage", "name": "Vsrc", "amplitude_V": 600, "frequency_Hz": 13.56e6},
            "load": {"name": "resistor", "R_ohm": 50},
            "measurement": {"load_current": "auto"},
            "solver": {"name": "ngspice_cli", "tran": {"step_s": 1e-11, "stop_s": 0.5 / 13.56e6}},
        }
    )
    assert validate_case(case, strict=True).ok
    case.data["target"] = {"objective": "rf_load"}
    [warning] = validate_case(case).issues
    assert warning.code == "solver.insufficient_rf_cycles"
    assert "at least 3 cycles" in warning.message

    case.data["source"] = {"type": "dc_voltage", "name": "Vsrc", "voltage_V": 1}
    assert "target.rf_load_without_fundamental" in codes(validate_case(case))

    case.data["target"]["fundamental_Hz"] = "$operating_f"
    case.data["study"] = {"scenarios": [{"id": "f1", "values": {"operating_f": 13.56e6}}]}
    assert validate_case(case, strict=True).ok
