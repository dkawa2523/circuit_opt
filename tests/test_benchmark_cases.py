"""The decision benchmarks must keep their physical question and controls distinct."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from pcd.case import load_case
from pcd.study_config import study_spec_from_case
from pcd.validation import validate_case

CASES = Path(__file__).resolve().parents[1] / "bench" / "cases"
FIXED = load_case(CASES / "match_fixed_nominal.yaml")
LIMITED = load_case(CASES / "match_limited_tuner.yaml")
FULL = load_case(CASES / "match_full_tuner.yaml")
SEARCH = load_case(CASES / "match_discrete_hardware_search.yaml")
ROLE_FACTORIAL = load_case(CASES / "role_factorial_search.yaml")
COMPONENT_CORNERS = load_case(CASES / "component_value_corner_stress.yaml")
CCP_FREQUENCY = load_case(CASES / "ccp_lumped_frequency_conformance.yaml")
ICP_FREQUENCY = load_case(CASES / "icp_transformer_frequency_conformance.yaml")
TOPOLOGY_CASES = tuple(load_case(path) for path in sorted(CASES.glob("topology_*_golden.yaml")))
REFERENCE_CASES = tuple(load_case(path) for path in sorted(CASES.glob("reference_plane_*.yaml")))
EXPECTATIONS = yaml.safe_load((CASES.parent / "expectations.yaml").read_text(encoding="utf-8"))


def _scenario_map(case):
    return {scenario.scenario_id: dict(scenario.values) for scenario in study_spec_from_case(case).scenarios}


def test_every_core_case_has_exactly_one_separate_expectation_entry():
    assert set(EXPECTATIONS) - {"_suite"} == {path.name for path in CASES.glob("*.yaml")}


def test_benchmark_cases_are_strictly_valid_and_share_one_scenario_envelope():
    reference = _scenario_map(FIXED)
    for case in (FIXED, LIMITED, FULL):
        assert validate_case(case, strict=True).ok
        assert _scenario_map(case) == reference


def test_the_synthetic_envelope_declares_only_observable_electrical_corners():
    scenarios = _scenario_map(FULL)
    nominal = scenarios["nominal"]
    assert set(nominal) == {"load_resistance_ohm", "load_reactance_ohm"}
    assert scenarios["low_R_nominal_X"]["load_reactance_ohm"] == pytest.approx(nominal["load_reactance_ohm"])
    assert scenarios["high_R_nominal_X"]["load_reactance_ohm"] == pytest.approx(nominal["load_reactance_ohm"])
    assert scenarios["low_R_nominal_X"]["load_resistance_ohm"] < nominal["load_resistance_ohm"]
    assert scenarios["high_R_nominal_X"]["load_resistance_ohm"] > nominal["load_resistance_ohm"]
    assert FULL.data["load"]["name"] == "impedance_point"


def test_cases_form_a_negative_negative_positive_control_sequence():
    fixed_study, limited_study, full_study = (case.data["study"] for case in (FIXED, LIMITED, FULL))
    assert "controls" not in fixed_study
    assert fixed_study["design_variables"] == ["C1", "L1", "C2"]
    assert limited_study["design_variables"] == full_study["design_variables"] == ["L1"]

    limited_c1 = set(limited_study["controls"]["variables"]["C1"]["values"])
    full_c1 = set(full_study["controls"]["variables"]["C1"]["values"])
    assert limited_c1 < full_c1
    assert EXPECTATIONS[FIXED.path.name]["expected"]["feasible"] is False
    assert EXPECTATIONS[LIMITED.path.name]["expected"]["feasible"] is False
    assert EXPECTATIONS[FULL.path.name]["expected"]["feasible"] is True


def test_benchmark_inputs_use_the_public_schema_and_keep_expectations_separate():
    for case in (FIXED, LIMITED, FULL, SEARCH):
        assert case.authored_data["schema"] == "pcd.rf.v1"
        assert "benchmark" not in case.authored_data
        assert case.resolved_plan is not None


def test_discrete_search_benchmark_has_a_complete_known_hardware_shortlist():
    assert validate_case(SEARCH, strict=True).ok
    assert SEARCH.resolved_plan is not None
    assert SEARCH.resolved_plan["execution"]["optimizer"] == "grid"
    assert SEARCH.resolved_plan["execution"]["trials"] == 3
    expected = EXPECTATIONS[SEARCH.path.name]["expected"]
    assert expected["n_candidates"] == 3
    assert expected["feasible_candidates"] == 1


def _parallel(*impedances: complex) -> complex:
    return 1.0 / sum(1.0 / impedance for impedance in impedances)


def _analytic_topology_impedance(case) -> complex:
    authored = case.authored_data
    omega = 2.0 * math.pi * float(authored["frequency_Hz"])
    network = authored["network"]
    fixed = network["fixed"]
    load = authored["load"]
    load_z = complex(load["resistance_ohm"], load["reactance_ohm"])
    if network["type"] == "l_match":
        output = _parallel(load_z, 1.0 / (1j * omega * fixed["C1"]))
        return 1j * omega * fixed["L1"] + output

    output_impedances = [load_z, 1.0 / (1j * omega * fixed["C2"])]
    if network["type"] == "pi_match_harmonic":
        output_impedances.append(1j * omega * fixed["Lh"] + 1.0 / (1j * omega * fixed["Ch"]))
    series_branch = 1j * omega * fixed["L1"] + _parallel(*output_impedances)
    return _parallel(series_branch, 1.0 / (1j * omega * fixed["C1"]))


def test_each_public_network_topology_has_an_independent_complex_impedance_golden():
    assert {case.authored_data["network"]["type"] for case in TOPOLOGY_CASES} == {
        "l_match",
        "pi_match",
        "pi_match_harmonic",
    }
    for case in TOPOLOGY_CASES:
        assert validate_case(case, strict=True).ok
        expected = EXPECTATIONS[case.path.name]["expected"]["input_impedance_ohm"]["nominal"]
        assert _analytic_topology_impedance(case) == pytest.approx(complex(*expected), rel=1e-12, abs=1e-12)


def test_ccp_frequency_case_golden_is_derived_from_the_declared_effective_port_only():
    assert validate_case(CCP_FREQUENCY, strict=True).ok
    authored = CCP_FREQUENCY.authored_data
    load = authored["load"]["parameters"]
    network = authored["network"]["fixed"]
    expected = EXPECTATIONS[CCP_FREQUENCY.path.name]["expected"]["input_impedance_ohm"]
    for condition in authored["conditions"]:
        frequency = float(condition["frequency_Hz"])
        omega = 2.0 * math.pi * frequency
        load_z = complex(
            load["R_eff_ohm"],
            omega * load["L_eff_H"] - 1.0 / (omega * load["C_sheath_eq_F"]),
        )
        output = _parallel(load_z, 1.0 / (1j * omega * network["C1"]))
        input_z = 1j * omega * network["L1"] + output
        assert input_z == pytest.approx(complex(*expected[condition["id"]]), rel=1e-12, abs=1e-12)


def test_icp_frequency_case_golden_is_derived_from_the_declared_terminal_equation_only():
    assert validate_case(ICP_FREQUENCY, strict=True).ok
    authored = ICP_FREQUENCY.authored_data
    load = authored["load"]["parameters"]
    network = authored["network"]["fixed"]
    expected = EXPECTATIONS[ICP_FREQUENCY.path.name]["expected"]["input_impedance_ohm"]
    for condition in authored["conditions"]:
        frequency = float(condition["frequency_Hz"])
        omega = 2.0 * math.pi * frequency
        series = complex(float(load["R_coil_ohm"]), omega * float(load["L_coil_H"])) + (
            omega**2
            * float(load["reflected_inductance_H"])
            / complex(float(load["secondary_damping_rate_rad_s"]), omega)
        )
        load_z = 1.0 / (1.0 / series + 1j * omega * float(load["C_parallel_F"]))
        output = _parallel(load_z, 1.0 / (1j * omega * float(network["C1"])))
        input_z = 1j * omega * float(network["L1"]) + output
        assert input_z == pytest.approx(complex(*expected[condition["id"]]), rel=1e-12, abs=1e-12)


def test_role_factorial_case_exercises_candidate_scenario_and_control_together():
    assert validate_case(ROLE_FACTORIAL, strict=True).ok
    assert ROLE_FACTORIAL.authored_data["schema"] == "pcd.rf.v1"
    assert ROLE_FACTORIAL.resolved_plan is not None
    plan = ROLE_FACTORIAL.resolved_plan
    assert plan["execution"]["trials"] == 2
    study = plan["case"]["study"]
    assert study["design_variables"] == ["L1"]
    assert set(study["scenario_table"]["values"]) == {"load_resistance_ohm", "load_reactance_ohm"}
    assert study["controls"]["variables"] == {"C1": {"values": [2.0e-11, 8.0e-11]}}
    assert EXPECTATIONS[ROLE_FACTORIAL.path.name]["expected"]["n_evaluations"] == 2 * 2 * 2


def test_component_value_corners_are_full_factorial_scenarios_with_independent_goldens():
    assert validate_case(COMPONENT_CORNERS, strict=True).ok
    assert COMPONENT_CORNERS.authored_data["schema"] == "case_yaml.v1"
    assert COMPONENT_CORNERS.data["study"]["design_variables"] == ["C1_nominal", "L1_nominal", "C2_nominal"]

    scenarios = _scenario_map(COMPONENT_CORNERS)
    assert len(scenarios) == 2**3
    nominal = {
        name: float(spec["default"])
        for name, spec in COMPONENT_CORNERS.data["variables"].items()
        if name.endswith("_nominal")
    }
    for name in ("C1_factor", "L1_factor", "C2_factor"):
        assert {float(values[name]) for values in scenarios.values()} == {0.85, 1.15}

    authored = COMPONENT_CORNERS.authored_data
    frequency = float(authored["source"]["frequency_Hz"])
    omega = 2.0 * math.pi * frequency
    load_z = complex(float(authored["load"]["resistance_ohm"]), float(authored["load"]["reactance_ohm"]))
    expected = EXPECTATIONS[COMPONENT_CORNERS.path.name]["expected"]["input_impedance_ohm"]
    for scenario_id, values in scenarios.items():
        c1_actual = nominal["C1_nominal"] * float(values["C1_factor"])
        l1_actual = nominal["L1_nominal"] * float(values["L1_factor"])
        c2_actual = nominal["C2_nominal"] * float(values["C2_factor"])
        output = _parallel(load_z, 1.0 / (1j * omega * c2_actual))
        series = 1j * omega * l1_actual + output
        input_z = _parallel(series, 1.0 / (1j * omega * c1_actual))
        assert input_z == pytest.approx(complex(*expected[scenario_id]), rel=1e-12, abs=1e-12)

    assert EXPECTATIONS[COMPONENT_CORNERS.path.name]["expected"]["n_evaluations"] == 2**3


def test_reference_plane_cases_encode_one_equivalence_and_one_double_count_negative_control():
    assert len(REFERENCE_CASES) == 3
    for case in REFERENCE_CASES:
        assert validate_case(case, strict=True).ok

    by_name = {case.path.name: case.authored_data for case in REFERENCE_CASES}
    explicit = by_name["reference_plane_fixture_explicit.yaml"]
    embedded = by_name["reference_plane_fixture_embedded.yaml"]
    doubled = by_name["reference_plane_fixture_double_counted.yaml"]
    omega = 2.0 * math.pi * float(explicit["source"]["frequency_Hz"])
    fixture_l = next(item["value"] for item in explicit["circuit"]["components"] if item["ref"] == "Lfixture")
    fixture_r = next(item["value"] for item in explicit["circuit"]["components"] if item["ref"] == "Rfixture")
    plasma_z = complex(explicit["load"]["resistance_ohm"], explicit["load"]["reactance_ohm"])
    embedded_z = complex(embedded["load"]["resistance_ohm"], embedded["load"]["reactance_ohm"])
    doubled_z = complex(doubled["load"]["resistance_ohm"], doubled["load"]["reactance_ohm"])

    fixture_z = fixture_r + 1j * omega * fixture_l
    assert plasma_z + fixture_z == pytest.approx(embedded_z, abs=1e-12)
    assert doubled_z == embedded_z
    assert doubled_z + fixture_z != pytest.approx(embedded_z, abs=1e-6)
