"""Parameter ownership is resolved once before simulation or search."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcd.case import default_params, load_case
from pcd.problem import ParameterRole, project_candidate_case, resolve_parameter_set

CASES = Path(__file__).resolve().parents[1] / "bench" / "cases"


def test_advanced_parameters_have_one_explicit_owner(make_case):
    case = make_case(
        {
            "case_id": "roles",
            "variables": {
                "fixture_C": {"default": 10e-12},
                "probe_delay": {"role": "calibration", "default": 2e-9},
                "matching_C": {"bounds": [10e-12, 1e-9], "default": 100e-12},
                "plasma_R": {"role": "latent", "bounds": [1.0, 100.0], "default": 20.0},
                "pressure": {"default": 5.0},
            },
            "study": {
                "scenarios": [{"id": "low", "values": {"pressure": 3.0}}],
                "controls": {"variables": {"tuner_C": {"values": [20e-12, 40e-12]}}},
            },
        }
    )

    parameters = resolve_parameter_set(case)

    assert parameters.names(ParameterRole.FIXED) == ("fixture_C",)
    assert parameters.names(ParameterRole.CALIBRATION) == ("probe_delay",)
    assert parameters.names(ParameterRole.DESIGN) == ("matching_C",)
    assert parameters.names(ParameterRole.LATENT) == ("plasma_R",)
    assert parameters.names(ParameterRole.OPERATING) == ("pressure",)
    assert parameters.names(ParameterRole.CONTROL) == ("tuner_C",)


def test_public_fixed_search_operating_and_control_roles_remain_distinct():
    fixed = resolve_parameter_set(load_case(CASES / "match_full_tuner.yaml"))
    searched = resolve_parameter_set(load_case(CASES / "match_discrete_hardware_search.yaml"))

    assert fixed.names(ParameterRole.FIXED) == ("L1",)
    assert fixed.names(ParameterRole.DESIGN) == ()
    assert fixed.names(ParameterRole.OPERATING) == ("load_resistance_ohm", "load_reactance_ohm")
    assert fixed.names(ParameterRole.CONTROL) == ("C1", "C2")
    assert searched.names(ParameterRole.FIXED) == ("L1", "C2")
    assert searched.names(ParameterRole.DESIGN) == ("C1",)


def test_design_projection_excludes_fixed_and_latent_values(make_case):
    case = make_case(
        {
            "case_id": "projection",
            "variables": {
                "fixed_R": {"default": 50.0},
                "design_C": {"bounds": [1e-12, 1e-9], "default": 1e-10},
                "plasma_R": {"role": "latent", "bounds": [1.0, 100.0], "default": 20.0},
            },
        }
    )

    projected = project_candidate_case(case)

    assert set(projected.data["variables"]) == {"design_C"}
    assert default_params(case) == {"fixed_R": 50.0, "design_C": 1e-10, "plasma_R": 20.0}


def test_conflicting_role_assignments_fail_at_the_input_boundary(make_case):
    both_runtime_roles = make_case(
        {
            "case_id": "conflict",
            "study": {
                "scenarios": [{"id": "one", "values": {"x": 1.0}}],
                "controls": {"defaults": {"x": 1.0}},
            },
        },
        name="runtime-conflict.yaml",
    )
    with pytest.raises(ValueError, match=r"x.*operating.*control"):
        resolve_parameter_set(both_runtime_roles)

    declared_conflict = make_case(
        {
            "case_id": "declared_conflict",
            "variables": {"x": {"role": "latent", "default": 1.0}},
            "study": {"design_variables": ["x"]},
        },
        name="declared-conflict.yaml",
    )
    with pytest.raises(ValueError, match=r"latent.*design"):
        resolve_parameter_set(declared_conflict)


def test_invalid_roles_and_study_shapes_fail_during_role_resolution(make_case):
    bad_role = make_case(
        {"case_id": "bad_role", "variables": {"x": {"role": "optimizer", "default": 1.0}}},
        name="bad-role.yaml",
    )
    with pytest.raises(ValueError, match="role must be one of"):
        resolve_parameter_set(bad_role)

    bad_study = make_case({"case_id": "bad_study", "study": []}, name="bad-study.yaml")
    with pytest.raises(ValueError, match="study must be a mapping"):
        resolve_parameter_set(bad_study)

    empty_name = make_case(
        {"case_id": "empty_name", "variables": {"": {"default": 1.0}}},
        name="empty-name.yaml",
    )
    with pytest.raises(ValueError, match="name must not be empty"):
        resolve_parameter_set(empty_name)


def test_design_projection_removes_variables_from_every_source(make_case):
    case = make_case(
        {
            "case_id": "source_roles",
            "sources": [
                {
                    "type": "voltage_sine",
                    "variables": {
                        "fixed_amplitude": {"default": 10.0},
                        "design_frequency": {"bounds": [1e6, 2e6], "default": 1.5e6},
                    },
                }
            ],
        }
    )

    projected = project_candidate_case(case)

    assert projected.data["variables"] == {"design_frequency": {"bounds": [1e6, 2e6], "default": 1.5e6}}
    assert "variables" not in projected.data["sources"][0]


def test_latent_candidate_projection_is_separate_from_design(make_case):
    case = make_case(
        {
            "case_id": "identify",
            "variables": {
                "design_C": {"bounds": [1e-12, 1e-9], "default": 1e-10},
                "plasma_R": {"role": "latent", "bounds": [1.0, 100.0], "default": 20.0},
            },
            "study": {"candidate_role": "latent"},
        }
    )

    projected = project_candidate_case(case)

    assert set(projected.data["variables"]) == {"plasma_R"}
    assert projected.data["study"]["candidate_role"] == "latent"
