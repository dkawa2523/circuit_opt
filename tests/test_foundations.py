"""Small shared helpers: cases, artifacts, search sampling, and SPICE values."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcd.artifacts import write_json
from pcd.case import (
    NO_SOURCE_WARNING,
    case_warnings,
    load_case,
    variable_specs,
)
from pcd.core.spaces import grid_values
from pcd.search import sample_param
from pcd.spice import is_spice_literal, resolve_value, spice_value

# Component values span many decades in this domain: pF to mH, mΩ to GΩ.
component_values = st.floats(min_value=1e-15, max_value=1e12, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=2**32 - 1)
PROPERTY = settings(max_examples=200, deadline=None)


# --- SPICE value rendering -------------------------------------------------


def test_values_render_as_literals_or_parameter_references():
    assert spice_value("1k") == "1k"
    assert spice_value("C1") == "{C1}"
    assert spice_value("$C1") == "{C1}"
    assert spice_value("{C1}") == "{C1}"


@pytest.mark.property
@PROPERTY
@given(value=component_values)
def test_rendering_a_component_value_keeps_12_significant_digits(value: float) -> None:
    """``spice_value`` formats with %.12g, so the netlist carries 12 digits.

    The tolerance *is* that contract: tighter would assert precision the format
    does not provide, looser would hide a real precision regression.
    """

    assert float(spice_value(value)) == pytest.approx(value, rel=1e-11)


@pytest.mark.property
@PROPERTY
@given(value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12))
def test_finite_numbers_are_spice_literals(value: float) -> None:
    assert is_spice_literal(value)


def test_non_finite_numbers_are_never_spice_literals():
    """NaN/inf must not reach a netlist as a bare number."""

    for value in (float("nan"), float("inf"), float("-inf")):
        assert not is_spice_literal(value)


# --- parameter references --------------------------------------------------


def test_parameter_references_resolve_from_params():
    params = {"freq": 13.56e6}
    assert resolve_value("freq", params) == 13.56e6
    assert resolve_value("$freq", params) == 13.56e6
    assert resolve_value("unknown", params, 1.0) == "unknown"
    assert resolve_value(None, params, 2.0) == 2.0


# --- design variable discovery ---------------------------------------------


def test_variables_are_collected_from_every_supported_section(make_case):
    case = make_case(
        {
            "case_id": "vars",
            "variables": {"a": {"default": 1}},
            "source": {"variables": {"b": {"default": 2}}},
            "circuit": {"variables": {"c": {"default": 3}}},
            "load": {"variables": {"d": {"default": 4}}},
        }
    )
    assert set(variable_specs(case)) == {"a", "b", "c", "d"}


def test_a_scalar_variable_is_normalized_to_a_default(make_case):
    case = make_case({"case_id": "scalar", "variables": {"x": 5}})
    assert variable_specs(case)["x"] == {"default": 5}


# --- sampling --------------------------------------------------------------


@pytest.mark.property
@PROPERTY
@given(lo=st.floats(min_value=1e-12, max_value=1e6), width=st.floats(min_value=1.0, max_value=1e6), seed=seeds)
def test_log_sampling_stays_inside_the_declared_bounds(lo: float, width: float, seed: int) -> None:
    hi = lo * width
    value = sample_param(np.random.default_rng(seed), {"bounds": [lo, hi], "scale": "log"})
    assert lo <= value <= hi
    assert math.isfinite(value)


@pytest.mark.property
@PROPERTY
@given(lo=st.integers(min_value=-1000, max_value=1000), span=st.integers(min_value=0, max_value=1000), seed=seeds)
def test_integer_sampling_is_inclusive_of_both_bounds(lo: int, span: int, seed: int) -> None:
    value = sample_param(np.random.default_rng(seed), {"bounds": [lo, lo + span], "type": "int"})
    assert isinstance(value, int)
    assert lo <= value <= lo + span


categorical_choices = st.lists(
    st.one_of(st.text(min_size=1, max_size=12), st.integers(), st.floats(allow_nan=False)),
    min_size=1,
    max_size=8,
    unique=True,
)


@pytest.mark.property
@PROPERTY
@given(choices=categorical_choices, seed=seeds)
def test_categorical_sampling_returns_the_declared_object_unchanged(choices: list[object], seed: int) -> None:
    """No dtype coercion: a mixed list like [0, "auto"] must survive intact."""

    value = sample_param(np.random.default_rng(seed), {"choices": choices})
    assert value in choices
    assert type(value) in {type(c) for c in choices}


def test_categorical_sampling_is_reproducible_for_a_seed():
    """Indexing draws from the same RNG stream that rng.choice used."""

    spec = {"choices": ["l_match", "pi_match", "pi_match_harmonic"]}
    first = [sample_param(np.random.default_rng(11), spec) for _ in range(5)]
    second = [sample_param(np.random.default_rng(11), spec) for _ in range(5)]
    assert first == second


def test_log_scale_rejects_non_positive_bounds():
    with pytest.raises(ValueError, match="log-scale bounds must be positive"):
        sample_param(np.random.default_rng(0), {"bounds": [-1.0, 10.0], "scale": "log"})


# --- grids -----------------------------------------------------------------


@pytest.mark.property
@PROPERTY
@given(
    lo=st.floats(min_value=1e-9, max_value=1e3),
    width=st.floats(min_value=1.0, max_value=1e6),
    levels=st.integers(min_value=1, max_value=12),
)
def test_log_grid_is_monotonic_and_bounded(lo: float, width: float, levels: int) -> None:
    hi = lo * width
    values = grid_values({"bounds": [lo, hi], "scale": "log", "grid": levels})
    assert len(values) == levels
    assert values == sorted(values)
    assert values[0] == pytest.approx(lo, rel=1e-9)
    if levels > 1:
        # geomspace(lo, hi, 1) yields [lo]; only a multi-point grid spans the range.
        assert values[-1] == pytest.approx(hi, rel=1e-9)


def test_grid_uses_choices_and_explicit_values_verbatim():
    assert grid_values({"choices": ["a", "b"]}) == ["a", "b"]
    assert grid_values({"values": [1, 2, 3]}) == [1, 2, 3]
    assert grid_values({"default": 7}) == [7]


# --- loading case files ----------------------------------------------------


def test_a_json_case_loads_like_a_yaml_one(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        '{"schema": "case_yaml.v1", "case_id": "from_json", "variables": {"x": {"default": 1}}}',
        encoding="utf-8",
    )
    assert load_case(path).case_id == "from_json"


def test_an_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "case.txt"
    path.write_text("case_id: x", encoding="utf-8")
    with pytest.raises(ValueError, match="case file must be YAML or JSON"):
        load_case(path)


def test_a_case_whose_root_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="case file root must be a mapping"):
        load_case(path)


def test_a_case_without_a_schema_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="case schema is required"):
        load_case(path)


def test_an_unknown_schema_is_rejected_instead_of_becoming_an_advanced_case(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text("schema: future.v9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported case schema"):
        load_case(path)


def test_the_case_id_falls_back_to_the_file_stem(tmp_path):
    path = tmp_path / "unnamed.yaml"
    path.write_text("schema: case_yaml.v1\nsource: {}\n", encoding="utf-8")
    assert load_case(path).case_id == "unnamed"


# --- JSON writing ----------------------------------------------------------


def test_numpy_and_path_values_are_json_encodable(tmp_path):
    out = tmp_path / "nested" / "payload.json"
    write_json(
        out,
        {
            "path": tmp_path,
            "array": np.array([1.0, 2.0]),
            "float": np.float64(1.5),
            "int": np.int64(3),
            "bool": np.bool_(True),
        },
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["array"] == [1.0, 2.0]
    assert payload["float"] == 1.5
    assert payload["int"] == 3
    assert payload["bool"] is True
    assert payload["path"].endswith(tmp_path.name)


def test_an_unencodable_value_names_its_type(tmp_path):
    with pytest.raises(TypeError, match="object"):
        write_json(tmp_path / "bad.json", {"x": object()})


# --- warnings recorded into a run ------------------------------------------


def test_a_variable_declared_twice_is_reported(make_case):
    case = make_case(
        {
            "case_id": "dup",
            "source": {"type": "sine_voltage", "variables": {"C1": {"default": 1}}},
            "circuit": {"variables": {"C1": {"default": 2}}},
        }
    )
    warnings = case_warnings(case)
    assert any("C1" in w and "later value wins" in w for w in warnings)


def test_a_case_without_a_source_is_reported(make_case):
    assert NO_SOURCE_WARNING in case_warnings(make_case({"case_id": "nosrc"}))


# --- the drive frequency ----------------------------------------------------


def test_the_fundamental_frequency_resolves_a_parameter_reference(make_case):
    """source.frequency_Hz may name a design variable, as render_source allows."""

    from pcd.spice import fundamental_hz

    case = make_case({"case_id": "ref", "source": {"frequency_Hz": "freq"}})
    assert fundamental_hz(case, {"freq": 2.0e6}) == pytest.approx(2.0e6)


def test_an_explicit_target_frequency_wins_over_the_source(make_case):
    from pcd.spice import fundamental_hz

    case = make_case({"case_id": "t", "source": {"frequency_Hz": 1.0e6}, "target": {"fundamental_Hz": 13.56e6}})
    assert fundamental_hz(case, {}) == pytest.approx(13.56e6)


def test_an_unresolvable_frequency_is_reported_clearly(make_case):
    from pcd.spice import fundamental_hz

    case = make_case({"case_id": "bad", "source": {"frequency_Hz": "not_a_number"}})
    with pytest.raises(ValueError, match="must resolve to a number"):
        fundamental_hz(case, {})
