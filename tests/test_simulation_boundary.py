"""Typed simulation requests and their ngspice control rendering."""

from __future__ import annotations

import pytest

from pcd.netlist import render_control_lines
from pcd.probes import NamedProbe, ProbePlan
from pcd.simulation import AcSweep, AnalysisRequest
from pcd.simulation_input import SolverSettings, resolve_simulation_case


def _probe_plan(*transient_vectors: str, ac_vectors: tuple[str, ...] = ()) -> ProbePlan:
    transient = tuple(NamedProbe(vector, vector) for vector in transient_vectors)
    transient += (NamedProbe("v(src)", "source_voltage_V"),)
    return ProbePlan(
        source_name="Vsrc",
        source_voltage_vector="v(src)",
        load_current_column=None,
        transient=transient,
        ac=tuple(NamedProbe(vector, vector) for vector in ac_vectors),
    )


def test_a_case_resolves_to_one_small_solver_input(make_case):
    case = make_case(
        {
            "source": {"name": "Vrf", "p": "src", "n": "return"},
            "measurement": {"current_source": "Vrf", "probes": {"mid_V": "v(mid)"}},
            "solver": {
                "name": "custom",
                "executable": "solver.exe",
                "timeout_s": 12.5,
                "options": {"reltol": "1e-5"},
                "ac": {"frequency_Hz": "frequency"},
            },
        }
    )

    resolved = resolve_simulation_case(case, {"frequency": 13.56e6}, solver_override="override")

    assert resolved.solver.name == "override"
    assert resolved.solver.executable == "solver.exe"
    assert resolved.solver.timeout_s == 12.5
    assert resolved.analysis.ac == AcSweep("lin", 1, 13.56e6, 13.56e6)
    assert resolved.probes.source_voltage_vector == "v(src,return)"
    assert resolved.probes.ac_columns == ("mid_V",)
    assert resolved.measurement.load_negative == "0"
    assert resolved.netlist_options == (("reltol", "1e-5"),)


def test_a_transient_only_case_asks_for_no_sweep():
    request = AnalysisRequest.from_config({"tran": {"step_s": 1e-9, "stop_s": 1e-6}})
    assert request.ac is None


def test_an_ac_sweep_is_read_from_the_case():
    sweep = AnalysisRequest.from_config({"ac": {"sweep": "lin", "points": 50, "start_Hz": 1e6, "stop_Hz": 2e7}}).ac
    assert sweep is not None
    assert sweep == AcSweep(sweep="lin", points=50, start_hz=1e6, stop_hz=2e7)


@pytest.mark.parametrize("reference", ["rf_frequency_Hz", "$rf_frequency_Hz"])
def test_a_parameterized_ac_point_is_resolved_for_each_scenario(reference):
    sweep = AnalysisRequest.from_config(
        {"ac": {"frequency_Hz": reference}},
        {"rf_frequency_Hz": 27.12e6},
    ).ac
    assert sweep is not None
    assert sweep == AcSweep(sweep="lin", points=1, start_hz=27.12e6, stop_hz=27.12e6)


@pytest.mark.parametrize(
    "solver",
    [
        {"tran": "not-a-mapping"},
        {"tran": {"step_s": "fast", "stop_s": 1.0}},
        {"tran": {"step_s": -1.0, "stop_s": 1.0}},
        {"tran": {"step_s": 2.0, "stop_s": 1.0}},
        {"ac": "not-a-mapping"},
        {"ac": {"sweep": "random"}},
        {"ac": {"points": "many"}},
        {"ac": {"points": 1.5}},
        {"ac": {"points": 0}},
        {"ac": {"points": 10, "start_Hz": 2e6, "stop_Hz": 1e6}},
        {"ac": {"frequency_Hz": 0}},
        {"ac": {"frequency_Hz": 1e6, "points": 1}},
    ],
)
def test_invalid_analysis_settings_are_rejected_at_the_typed_boundary(solver):
    with pytest.raises((TypeError, ValueError)):
        AnalysisRequest.from_config(solver)


def test_typed_execution_models_keep_their_own_invariants():
    with pytest.raises(ValueError, match="at least one"):
        AnalysisRequest(transient=None, ac=None)
    with pytest.raises(ValueError, match="name"):
        SolverSettings(name="")
    with pytest.raises(ValueError, match="timeout_s"):
        SolverSettings(timeout_s=0)


@pytest.mark.parametrize(
    "declaration",
    [
        {"source": [1, 2]},
        {"sources": 1},
        {"source": {"name": "V1"}, "sources": [{"name": "V2"}]},
        {"sources": [{"name": "V1"}, "not-a-source"]},
        {"sources": [{"name": "Vsame"}, {"name": "Vsame"}]},
        {"measurement": []},
        {"measurement": {"current_source": "Vmissing"}},
        {"measurement": {"probes": "v(out)"}},
        {"measurement": {"probes": {"voltage_V": "v(extra)"}}},
        {"solver": {"options": []}},
        {"load": []},
        {"load": {"ports": []}},
    ],
)
def test_source_and_observation_shape_is_owned_by_the_simulation_boundary(make_case, declaration):
    case = make_case({"source": {"name": "Vsrc"}, **declaration})

    with pytest.raises((TypeError, ValueError)):
        resolve_simulation_case(case)


def test_the_control_block_runs_only_a_transient_by_default():
    request = AnalysisRequest.from_config({"tran": {"step_s": 1e-9, "stop_s": 1e-6}})
    lines = render_control_lines(request, "out", _probe_plan())
    assert "tran 1e-09 1e-06 0 1e-09" in lines
    assert "wrdata waveform.csv time v(out) i(Vsrc) v(src)" in lines
    assert not any(line.startswith("ac ") for line in lines)


def test_requesting_a_sweep_adds_it_to_the_same_run():
    """One solver invocation produces both files; there is no second run."""

    solver = {"tran": {"step_s": 1e-9, "stop_s": 1e-6}, "ac": {"points": 10}}
    lines = render_control_lines(AnalysisRequest.from_config(solver), "out", _probe_plan())
    assert any(line.startswith("tran ") for line in lines)
    assert any(line.startswith("ac dec 10") for line in lines)
    assert "set numdgt=15" in lines
    assert "wrdata ac.csv v(src) i(Vsrc) v(out)" in lines


def test_an_ac_only_case_does_not_run_or_write_a_transient():
    solver = {"ac": {"sweep": "lin", "points": 3, "start_Hz": 1e6, "stop_Hz": 2e6}}
    request = AnalysisRequest.from_config(solver)
    lines = render_control_lines(request, "out", _probe_plan())
    assert request.transient is None
    assert not any(line.startswith("tran ") for line in lines)
    assert not any("waveform.csv" in line for line in lines)
    assert "wrdata ac.csv v(src) i(Vsrc) v(out)" in lines


def test_a_scenario_frequency_generates_one_exact_ac_point():
    request = AnalysisRequest.from_config(
        {"ac": {"frequency_Hz": "rf_frequency_Hz"}},
        {"rf_frequency_Hz": 6.78e6},
    )
    lines = render_control_lines(request, "out", _probe_plan())
    assert "ac lin 1 6.78e+06 6.78e+06" in lines


def test_extra_probes_are_saved_alongside_the_standard_vectors():
    lines = render_control_lines(AnalysisRequest.from_config({}), "out", _probe_plan("i(L1)", "v(mid)"))
    assert "wrdata waveform.csv time v(out) i(Vsrc) i(L1) v(mid) v(src)" in lines
    assert lines[0].startswith(".save v(out) i(Vsrc) i(L1) v(mid)")


def test_ac_probes_are_written_after_the_standard_source_and_load_vectors():
    lines = render_control_lines(
        AnalysisRequest.from_config({"ac": {"frequency_Hz": 1e6}}),
        "v(load)",
        _probe_plan(ac_vectors=("i(Vobserve_L1)",)),
    )
    assert "wrdata ac.csv v(src) i(Vsrc) v(load) i(Vobserve_L1)" in lines
