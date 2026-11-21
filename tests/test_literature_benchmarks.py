from pathlib import Path

import pytest
import yaml

from bench import run_suite as core_suite
from bench.literature import run_suite as literature_suite
from bench.literature.p0_lee2021_bias.run import _table_case
from bench.literature.p1_colpo1999_icp.digitized.run_uncertainty_challenge import _decision_stability
from bench.literature.p1_gec_ccp import run_all32_benchmark as hargis_source
from bench.literature.p1_gec_ccp.run_hardware_family_comparison import (
    _candidate_summary,
    _materialize_family_case,
)
from bench.literature.p1_lee2020_icp import run_benchmark as lee_equations
from pcd.api import prepare_case
from pcd.case import load_case

ROOT = Path(__file__).resolve().parents[1]
GEC = ROOT / "bench" / "literature" / "p1_gec_ccp"
LEE = ROOT / "bench" / "literature" / "p0_lee2021_bias"


def test_benchmark_suites_require_new_output_roots(tmp_path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    with pytest.raises(FileExistsError):
        literature_suite.run(root)
    with pytest.raises(FileExistsError):
        core_suite.main(["--run-root", str(root)])


@pytest.mark.parametrize("benchmark", [hargis_source, lee_equations], ids=["hargis", "lee"])
def test_literature_cli_requires_ngspice_by_default(benchmark, monkeypatch, capsys, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr(benchmark.shutil, "which", lambda _name: None)
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--output", str(output)])

    assert benchmark.main() == 2
    captured = capsys.readouterr()
    assert "ngspice" in captured.err
    assert "PASS" not in captured.out
    assert not output.exists()


def test_hargis_source_runner_checks_committed_views_without_rewriting_them() -> None:
    paths = [
        GEC / "raw_tables_iii_iv.csv",
        GEC / "derived_impedance_all32.csv",
        GEC / "derived_impedance_66pa.csv",
        GEC / "reported_spread_envelope.csv",
    ]
    before = {path: path.read_bytes() for path in paths}

    result = hargis_source.run(run_ngspice=False)

    assert result["passed"]
    assert not result["ngspice"]["executed"]
    assert "ngspice_replay" not in result["checks"]
    assert result["checks"]["committed_views"]
    assert {path: path.read_bytes() for path in paths} == before


def test_hargis_hardware_comparison_declares_independent_unweighted_families() -> None:
    spec = yaml.safe_load((GEC / "hardware_family_spec.yaml").read_text(encoding="utf-8"))

    assert [item["id"] for item in spec["families"]] == [
        "central_operating_conditions",
        "reported_apparatus_spread",
        "phase_model_minus6",
        "phase_model_plus6",
    ]
    assert "scenario_count" not in spec
    assert "weights" not in spec


def test_hargis_hardware_comparison_generates_a_valid_exact_candidate_case(tmp_path: Path) -> None:
    spec = yaml.safe_load((GEC / "hardware_family_spec.yaml").read_text(encoding="utf-8"))
    case_path, integrity, _ = _materialize_family_case(spec, spec["families"][0], tmp_path)

    case = load_case(case_path)

    assert integrity["passed"]
    assert case.resolved_plan is not None
    assert case.resolved_plan["execution"]["optimizer"] == "grid"
    assert case.resolved_plan["execution"]["trials"] == len(spec["candidate_L1_H"])


def test_hargis_hardware_summary_uses_the_flat_evaluation_table() -> None:
    candidate = {
        "design.L1": 1.5e-6,
        "success_fraction": 1.0,
        "control_margin": 0.5,
    }
    evaluations = [
        {
            "scenario_id": "a",
            "selected_control": True,
            "status": "ok",
            "feasible": True,
            "metric.reflection_magnitude": 0.1,
            "control.C1": 2.0,
            "control.C2": 20.0,
        },
        {
            "scenario_id": "a",
            "selected_control": False,
            "status": "ok",
            "feasible": False,
            "metric.reflection_magnitude": 0.8,
            "control.C1": 1.0,
            "control.C2": 10.0,
        },
        {
            "scenario_id": "b",
            "selected_control": True,
            "status": "ok",
            "feasible": False,
            "metric.reflection_magnitude": 0.4,
            "control.C1": 1.0,
            "control.C2": 20.0,
        },
    ]
    metadata = {
        "a": {"apparatus_group_MHz": "24"},
        "b": {"apparatus_group_MHz": "34"},
    }
    controls = {"C1_F": [1.0, 2.0, 3.0], "C2_F": [10.0, 20.0, 30.0]}

    summary = _candidate_summary(candidate, evaluations, metadata, controls, limit=0.5)

    assert summary["scenario_count"] == 2
    assert summary["feasible_scenarios"] == 1
    assert summary["endpoint_scenarios"] == 1
    assert summary["n_evaluations"] == 3
    assert summary["minimum_control_margin"] == 0.5
    assert summary["infeasible_scenarios"] == ["b"]


def test_lee_metadata_and_plane_substitutions_are_explicit() -> None:
    source = yaml.safe_load((LEE / "source.yaml").read_text(encoding="utf-8"))
    expectations = yaml.safe_load((LEE / "expectations.yaml").read_text(encoding="utf-8"))

    assert source["source"]["journal"] == "AIP Advances"
    assert set(expectations["matching_cases"]) == {
        "matcher_output_probe.yaml",
        "post_coax_plane_sensitivity.yaml",
        "plasma_terminal_plane_sensitivity.yaml",
    }


def test_lee_table_i_prepares_the_standard_ac_port(tmp_path: Path) -> None:
    source = yaml.safe_load((LEE / "source.yaml").read_text(encoding="utf-8"))
    case = _table_case(source, tmp_path)

    record = prepare_case(case, run_root=tmp_path / "prepared")

    assert record.measurement["source_voltage_vector"] == "v(src)"
    assert record.measurement["source_current_vector"] == "i(Vsrc)"
    assert record.diagnostics["source_scope"]["ac_excitations"] == ["Vsrc"]


def test_colpo_corner_results_are_grouped_by_parent_condition() -> None:
    scenarios = [
        {"scenario_id": "condition_a__rlo_xlo", "feasible": True},
        {"scenario_id": "condition_a__rhi_xhi", "feasible": True},
        {"scenario_id": "condition_b__rlo_xlo", "feasible": True},
        {"scenario_id": "condition_b__rhi_xhi", "feasible": False},
        {"scenario_id": "condition_c__rlo_xlo", "feasible": False},
        {"scenario_id": "condition_c__rhi_xhi", "feasible": False},
    ]

    stability = _decision_stability(scenarios)

    assert stability["parent_condition_count"] == 3
    assert stability["robust_parent_conditions"] == ["condition_a"]
    assert stability["mixed_parent_conditions"] == ["condition_b"]
    assert stability["failed_parent_conditions"] == ["condition_c"]
