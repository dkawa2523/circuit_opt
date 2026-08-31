from pathlib import Path

import yaml

from bench.literature.p0_lee2021_bias.run import _table_spice_netlist
from bench.literature.p1_colpo1999_icp.digitized.run_uncertainty_challenge import _decision_stability
from bench.literature.p1_gec_ccp.run_all32_benchmark import run as run_hargis_source

ROOT = Path(__file__).resolve().parents[1]
GEC = ROOT / "bench" / "literature" / "p1_gec_ccp"
LEE = ROOT / "bench" / "literature" / "p0_lee2021_bias"


def test_hargis_source_runner_checks_committed_views_without_rewriting_them() -> None:
    paths = [
        GEC / "raw_tables_iii_iv.csv",
        GEC / "derived_impedance_all32.csv",
        GEC / "derived_impedance_66pa.csv",
        GEC / "reported_spread_envelope.csv",
    ]
    before = {path: path.read_bytes() for path in paths}

    result = run_hargis_source(run_ngspice=False, require_ngspice=False)

    assert result["status"] == "PASS"
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
    assert not (GEC / "hardware_authority_spec.yaml").exists()


def test_lee_metadata_and_plane_substitutions_are_explicit() -> None:
    source = yaml.safe_load((LEE / "source.yaml").read_text(encoding="utf-8"))
    expectations = yaml.safe_load((LEE / "expectations.yaml").read_text(encoding="utf-8"))

    assert source["source"]["journal"] == "AIP Advances"
    assert set(expectations["matching_cases"]) == {
        "matcher_output_probe.yaml",
        "post_coax_plane_sensitivity.yaml",
        "plasma_terminal_plane_sensitivity.yaml",
    }


def test_lee_table_i_netlist_writes_the_complete_ac_port_contract() -> None:
    source = yaml.safe_load((LEE / "source.yaml").read_text(encoding="utf-8"))
    values = source["table_i"]["equivalent_circuit"]
    frequency = float(source["operating_condition"]["frequency_Hz"])

    netlist = _table_spice_netlist(values, frequency)

    assert "wrdata ac.csv v(src) i(Vsrc) v(src)" in netlist


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
