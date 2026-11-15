"""Regression for aggregate AC power inferred from an incomplete source boundary."""

from pathlib import Path

import pandas as pd
import pytest

from pcd.api import load_case, measure_response, run_case_study
from pcd.records import load_simulation_result, read_sim_record
from pcd.reporting import analyze_run


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_ac_controlled_source_withholds_aggregate_power_in_study_and_reports(tmp_path):
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        "schema: case_yaml.v1\n"
        "case_id: ac_controlled_power_balance\n"
        "source: {type: dc_voltage, name: V1, p: in, voltage_V: 0, ac_magnitude_V: 2}\n"
        "circuit:\n"
        "  builder: from_yaml\n"
        "  output_node: out\n"
        "  components:\n"
        "    - {ref: R1, n1: in, n2: out, value: 50}\n"
        "    - {raw: 'Eaux aux 0 in 0 0.7'}\n"
        "    - {ref: R2, n1: aux, n2: out, value: 50}\n"
        "load: {name: resistor, R_ohm: 50, ports: {p: out, n: 0}}\n"
        "measurement: {load_current: auto}\n"
        "solver: {ac: {frequency_Hz: 1000000}}\n"
        "target: {objective: impedance_match, constraints: {metric_bounds: {network_loss_W: {max: 0.006}}}}\n",
        encoding="utf-8",
    )

    case = load_case(case_path)
    study = run_case_study(case, tmp_path / "study")
    assert study["best"]["status"] == "incomplete_evidence"
    root = Path(study["run_root"])
    evaluations = pd.read_csv(root / study["artifacts"]["evaluation_table"])
    row = evaluations.iloc[0]
    assert row["metric.source_real_power_W"] == pytest.approx(0.0173333333333333)
    assert row["metric.load_real_power_W"] == pytest.approx(0.0128444444444444)
    assert row["metric.reflection_magnitude"] == pytest.approx(0.395348837209302)
    assert pd.isna(row["constraint.max_network_loss_W.value"])
    assert "Eaux" in row["metric.network_loss_W_unavailable_reason"]

    record = read_sim_record(root / row["artifact.manifest"])
    # A saved record from the earlier schema lacks this diagnostic. Its
    # canonical deck still contains the evidence needed to rederive it.
    del record["diagnostics"]["source_scope"]["ac_power_balance_unavailable_reason"]
    legacy_metrics = measure_response(case, load_simulation_result(record))
    assert legacy_metrics["network_loss_W"] is None
    assert "saved source inventory" in legacy_metrics["network_loss_W_unavailable_reason"]
    assert legacy_metrics["source_real_power_W"] == pytest.approx(row["metric.source_real_power_W"])
    report = analyze_run(record, output={"schematic": False, "plots": []})
    ac = report["analyses"]["ac"]
    assert ac["network_loss_W"] is None
    assert ac["transfer_efficiency"] is None
    assert ac["resistance_ohm"] == pytest.approx(115.384615384615)
    assert "Eaux" in ac["transfer_efficiency_unavailable_reason"]
    frequency_table = pd.read_csv(Path(report["output_dir"]) / report["artifacts"]["frequency_metrics"])
    assert frequency_table["network_loss_W"].isna().all()
    assert frequency_table["network_loss_W_unavailable_reason"].str.contains("Eaux").all()
