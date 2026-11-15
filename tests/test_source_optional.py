"""A transient circuit need not have an independent source branch."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcd.api import simulate_case
from pcd.records import load_simulation_result
from pcd.reporting import analyze_run
from pcd.simulation_input import resolve_simulation_case
from pcd.validation import validate_case


@pytest.mark.e2e
@pytest.mark.usefixtures("ngspice")
def test_unpowered_rc_discharge_preserves_real_probes_without_creating_a_source(make_case, tmp_path):
    case = make_case(
        {
            "case_id": "discharge",
            "circuit": {
                "builder": "from_yaml",
                "output_node": "out",
                "components": [
                    {"ref": "R1", "n1": "out", "n2": "0", "value": 1000, "observe": True},
                    {"ref": "C1", "n1": "out", "n2": "0", "value": 1e-9},
                    {"raw": ".ic v(out)=1"},
                ],
            },
            "solver": {"tran": {"step_s": 1e-8, "stop_s": 3e-6}},
        }
    )
    assert validate_case(case, strict=True).ok
    record = simulate_case(case, run_root=tmp_path)
    assert record.status == "ok"
    assert record.measurement["current_source"] is None
    assert record.measurement["source_voltage_vector"] is None
    assert record.measurement["source_current_vector"] is None

    wave = pd.read_csv(record.run_dir / record.waveform_file)
    assert list(wave.columns[:3]) == ["time_s", "voltage_V", "current_A"]
    assert wave["voltage_V"].iloc[0] == pytest.approx(1)
    assert np.interp(1e-6, wave["time_s"], wave["voltage_V"]) == pytest.approx(math.exp(-1), rel=1e-3)
    assert np.allclose(wave["component_R1_current_A"], wave["voltage_V"] / 1000, rtol=1e-3, atol=1e-12)
    assert wave["current_A"].isna().all()
    assert "source_voltage_V" not in wave
    assert load_simulation_result(record.run_dir).current_A is None

    report = analyze_run(record.run_dir, output={"plots": [], "schematic": False})
    assert report["status"] == "complete"
    assert report["measurement"]["observed_source"] is None
    assert report["measurement"]["impedance_reference_plane"] is None
    assert "energy" not in report["analyses"]["transient"]
    catalog = json.loads((Path(report["output_dir"]) / "signals.json").read_text(encoding="utf-8"))
    assert {"current_A", "source_voltage_V"}.isdisjoint(catalog["signals"])


def test_an_existing_raw_default_source_keeps_its_observation(make_case):
    case = make_case({"circuit": {"components": [{"raw": "Vsrc src 0 DC 1"}]}})
    probes = resolve_simulation_case(case).probes
    assert (probes.source_name, probes.source_voltage_vector, probes.source_current) == ("Vsrc", "v(src)", "i(Vsrc)")
