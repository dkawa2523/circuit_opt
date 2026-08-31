from __future__ import annotations

import numpy as np

from pcd.metric_registry import register as metric_register
from pcd.netlist import Circuit
from pcd.sim_registry import register as sim_register


@sim_register("circuit", "custom_series_lc")
def custom_series_lc(case, params):
    c = Circuit(output_node="out")
    c.params.update(params)
    c.add("Lx", "src", "mid", "Lx")
    c.add("Cx", "mid", "out", "Cx")
    c.add("Rleak", "out", "0", 1e12)
    return c


@metric_register("peak_voltage")
def peak_voltage(case, record, waveform):
    peak = float(np.nanmax(np.abs(waveform["voltage_V"].to_numpy(float)))) if len(waveform) else 1e30
    return {"loss": peak, "peak_abs_voltage_V": peak, "objective": "peak_voltage"}
