"""Public electrical-analysis facade.

The implementation is grouped by physical responsibility.  Existing callers
can keep importing from :mod:`pcd.analysis`; new internal code should import
the owning submodule directly.
"""

from .ac import DEFAULT_Z0, ac_power_flow, at_frequency, frequency_sweep_metrics, half_power_bandwidth, input_impedance
from .stress import ac_component_metrics, component_loss_balance, transient_component_metrics
from .transient import (
    DEFAULT_HARMONIC_COUNT,
    DEFAULT_PERIODIC_CYCLES,
    DEFAULT_SETTLING_COMPARISONS,
    DEFAULT_SETTLING_TOLERANCE,
    harmonic_spectrum,
    power_flow,
    rf_measurement_options,
    rf_port_metrics,
)

__all__ = [
    "DEFAULT_HARMONIC_COUNT",
    "DEFAULT_PERIODIC_CYCLES",
    "DEFAULT_SETTLING_COMPARISONS",
    "DEFAULT_SETTLING_TOLERANCE",
    "DEFAULT_Z0",
    "ac_component_metrics",
    "ac_power_flow",
    "at_frequency",
    "component_loss_balance",
    "frequency_sweep_metrics",
    "half_power_bandwidth",
    "harmonic_spectrum",
    "input_impedance",
    "power_flow",
    "rf_measurement_options",
    "rf_port_metrics",
    "transient_component_metrics",
]
