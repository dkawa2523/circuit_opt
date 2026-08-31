# Component loss and electrical stress

## Responsibility

PCD evaluates electrical feasibility at declared operating points. It can
represent an effective capacitor ESR or inductor DCR/core-loss resistance,
measure terminal voltage/current, and compare electrical loss or stress with
declared limits. It is not a vendor component database, electromagnetic model,
or thermal solver.

In the public `pcd.rf.v1` input, an explicit `drive_peak_V` makes PCD observe
every named component in the selected matching topology. Use
`network.loss_ohm`, `acceptance.component_limits`, and optional
`acceptance.source_limits` for the limits that matter to the design. The
`from_yaml` form below is the resolved implementation and remains available
for advanced custom circuits, where observation stays explicit.

Use the least detailed evidence that supports the decision:

- omit `series_resistance_ohm` for an ideal exploratory topology;
- supply effective ESR/DCR for loss and efficiency decisions;
- vary that resistance by scenario when measured frequency or temperature data
  justify the variation;
- use a separate qualified thermal model to turn watts into temperature.

## Structured two-terminal component

The generic `from_yaml` builder accepts two optional fields:

```yaml
circuit:
  builder: from_yaml
  components:
    - ref: L1
      n1: src
      n2: out
      value: L1
      series_resistance_ohm: L1_DCR
      observe: true
```

`series_resistance_ohm` inserts one explicit series resistor. For a capacitor
it is effective ESR; for an inductor it may represent DCR plus independently
qualified effective RF loss. `observe: true` inserts an ideal zero-volt current
meter and records the original two-terminal package voltage.

Raw SPICE lines remain available for advanced models, but automatic component
observation requires the structured `ref/n1/n2/value` form. This keeps probe
names, internal nodes, and metric names deterministic.

## Calculations

For sinusoidal AC, the source AC magnitude and component phasors are peak
values:

```text
V_rms = |V_peak| / sqrt(2)
I_rms = |I_peak| / sqrt(2)
P_loss = I_rms^2 R_series
```

Real source and load power use peak phasors:

```text
P = 0.5 Re(V I*)
network loss = source real power - load real power
```

For transient analysis, the same metrics are calculated over the final common
whole-cycle window using time-weighted RMS. Harmonics therefore contribute to
RMS current and effective-resistance loss.

The source's `amplitude_V` is used as the AC peak magnitude. Set
`ac_magnitude_V` only when the small-signal stress run intentionally uses a
different amplitude. Reflection and impedance are amplitude independent;
voltage, current, and loss are not.

## Metrics and limits

For an observed `L1`, both AC `impedance_match` and transient `rf_load` return:

```text
component_L1_voltage_peak_V
component_L1_voltage_rms_V
component_L1_current_peak_A
component_L1_current_rms_A
component_L1_loss_W                 # when series resistance is declared
```

Runs with explicit loss on one or more observed components also return:

```text
modeled_component_loss_W
component_loss_balance_residual_W
component_loss_balance_fraction_of_source
```

The simulated source terminal also returns:

```text
source_current_rms_A
source_apparent_power_VA
source_real_power_W
```

The first two can be limited in the public input with
`acceptance.source_limits`. They characterize the ideal source terminal in the
declared circuit, not directional-coupler forward power or a generator's
internal protection model.

Limits use the existing generic metric-bound mechanism:

```yaml
target:
  objective: impedance_match
  constraints:
    metric_bounds:
      reflection_magnitude: {max: 0.316227766}
      component_L1_current_rms_A: {max: 1.0}
      component_L1_loss_W: {max: 0.5}
      component_loss_balance_fraction_of_source: {max: 1.0e-5}
```

The balance residual is evidence that all intended network losses were
represented and observed. A nonzero residual is not automatically an error:
it may be an unobserved resistor, fixture loss, or numerical tolerance.

## Limits of interpretation

- Effective ESR/DCR is exact only for the frequency, amplitude, and condition
  for which it was supplied.
- `I_rms^2 R` does not predict junction, winding, or case temperature.
- AC stress is linear sinusoidal evidence; use transient analysis for switching,
  nonlinear loads, or harmonic heating.
- Terminal voltage does not resolve internal winding or dielectric field peaks.
- A good match can still violate voltage, current, loss, or thermal ratings.

See `examples/rf_component_stress.yaml` for the concise public form and B5 in
`bench/cases/match_high_drive_stress.yaml` for a reproducible
matching-network feasibility decision.
