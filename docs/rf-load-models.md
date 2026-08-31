# RF load modelling boundary

New studies select these models through the small `pcd.rf.v1` `load` block;
see [the RF input reference](input-format.md). Longer mappings shown below are
the resolved internal representation and advanced compatibility form, not
fields a normal user must repeat.

## Purpose

PCD needs electrical loads that are simple enough for repeated matching-network
studies and explicit enough that their evidence can be audited. It does not
need a discharge solver. The boundary is therefore the chamber or coil
reference plane: upstream is the source/matching network; downstream is a
qualified one-port electrical representation.

Every RF load case records:

- model name and parameters;
- reference plane;
- parameter origin in `characterization`;
- model/measurement frequency or qualified frequency interval;
- external scenario ID and weight.

Geometry, density, collision rate, chemistry, sheath area, and species power
are intentionally absent unless an external tool has already converted them
to electrical parameters.

## Model selection

Use the least structured model supported by the observations:

1. Use `impedance_point` for one measured or supplied R+jX value. For a
   measured frequency table, evaluate its rows as independent scenarios.
2. Use `ccp_lumped` only when a series R-L-C fit is independently justified
   over multiple frequencies or conditions.
3. Use `icp_transformer` only when coil/dummy/plasma-on or broadband data can
   distinguish coil loss, reflected loss, and optional capacitive loading.
4. For hybrid ICP+CCP equipment, run separate coil-port and electrode-port
   studies with correlated scenario IDs. A true multiport model is a future
   feature; summing two one-port powers would double-count coupling.

Model complexity is not fidelity by itself. A six-parameter circuit fitted to
one complex datum is less identifiable than the two-parameter datum itself.

## Impedance point

At the declared anchor frequency:

```text
Z(f_model) = R + jX,  R >= 0
```

SPICE realization is series R-L for X > 0, series R-C for X < 0, and R alone
for X = 0:

```text
L = X / (2 pi f_model)
C = -1 / (2 pi f_model X)
```

This is exact at `model_frequency_Hz`. Although the generated L or C has a
mathematical continuation, PCD requires a one-point AC solve for this model and
requires the run fundamental to equal the anchor. Bandwidth conclusions require
independent measured frequency scenarios or a qualified structured model.

Minimum fields:

```yaml
load:
  name: impedance_point
  reference_plane: electrode_terminal
  characterization: {origin: measured_vna, dataset: chamber_A_2026_08.csv}
  resistance_ohm: load_R
  reactance_ohm: load_X
  model_frequency_Hz: 13560000
```

## Measured frequency points without an R-L-C fit

A measured frequency table is data, not a fourth equivalent-circuit topology.
Each row remains one independent `impedance_point`, and PCD runs one AC point
at that row's frequency:

```yaml
variables:
  rf_frequency_Hz: {default: 13560000}
  load_R: {default: 25}
  load_X: {default: -70}
source: {frequency_Hz: rf_frequency_Hz}
load:
  name: impedance_point
  resistance_ohm: load_R
  reactance_ohm: load_X
  model_frequency_Hz: rf_frequency_Hz
solver:
  ac: {frequency_Hz: rf_frequency_Hz}
study:
  scenario_table:
    table_file: measured_impedance.csv
    values:
      rf_frequency_Hz: frequency_Hz
      load_R: resistance_ohm
      load_X: reactance_ohm
```

The same scenario value therefore sets the source frequency, the exact
R+jX realization frequency, and the solver frequency. No R-L-C fit, frequency
interpolation, or out-of-range extrapolation is performed. A fixed candidate
network can be evaluated across all rows and aggregated by worst case or a
declared weighting.

This representation intentionally does not manufacture a continuous curve.
If a value between measured points is required, add a justified measurement,
or qualify a passive structured model outside PCD and use `ccp_lumped` or
`icp_transformer`. Interpolating R and X independently can violate causality or
passivity and is therefore not a silent platform default.

## Effective CCP R-L-C

The implemented one-port is:

```text
Z_CCP(w) = R_eff + j w L_eff + 1 / (j w C_sheath_eq)
```

Interpretation is deliberately limited:

- `R_eff` is all real RF power absorbed downstream of the reference plane;
- `L_eff` is an effective inductive/inertial term;
- `C_sheath_eq` is the equivalent series capacitive term seen at the port.

`R_eff` is not electron power, ion power, stochastic heating, or ohmic heating
separately. A one-port V/I waveform cannot identify those paths. The model is
useful for matching, resonance movement, sensitivity, and electrical corner
studies over the interval for which its fitted residual is acceptable.

One complex measurement supplies only two real observations while this model
has three parameters. Do not fit all three freely from a single frequency.
Fix one from independent evidence or use multiple frequencies/conditions.

## Effective ICP transformer

The primary contains coil resistance and inductance. Plasma-on reflected
loading is represented by the two combinations that the coil terminal can
actually identify:

```text
Z_series(w) = R_coil + j w L_coil
            + w^2 L_reflected / (gamma_secondary + j w)
```

where `L_reflected >= 0`, `L_reflected <= L_coil`, and
`gamma_secondary > 0`. The damping rate is a terminal fit parameter, not a
collision frequency unless independent plasma evidence establishes that
interpretation.

With an independently identified port capacitance:

```text
Z_ICP(w) = 1 / (1 / Z_series(w) + j w C_parallel)
```

`C_parallel` is an independently identified ideal terminal susceptance. It has
no real-power path and therefore cannot represent capacitive plasma heating or
separate inductive and capacitive plasma power.

Useful identification evidence includes:

- coil/dummy-fixture data for `R_coil`, `L_coil`, and fixture capacitance;
- plasma-off versus plasma-on complex impedance;
- a frequency sweep or multiple reproducible operating points;
- fit residuals and a held-out condition.

Without those observations, prefer `impedance_point`. PCD validates passivity
and bounds but cannot make an unsupported physical interpretation valid.

## Reference plane and data rules

R and X are meaningless without a frequency and reference plane. Calibration
or de-embedding should be completed before the data enter PCD. Cable,
feedthrough, coil, matching network, and chamber parasitics must not be included
both in the circuit candidate and again in the load.

Recommended single-frequency corner columns are:

```text
scenario_id, resistance_ohm, reactance_ohm, weight, condition_id
```

For a measured frequency table, add `frequency_Hz` and map it to the shared
`rf_frequency_Hz` scenario parameter. Duplicate frequencies are allowed only
when their scenario IDs denote distinct repeat measurements or process
conditions; the platform does not average away that variation.

Keep raw process variables as metadata in the source dataset if useful, but map
only electrical parameters into the circuit scenario. Scenario IDs should be
stable across coil-port and electrode-port studies.

At execution, PCD stores each built-in runtime data dependency once under a
bounded `inputs/<sha256-prefix>` path and records its full digest, declared
path, size, and archived path in `input_manifest.json`. A prefix collision is
rejected after checking the full digest. `input_case.yaml` remains the exact authored
record; `case.yaml` and `resolved_plan.yaml` point to the archived snapshot.
Plugins remain separately hashed executable code and are not copied into the
data-input bundle.

## Measurement and power

Transient RF-port metrics use differential load voltage and exact series load
current. With `measurement.load_current: auto`, PCD inserts a zero-volt source
at the load port. It then reports:

- peak/RMS/DC load voltage and RMS load current;
- real load power and real source power;
- network loss and transfer efficiency;
- fundamental load resistance/reactance;
- voltage/current harmonics;
- periodic-window residual and settled status.

`load_real_power_W` is accepted real power at the named load port. For the ICP
model it includes coil loss and effective reflected loss; for the CCP model it
is the aggregate power represented by `R_eff`. No electron/ion/sheath or
inductive/capacitive plasma-power allocation is returned. Power conservation
at this layer means checking source power = accepted load-port power + network
loss, subject to sign convention and numerical tolerance. It does not validate
a microscopic energy balance inside the discharge.

## Main risks and controls

| risk | consequence | control |
|---|---|---|
| wrong reference plane | duplicated or missing parasitics | named plane and de-embedded source data |
| one-point model used as broadband truth | false bandwidth/resonance claim | anchor frequency in model and report |
| over-parameterized CCP/ICP fit | unstable, non-unique scenarios | prefer R+jX; use the identifiable ICP form |
| negative fitted resistance | active/nonphysical passive load | reject negative resistance |
| source current used as load current | wrong delivered power with shunt branches | explicit series load ammeter |
| partial-cycle averaging | reactive power leaks into real power | final whole-cycle window and residual |
| AC result treated as large-signal RF | missed harmonics and state dependence | label AC as small-signal; add transient evidence when needed |
| hybrid powers simply added | coupling/double counting | separate correlated studies until multiport support |

## Implemented and deferred scope

Implemented now:

- three explicit RF load models;
- AC-only execution;
- exact one-point AC execution resolved from each scenario;
- complete enumeration of every declared discrete equipment-control state;
- requested-frequency interpolation of simulated AC phasors, without extrapolation; measured load rows are never interpolated;
- effective upstream component ESR/DCR and electrical stress/loss metrics;
- differential load-port measurement and exact load current;
- reusable CSV scenario tables;
- passive-parameter checks and model metadata validation;
- analytic and real-ngspice regression tests.

Deferred until supported by qualified data and a clear use case:

- multiport ICP+CCP coupling;
- temperature prediction and detailed frequency-dependent component models;
- tuner motion, quantization, delay, and closed-loop dynamics;
- parameter identification and held-out-fit diagnostics;
- nonlinear harmonic loads or an external plasma solver adapter.
