# RF study input (`pcd.rf.v1`)

This is the user-facing input for matching-network studies. It contains only
engineering choices and evidence. PCD compiles it into the explicit internal
case consumed by validation, simulation, caching, and reporting.

## Required top-level fields

```yaml
schema: pcd.rf.v1
case_id: unique_name
frequency_Hz: 13560000       # omit only when every table row/condition supplies it
network: {...}
load: {...}
acceptance: {...}
```

`drive_peak_V` is optional for reflection-only AC studies because impedance
ratios do not depend on amplitude. It becomes required when an absolute
component or source-terminal limit is declared. When it is supplied, terminal
stress is reported for every named component in the public matching topology;
users need not list probes.

## Network

Supported named networks are `l_match`, `pi_match`, and
`pi_match_harmonic`. Their component references are fixed:

| type | required references |
|---|---|
| `l_match` | `L1`, `C1` |
| `pi_match` | `C1`, `L1`, `C2` |
| `pi_match_harmonic` | `C1`, `L1`, `C2`, `Lh`, `Ch` |

Each reference appears in exactly one role:

```yaml
network:
  type: pi_match
  fixed:                       # one chosen hardware value
    L1: 6.43e-7
  search:                      # candidate hardware search
    C1: {range: [1e-11, 1e-9], scale: log}
  tuning:                      # exact settings available per condition
    C2: [4.5e-11, 1.6e-10, 2.1e-10]
```

A search axis uses either `values` or `range`; `scale: log` applies to a range,
and an explicit `default` is optional for either form. Values must be positive component values.
When no default is supplied, the resolved plan records the first discrete
value or the range center. A `values` shortlist is enumerated completely and
exactly once; a `range` uses a sampled optimizer. Tuning likewise accepts only
explicit discrete values because a partial inner search cannot establish
infeasibility.

Effective series loss is a fixed qualified value at the operating point:

```yaml
network:
  loss_ohm: {C1: 0.1, L1: 0.5, C2: 0.1}
```

It is used for terminal electrical loss and loss closure, not for internal
temperature prediction.

## Load

Every load requires `reference_plane`. An optional `evidence` mapping records
the parameter origin, de-embedding, dataset revision, or qualified range.

### Independent impedance points

```yaml
load:
  type: impedance_table
  file: chamber_impedance.csv
  reference_plane: electrode_terminal
  evidence: {origin: measured, deembedding: fixture_v3}
```

Canonical CSV columns are:

```csv
scenario_id,resistance_ohm,reactance_ohm,weight
nominal,25,-80,1
```

`weight` is optional. `frequency_Hz` and `drive_peak_V` are optional columns.
A frequency column independently sets the source frequency, impedance-point
anchor, and one-point AC solve; a top-level frequency must then be omitted. A
drive column makes each row a complete electrical/stress operating point and
satisfies absolute component-limit input. Extra provenance columns are
retained in the dataset but do not become solver parameters.

For one point, put the values directly in the case:

```yaml
load:
  type: impedance_point
  resistance_ohm: 25
  reactance_ohm: -80
  reference_plane: electrode_terminal
```

### Effective CCP load

```yaml
load:
  type: ccp_lumped
  reference_plane: electrode_terminal
  parameters:
    R_eff_ohm: 8
    L_eff_H: 0.4e-6
    C_sheath_eq_F: 120e-12
```

This is a qualified series effective R-L-C one-port. It does not compute a
sheath capacitance from geometry or divide electron and ion power.

### Effective ICP load

```yaml
load:
  type: icp_transformer
  reference_plane: coil_terminal
  parameters:
    R_coil_ohm: 0.2
    L_coil_H: 2.0e-6
    reflected_inductance_H: 0.18e-6
    secondary_damping_rate_rad_s: 4.0e6
    C_parallel_F: 20e-12       # optional
```

This represents an identifiable effective plasma-on coil loading fit. The two
reflected terms describe the terminal response; the damping rate is not a
collision frequency without independent evidence. `C_parallel_F` is an ideal
terminal susceptance and does not represent capacitive plasma heating. The
model does not advance a discharge state.

## Conditions

Use `conditions` for a small inline set of non-table operating conditions:

```yaml
conditions:
  - {id: low_drive, drive_peak_V: 25}
  - {id: high_drive, drive_peak_V: 100}
```

`frequency_Hz` may also vary by condition. If one condition supplies frequency
or drive, every condition must supply it. Conditions are intentionally not
combined with an impedance table: implicit Cartesian products are easy to
misread, so each required R+jX/frequency point belongs in that table.

## Acceptance

Exactly one reflection limit is required:

```yaml
acceptance:
  reflected_power_fraction_max: 0.10
```

or:

```yaml
acceptance:
  reflection_magnitude_max: 0.31623
```

The power-fraction form is compiled to `|Gamma| <= sqrt(limit)`. Optional
component limits are:

- `current_rms_A_max`
- `current_peak_A_max`
- `voltage_rms_V_max`
- `voltage_peak_V_max`
- `loss_W_max`

Example:

```yaml
acceptance:
  reflected_power_fraction_max: 0.10
  component_limits:
    L1: {current_rms_A_max: 1.0, loss_W_max: 0.5}
  source_limits:
    current_rms_A_max: 1.0
    apparent_power_VA_max: 50.0
  control_margin_min: 0.20
  loss_balance_fraction_max: 1.0e-5
```

`loss_W_max` requires `network.loss_ohm` for that component. Loss-balance
acceptance requires at least one declared effective loss. `source_limits`
refer to the simulated ideal-source terminal, not forward power inside a real
generator. `control_margin_min` is optional and requires `network.tuning`; it
uses the same normalized 0=edge, 1=center value reported in the results.

## Execution

Most studies need no execution block. Defaults are written into the resolved
plan: `ngspice_cli`, seed 0, one fixed candidate, complete grid enumeration for
discrete `values`, or 30 random candidates when any search axis uses `range`.

```yaml
execution:
  solver: ngspice_cli
  optimizer: random
  seed: 7
  trials: 50
  candidate_state_limit: 250
  control_state_limit: 250
```

A case without a search always runs one candidate. With `optimizer: grid`,
`trials` is inferred from the Cartesian product and cannot truncate it. The
candidate- and tuning-state limits are safety bounds, not sampling budgets;
every state below them is evaluated.

## Persisted truth

Each public-input run stores:

| file | purpose |
|---|---|
| `input_case.yaml` | exactly what the user authored |
| `resolved_plan.yaml` | defaults, inferences, execution settings, and explicit case |
| `case.yaml` | executable internal case used by the numerical path |
| `netlist.cir` | exact generated circuit for an evaluation |
| `sim_manifest.json` | solver identity, parameters, hashes, diagnostics, and artifacts |

Study results additionally retain every candidate, scenario, control
evaluation, aggregation, and content-addressed raw simulation result.
Candidate selection first prefers complete solver evidence, then condition
coverage, constraint violation, and the declared objectives. If
`control_margin_min` is present, an electrically valid edge setting remains
identified as reachable but is not accepted as having sufficient reserve. If
jointly acceptable settings are equal, PCD prefers the one with more numeric
tuning headroom. The reported
`control_margin` is 0 at a declared tuning-grid edge and 1 at its center; the
worst axis and condition are retained.  A single numeric setting is neutral at
1, while categorical controls are excluded. `edge_limited` is true only when
margin is the sole barrier to full feasibility.
When `pcd run` receives execution overrides such as `--trials` or `--solver`,
their effective values replace the corresponding fields in `resolved_plan.yaml`
and `case.yaml` before validation and hashing. `input_case.yaml` remains the
unaltered authored input.

## Responsibility boundary

The public format deliberately does not accept plasma geometry, cross
sections, reaction chemistry, electron density, or inferred sheath state.
Those belong in a separately qualified plasma model or dataset. PCD accepts
their electrical consequence at a declared terminal and answers circuit
selection, tuning-authority, frequency replay, and component-feasibility
questions within that evidence boundary.
