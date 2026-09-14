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
  search:                      # finite candidate hardware shortlist
    C1: {values: [1e-10, 2.6e-10, 1e-9]}
  tuning:                      # exact settings available per condition
    C2: [4.5e-11, 1.6e-10, 2.1e-10]
```

A search axis contains a non-empty `values` list and may select one of those
values as its replay `default`. Values must be positive and unique. When no
default is supplied, the first value is used only for standalone netlist
preview. Every hardware combination is evaluated exactly once. Tuning likewise
accepts only explicit settings because an incomplete inner search cannot
establish infeasibility.

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

Most studies need no execution block. The resolved plan uses `ngspice_cli`, one
fixed candidate, or the complete Cartesian product of `network.search.values`.

```yaml
execution:
  solver: ngspice_cli
  candidate_state_limit: 250
  control_state_limit: 250
```

A case without a search always runs one candidate. Candidate and tuning-state
limits are safety bounds, not sampling budgets; every declared state below
them is evaluated. `--optimizer`, `--trials`, and `--seed` are advanced-case
options and cannot change a resolved `pcd.rf.v1` candidate set. `--solver` may
select `ngspice_cli`, or an adapter already registered by an advanced Python
integration, without changing the problem. `--require-acceptance` is an
optional automation gate: it exits nonzero when the completed decision is not
`meets_declared_acceptance`, even if every electrical solve succeeded.

## Advanced external circuit input

`case_yaml.v1` can import an existing circuit while PCD continues to own the
source, AC/transient analysis, output vectors, and final parameter overrides:

```yaml
circuit:
  builder: from_netlist
  netlist_file: exported_match.cir
  netlist_mode: deck           # deck | fragment
  source_policy: replace_named # replace_named | preserve
  output_node: electrode
```

`deck` discards the standard first-line SPICE title; `fragment` treats its
first line as executable circuit input and is the default for compatibility.
`replace_named` removes only top-level independent sources whose names exactly
match structured case sources. It does not infer conflicts from shared nodes.
Use `preserve` when the imported sources are intentionally part of the circuit.

Top-level execution directives such as `.control`, `.ac`, `.tran`, `.op`,
`.dc`, `.noise`, and `.end` are not imported. Consequently `from_netlist` is a
circuit-import adapter for PCD AC/transient studies, not an arbitrary ngspice
deck runner.

## Persisted truth

Each public-input run stores the following files in one immutable
`generations/g_<id>/` directory:

| file | purpose |
|---|---|
| `input_case.yaml` | exactly what the user authored |
| `resolved_plan.yaml` | defaults, inferences, execution settings, and explicit case |
| `case.yaml` | executable internal case used by the numerical path |
| `netlist.cir` | exact generated circuit for an evaluation |
| `sim_manifest.json` | solver identity, parameters, hashes, diagnostics, and artifacts |
| `evaluations.csv` | one flat candidate/scenario/control row per electrical solve |

Study results additionally retain every candidate, scenario, control
evaluation, aggregation, and content-addressed raw simulation result.
For an advanced external SPICE deck, recursively included files and selected
library sections are inlined into `imported_netlist.cir`; their original bytes
are also content-addressed and listed separately in `input_manifest.json`.
The root `study_result.json` is written last and points to the active generation.
Rerunning the same case publishes a new complete generation while preserving
the previous generation and reusable raw simulation cache entries. If a rerun
is interrupted, the previous `study_result.json` and its candidate inventory
remain authoritative; an unpublished generation may remain for inspection or
later cleanup. `pcd result-prune STUDY_ROOT --keep 3` previews old immutable
generations; add `--apply` to remove only those listed. Raw and evaluation
caches are never removed by this command.
`study_result.json.best` is the compact decision surface: selected fixed
candidate, acceptance status, solved/accepted condition counts, and each
condition's selected control, objective values, and failed constraints.
`best.status` describes evidence for the selected candidate itself. The
separate `best.search_completeness` field becomes `incomplete_evidence` when
any explored candidate or control solve failed, because global optimality may
then be unknown even when the selected candidate demonstrably meets its
declared acceptance limits.
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
When `pcd run` receives a solver override, its effective value replaces the
solver field in `resolved_plan.yaml` and `case.yaml` before validation and
hashing. Candidate enumeration remains derived from the authored RF input.
`input_case.yaml` remains unaltered.

`evaluations.csv` repeats design, scenario, and control values with explicit
prefixes and includes raw status, selection flag, metrics, constraints, cache
identity, duration, and artifact paths. `table_schema`, generation-specific
`dataset_id`, case schemas, and runtime/solver fingerprints make rows safe to
combine across studies without relying on directory names. It is intended as
the direct analysis or machine-learning handoff; candidate-level
`study_history.json` remains the compact optimizer trace. `selected_control`
is an outcome of evaluating all controls, not an independent input feature;
using it to predict that same outcome would leak target information.

## Responsibility boundary

The public format deliberately does not accept plasma geometry, cross
sections, reaction chemistry, electron density, or inferred sheath state.
Those belong in a separately qualified plasma model or dataset. PCD accepts
their electrical consequence at a declared terminal and answers circuit
selection, tuning-authority, frequency replay, and component-feasibility
questions within that evidence boundary.
