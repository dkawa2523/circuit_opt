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

### Quasi-static impedance profile

Use `impedance_profile` when measured terminal R+jX changes on an envelope
time scale that is slow relative to the RF period:

```yaml
frequency_Hz: 13560000
drive_peak_V: 100
load:
  type: impedance_profile
  file: plasma_impedance_profile.csv
  reference_plane: electrode_terminal
  evidence: {origin: measured, deembedding: fixture_v3}
```

The canonical CSV shape is:

```csv
time_s,resistance_ohm,reactance_ohm
0,18,-100
1.0e-5,22,-80
2.0e-5,30,-60
```

`time_s` must be finite, non-negative, and strictly increasing. Resistance
must be passive and every R/X value finite. Optional `frequency_Hz`,
`drive_peak_V`, and `weight` columns follow the same rules as an impedance
table. Frequency and drive may be fixed at the case level or supplied in every
row, but not both. An absolute drive is mandatory because the profile output
reports power in watts.

Every row becomes one exact-frequency `impedance_point` scenario. PCD does not
interpolate between rows, advance a plasma state, or model tuner dynamics. A
fixed hardware candidate is shared across the profile; declared
`network.tuning` settings may be selected independently for each snapshot.
The active study generation adds `snapshot_response.csv`, ordered by `time_s`,
with declared load R/X, calculated matching-input R/X, `|Gamma|`, reflected
power fraction, forward/reflected/load power, network loss, selected controls,
and acceptance status. Wave powers use the configured source-side reference
impedance; declared load R/X remains at `load.reference_plane`.

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
combined with an impedance table or profile: implicit Cartesian products are
easy to misread, so each required R+jX/frequency/time point belongs in that
table.

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

New Python solver adapters register with `pcd.api.register_solver` and accept a
single `SolverRunRequest`. The request contains paths plus resolved execution,
analysis, probe, and measurement-reference settings; adapters should not parse
the full Case mapping.

Advanced `case_yaml.v1` inputs may declare the solver request directly:

```yaml
solver:
  name: ngspice_cli
  timeout_s: 300
  tran: {step_s: 1.0e-10, stop_s: 1.0e-6}
  ac: {sweep: dec, points: 40, start_Hz: 1.0e6, stop_Hz: 1.0e8}
```

`timeout_s`, transient bounds, and AC frequencies must be positive and finite;
the transient step cannot exceed its stop time. An AC request uses either one
`frequency_Hz` value or the `sweep`/`points`/`start_Hz`/`stop_Hz` form, never
both. These rules are enforced when the mapping becomes an `AnalysisRequest`,
before any run directory or ngspice input is created. Invalid explicit values
are not replaced by defaults; defaults apply only when a field is omitted.

For an advanced structured circuit that will be exported with `ml-corpus`,
declare its stable family separately from the rendering builder:

```yaml
circuit:
  builder: from_yaml
  topology_family: custom_l_match
  output_node: electrode
  components:
    - {ref: L1, n1: src, n2: electrode, value: L1, observe: true}
    - {ref: C1, n1: electrode, n2: '0', value: C1, observe: true}
```

Each structured two-terminal component becomes one logical graph component.
`observe: true` and `series_resistance_ohm` may change ngspice rendering and
available targets, but observation meters and internal loss nodes are never
learned as topology. A raw SPICE line, imported netlist, magnetic coupling, or
prescribed time profile is still valid for its supported simulation path but
is not graph-ready until its builder declares equivalent graph semantics.

## Advanced continuous optimization

The explicit `case_yaml.v1` format may optimize bounded continuous design
variables without replacing the public RF format's complete discrete search:

```yaml
run: {trials: 24}
circuit:
  variables:
    R1: {bounds: [100, 10000], scale: log, default: 1000}
    C1: {bounds: [1e-10, 1e-7], scale: log, default: 1e-9}
target:
  objective: waveform_l2
  waveform_file: target.csv
  constraints:
    metric_bounds:
      peak_abs_voltage_V: {max: 4.9}
study:
  objectives:
    - {metric: normalized_rmse, direction: minimize, aggregation: worst}
    - {metric: peak_abs_voltage_V, direction: minimize, aggregation: worst}
optimizer: {name: differential_evolution, seed: 3}
```

`differential_evolution` implements bounded `DE/rand/1/bin`. It accepts only
numeric continuous axes with increasing finite `bounds`; `scale` is `linear`
or positive `log`. A variable with one `choices` entry or only a `default` is
fixed. Multiple choices, integer/bool axes, and unbounded axes are rejected so
topology selection is not silently treated as a continuous coordinate.

The population has `4 * number_of_continuous_variables` members, with a
minimum of four. `run.trials` must cover initialization plus at least one full
generation. The declared default candidate is evaluated first when all axes
provide an in-range default; the remaining initial points are stratified over
the normalized bounds. The implementation uses fixed mutation 0.8 and
crossover 0.7, while the archived seed and implementation fingerprint make the
proposal sequence reproducible.

Every proposed fixed candidate is evaluated through the unchanged
Candidate x Scenario x Control pipeline. Child/parent replacement uses the
same lexicographic order as the final decision: complete solver evidence,
engineering feasibility, normalized constraint violation, then declared
objectives. It does not add a tunable penalty to the objective. The compact
`study_history.json` records parameters, proposal phase/generation and
acceptance, objective aggregates, failed constraints/evaluations, scenario and
evaluation counts, seed, and duration. Since every search candidate already
uses every declared scenario and control, selecting the best candidate does
not trigger a duplicate solver run. A future surrogate-proposed candidate must
still be evaluated by ngspice through this full pipeline.

When a study declares at least two objectives, the active generation also
contains `pareto_front.csv`. Dominance respects each objective's declared
`minimize` or `maximize` direction. A candidate is eligible only when every
explored control solved, every scenario has an accepted selected control, and
all aggregated objective values are finite. Equal objective points are retained
because different component values can represent distinct engineering choices.

`study_result.json.pareto.scope` is `observed_candidates` for sampled continuous
searches: the table is the nondominated set among evaluated candidates, not a
claim that the continuous design space was exhausted. `declared_grid` is used
only for exact candidate enumeration. The existing feasibility-first,
lexicographic `best` decision remains unchanged and is marked by the `selected`
column in the Pareto table.

The runnable target-waveform example is
[`examples/advanced/generic_rc_filter.yaml`](../examples/advanced/generic_rc_filter.yaml).

## Advanced prescribed resistance profile

An explicit `case_yaml.v1` circuit or `from_yaml` load may prescribe a
time-varying resistor from CSV:

```yaml
circuit:
  builder: from_yaml
  components:
    - ref: Rchamber
      n1: out
      n2: 0
      observe: true
      value:
        profile: chamber_resistance.csv
        time_column: time_s
        value_column: resistance_ohm
        interpolation: linear  # linear | hold
        repeat_period_s: 75e-6 # optional
solver:
  tran: {step_s: 1e-8, stop_s: 150e-6}
```

The CSV needs at least two rows, starts at `time_s=0`, and has finite,
strictly increasing time and strictly positive resistance. `linear` uses
piecewise-linear interpolation and holds the final value after a non-repeated
profile; `hold` uses the preceding sample until the next timestamp. A repeated
profile must end at `repeat_period_s` and repeat its initial resistance at that
boundary.

This feature is transient-only. `solver.tran.step_s` may not exceed the
shortest profile interval, and PCD writes that step as ngspice's explicit
maximum internal step (`tmax`). `observe: true` adds the resistor terminal
voltage/current columns to `data/transient.csv`. The source CSV is included in
the input manifest, archived by content hash, and included in the simulation
fingerprint.

This is a one-way imposed `R(t)`, not a plasma-state solver. Time-varying C/L
are intentionally unsupported because a scalar substitution alone does not
define charge/flux or external energy exchange. The runnable example is
[`examples/advanced/time_varying_resistor.yaml`](../examples/advanced/time_varying_resistor.yaml).

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

One direct `sim-run` stores the following responsibility-based layout:

| file | purpose |
|---|---|
| `summary.json` | concise status, solver version, warnings, and result paths |
| `data/transient.csv` | canonical transient samples, or an empty table for an AC-only run |
| `data/ac.csv` | canonical AC phasors when AC was requested |
| `debug/manifest.json` | parameters, hashes, detailed diagnostics, and every artifact path |
| `debug/case.yaml` | executable case used by the numerical path |
| `debug/netlist.cir` | exact generated circuit |
| `debug/solver.log` | exact solver output, or the prepared-only marker before adapter execution |

For a completed `impedance_profile` study, the active immutable generation also
contains `snapshot_response.csv`. It is the concise selected-candidate time
projection; `evaluations.csv` remains the complete Candidate x Scenario x
Control table used for audit and machine learning.

Referenced data, an authored public input, and its resolved plan are also kept
under `debug/` when applicable. Parameters are not duplicated into a separate
`params.json`. Writers and readers use the same responsibility-based layout.

Each study stores its authored input, resolved plan, executable case, and
input manifest once in an immutable `generations/g_<id>/` directory. Individual
electrical evaluations use the same `summary.json` / `data/` / `debug/` run
layout under the study artifact area. `evaluations.csv` contains one flat
candidate/scenario/control row per electrical solve and is the detailed source
for every explored candidate. `study_history.json` keeps one compact optimizer
row per candidate. Only the selected design is also stored as
`best_candidate.json`; it keeps each of that candidate's control trials once
and identifies the selected trial by `selected_trial`. The
`pcd.results.selected_evaluation` reader follows this single indexed form.
Content-addressed raw
simulation results remain separate because they are reusable computation
cache, not published engineering decisions.
For an advanced external SPICE deck, recursively included files and selected
library sections are inlined into `imported_netlist.cir`; their original bytes
are also content-addressed and listed separately in `input_manifest.json`.
The root `study_result.json` is written last and points to the active generation.
Rerunning the same case publishes a new complete generation while preserving
the previous generation and reusable raw simulation cache entries. If a rerun
is interrupted, the previous `study_result.json` and its declared artifacts
remain authoritative; an unpublished generation may remain for inspection or
later cleanup. `pcd result-prune STUDY_ROOT --keep 3` previews old immutable
generations; add `--apply` to remove only those listed. The reusable raw cache
is never removed by this command. The former write-only `evaluations/` JSON
cache was removed; evaluation interpretation is recomputed from raw data and
published once in the active generation.
`study_result.json.best` is the compact decision surface: selected fixed
candidate, acceptance status, solved/accepted condition counts, and each
condition's selected control, objective values, and failed constraints.
`study.metadata` also states the analysis type, fixed frequency when one
exists, reference plane, and reference impedance. `artifacts.best_candidate`
points directly to the selected candidate JSON, which in turn identifies the
exact ngspice manifest for every selected condition; no copied `best/`
directory is created.
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
prefixes and includes raw status, selection flag, metrics, constraints, raw
cache identity, duration, and artifact paths. Constant circuit metadata and
merged parameters are not repeated as `observation.*` columns. `table_schema`, generation-specific
`dataset_id`, case schemas, and runtime/solver fingerprints make rows safe to
combine across studies without relying on directory names. It is intended as
the direct analysis source and the input to `ml-prepare`; candidate-level
`study_history.json` remains the compact optimizer trace. `selected_control`
is an outcome of evaluating all controls, not an independent input feature;
using it to predict that same outcome would leak target information.
For numeric max/min limits, each constraint also has a signed `.margin`
column in the metric's physical unit: `limit - value` for a maximum and
`value - limit` for a minimum. Positive values are feasible reserve, zero is
the boundary, and negative values are violations. `.violation` remains the
non-negative normalized quantity used by feasibility-first ranking. Missing
measurements have no numeric margin rather than a fabricated large penalty.
`study_history.json.constraint_margins` stores the worst observed selected-
scenario margin per named constraint.
For a multi-objective study, `pareto_front.csv` is a smaller decision projection
over the same candidate evidence. It is not a second metric cache and does not
contain failed or constraint-rejected candidates.

### Preparing a surrogate dataset

Prepare one committed study without rerunning ngspice:

```powershell
uv run pcd ml-prepare runs/<study> --out runs/ml/<study> --test-fraction 0.2 --seed 0
```

The output directory contains only `dataset.csv` and `manifest.json`; it must
not be the active immutable generation. The preparation boundary verifies the
`evaluation_table.v2` identity, committed row/candidate counts, the unique
Candidate x Scenario x Control grain, and finite declared objectives on every
successful solve. Failed solves remain rows for failure classification and are
marked ineligible for objective regression instead of being silently dropped.
The manifest also reports train/test class counts for solver success,
feasibility, and every constraint label, so a one-class holdout cannot be
mistaken for calibrated constraint evaluation.

Features are only `design.*`, `scenario.*`, and `control.*` columns. Declared
`metric.*` objectives, other response metrics, constraint labels, feasibility,
and solver status are explicit targets or context. Cache keys, timings, errors,
artifact paths, and constant fingerprints stay in the manifest rather than the
training table. `selected_control` is omitted entirely.

The deterministic split hashes the complete fixed-design value vector, not a
candidate label, so a repeated design and all its Scenario/Control evaluations
remain together. Its validity claim is limited to unseen fixed designs inside
the same archived case. Cross-case or independent measurement-series
generalization requires explicit series identity and case-level predictors and
is not inferred from this export. Model fitting and candidate proposal are not
performed by `ml-prepare`; every later proposal still needs the normal complete
ngspice evaluation.

Evaluate the prepared fixed holdout without rerunning ngspice:

```powershell
uv run pcd ml-evaluate runs/ml/<study> --out runs/ml-evaluation/<study>
```

The evaluator accepts finite numeric features only. It applies a declared
`log10` transform to log-scale numeric variables, standardizes from training
rows, and compares a fixed distance-weighted 3-neighbour model against the
training mean (regression) and majority class (classification). There is no
hyperparameter search or split selection. `predictions.csv` retains test-row
identity and actual/baseline/surrogate values; `evaluation.json` reports
MAE/RMSE/R², confusion counts, balanced accuracy when both classes exist,
runtime, and an explicit evidence gate. Source evaluation timing is summarized
only from uncached rows and remains excluded from features.

A one-class train or test label is insufficient classification evidence even
if accuracy is 100%. The evidence gate always leaves Bayesian optimization
disabled: this command measures response approximation, not uncertainty,
candidate-ranking regret, or a reduction in ngspice calls.

If the fixed holdout lacks both constraint classes, do not change its seed to
obtain a favorable split. Run a separately declared fixed-design study,
prepare it with `ml-prepare`, and supply it as validation evidence:

```powershell
uv run pcd ml-evaluate runs/ml/<study> `
  --constraint-validation runs/ml/<validation-study> `
  --out runs/ml-evaluation/<study>
```

The two manifests must declare identical features, transforms, objectives, and
classification targets. Their committed dataset IDs and fixed-design values
must differ. The evaluator fits scaling and the locked 3-neighbour model only
on the original training split, then scores every row of the separate dataset;
that dataset's own train/test labels are intentionally irrelevant. It writes
`constraint_validation_predictions.csv` and reports class counts, confusion,
balanced accuracy, and the majority-class baseline. This establishes only
interpolation evidence for a separate simulated design set in the same circuit
family, not measurement validation or cross-circuit generalization. The
reproducible RC protocol is
[`bench/ml/generic_rc_constraint_boundary.yaml`](../bench/ml/generic_rc_constraint_boundary.yaml).

Candidate ranking is not an input-schema or CLI feature. A preregistered
offline gate in [`bench/ml/`](../bench/ml/README.md) replays a fixed
three-neighbour order over two fully evaluated 81-candidate pools, compares it
with 101 fixed random orders sharing the same six initial observations, and
then reruns the selected candidate through the normal study path. The observed
savings were 14.3% for transient RC sizing and 29.6% for RF matching, both
below the fixed 30% requirement. Consequently there is no sequential ML
proposal or Bayesian-optimization option to configure; use exact grids or the
advanced seeded Differential Evolution path for design selection.

## Responsibility boundary

The public format deliberately does not accept plasma geometry, cross
sections, reaction chemistry, electron density, or inferred sheath state.
Those belong in a separately qualified plasma model or dataset. PCD accepts
their electrical consequence at a declared terminal and answers circuit
selection, tuning-authority, frequency replay, and component-feasibility
questions within that evidence boundary.
