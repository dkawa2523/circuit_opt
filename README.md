# PCD — semiconductor-equipment RF circuit studies

PCD evaluates whether one fixed RF matching network remains electrically
acceptable over a declared chamber-load window after only the tuner settings
available in the equipment are adjusted.

It is a circuit-model foundation, not a plasma solver. Chamber or plasma
information enters as a qualified electrical one-port at a named reference
plane: measured or supplied R+jX points, a time-ordered quasi-static R+jX
profile, an effective CCP R-L-C fit, or an effective ICP transformer fit. PCD does not infer density, sheath geometry,
chemistry, species power, temperature, or self-consistent plasma/circuit
feedback.

Advanced explicit transient cases may also impose a measured or externally
calculated positive `R(t)` on a declared resistor. This is one-way circuit
excitation, not self-consistent plasma evolution; its source CSV is archived
and its terminal voltage/current can be observed directly.

A case-local etch-CCP verification problem now exercises a richer 60 MHz
upper RF + 2 MHz lower RF + pulsed negative-DC circuit. It keeps matching,
feedthrough, blocking, choke, electrode/ESC stray, and wall-loss elements
explicit, while charge/flux laws define time-varying bulk `R/L` and both
sheath capacitances. An independent MNA target checks voltage, current, charge,
flux, and reflected voltage; an 81-point exact grid then identifies four
bounded on-state values from wafer voltage alone. This is evidence for explicit
circuit modeling, not a new generic plasma-physics feature. See the
[problem statement and results](docs/etch-ccp-dcs-dynamic-impedance-study-ja.md)
and the [six evidence figures](bench/figures/README.md#dynamic-impedance-etch-ccp-with-dual-rf-and-dc-superposition).

A worked dual-frequency CCP example combines a 40 MHz upper excitation, an
800 kHz rectangular lower bias, fixed effective sheaths, and a prescribed
plasma `R(t)`. It verifies wafer/chuck voltage and the upper-port reflected
voltage wave against an independent charge equation. Two separate exact-grid
inverse studies then recover a bounded four-point `R(t)` profile from either
observable alone and check the unused observable as held-out circuit evidence.
See the [problem statement and results](docs/dual-frequency-ccp-pulsed-bias-study-ja.md)
and the [four evidence figures](bench/figures/README.md#dual-frequency-ccp-with-rectangular-lower-bias).

PCD v1 optimizes component values and selects among declared, validated circuit
templates. It does not generate arbitrary circuit graphs, optimize physical
PCB/chamber placement, solve plasma chemistry, or perform self-consistent
plasma-circuit evolution. The three ordinary workflows are `sim-run` for one
solve, `analyze` for saved-result measurements and figures, and `run` for a
complete Scenario/Control design study or optimization. ML commands are
optional evidence tools and do not replace these workflows.

## Smallest useful study

```yaml
schema: pcd.rf.v1
case_id: production_window
frequency_Hz: 13560000

network:
  type: pi_match
  fixed: {L1: 6.43e-7}
  tuning:
    C1: [4.8e-10, 6.7e-10, 8.9e-10]
    C2: [4.5e-11, 1.6e-10, 2.1e-10]

load:
  type: impedance_table
  file: load_points.csv
  reference_plane: electrode_terminal

acceptance:
  reflected_power_fraction_max: 0.10
```

`load_points.csv` has one stable shape:

```csv
scenario_id,resistance_ohm,reactance_ohm,weight
nominal,25,-80.77,1
high_R,50,-80.77,1
low_R,12.5,-80.77,1
```

Run it with one command:

```powershell
uv sync --group dev
uv run pcd solver-diagnose
uv run pcd run case.yaml
```

When using Codex inside this repository, the project-local `$pcd-runner` skill
can select the appropriate validation, one-simulation, study, benchmark, or
result-inspection workflow. Its instructions live under
`.agents/skills/pcd-runner`; it is not installed as a user-wide skill.

ngspice must be installed and available on `PATH`; on Windows the console
binary `ngspice_con.exe` is preferred for unattended runs. If local application
policy blocks generated console-script launchers, the equivalent entry point
is `uv run python -m pcd`. `solver-diagnose` checks discovery and reports the
exact executable/version before a long study starts.

Results default to `runs/`. The terminal shows the selected fixed candidate,
analysis type, fixed frequency when applicable, reference plane, condition
coverage, selected tuner state per condition, declared objectives, failed
limits, and the result directory. It also prints the selected candidate
evidence path. `study_result.json` retains the same compact decision together
with every detailed result reference. Use `--json`
for the machine-readable result or `--output path` to choose the root
directory. For CI or unattended qualification, add `--require-acceptance` so
an electrically solved but rejected design also produces a nonzero exit code.
The active generation contains the complete flat `evaluations.csv`, compact
candidate-level `study_history.json`, and one structured `best_candidate.json`
for the selected design. Reusable raw solver output is cached separately;
non-selected candidate JSON and interpreted evaluation JSON are not written a
second time.

The five real-ngspice PCD v1 user paths, their executable release suite, and
the latest separated circuit/ML decisions are kept in
[`bench/release/README.md`](bench/release/README.md). The 2026-09-30 closure
reproduced all seven inputs with 131 fresh ngspice evaluations: the circuit
foundation is GO, while ML candidate proposal and the full originally
requested scope remain NO-GO. A later topology-aware ML renewal does not alter
that historical decision: it first adds a solver-neutral graph/corpus boundary
for declared structured templates, and still provides no online proposal
command until held-out-topology and solver-saving gates pass.

### Optional surrogate evidence

For topology-aware response learning, export any number of completed AC
studies without rerunning ngspice:

```powershell
uv run pcd ml-corpus runs/<study-a> runs/<study-b> --out runs/ml-corpus/<name> --seed 0
```

This writes `graphs.jsonl`, `samples.csv`, `component_responses.csv`, and
`manifest.json`. The graph contains logical components and ports, while
zero-volt observation sources and internal rendering nodes remain solver
details. Samples retain source/load/scenario context and complex port phasors;
component phasors are present when the source case observed that component.
The manifest separates design, external-condition, and topology-family
holdouts. A split is marked unavailable instead of making a generalization
claim when there are too few groups or the split is confounded with topology.
This command prepares data only. Compare the fixed M2 reference models without
rerunning ngspice:

```powershell
uv run pcd ml-corpus-evaluate runs/ml-corpus/<name> --out runs/ml-comparison/<name> --seed 0
```

The comparison uses the corpus-declared design, external-condition, and
leave-one-topology-family-out protocols for a training mean, ridge regression,
a two-hidden-layer MLP, and a small relation-specific component/net GNN. It
does not tune on test rows, save a deployable model, propose components, or
replace final ngspice evaluation. Read `evaluation.json` before using
`predictions.csv`; an unavailable protocol or a constant-baseline winner is a
stop result, not permission to reshuffle or add optimization.

The older fixed-topology metric surrogate remains available for its historical
holdout evidence. Create its role-limited export outside the immutable
generation:

```powershell
uv run pcd ml-prepare runs/<study> --out runs/ml/<study> --seed 0
uv run pcd ml-evaluate runs/ml/<study> --out runs/ml-evaluation/<study>
```

When a separately executed fixed-design study was reserved for constraint
validation, prepare it in the same way and attach it without changing the
original holdout:

```powershell
uv run pcd ml-prepare runs/<validation-study> --out runs/ml/<validation-study> --seed 0
uv run pcd ml-evaluate runs/ml/<study> --constraint-validation runs/ml/<validation-study> --out runs/ml-evaluation/<study>
```

This writes `dataset.csv` and `manifest.json`. The manifest explicitly names
input features, objective/constraint labels, failed-row eligibility, excluded
runtime metadata, and the deterministic train/test groups. Identical fixed
design values remain in one split across every Scenario/Control row, and
`selected_control` is excluded because it is a post-evaluation outcome. The
split measures unseen designs within that one archived case; it does not claim
generalization to another circuit or independent measurement campaign.
`ml-evaluate` then writes holdout `predictions.csv` and `evaluation.json`. It
compares a fixed distance-weighted three-neighbour model with the training mean
or majority-class baseline; declared log-scale numeric variables are transformed
before standardization. Only finite numeric features are accepted. One-class
train/test labels are reported as insufficient classification evidence even
when their raw accuracy is 100%. The command does not tune a model, propose a
candidate, claim saved ngspice evaluations, or enable Bayesian optimization.
The optional validation dataset must have the same roles and transforms, a
different committed dataset identity, and no overlapping fixed-design values.
Its rows are used only for constraint scoring; fitting still uses the original
training split. Per-row results are written to
`constraint_validation_predictions.csv`.

The separate preregistered ranking gate is documented in
[`bench/ml/README.md`](bench/ml/README.md). Its fixed RC and RF pools achieved
14.3% and 29.6% fewer evaluations than their fixed-random medians, below the
required 30% in both cases. PCD therefore has no ML candidate-proposal command
and makes no claim of saved ngspice calls. Deterministic grid and Differential
Evolution remain the decision optimizers; Latin hypercube is available as a
non-adaptive initial-design and comparison baseline.

When two or more objectives are declared, it also contains `pareto_front.csv`.
That table includes only fully solved, fully feasible candidates and identifies
the ordinary lexicographic selection without replacing it. A sampled continuous
study reports an observed front, not a claim about the unsearched design space.
For an `impedance_profile`, the active generation also contains the concise
`snapshot_response.csv`: one selected row per time point with load and input
R/X, reflection, forward/reflected/load power, selected tuning, and acceptance.

For one simulation without candidate scoring, use `pcd sim-run case.yaml`.
Its saved result can be inspected again without rerunning ngspice:

```powershell
uv run pcd analyze runs/<simulation-run>
```

A new simulation run has three top-level entries: `summary.json` is the small
user-facing outcome, `data/` contains the canonical transient/AC tables, and
`debug/` contains the replay case, exact netlist, solver log, and detailed
manifest. Parameters are recorded once in the debug manifest instead of a
duplicate `params.json`. Saved-run tools read this same layout, so artifact
creation and replay have one path rather than a historical-format branch.

`analyze` writes a concise `summary.json` plus the applicable impedance/Smith,
transient, power, and component-stress figures. It reports electrical
measurements only; engineering acceptance remains the responsibility of a
study so a solver success is never presented as a design pass.

For a multi-point AC sweep, `analyze` also reports the sampled best match and,
only when both half-power crossings are inside the sweep, the power-response
resonance, -3 dB bandwidth, fractional bandwidth, and loaded Q. The response
uses measured load real power when its probes are present, otherwise
`1 - |Gamma|^2`; this is not an unloaded component Q. Periodic transient runs
also receive a voltage/current harmonic comparison relative to H1.

## What users specify

| input | engineering meaning | changes when |
|---|---|---|
| `network.fixed` | selected hardware | a new candidate is built |
| `network.search` | hardware values PCD may search | a new candidate is proposed |
| `network.tuning` | discrete equipment settings | independently for each load condition |
| load table / `conditions` | external electrical conditions | imposed on the candidate |
| `acceptance` | pass/fail engineering limits | never optimized away |

The internal Candidate, Scenario, and ControlState types enforce those roles,
but users do not need to write them. A parameter cannot belong to more than one
role.

PCD enumerates every declared tuning combination for every condition. It does
not sample an incomplete tuner grid and call the result infeasible. The
default safety limit is 250 states; a deliberately larger study may set
`execution.control_state_limit`.

The same rule applies to hardware selection: every `network.search.values`
combination is compared exactly once and the candidate count is inferred.
The public RF input does not turn a continuous range into an arbitrary sample
and present that sample as a completed design decision. Exploratory continuous
optimization remains available only in the advanced explicit extension path.
There, `differential_evolution` searches bounded numeric design variables with
a recorded seed and uses the same feasibility-first scenario result as final
candidate ranking; engineering constraints are not hidden inside the waveform
objective. `random` remains a simple mixed/categorical exploration baseline,
not the recommended continuous optimizer.
For a multi-objective advanced case, PCD derives a direction-aware observed
Pareto front from the completed candidate evidence. It does not turn
Differential Evolution into a hidden weighted-sum optimizer, and it labels a
front as complete only for an explicitly enumerated grid.

## What PCD resolves automatically

The public input is compiled once before validation or simulation. The
resolved plan explicitly contains:

- the RF source, standard nodes, load ports, and 50-ohm measurement plane;
- the same frequency for source, impedance anchor, and one-point AC solve;
- scenario-column mappings and a complete tuning-state budget;
- reflection objective, engineering constraints, solver, and the exact
  candidate count;
- component probes and effective series-loss elements only when requested.

No numerical layer reinterprets the short input independently. Every run
archives `input_case.yaml`, `resolved_plan.yaml`, and executable `case.yaml`,
so inferred defaults remain reviewable and replayable. Command-line execution
overrides are applied before validation, hashing, and archival; the stored
resolved plan therefore describes the run that actually occurred.

See [the RF input reference](docs/input-format.md) for all supported fields and
[RF load models](docs/rf-load-models.md) for model responsibility and limits.
The current code audit, simplification decisions, and staged implementation
plan are in [the foundation renewal plan](docs/foundation-renewal-plan-ja.md).

## Component stress and effective loss

Absolute stress requires a declared drive amplitude:

```yaml
drive_peak_V: 100
network:
  type: pi_match
  fixed: {C1: 2.58e-10, L1: 1.20e-6, C2: 7.45e-12}
  loss_ohm: {C1: 0.1, L1: 0.5, C2: 0.1}
acceptance:
  reflected_power_fraction_max: 0.10
  component_limits:
    L1: {current_rms_A_max: 1.0, loss_W_max: 0.5}
  loss_balance_fraction_max: 1.0e-5
```

PCD reports terminal RMS/peak voltage and current, effective loss, network
efficiency, source-terminal RMS current/apparent power, and electrical loss
closure. With an explicit drive, every named component in the public matching
topology is observed automatically; limits remain optional. Effective ESR/DCR
is not an internal temperature or lifetime model. Details are in
[component loss and stress](docs/component-loss-stress.md).

## RF load choices

| public load type | appropriate input | boundary |
|---|---|---|
| `impedance_table` | independent R+jX points, optionally with `frequency_Hz` per row | no interpolation between supplied points |
| `impedance_profile` | strictly time-ordered `time_s, resistance_ohm, reactance_ohm` observations | independent AC snapshots; no state propagation |
| `impedance_point` | one R+jX value at one frequency | exact only at that anchor |
| `ccp_lumped` | qualified effective series R-L-C parameters | no sheath state or species-power inference |
| `icp_transformer` | qualified coil plus identifiable reflected-loading fit | terminal model only; no density or plasma-power split |

`reference_plane` is required. `evidence` is optional so exploratory work can
run, but strict validation warns when applicability has not been documented.
Referenced scenario, target-waveform, and external-netlist files are archived
once per study by content hash, so the executable `case.yaml` replays from the
study bundle rather than depending on the original file location.
Advanced component resistance profiles use that same archive and fingerprint
path. See
[`time_varying_resistor.yaml`](examples/advanced/time_varying_resistor.yaml)
for a runnable transient whose ngspice result is checked against a closed-form
resistive divider.

A quasi-static profile additionally requires an absolute `drive_peak_V`,
either once in the case or in every CSV row, because its output includes power
in watts. Each row is solved at its declared frequency as an independent
small-signal AC condition. PCD does not interpolate the measured profile or
claim that the circuit/plasma state evolves between rows. The runnable
[`rf_quasi_static_profile.yaml`](examples/rf_quasi_static_profile.yaml)
demonstrates the complete input and output path.

## Core electrical benchmarks

The core suite contains twelve concise public RF cases and four explicit
advanced/boundary cases:

- A1-A3: independent complex-impedance goldens for every public topology;
- A4: multi-frequency E2E for the effective CCP R-L-C port;
- A5: multi-frequency public-input E2E for the effective ICP transformer port;
- B1-B3: fixed, limited-control, and full-control design decisions;
- B4: independent synthetic frequency-point replay;
- B5: match passes but high-drive component limits fail;
- B6: complete three-value hardware search;
- B7: Candidate x Scenario x Control orthogonal enumeration;
- B8: deterministic full-factorial component-value corner stress;
- D1-D3: equivalent lossy reference-plane representations and a fixture
  double-counting negative control.

Expectations and explanatory text live separately in
`bench/expectations.yaml`. Reproduce all 411 real-ngspice evaluations with:

```powershell
uv run python bench/run_suite.py --run-root runs/benchmark_suite
```

These cases establish circuit-pipeline behavior and bounded electrical design
decisions, not a qualified reactor process window.

The concise benchmark report is stored at
[`output/reports/benchmark-report.md`](output/reports/benchmark-report.md),
with a machine-readable summary and flat core/literature evidence tables in
the same directory. See [`bench/reports/README.md`](bench/reports/README.md)
for its reproducible generation sequence and interpretation boundary.

## Code structure

```text
.agents/skills/pcd-runner/ repository-local Codex execution workflow
pcd/plan.py       public RF input -> explicit resolved execution plan
pcd/case.py       schema routing, case paths, and design-variable discovery
pcd/api.py        supported imports for Python callers and plugins
pcd/study_config.py advanced Candidate/Scenario/Control translation
pcd/core/         role types, scenario/control selection, aggregation
pcd/study.py      evaluation execution, caching, and study result assembly
pcd/search.py     optimizer interface, exact grids, and random baseline
pcd/continuous_search.py bounded differential-evolution candidate search
pcd/ml/           offline dataset, surrogate, and fixed-pool ranking evidence
pcd/sim_core.py   one simulation and its immutable run artifacts
pcd/simulation.py solver-neutral analysis/result types
pcd/simulation_input.py Case -> typed solver execution input
pcd/probes.py     Case-free waveform/AC/component probe declarations
pcd/component_models.py Case -> observed component and ESR/DCR resolution
pcd/component_profiles.py prescribed R(t) CSV validation and SPICE rendering
pcd/netlist.py    circuit IR and ngspice rendering
pcd/circuit_graph.py structured component-terminal-net records for offline ML data
pcd/solver.py     typed ngspice process adapter
pcd/ngspice_io.py ngspice output -> canonical arrays
pcd/netlist_import.py execution-preserving external-netlist import
pcd/netlist_parse.py drawing-oriented topology parsing
pcd/netlist_viz.py parsed-topology rendering
pcd/sim_methods.py named circuit, load, and solver implementations
pcd/spice.py      SPICE parameter resolution and value formatting
pcd/artifacts.py  run serialization and implementation identity
pcd/records.py    stable reads and artifact-path resolution for saved runs
pcd/analysis/     AC, periodic-transient, and component-stress measurements
pcd/reporting/    saved-run electrical summary and standard figures
pcd/metrics.py    metrics and engineering-limit evaluation
pcd/rf_loads.py   electrical RF-load equations and parameter validation
pcd/results/      content-addressed result storage and summaries
pcd/signals/      reusable time-series and phasor primitives
```

The dependency flow and directory ownership are summarized in
[the architecture guide](docs/architecture.md). Third-party extensions should
import documented types and decorators from `pcd.api`; internal module helpers
may change within a minor release. Each extension receives an isolated case and
parameter snapshot, so accidental mutation cannot alter the study fingerprint
or another evaluation.

Solver extensions use `register_solver` and accept one `SolverRunRequest`.
Every solver follows this boundary; adapters do not receive the whole Case
dictionary merely to obtain analysis or execution settings.

The explicit `case_yaml.v1` form remains an advanced extension path for custom
netlists, plugins, transient studies, and generic waveform objectives. In a
case, use either the single `source` mapping or the `sources` list, not both.
In a multi-source advanced case, `measurement.current_source` selects the single AC
excitation and measured input port; other structured sources are AC-grounded,
and a floating source is measured across its declared `p`/`n` terminals. Both
public and advanced studies use `pcd run`; `sim-run` is reserved for one
simulation without scoring. A solver-reported failed `sim-run` exits nonzero
while retaining its run record; `--allow-failure` is available for deliberate
collection of those known solver outcomes. Invalid input exits with code 2 and
is not converted into a fake waveform or bypassed by `--allow-failure`.
Source selection, probe columns, solver analysis, and measurement/load-port
shape are resolved before a run directory is created, so validation and
execution use the same interpretation. Circuit and load builders are also run
in memory before artifact allocation; missing RF-model parameters, nonphysical
values, malformed components, and invalid imported-netlist policies therefore
do not leave partial run directories. Unexpected extension errors propagate.
`Circuit.add` supplies both rendering and logical graph semantics. A custom
builder with solver-only meters/internal nodes should declare the physical
element with `add_graph_component` and render only those internal statements
with `raw(..., graph_neutral=True)`; raw physical SPICE without declared
terminal semantics is intentionally excluded from `ml-corpus`.
New RF matching studies should use `pcd.rf.v1`.

For `circuit.builder: from_netlist`, authored `.param`, `.model`, `.options`,
conditionals, and subcircuits are retained. Relative `.include` dependencies
and selected `.lib` sections are resolved recursively and inlined into the
portable execution snapshot; every contributing source file is hashed in the
input manifest. The case owns the source, analysis/control block, output
files, and final design-parameter overrides, so imported `.control`, `.ac`,
`.tran`, and `.end` statements are not copied into the generated run deck.
Set `circuit.netlist_mode` to `deck` for a complete SPICE file with its
mandatory first-line title, or leave the default `fragment` for circuit
statements with no title. The default `source_policy: replace_named` removes
only imported independent sources whose names exactly match generated case
sources; `preserve` keeps every imported source. PCD never guesses a source
conflict from a shared node.

Advanced transient RF cases may tune the measurement policy without replacing
the metric implementation:

```yaml
measurement:
  load_current: auto
  periodic_cycles: 5
  settling_comparisons: 3
  settling_tolerance: 1.0e-4
  harmonic_count: 7
```

Defaults remain 3 measured cycles, 2 adjacent-cycle comparisons, `1e-3`
normalized residual, and 3 harmonics. A measurement is accepted only when the
trace contains both the requested measured cycles and enough history for every
settling comparison; PCD reports required and available cycle counts.

## Verification

```powershell
uv run --frozen python -m pytest -q
uv run --frozen python -m nox -s quality-pr
uv run --frozen python -m nox -s architecture-audit
uv run --frozen python bench/run_suite.py --run-root runs/benchmark_suite
```

Runtime output belongs under `runs/`; examples document syntax; tests verify
software and numerical behavior; benchmarks support only their stated
electrical decisions. The architecture audit combines Ruff complexity checks,
Pyrefly, Import Linter, and Radon hotspot reports; Radon scores are review
evidence rather than blanket pass/fail thresholds.
