# Architecture and ownership

PCD has one numerical path for public RF studies and advanced explicit cases:

```text
authored YAML/JSON
        |
        v
case.py -> plan.py (pcd.rf.v1 only) -> executable Case
        |                                      |
        v                                      v
simulation_input.py                       circuit/load
        |                                      |
        v                                      |
ResolvedSimulationCase                         |
(SolverSettings, AnalysisRequest, ProbePlan,    |
 MeasurementReference)                         |
        \______________________________________/
                              |
                              v
                         netlist.py
                              |
                              v
                 SolverRunRequest -> solver.py
                                         |
                                         v
                             simulation.SimulationResult
                                         |
                                         v
                             sim_core.py -> SimRecord
                                         |
                                         v
                    records.py -> analysis/ -> metrics.py -> core/StudyRunner
                         |                            |
                         v                            v
                    reporting/           results/ + evaluations.csv -> ml/dataset
                         |                            |                |
                         v                            |                v
                    summary + standard figures       |          dataset + manifest
                                                      |                |
                                                      v                v
                                                ml/ranking       ml/evaluation
                                                      |                |
                                                      v                v
                                           retrospective gate   holdout evidence
```

The arrows show data flow, not every Python import. Public RF input is compiled
once; simulation, caching, metrics, and reporting consume the same resolved
case rather than interpreting shorthand independently.

The v1 product boundary is deliberately smaller than the possible package
graph: one simulation (`sim-run`), saved-result analysis (`analyze`), and a
complete design study/optimization (`run`). Plasma enters through a qualified
electrical terminal model. Topology optimization means exact comparison of
declared, validated templates; arbitrary graph generation and physical layout
are outside v1. The topology-aware ML renewal may learn across those declared
templates, but it remains offline until held-out-topology prediction and
measured solver-call savings both pass. Every selected candidate returns
through `run`; a learned prediction is never the final circuit result.

Completed AC studies have one additional offline projection:

```text
Circuit -> circuit_graph.py -> circuit_graph.v1 ----------------\
                                                                > ml/corpus.py -> corpus tables
saved study -> corpus.py -> canonical AC responses + context --/                   |
                                                                                  v
                                                                  ac_graph_corpus.v1 files
                                                                                  |
                                                                                  v
                                                   corpus_evaluation.py -> ml/ac_evaluation.py
```

This path records component, terminal, net, and boundary-port semantics. It
does not parse arbitrary SPICE, execute a solver, fit a model, calculate
engineering acceptance, or introduce a second simulation path.

## Directory responsibilities

| path | owns | does not own |
|---|---|---|
| `pcd/core/` | immutable study roles, evaluation order, feasibility-first aggregation, Pareto eligibility and dominance | circuits, ngspice, search algorithms |
| `pcd/signals/` | cleaned numerical series, periodic windows, phasors, time-weighted power | case schemas or engineering limits |
| `pcd/analysis/` | pure AC, periodic-transient, and component-stress measurements | Case parsing, persistence, solver execution, report rendering |
| `pcd/reporting/` | saved-run orchestration, concise electrical summaries, and standard figures | solver execution, engineering acceptance, publication-specific layouts |
| `pcd/circuit_graph.py` | framework-neutral component-terminal-net records and stable structure/instance fingerprints for structured circuit templates | SPICE parsing, solver execution, ML tensors, topology generation |
| `pcd/corpus.py` | committed-study discovery, identity checks, physical graph/context reconstruction, provenance, and corpus file writing | response shaping, split policy, solver execution, model fitting, online proposal |
| `pcd/corpus_evaluation.py` | corpus artifact/hash verification and model-comparison file writing | model mathematics, solver execution, candidate proposal |
| `pcd/ml/` | pure AC corpus shaping, graph encoding, fixed ridge/MLP/GNN comparison, leakage-aware groups/splits, dataset validation, external constraint scoring, and retrospective ranking | file discovery, persistence, online candidate proposal, solver execution |
| `pcd/` | application planning, circuits, execution, measurements, persistence | apparatus qualification data |
| `examples/` | runnable syntax and extension examples | regression truth |
| `tests/` | software, numerical, and boundary regression checks | scientific qualification claims |
| `bench/` | reproducible electrical and literature-derived evidence | runtime package behavior |
| `bench/release/` | public-CLI user-workflow orchestration and separated circuit/ML release decisions | solver implementation, metric ownership, process qualification |
| `docs/` | supported inputs, interpretation, and model limits | generated run artifacts |
| `runs/` | ignored, reproducible execution/cache data | reviewed publication outputs |
| `output/` | reviewed, versioned report and figure products | mutable run caches |
| `quality/` | the deliberate coverage floor | custom test frameworks |

## Module boundaries

- `case.py` loads one schema and owns path resolution. `plan.py` is the only
  public-RF shorthand compiler. A public quasi-static impedance profile is
  compiled into ordered, independent one-point AC scenarios; it does not add a
  second solver path or a dynamic state model.
- `circuit_graph.py` owns the portable graph record used to describe a
  structured circuit independently of ngspice and any ML framework.
  `netlist.circuit_to_graph` is the small adapter from an already-built
  `Circuit`. A builder declares logical physical components separately from
  its ngspice rendering, so observation meters and internal ESR/DCR nodes do
  not alter the topology. Raw SPICE and prescribed time profiles remain
  graph-unavailable instead of receiving guessed terminal semantics.
  `topology_family` is the declared cross-design grouping key;
  `topology_fingerprint` detects authored wiring changes and is explicitly not
  a general graph-isomorphism proof.
- `corpus.py` is the offline application adapter for committed studies. It
  rebuilds only the structured logical graph, follows canonical saved AC
  artifacts, and writes the prepared files with source provenance. Pure
  response validation, target shaping, and split policy live in
  `ml/corpus.py`, so they can be tested or reused without filesystem or study
  dependencies. The result has distinct graph, sample/port, and long-form
  component-response tables. It groups identical designs and
  external conditions before splitting, requires train/test topology coverage
  for a claimed known-topology split, and marks insufficient/confounded splits
  unavailable. It neither invokes ngspice nor imports study/search execution.
- `corpus_evaluation.py` is the file adapter for `ac_graph_corpus.v1` model
  comparison. `ml/graph_encoding.py` owns flat and component/net encodings,
  `ml/neural.py` owns the fixed NumPy MLP and relation-specific message passing,
  and `ml/ac_evaluation.py` alone owns folds, metrics, and comparison decisions.
  These model modules cannot import corpus files, study orchestration, a solver,
  or optimization. The command writes evidence, not a deployable model.
- `study_config.py` translates an executable case into Candidate, Scenario,
  ControlState, Objective, and Constraint objects. It also copies only the
  small electrical interpretation context (analysis type, fixed frequency,
  reference plane, and Z0) into study metadata; measurement calculations stay
  in `analysis/`.
- `search.py` owns the small optimizer contract, exact finite candidate grid,
  registry adapters, and mixed-space random baseline. `continuous_search.py`
  owns only bounded-coordinate translation plus `DE/rand/1/bin` proposal and
  replacement state. Neither can execute a solver, reinterpret constraints,
  scenarios, or controls, or alter the public RF candidate grid.
- `component_profiles.py` alone loads, validates, and renders CSV-backed
  prescribed `R(t)` declarations. It owns interpolation, repeat-boundary,
  passivity, and transient-resolution rules; it does not model plasma state or
  support C/L without an energy-consistent constitutive law.
- `simulation.py` owns the solver-neutral `AnalysisRequest`, canonical result
  columns, and `SimulationResult`. `TransientAnalysis` and `AcSweep` enforce
  the positive, finite, internally consistent values required by every solver.
  This module knows neither Case nor ngspice syntax.
- `probes.py` owns Case-free `ProbePlan` and `ComponentObservation`
  declarations plus deterministic probe naming. `component_models.py` alone
  resolves observed component declarations and ESR/DCR from a Case.
  `simulation_input.py` is the single Case-to-solver-input adapter. It resolves
  source declarations, `SolverSettings`, `AnalysisRequest`, `ProbePlan`,
  `MeasurementReference`, and netlist options into a deliberately small
  `ResolvedSimulationCase`. `validation.py` compiles that same input and the
  same in-memory circuit/load model; it does not maintain second parsers or
  RF-load equations. It owns report formatting and cross-section engineering
  checks that cannot belong to one typed value.
- `netlist.py` builds the in-memory circuit and renders deterministic ngspice
  text, including the ngspice control block. Circuit/load section shape,
  builder selection, and builder return types are resolved into one
  `NetlistInputs`; built-in builder parameter requirements and physical domains
  live with their implementation in `sim_methods.py` and `rf_loads.py`.
  Netlist source rendering and imported-source replacement both use the source
  resolver from `simulation_input.py`. `netlist_import.py` preserves executable external statements;
  `netlist_parse.py` intentionally extracts only the topology needed to draw.
  Transient rendering sets an explicit ngspice `tmax` equal to the typed output
  step, so time-profile resolution is independent of ambient ngspice settings.
- `solver.py` executes ngspice from a typed `SolverRunRequest`; it does not read
  Case dictionaries. `ngspice_io.py` alone translates historical `wrdata`
  layouts into canonical arrays. `sim_core.py` allocates a run directory only
  after simulation and circuit/load inputs have been constructed, persists
  solver outcomes without reclassifying exceptions, and owns the three-part layout:
  user-facing `summary.json`, canonical `data/`, and replay-only `debug/`.
  `records.py` is the read-only artifact boundary and follows that same layout.
- `analysis/ac.py` owns frequency-domain impedance, reflection, and port-power
  calculations. For actual multi-point solver sweeps it also owns the dominant
  power-response peak, bracketed half-power bandwidth, and loaded Q; it does
  not infer an unloaded component Q from a single point. `analysis/transient.py`
  owns periodic windows, settling evidence, time-domain power, harmonics, and
  RF-port measurements.
  `analysis/stress.py` owns component terminal stress and modeled-loss balance.
  `analysis/__init__.py` is a calculation-free compatibility facade for the
  established imports. None of these modules reads Case, persistence, netlist,
  or solver state. `metrics.py` selects a named metric and evaluates
  engineering limits.
- `reporting/` is the read-only saved-run use case. It resolves the archived
  executable Case, loads canonical artifacts through `records.py`, calls the
  pure analysis owners, and writes `summary.json` plus applicable standard
  figures. `harmonics.py` owns only the H1-relative voltage/current comparison;
  the other standard figures remain in `figures.py`. Reporting never reruns the
  solver or decides whether a design passes; the study layer owns acceptance
  and publication-specific figures remain in `bench/`.
- `bench/reports/generate_case_report.py` consumes only the already interpreted
  core and literature suite summaries. It projects them into one concise
  Markdown report, one JSON status, and two flat CSV audit tables. It does not
  import candidate readers, inspect raw solver artifacts, or define another
  benchmark/acceptance contract.
- `bench/release/run_suite.py` is an outside-in release adapter. It calls the
  public CLI, follows the paths returned by simulation and study summaries,
  and checks only the five supported user outcomes. It does not import solver,
  study, analysis, or ML implementation modules. Expected electrical rejection
  is accepted only when the named failed limits and coverage are reproduced.
  An attached P3 result is integrity-checked and reported as a separate ML
  decision; a circuit PASS cannot enable candidate proposal.
- `results/` owns two distinct study stores: reusable physical `raw/` cache
  entries and immutable published `generations/`. The root `study_result.json`
  is the commit pointer written only after a complete generation exists.
  `evaluations.csv` is the complete flat evaluation projection in that
  generation and the only detailed record for non-selected candidates. There
  is no second interpreted evaluation cache: metrics and constraints are
  recomputed from raw data.
  `study_history.json` is the compact optimizer trace and keeps proposal phase,
  seed, duration, objective aggregates, failed constraints/evaluations, and
  scenario coverage without replacing those fields with an opaque penalty.
  For two or more declared objectives, `core.aggregation` derives the
  direction-aware nondominated set only from candidates with complete control
  evidence and full scenario feasibility. `results/report.py` projects it to
  `pareto_front.csv`; it does not change candidate selection or rerun a solver.
  Sampled searches label this as an observed front, while exact enumeration may
  label the declared grid complete.
  The root result names one `best_candidate.json` directly; it is the single
  structured link to every selected condition's ngspice evidence. The execution
  pipeline does not publish candidate files while ranking; the study boundary
  writes this file only after selection completes.
  `results/snapshot.py` is a pure selected-candidate projection for a
  quasi-static study. It writes no solver data and performs no interpolation;
  `study.py` persists its `snapshot_response.csv` beside the full evaluation
  table.
- `ml/dataset.py` consumes an already loaded committed result and evaluation
  table. It verifies dataset identity, Candidate x Scenario x Control grain,
  declared objectives, and failure rows; assigns every retained column an
  explicit feature/target/context role; and keeps identical fixed-design
  values in one deterministic train/test group. It does not discover files,
  fit a model, import a solver, or claim cross-case generalization. The CLI is
  the thin adapter that follows the committed artifact path and writes the two
  derived export files outside the immutable generation.
- `ml/evaluation.py` consumes only that prepared table and manifest. It applies
  declared linear/log feature transforms, compares fixed three-neighbour
  regression/classification with constant training baselines, and reports
  one-class labels as insufficient evidence. It does not search
  hyperparameters, propose circuit values, import study/solver persistence, or
  claim saved ngspice calls. The CLI verifies the dataset hash and writes the
  prediction/evidence artifacts outside the prepared dataset directory.
- `ml/neighbors.py` owns only declared feature transformation, training-row
  standardization, and stable inverse-distance prediction. Both ML evaluations
  use it, so the fixed model is not reimplemented in benchmark code.
- `ml/ranking.py` replays candidate order inside an already evaluated finite
  pool using candidate-level feasibility and objective values. It shares the
  same initial observations with 101 fixed random-order baselines and does not
  import study execution or persistence. `bench/ml/run_ranking_benchmark.py`
  owns case execution, artifact paths, and the fresh full Scenario x Control
  verification. The fixed RC/RF gate failed its 30% requirement, so this
  remains historical offline evidence and no online proposal loop was added.
  The topology-aware renewal does not revise that result. It starts a separate
  evidence path: graph contract, PCD-specific corpus, identical baseline/GNN
  splits, held-out topology tests, uncertainty/OOD reporting, and only then a
  new solver-saving gate. Model code remains downstream of saved data and may
  not import study orchestration or a concrete solver.
- Signed constraint margin is created where a constraint is evaluated, then
  carried as ordinary result data. `study.py` aggregates the worst observed
  margin for result/history output and writes row-level margins to the
  evaluation table. `ml/dataset.py` may label that column as a response target;
  it does not calculate engineering limits or invoke an optimizer.
- `artifacts.py` owns the list of built-in external data dependencies. A
  component profile follows the same content-addressed archive, path rewrite,
  and fingerprint path as scenario tables and target traces.

## Supported extension boundary

Plugins and external Python integrations import from `pcd.api`. A circuit
builder returns `Circuit`, a solver returns `SimulationResult`, a metric returns
a non-empty mapping, and an optimizer derives from `BaseOptimizer`. An
optimizer implements `ask`/`tell`; it may override `proposal_metadata` only for
small JSON-ready provenance fields used by `study_history.json`. Extension
calls receive detached case/parameter data so they cannot mutate the cached
study definition accidentally.

Solver integrations use `register_solver` and receive one
`SolverRunRequest`; they do not receive the full Case dictionary. There is no
signature inspection or Case-based compatibility invocation in the registry.
The nested `SolverSettings`, `AnalysisRequest`, `ProbePlan`, and
`MeasurementReference` types are also exported from `pcd.api` as the solver
extension contract.
Circuit and load plugins continue to receive isolated Case data while their
separate input boundaries are migrated. `Circuit.add` declares the same
two-terminal element for rendering and graph export. A plugin that needs
solver-only instrumentation declares the logical element once with
`add_graph_component`, then renders meters/internal elements with
`raw(..., graph_neutral=True)`. Authored physical raw SPICE must not be marked
neutral merely to force corpus export.

## Failure boundary

Failures become data only at the layer that can identify their meaning:

- `solver.py` returns a failed `SimulationResult` for missing executables,
  timeout, nonzero solver exit, missing requested output, and malformed solver
  output;
- an evaluator may explicitly return a failed `RawResult` for a known
  evaluation outcome;
- `StudyRunner` converts only `UnsettledMeasurementError` to `not_settled`.

`sim_core.simulate_case` does not catch arbitrary exceptions. Invalid model or
method names, incomplete objective definitions, duplicate metric outputs, and
extension implementation errors propagate to the caller. Solver and analysis
settings are resolved before a run directory is allocated. If solver execution
has already been prepared and then an adapter fails, the honest `prepared`
record remains for debugging;
the platform does not create an empty waveform and label it as an observation.
The CLI reports invalid user input separately with exit code 2. `--allow-failure`
applies only to an explicit solver-reported failed result, never to invalid input
or an implementation exception.

The advanced `case_yaml.v1` schema remains an explicit integration format; the
smaller `pcd.rf.v1` schema is the stable user format for RF matching studies.
Adding a new plasma-equivalent model belongs in the load-model boundary only
when its terminal equation, parameters, reference plane, and applicability
evidence are well defined.

## Architecture audit

Run the repeatable static review with:

```powershell
uv run --frozen python -m nox -s architecture-audit
```

Ruff identifies clear local complexity regressions, Pyrefly checks typed
interactions, and Import Linter enforces the small set of dependency directions
that are already intentional. In particular, simulation preparation/execution
cannot import measurement, study, search, reporting, or CLI modules; analysis
cannot import Case interpretation, persistence, solver, or application layers;
ML preparation/evaluation/ranking cannot import concrete solver, study orchestration,
or persistence modules.
Radon lists C-or-worse functions and B-or-worse
modules for human review. Its scores are not blanket gates: cohesive numerical
code must not be fragmented merely to improve a metric.

The current ownership map above describes the repository as it runs today.
[The foundation renewal plan](foundation-renewal-plan-ja.md) records the
measured coupling, desired simulation/measurement/study/optimization
boundaries, extension map, and migration completion criteria.
