# Changelog

## Unreleased

- start the topology-aware ML renewal without changing the solver or optimizer:
  add a framework-neutral `circuit_graph.v1` component-terminal-net record for
  declared structured templates, preserve explicit source/load/ground ports,
  separate wiring and value fingerprints, and reject raw SPICE rather than
  guessing its terminal semantics; keep arbitrary topology generation and
  online ML proposal outside the implemented scope;
- add the offline `ml-corpus` path from one or more completed AC studies to
  physical graphs, source/load/scenario context, complex port targets, and
  long-form component voltage/current targets. Keep observation meters out of
  graph identity, preserve declared series loss on logical components, group
  designs and external conditions before holdout, and mark insufficient or
  topology-confounded splits unavailable instead of overstating evidence.
  Keep study discovery/provenance/writes in the application adapter and pure
  response shaping/split policy in `pcd.ml.corpus`;
- add the fixed `ml-corpus-evaluate` M2 comparison: training mean, ridge, a
  two-hidden-layer NumPy MLP, and a component/net relational GNN use the same
  declared design, condition, and topology-family folds. Keep graph encoding,
  neural calculation, evaluation policy, and artifact I/O in separate owners;
  do not persist models or add a candidate-proposal path. The first fixed
  three-topology, 108-evaluation ngspice corpus selected the constant baseline
  in every aggregate protocol, so model persistence remains blocked and the
  next evidence step is a separately locked M2R corpus rather than tuning on
  the observed holdouts;
- carry signed engineering constraint margins through result tables, study
  histories, ML response targets, and the CLI. This gives
  calculation, offline learning, and candidate search one useful feasibility
  signal without adding another optimizer, model, dependency, or execution
  path;
- add one public-CLI release runner for the five supported user workflows;
  strict-validate seven inputs, require a fresh run root, verify concise
  electrical results and replay artifacts, distinguish intentional engineering
  rejection from solver failure, and combine the circuit result with the
  preregistered ML evidence as separate GO/NO-GO decisions; the P4 run
  reproduced 131 uncached ngspice evaluations, yielding circuit foundation GO
  and ML/full-requested-scope NO-GO;
- add a preregistered retrospective ML-ranking gate with fixed 81-candidate
  transient-RC and RF-matching pools, shared initial observations, 101 fixed
  random baselines, and fresh selected-candidate ngspice verification; retain
  the measured 14.3%/29.6% savings as a failed 30% gate and deliberately do not
  add sequential candidate proposal or Bayesian optimization;
- share one small feature-transform/standardization/three-neighbour owner
  between holdout evaluation and fixed-pool ranking without adding a solver,
  study, or persistence dependency to `pcd.ml`; add development-only PyYAML
  stubs so the new typed boundary and existing YAML readers remain visible to
  Pyrefly without suppressions;
- stop publishing one structured JSON file per explored candidate; keep all
  candidate summaries in `study_history.json`, all Candidate x Scenario x
  Control rows in `evaluations.csv`, and only the selected design in
  `best_candidate.json`; move benchmark, figure, and literature readers to
  those explicit sources and remove candidate-directory discovery from the
  execution pipeline;
- replace the 2,295-line custom report-surface generator and 244 kB widget
  artifact with a source-summary projection that writes a readable Markdown
  report, compact JSON status, and flat core/literature CSV tables; preserve
  source hashes and the separation between benchmark reproduction, electrical
  feasibility, and apparatus qualification while removing candidate-file and
  UI-schema coupling;
- remove three unreferenced compatibility branches after fixing the P1 paths:
  root `sim_manifest.json` reads, duplicated historical candidate `selected`
  values, and four-argument solver callables; use the current
  `summary/data/debug`, indexed trial, and `SolverRunRequest` boundaries in
  production and test adapters alike;
- close the P1 real-ngspice user-workflow gate with one release matrix covering
  saved-run analysis, Candidate/Scenario/Control studies, resonance/Q,
  quasi-static R+jX and prescribed R(t), and target-waveform sizing; expose
  study analysis/frequency/reference-plane context and a direct selected-
  candidate evidence path, classify nonperiodic transients as periodic
  `not_applicable`, and replace opaque raw components in acceptance examples;
- replace the open-ended staged roadmap with a PCD v1 completion path: first
  close five real-ngspice user workflows, then remove unused complexity,
  demonstrate numeric-sizing solver savings, and finish release acceptance;
  keep arbitrary circuit-graph generation, physical placement, self-consistent plasma,
  and unproven Bayesian optimization outside the v1 critical path;
- add optional independent constraint validation to `ml-evaluate`; fit only on
  the original training split, reject reused dataset identities/design values,
  compare every separate validation row with the majority baseline, and retain
  per-row predictions without changing the fixed holdout or enabling Bayesian
  optimization;
- add a fixed 25-point generic-RC boundary study demonstrating real ngspice
  constraint-validation evidence independently of surrogate fitting;
- add `ml-evaluate`, a dependency-free fixed-holdout comparison of a declared-
  scale, distance-weighted 3-neighbour surrogate against training mean/majority
  baselines; publish per-objective errors, constraint class evidence, uncached
  source-cost comparison, predictions, and an explicit gate that cannot claim
  Bayesian-optimization readiness or ngspice savings;
- add a model-independent ML dataset boundary and `ml-prepare` command that
  validates committed evaluation identity/grain, preserves failed outcomes,
  assigns explicit feature/objective/constraint roles, excludes
  `selected_control` and runtime artifacts, and creates a deterministic
  fixed-design group holdout without adding scikit-learn to core dependencies;
- add a direction-aware Pareto decision table for multi-objective studies;
  include only candidates with complete control-solver evidence and full
  scenario feasibility, retain equal trade-off points, preserve the existing
  lexicographic best decision, and distinguish sampled observed fronts from
  complete declared grids;
- add a seeded, dependency-free `DE/rand/1/bin` optimizer for bounded continuous
  advanced-case variables; preserve feasibility-first constraint ordering and
  record proposal generation, acceptance, failures, constraints, and duration
  in the compact study history without changing public RF grid enumeration;
- add an advanced, CSV-backed prescribed `R(t)` component with linear/hold
  interpolation, explicit periodic closure, positive-resistance and timestep
  checks, behavioral-ngspice rendering, content-addressed replay, fingerprinting,
  observed terminal probes, and a real-solver analytic-divider regression;
- add a bounded quasi-static `impedance_profile` workflow that validates a
  monotonic passive R+jX time table, reuses exact one-point AC scenarios, and
  publishes selected tuning, reflection, and power in `snapshot_response.csv`
  without claiming dynamic plasma-state propagation;
- report source-side forward and reflected AC wave power using the declared
  reference impedance;
- make `sim-run` concise by default and return the same user-facing summary
  from `--json`;
- separate a direct simulation into `summary.json`, canonical `data/`, and
  replay-only `debug/`; remove duplicate `params.json` and the unreferenced
  historical root-manifest read branch;
- keep solver-reported operational failures as inspectable results while
  allowing invalid configuration and unexpected evaluator, metric, parser, or
  solver-adapter exceptions to propagate instead of fabricating failed data;
- make typed solver settings and analysis requests the sole owner of
  AC/transient/timeout validation, remove the duplicate per-rule diagnostics,
  and reject invalid explicit timeouts instead of silently substituting a
  default before creating a run directory;
- make the simulation-input adapter the sole parser for source containers and
  names, probe columns, and measurement/load-port mapping shape; reuse its
  source resolver in netlist generation and replace parallel validation codes
  with one actionable simulation-input diagnostic;
- make circuit/load builders own section shape, required model parameters,
  physical domains, imported-netlist policy, and return types; validate by
  constructing the same in-memory model and allocate run artifacts only after
  that construction succeeds;
- remove the write-only interpreted-evaluation cache and its IDs; keep raw
  simulation cache, immutable published generations, and the flat evaluation table as
  separate responsibilities;
- remove the unused final optimizer-state duplicate from `study_result.json`;
- store each candidate control trial once, reference the selected trial by
  index, and remove the historical duplicated-selection reader branch;
- remove duplicated circuit/parameter observations from study rows and advance
  the compact candidate and evaluation-table schemas to v2;
- remove the redundant `--strict-exit` compatibility flag;
- remove legacy flat artifact keys and implicit v1 artifact-path fallbacks;
- invoke pytest and coverage through Python so the quality gate works under
  Windows application-control policies;
- add Radon and a single `architecture-audit` session combining Ruff
  complexity checks, Pyrefly, Import Linter, and non-gating hotspot reports;
- separate solver-neutral analysis requests/results, Case-to-probe compilation,
  and ngspice output translation from electrical measurement calculations;
- enforce that netlist/solver/simulation modules cannot import measurement or
  metric implementations;
- resolve solver settings, analyses, probes, measurement reference, and
  netlist options once into a
  typed `ResolvedSimulationCase`; make every solver consume
  `SolverRunRequest` and remove the former Case-based compatibility invocation;
- replace the monolithic analysis module with AC, periodic-transient, and
  component-stress owners behind the existing import facade; reduce the
  periodic-window calculation from Radon D to B and verify that persisted
  canonical waveforms reproduce the in-memory RF metrics;
- add a saved-run `analyze` use case that regenerates a concise electrical
  summary and standard impedance/Smith, transient, power, and component-stress
  figures without rerunning ngspice or conflating solver success with design
  acceptance;
- add multi-point power-response resonance, bracketed -3 dB bandwidth, and
  loaded-Q measurements, plus an H1-relative voltage/current harmonic figure;
  omit Q rather than extrapolating when a sweep does not contain both
  half-power crossings;
- remove the narrower `visualize-response` command and its duplicate
  `response_plot.py` implementation after folding that workflow into
  `analyze`;
- replace the stale snapshot specification with a current, evidence-based
  foundation renewal plan, including measured dependency/type/complexity
  findings and staged responsibility boundaries, and remove superseded review
  artifacts.

## 0.16.0

- distinguish complete SPICE decks from title-less circuit fragments;
- replace imported sources by exact name rather than inferred node conflict;
- preserve unavailable current as missing data and enforce requested RF cycles;
- provide the stable `pcd.api` extension surface and isolate plugin inputs;
- add dataset identities and runtime fingerprints to `evaluations.csv`;
- centralize saved-run artifact resolution and make preparation failures reuse
  their original run directory;
- provide an explicit, dry-run-by-default retention command for old study
  generations without touching reusable caches;
- document module ownership, supported extension boundaries, and generated
  output responsibilities;
- add the repository-scoped `$pcd-runner` Codex skill for validated ngspice,
  study, benchmark, visualization, and result-inspection workflows.
