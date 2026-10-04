# Changelog

## Unreleased

- resolve every executable parameter into one fixed, calibration, operating,
  control, design, or latent role; reject duplicate/conflicting ownership,
  project optimizer inputs onto design variables only, retain constants as
  ordinary solver defaults, and publish role-separated values in study result
  tables without adding a second execution path;
- reset the unsupported ML product surface after its evidence gates failed:
  remove the four ML/corpus CLI commands, runtime ML/corpus/graph modules,
  graph-only circuit metadata, fixed-pool benchmark code, and ML release-gate
  attachment; keep the measured negative outcome as a short design record and
  return `Circuit` to one ngspice-rendering responsibility. The optional
  OptunaHub group remains isolated for reproduction of the existing
  case-specific optimization figures and is not a runtime dependency;
- route direct simulation, studies, and terminal identification through one
  `execute_case` forward solve returning the canonical in-memory
  `SimulationResult` together with its persisted `SimRecord`; add one
  Case-specific evaluation adapter, calculate fresh metrics without a CSV
  round trip, and restore cached/saved data to the same response type before
  measurement;
- add `pcd identify` for bounded latent effective-terminal parameters: fit
  declared complex V/I scenarios, freeze the result for disjoint holdout
  scenarios, replay both phases through ordinary ngspice without cache reuse,
  and reject a low-loss result when its normalized local sensitivity matrix is
  rank-deficient or ill-conditioned;
- independently replay the selected design through the ordinary solver with
  raw-cache reuse disabled; publish the replay as `best_candidate.json` and
  the final acceptance decision while keeping search evaluations and history
  separate;
- centralize L, pi, and harmonic-pi matching connectivity and component roles
  in one reviewed topology catalog consumed by both public-input compilation
  and netlist generation; keep values, frequencies, losses, and limits explicit
  in each case;
- align the advanced RC sizing example's pulse window with its stored step-
  response target so the declared waveform objective is attainable instead of
  comparing against an incompatible source waveform;
- keep the RC sizing example's installed resistor fixed and optimize only its
  capacitor because one voltage waveform identifies the RC time constant, not
  R and C separately;
- reflect Differential Evolution mutations at continuous bounds instead of
  clipping many distinct mutations onto duplicate endpoint proposals, and use
  a seeded uniform fallback when evolution still repeats an evaluated point;
- carry signed engineering constraint margins through result tables, study
  histories, and the CLI, giving calculation and candidate search one useful
  feasibility signal without adding another optimizer or execution path;
- add one public-CLI release runner for the six supported user workflows;
  strict-validate eight inputs, require a fresh run root, verify concise
  electrical results and replay artifacts, distinguish intentional engineering
  rejection from solver failure, and publish the circuit decision only; machine
  learning is outside this release gate;
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
