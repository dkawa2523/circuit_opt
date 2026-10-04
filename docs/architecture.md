# Architecture and ownership

This document describes the code that is currently executable. Planned
machine-learning work is kept separately in the
[platform reset plan](platform-reset-plan-ja.md), so a future design is not
confused with an implemented contract.

## Product boundary

PCD currently has four user workflows:

- `sim-run`: resolve one case and execute one ngspice calculation;
- `analyze`: calculate electrical measurements and standard figures from a
  saved run without executing ngspice again;
- `run`: evaluate Candidate x Scenario x Control combinations and select a
  design using declared objectives and constraints.
- `identify`: fit bounded latent terminal parameters on declared observations,
  verify unseen scenarios, and reject locally non-identifiable fits.

`validate-case`, `sim-netlist`, `visualize-netlist`, `result-summary`, and
`result-prune` support those workflows. Machine-learning training or proposal,
arbitrary topology generation, physical placement, plasma chemistry, and
self-consistent plasma/circuit evolution are not current product functions.

Plasma and chamber knowledge enters the current runtime only as an electrical
terminal model with a named reference plane: R+jX data, an ordered
quasi-static R+jX profile, a qualified effective CCP/ICP model, or an explicit
externally supplied component profile.

## One calculation path

```text
authored YAML/JSON
        |
        v
case.py ----> plan.py (pcd.rf.v1 only) ----> executable Case
        |                                         |
        +----> problem.py (one parameter role)    |
        |                                         |
        v                                         v
simulation_input.py                         circuit + load
        |                                         |
        +-------------------+---------------------+
                            v
                         netlist.py
                            |
                            v
                 SolverRunRequest -> solver.py -> ngspice
                                               |
                                               v
                                  simulation.SimulationResult
                                      |                    |
                                      v                    v
                         metrics.measure_response      sim_core.py
                                      |                    |
                                      |                    v
                                      |                SimRecord
                                      |                    |
                                      +----> evaluation.py <---- records.py
                                                  |
                                                  v
                                      core/StudyRunner + search
                                                  |
                                                  v
                                     results/ + evaluations.csv

saved SimRecord ----> reporting/ + analysis/ ----> summary + standard figures
```

There is no second ML simulation path. Validation, direct simulation, and a
study build the same typed simulation request and the same in-memory
circuit/load representation.

## Responsibility map

| area | owns | must not own |
|---|---|---|
| `case.py`, `plan.py`, `problem.py` | file loading, public RF compilation, one-time parameter-role resolution | solver execution, metric calculation |
| `simulation.py`, `simulation_input.py`, `probes.py` | solver-neutral requests, canonical result shapes, probes and measurement reference | ngspice syntax, study ranking |
| `topology_catalog.py`, `netlist.py`, `sim_methods.py`, `rf_loads.py` | reviewed matching connections, circuit/load construction and deterministic ngspice rendering | subprocess execution, acceptance decisions |
| `solver.py`, `ngspice_io.py` | ngspice process execution and output decoding | Case interpretation, objective scoring |
| `sim_core.py`, `records.py` | one-run lifecycle, persistence, stable saved-run reads | metric definitions, optimizer policy |
| `signals/`, `analysis/` | pure numerical signal, AC, transient, power and stress calculations | Case parsing, persistence, solver execution |
| `metrics.py` | named measurements and engineering-limit evaluation | candidate generation, report layout |
| `evaluation.py` | one request-to-response-to-metrics adapter and fresh-response reuse | scenario aggregation, candidate proposal, solver implementation |
| `core/`, `study_config.py` | Candidate/Scenario/Control roles and feasibility-first aggregation | circuits, ngspice, search algorithms |
| `search.py`, `continuous_search.py` | proposal order and optimizer state | solver calls, physics, constraint meaning |
| `study.py`, `results/` | orchestration, cache use, immutable generations and decisions | alternative physics or metric formulas |
| `identification.py` | fit/holdout partition, latent-only orchestration, local sensitivity rank | circuit equations, solver implementation, design hardware search |
| `reporting/`, `figures/` | read-only summaries and reusable standard figures | solver execution, design acceptance |
| `bench/` | reproducible electrical and literature-derived evidence | runtime package contracts |

## Important boundaries

### Simulation

`simulation_input.py` is the only Case-to-solver-input adapter. It resolves
`SolverSettings`, `AnalysisRequest`, `ProbePlan`, and
`MeasurementReference`. `netlist.py` consumes that resolved input and a
`Circuit`; it neither starts a process nor writes result artifacts.

`Circuit.add` declares an ordinary two-terminal element. `Circuit.raw` adds an
explicit SPICE statement used for imported topology, meters, internal loss
nodes, coupling, or behavioral components. The runtime circuit API contains
no machine-learning graph metadata.

`solver.py` receives one `SolverRunRequest`. It returns solver-reported
operational failures as a `SimulationResult`; it does not read a full Case or
calculate engineering success.

`sim_core.execute_case` is the one forward-solve entry point. It returns a
`SimulationRun` containing the canonical in-memory `SimulationResult` and its
persisted `SimRecord`. The older-looking `simulate_case` name is only the
direct-simulation wrapper that returns the record from that same call.

### Measurement and acceptance

`analysis/` consumes canonical arrays and foundational types only. It cannot
read a Case, invoke ngspice, discover result directories, or decide which
candidate wins. `metrics.py` connects a named engineering request to those
calculations and evaluates declared limits. `measure_response` operates on the
in-memory `SimulationResult`; `measure_record` restores that same type before
calling it. A fresh study evaluation therefore does not write and reread CSV
merely to calculate its metrics, while a cached or saved evaluation follows
the identical measurement path after restoration.

A successful ngspice process, a successful measurement, and an acceptable
design are different states. Only the study layer can claim the last one after
complete Scenario/Control evidence is available.

### Search and studies

`problem.py` assigns each named input exactly one of fixed, calibration,
operating, control, design, or latent. Public `network.search` values and only
advanced variables classified as design reach a `run` optimizer; only bounded
latent variables reach `identify`. All other defaults still reach simulation
as ordinary resolved numbers; scenarios and controls enter through their
existing typed runtime objects. Conflicting roles and duplicate declarations
fail at this input boundary.

`search.py` exposes the small ask/tell optimizer boundary and exact finite
grids. One invocation proposes exactly one role: `design` for `run` or
`latent` for `identify`. `continuous_search.py` owns bounded-coordinate
translation and the current seeded Differential Evolution implementation.
Neither receives a solver, rewrites constraints, or changes a public RF
candidate grid.

`topology_catalog.py` is the single connectivity and component-role source for
the three reviewed matching templates. Frequency, component values, losses,
search bounds, and engineering limits stay in each case because a topology
name cannot qualify those quantities by itself. Both the public input compiler
and registered circuit builders consume this catalog.

To add a reviewed matching topology, make the connectivity change in
`topology_catalog.py`, add only the thin registry entry in `sim_methods.py`,
and add an independent complex-impedance golden case plus a real-ngspice case.
Do not copy connectivity into the input compiler, and do not put component
values, search ranges, frequency claims, or stress limits in the catalog. A
custom topology that is not ready for the public catalog stays an explicit
`from_yaml` circuit or a case-local plugin.

`evaluation.py` is the Case-specific adapter used by `StudyRunner`. `study.py`
asks for a candidate, evaluates all required Scenario/Control states through
that adapter, sends the result back to the optimizer, and commits an immutable
generation only after the study is complete. Ranking may reuse raw search
results. The chosen parameter set and each scenario's chosen control are then
evaluated once more with cache reuse disabled; that replay, rather than the
search observation, supplies `best_candidate.json` and the final acceptance
decision. The complete control grid remains in `evaluations.csv` and is not
needlessly repeated during verification.

`evaluations.csv` is the complete Candidate x Scenario x Control audit table.
Its active candidate prefix is `design.*` for `run` and `latent.*` for
`identify`; `scenario.*` and `control.*` vary at their own levels. Other roles
remain constant columns without pretending they were optimizer proposals.
`study_history.json` is the compact optimizer trace. `best_candidate.json` is
the structured link to selected real-ngspice evidence. These are result data,
not a hidden surrogate dataset contract.

### Identification

`identification.py` is an outer orchestration layer, not an alternative
simulator or optimizer. It resolves the authored scenarios once, runs the fit
partition through the ordinary latent-candidate study, freezes the selected
latent values, and runs the holdout partition through the same study and
cache-free replay path. It then perturbs each latent coordinate within its
declared bounds and forms a normalized output-sensitivity matrix from the
configured electrical metrics.

An identification succeeds only when fit and holdout limits pass, solver
evidence is complete, and the local matrix has full column rank within the
declared condition-number limit. Optional known truth is benchmark metadata;
it is never used to propose a value. `terminal_vi_fit` converts a measured
complex V/I pair into terminal impedance at the calibrated source plane. This
does not convert an effective R/L/C value into a claim about density,
chemistry, sheath geometry, or process yield.

### Reporting and benchmarks

`reporting/` reads saved canonical data and produces an electrical summary and
standard figures. It never reruns a solver or changes the study decision.

`bench/release/run_suite.py` is an outside-in circuit acceptance adapter. It
calls the public CLI and checks supported user outcomes. It has no ML gate and
does not qualify chamber process behavior. Publication-specific calculations
and layouts remain under `bench/`, outside the runtime package.

The optional `optuna-benchmark` dependency group exists only to reproduce a
case-specific optimizer study and its publication evidence. It is not imported
by PCD runtime code and is not a public optimizer contract.

## Extension boundary

External Python integrations import documented types and decorators from
`pcd.api`:

- a circuit builder returns `Circuit`;
- a solver accepts `SolverRunRequest` and returns `SimulationResult`;
- a metric accepts an isolated `Case`, resolved parameter mapping, and
  `SimulationResult`, then returns a non-empty mapping;
- an optimizer derives from `BaseOptimizer` and implements `ask`/`tell`.

Extensions receive detached case/parameter data so mutation cannot alter the
study fingerprint or another evaluation. Compatibility shims and signature
inspection are intentionally excluded.

New load models belong in the load-model boundary only when their terminal
equation, parameter units, reference plane, applicability evidence, and
verification case are explicit. A new optimizer may change proposal order,
but not the evaluation path or the meaning of feasibility.

## Failure boundary

Failures become data only where their meaning is known:

- `solver.py` returns a failed result for missing executable, timeout, nonzero
  solver exit, missing requested output, or malformed solver output;
- an evaluator may return a failed raw result for a known evaluation outcome;
- `StudyRunner` converts only a known unsettled-measurement condition.

Invalid configuration and unexpected implementation errors propagate. The CLI
reports invalid input with exit code 2. `--allow-failure` applies only to a
solver-reported failed simulation and never fabricates a waveform.

## Architecture audit

Run the repeatable review with:

```powershell
uv run --frozen python -m nox -s architecture-audit
```

- Ruff finds local correctness and clear complexity regressions.
- Pyrefly checks typed interactions.
- Import Linter protects the five high-value dependency directions: generic
  core, foundational signals, simulation independence, and pure analysis; it
  also keeps the CLI as a leaf.
- Radon reports complex or low-maintainability areas for human review; it is
  diagnostic rather than an incentive to fragment cohesive numerical code.

The next architecture decision is the R6 data gate in the
[platform reset plan](platform-reset-plan-ja.md). Machine-learning packages are
considered only after the completed forward, design-optimization, and terminal-
identification paths provide enough independent data to demonstrate a solver-
cost bottleneck and a generalization test.
