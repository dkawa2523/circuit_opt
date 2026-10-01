# PCD execution modes

Run commands from the repository root. Replace placeholders in braces with
the user's paths. Prefer forward slashes in arguments so the examples work in
PowerShell and POSIX shells.

## Common preflight

Check the public entry point and solver without changing the lock file:

```text
uv run --frozen python -m pcd --help
uv run --frozen python -m pcd solver-diagnose --json
```

If the environment has not been created yet, `uv run` may materialize it from
`uv.lock`. If locked dependencies are genuinely missing, use
`uv sync --frozen --group dev`; do not regenerate the lock file.

List registered circuit, load, solver, metric, and optimizer implementations:

```text
uv run --frozen python -m pcd list
```

## Validate or inspect without running ngspice

Validate normal executable input:

```text
uv run --frozen python -m pcd validate-case {case.yaml} --json
```

Add `--strict` only when warnings must fail a requested release or
qualification check.

Generate the resolved ngspice deck without solving it:

```text
uv run --frozen python -m pcd sim-netlist {case.yaml} --out runs/netlists/{case-name}.cir
```

Render a topology-oriented view of a generated or authored netlist:

```text
uv run --frozen python -m pcd visualize-netlist {netlist.cir} --out runs/visualizations/{name}.svg
```

Netlist visualization is explanatory parsing; it is not proof that ngspice
accepted or solved the deck.

## Run one simulation

Use this when the user wants waveforms or AC response for exactly one resolved
case, without candidate ranking or objective scoring:

```text
uv run --frozen python -m pcd sim-run {case.yaml} --run-root runs
```

The command prints the run directory. Read `summary.json` first. Canonical
waveform and AC data are under `data/`; the exact netlist, solver log, replay
case, and detailed `manifest.json` are under `debug/`. Resolve paths from the
summary or debug manifest rather than assuming filenames. Historical root
manifests are not part of the current read contract. A solver-reported failure
is nonzero by default but retains its run record; use `--allow-failure` only when that
known result is intentionally being collected. Invalid input exits with code 2
and does not manufacture a failed waveform. Unexpected implementation errors
propagate instead of being reclassified as simulation results.

Regenerate engineering measurements and standard figures from an existing
successful simulation directory or manifest, without rerunning ngspice:

```text
uv run --frozen python -m pcd analyze {run-dir-or-manifest} --out runs/analysis/{name}
```

Use `--frequency-Hz` only when the user supplies or requests a meaningful AC
operating point. Inspect `summary.json` and the applicable impedance/Smith,
transient, power, component-stress, and harmonic figures. A multi-point AC
sweep may also contain a power-response resonance, bracketed -3 dB bandwidth,
and loaded Q. Do not interpret that value as unloaded component Q, and do not
manufacture it for a single point or an unbracketed sweep. This mode reports
electrical measurements; use a study when engineering acceptance is required.

## Run a design study

Use this for public `pcd.rf.v1` studies and advanced `case_yaml.v1` studies
that require Candidate x Scenario x Control evaluation:

```text
uv run --frozen python -m pcd run {case.yaml} --output runs --json
```

Add `--require-acceptance` for CI, unattended qualification, or when the user
explicitly wants an unacceptable design to produce a nonzero exit. Do not add
`--optimizer`, `--trials`, or `--seed` to a public RF case unless the case and
user explicitly choose the advanced exploratory path.

After completion, inspect the printed `run_root` and its `study_result.json`.
Important fields are:

- `best`: selected candidate and aggregate decision;
- `study`: scenarios, objectives, and model metadata;
- `dataset`: dataset ID, table schema, and runtime/solver fingerprints;
- `artifacts.generation`: active immutable generation;
- `artifacts.evaluation_table`: generation-relative ML table;
- `artifacts.pareto_front`: optional complete-evidence nondominated candidates
  for a study with two or more objectives;
- `artifacts.snapshot_response`: optional selected-candidate time table for an
  `impedance_profile` study;
- `artifacts.best_candidate`: structured control and solver evidence for the
  selected design;
- `n_evaluations` and `n_failed_evaluations`: execution completeness.

For `pareto_front`, report `study_result.json.pareto.scope`. An
`observed_candidates` front covers only evaluated samples; only
`declared_grid` represents the complete declared discrete candidate set.

The selected candidate stores each control trial once and selects it by index.
Use `pcd.results.selected_evaluation` when reading `best_candidate.json`. Use
`study_history.json` for all-candidate summaries and `evaluations.csv` for
Candidate x Scenario x Control detail. The root `raw/` area is reusable
physical cache, not a second published decision table.

Produce a compact tabular summary when requested:

```text
uv run --frozen python -m pcd result-summary {study-root}
```

Prepare an explicit, leakage-aware dataset before any surrogate-model work:

```text
uv run --frozen python -m pcd ml-prepare {study-root} --out runs/ml/{name} --seed 0 --json
```

Read `manifest.json` before using `dataset.csv`. Confirm the declared feature,
objective, constraint, eligibility, and excluded-column roles. The holdout is
by fixed-design values across all Scenario/Control rows and supports only an
unseen-design claim within the same committed case. Do not describe it as
cross-case or independent-measurement validation. Check the per-split class
counts before claiming constraint calibration, and do not use
`selected_control`, runtime duration, cache keys, or artifact paths as model
features. A future proposed design must return through the complete ngspice
study path.

Prepare the topology-aware AC corpus from one or more completed studies:

```text
uv run --frozen python -m pcd ml-corpus {study-root-a} {study-root-b} --out runs/ml-corpus/{name} --seed 0 --json
```

Read `manifest.json` first. `graphs.jsonl` contains physical logical circuits,
`samples.csv` contains context plus complex port targets, and
`component_responses.csv` contains sample-by-component complex targets when
the source study declared observations. Check every split `status`; an
unavailable split is a dataset-coverage finding, not permission to reshuffle
rows. This command neither calls ngspice nor trains or selects a model.

Compare the fixed topology-aware reference models on that corpus:

```text
uv run --frozen python -m pcd ml-corpus-evaluate runs/ml-corpus/{name} --out runs/ml-comparison/{name} --seed 0 --json
```

Read `evaluation.json` first. Report design, external-condition, and topology
protocols separately, and include the training mean, ridge, MLP, and relational
GNN scores. An unavailable fold or constant-baseline winner is a stop result.
This command performs no test-set tuning, model persistence, candidate
proposal, ngspice call, or solver-saving claim.

Evaluate the prepared holdout without running ngspice:

```text
uv run --frozen python -m pcd ml-evaluate runs/ml/{name} --out runs/ml-evaluation/{name} --json
```

Read `evaluation.json` and report the constant-baseline comparison separately
from classification evidence. One-class train/test labels are explicitly not
evaluable. `bayesian_optimization_ready` remains false because this mode does
not measure uncertainty, candidate-ranking regret, or saved solver calls.

When a separately executed, fixed candidate study was reserved before reading
its outcomes, prepare it normally and attach it without changing the source
holdout seed:

```text
uv run --frozen python -m pcd ml-evaluate runs/ml/{name} --constraint-validation runs/ml/{validation-name} --out runs/ml-evaluation/{name} --json
```

Confirm that `fixed_design_overlap` is zero. The model must still be fitted
from `source_training_split_only`; the validation rows may only score the
locked constraint classifier. Report its class counts, confusion, balanced
accuracy, majority baseline, and same-circuit simulation limitation.

The repository's only candidate-ranking evidence is the preregistered P3
benchmark, not a user optimization command:

```text
uv run --frozen python bench/ml/run_ranking_benchmark.py --run-root runs/p3_ranking_20260930
```

Read `ranking_evaluation.json` and `ranking_trace.csv`. The fixed RC/RF run
failed the required 30% saving, so never turn this benchmark into a claim that
sequential ML proposal, Bayesian optimization, or saved ngspice calls are
available. Do not change its pools, seeds, model, or threshold after the fact.

## Inspect or prune generations

Never guess a generation path from timestamps. Start from the study root's
`study_result.json`, then follow the recorded artifact paths.

Preview retention:

```text
uv run --frozen python -m pcd result-prune {study-root} --keep 3 --json
```

Only after explicit deletion authorization and review of that exact preview:

```text
uv run --frozen python -m pcd result-prune {study-root} --keep 3 --apply --json
```

The active and newest generations are preserved; the reusable raw cache is
outside this pruning operation. The former interpreted-evaluation cache no
longer exists.

## Reproduce validation suites and curated outputs

Run these only when the user explicitly asks for the full benchmark,
regression evidence, or publication artifacts. The core suite performs 411
real-ngspice evaluations when no matching cache is available.

PCD v1 user-workflow release acceptance uses a new root and performs 131
real-ngspice evaluations. Attach a separately reproduced P3 result only when a
full release-closure decision is requested:

```text
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
uv run --frozen python bench/release/run_suite.py --run-root runs/release_closure --ml-ranking-result runs/p3_ranking/ranking_evaluation.json
```

Read `release_result.json` and `REPORT.md`. A suite PASS qualifies the expected
circuit workflow reproduction; the separate `release_decision` may still keep
ML proposal and the full requested scope at NO-GO. Never reuse an existing run
root to turn cached evaluations into release evidence.

Core electrical suite:

```text
uv run --frozen python bench/run_suite.py --run-root runs/benchmark_suite
```

Literature/source-conformance suite:

```text
uv run --frozen python bench/literature/run_suite.py --run-root runs/literature/final_evaluation
```

Regenerate the report and publication figures only after both source suites
complete successfully:

```text
uv run --frozen python bench/reports/generate_case_report.py
uv run --frozen python bench/figures/generate.py --run-root runs/benchmark_suite
```

Check `passed` in `runs/benchmark_suite/benchmark_result.json` and both
`passed` and `benchmark_integrity_passed` in the literature evaluation. A
benchmark case may intentionally be electrically infeasible and still PASS by
reproducing its expected classification.

For a requested code-quality gate after implementation work, use:

```text
uv run --frozen python -m nox -s quality-pr
```

Do not run the complete quality gate for an ordinary case execution unless the
user asks or source code changed as part of the same task.

## Case selection when the user asks for a demonstration

- Minimal real-ngspice smoke: `bench/cases/topology_l_match_golden.yaml`.
- Public matching study: `examples/rf_impedance_point_study.yaml`.
- Independent frequency points: `examples/rf_impedance_frequency_table.yaml`.
- Quasi-static time-ordered R+jX snapshots: `examples/rf_quasi_static_profile.yaml`.
- Component stress and effective loss: `examples/rf_component_stress.yaml`.
- Effective CCP one-port: `examples/rf_ccp_lumped.yaml`.
- Effective ICP terminal fit: `examples/rf_icp_transformer.yaml`.
- Advanced constrained continuous sizing: `examples/advanced/generic_rc_filter.yaml`.
- Advanced transient port measurement: `examples/advanced/rf_port_transient.yaml`.

Examples demonstrate syntax and model behavior. They are not qualified chamber
data or process recipes.
