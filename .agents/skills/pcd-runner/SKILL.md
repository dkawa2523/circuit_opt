---
name: pcd-runner
description: "Run, validate, inspect, and reproduce this repository's PCD RF circuit studies. Use only in this repository for requests involving PCD cases, ngspice simulations, design studies, benchmarks, result summaries, plots, or ML-ready evaluation data; do not use for unrelated circuit projects or ordinary source-code editing."
---

# PCD Runner

Operate the circuit-design platform through its public CLI and saved artifact
contracts. Keep simulation, engineering acceptance, benchmark reproduction,
and plasma-model applicability as separate claims.

## Confirm repository scope

Before executing project commands:

1. Resolve the Git repository root and work from it.
2. Confirm that `pyproject.toml` names `circuit-design-platform`, and that
   `pcd/` and `.agents/skills/pcd-runner/SKILL.md` exist.
3. If these checks fail, state that this skill is repository-scoped and do not
   run guessed PCD commands elsewhere.

Do not install this skill into a user or system skill directory.

## Choose the smallest execution mode

Read [references/execution-modes.md](references/execution-modes.md) before
running a command. Select one mode from the user's requested outcome:

- validation or netlist inspection without solver execution;
- one simulation without objective scoring;
- a complete scenario-aware design study;
- inspection, visualization, or export of an existing result;
- explicit full benchmark/report reproduction.

If the user asks only to smoke-test the platform and provides no case, use
`bench/cases/topology_l_match_golden.yaml`. It is a one-evaluation engine
conformance case, not a qualified chamber design.

## Execution rules

- Prefer `uv run --frozen python -m pcd` so the committed lock file and the
  Windows-safe module entry point are used. Do not update dependencies or the
  lock file merely to run a case.
- Before any solver-backed mode, run `solver-diagnose --json`. If ngspice is
  unavailable, report the diagnosis and stop; do not silently replace it with
  another solver or install software unless the user asks.
- Validate the selected case before execution. Use ordinary validation by
  default; add `--strict` only for a requested qualification, release, or CI
  check.
- Preserve the case, optimizer, solver, seed, run root, and acceptance policy
  requested by the user. Do not turn a public exhaustive RF grid into an
  exploratory sampled search.
- Put reproducible runtime data under `runs/` unless the user supplies another
  path. Do not overwrite committed report or figure artifacts unless the user
  asks to reproduce them.
- Treat a nonzero exit as meaningful. Inspect the saved failure record when
  available. Use `--allow-failure` only for an explicitly requested
  data-collection workflow.
- Preview `result-prune` first. Use `--apply` only when deletion is explicitly
  requested and the previewed study root is correct.
- Do not modify source code or input data while fulfilling a run-only request.

For an imported external netlist, confirm its declared `netlist_mode` and
`source_policy` before execution. A complete SPICE deck needs `deck`; a file
containing only circuit statements uses `fragment`. Never infer source
conflicts from shared nodes.

## Read and report results

Use paths recorded in `study_result.json` or `sim_manifest.json`; do not assume
that active files live directly under a study root. In a study, follow
`artifacts.generation` to the immutable generation and use its
`evaluations.csv` as the ML-ready table.

Report only the evidence produced by the selected mode. Include:

- case and schema, solver executable/version, and analysis mode;
- run/study root and immutable generation or manifest path;
- best fixed candidate, scenario coverage, acceptance outcome, failed limits,
  and failed evaluations when applicable;
- dataset ID plus runtime and solver fingerprints when handing data to ML;
- whether results came from cache;
- the relevant physical-model boundary.

Keep these meanings distinct:

- a successful ngspice process means the circuit solve completed;
- engineering feasibility means all declared case constraints passed;
- benchmark PASS means frozen behavior or a stated decision was reproduced;
- none of these qualifies plasma chemistry, thermal lifetime, or process yield.

For model interpretation, read `docs/rf-load-models.md`. For input fields and
external netlists, read `docs/input-format.md`. For extension boundaries, read
`docs/architecture.md`.
