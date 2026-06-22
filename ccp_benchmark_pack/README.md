# CCP GEC-like Argon Benchmark Pack for Circuit Design Platform v6

This pack defines a semiconductor plasma chamber benchmark that can be copied into the
root of `circuit_design_platform_v6_final` and executed with the existing v6 commands.

The physical reference is a GEC-like capacitively coupled argon discharge:

- RF: 13.56 MHz
- pressure: 100 mTorr ≈ 13.3 Pa
- powered electrode diameter: 10 cm
- inter-electrode gap: 2.45 cm
- gas: Ar

The pack intentionally separates three levels:

1. `ccp_gec_level1_fixed_match.yaml`  
   Fixed/state-derived plasma RLC load.  Tests basic ngspice netlist generation,
   ML scoring, and matching-network optimization.

2. `ccp_gec_level2_timevarying_plasma.yaml`  
   Time-varying `plasma_table_rlcq` load.  Tests the v6 boundary between external
   plasma data and circuit simulation.

3. `ccp_gec_level3_topology_and_load_choice.yaml`  
   Categorical topology/load search.  Tests the data-science workflow and surrogate
   learning on mixed continuous/categorical parameters.

Typical commands:

```bash
# From the v6 project root, after copying this directory as examples_ccp_gec/
pcd sim-netlist examples_ccp_gec/ccp_gec_level2_timevarying_plasma.yaml --out /tmp/ccp_level2.cir
pcd workflow-optimize examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml   --optimizer random --solver dummy --n-trials 20 --run-root runs/ccp_level3_demo
pcd ml-score examples_ccp_gec/ccp_gec_level3_topology_and_load_choice.yaml runs/ccp_level3_demo
pcd ml-fit-surrogate runs/ccp_level3_demo --out runs/ccp_level3_demo/surrogate.json
```

For physical validation, replace `--solver dummy` with `--solver ngspice_cli` after ngspice is installed.
Use `pcd validate-case <case> --strict` before production runs, and add
`--strict-exit` to simulation/scoring commands when they are executed from CI or
batch automation.  The `dummy` solver is intentionally a screening tool only; it
does not validate plasma physics, matching behavior, or power delivery.

## ngspice benchmark profiles

The ngspice rerun settings are kept outside the platform core in
`ngspice_benchmark_profiles.json`.

Profiles:

- `smoke`: 3 trials per Level 2/3 case, for connectivity checks.
- `standard`: 30 trials per Level 2/3 case, for dummy-vs-ngspice comparison.
- `category_extended`: 100 Level 3 trials, for topology/load risk analysis.
- `harmonic_focused`: 50 Level 2/3 trials with stronger A2/A3 weighting.
- `surrogate_feasible`: 100 Level 3 trials with a feasible-first objective.
- `extended`: compatibility alias for `category_extended`.

Run a smoke check:

```bash
py ccp_benchmark_pack/run_ngspice_benchmark.py --profile smoke --strict-exit
```

Run the standard profile:

```bash
py ccp_benchmark_pack/run_ngspice_benchmark.py --profile standard
```

The runner uses `ngspice_cli` and normally leaves `solver.executable` unset.  On
Windows, the platform core prefers `ngspice_con.exe` when it is on `PATH` so the
solver can run without a visible ngspice console window.  Use `--executable` only
for machine-specific overrides.

If the installer updated the user `PATH` but the current shell has not picked it
up yet, pass the console executable explicitly:

```powershell
py ccp_benchmark_pack/run_ngspice_benchmark.py --profile smoke --strict-exit `
  --executable "$env:LOCALAPPDATA\\Programs\\ngspice-46\\Spice64\\bin\\ngspice_con.exe"
```

Raw profile outputs go under `runs/ccp_ngspice_reeval/`, which is ignored by git.
Keep only reproducible settings, scripts, and compact curated summaries in this
benchmark pack.

Analyze an ngspice run against the tracked dummy benchmark output:

```bash
py ccp_benchmark_pack/analyze_ngspice_benchmark.py runs/ccp_ngspice_reeval/<run-id>
```

The analyzer writes compact CSV and Markdown summaries under
`ccp_benchmark_pack/results/<run-id>/`, including:

- dummy vs ngspice best/median/p90/max loss and penalty rate
- feasible/infeasible summaries and voltage/current risk
- Level 3 topology/load/combo category risk statistics
- feasible top candidates and low-loss infeasible candidates
- best-waveform harmonic amplitudes, target ratios, and phase errors
- Spearman correlations against loss
- all-trial, feasible-only, and robust-transform surrogate diagnostics

Interpretation rules:

- Treat best loss as secondary.
- Prefer feasible median, penalty rate, p90/max, and topology/load risk profile.
- Do not call waveform tailoring successful when A2/A3 target ratios remain low.
- Treat surrogate output as diagnostic until feasible-only CV performance is good.

The synthetic plasma table is generated from the simple reduced relations:

Lp = ell * m_e / (A * n_e * e^2)
Rp = nu_m * Lp
Csh = eps0 * A / s_sh

These are only screening approximations; high-fidelity validation should use PIC/MCC,
fluid, global model, or measured plasma diagnostics.
