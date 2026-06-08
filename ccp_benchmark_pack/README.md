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

The synthetic plasma table is generated from the simple reduced relations:

Lp = ell * m_e / (A * n_e * e^2)
Rp = nu_m * Lp
Csh = eps0 * A / s_sh

These are only screening approximations; high-fidelity validation should use PIC/MCC,
fluid, global model, or measured plasma diagnostics.
