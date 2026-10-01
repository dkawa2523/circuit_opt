# Surrogate evidence and ranking gate

These cases produce separate simulation evidence for a model whose features,
transforms, neighbors, threshold, and source holdout are already fixed. They
are not training data and are not general-purpose examples.

`generic_rc_constraint_boundary.yaml` declares a 5 x 5 component grid before
execution. Its RC products cross the 4.9 V peak-output limit while remaining
inside the source study's declared design bounds. Run and prepare it normally,
then attach the prepared dataset only as constraint validation:

```powershell
uv run --frozen python -m pcd run bench/ml/generic_rc_constraint_boundary.yaml --output runs/stage5_constraint_boundary
uv run --frozen python -m pcd ml-prepare runs/stage5_constraint_boundary/generic_rc_constraint_boundary --out runs/stage5_constraint_boundary_ml/generic_rc_constraint_boundary --seed 0
uv run --frozen python -m pcd ml-evaluate runs/stage5_ml_dataset/generic_rc_filter --constraint-validation runs/stage5_constraint_boundary_ml/generic_rc_constraint_boundary --out runs/stage5_constraint_validation/generic_rc_filter
```

Do not tune the grid, holdout seed, model, or threshold after reading these
outcomes. This checks interpolation and constraint classification for one
simulated RC family; it is not independent measurement or cross-circuit
validation.

## M2 topology-aware AC comparison

`m2_ac_corpus/` contains three synthetic topology-coverage studies with the
same three load conditions and twelve declared designs per topology. Rebuild
the evidence with a new run root, then compare every model on the corpus-owned
design, condition, and leave-one-topology-out splits:

```powershell
uv run --frozen python -m pcd run bench/ml/m2_ac_corpus/l_match.yaml --output runs/m2_ac_model_comparison/studies
uv run --frozen python -m pcd run bench/ml/m2_ac_corpus/pi_match.yaml --output runs/m2_ac_model_comparison/studies
uv run --frozen python -m pcd run bench/ml/m2_ac_corpus/pi_match_harmonic.yaml --output runs/m2_ac_model_comparison/studies
uv run --frozen python -m pcd ml-corpus runs/m2_ac_model_comparison/studies/m2_ac_l_match runs/m2_ac_model_comparison/studies/m2_ac_pi_match runs/m2_ac_model_comparison/studies/m2_ac_pi_match_harmonic --out runs/m2_ac_model_comparison/corpus --seed 23
uv run --frozen python -m pcd ml-corpus-evaluate runs/m2_ac_model_comparison/corpus --out runs/m2_ac_model_comparison/evaluation --seed 23
```

The 2026-10-01 run completed all 108 ngspice evaluations, but the training-mean
constant baseline won all three aggregate protocols. M2 is therefore an
implemented negative result: do not tune on these holdouts or proceed to model
persistence. The next permitted evidence step is a separately fixed M2R corpus
with additional topology, frequency, and load groups.

## P3 fixed candidate-ranking gate

`ranking_protocol.yaml` freezes two 81-candidate pools, six initial
observations, both seeds, the three-neighbour model, the feasible top-10%
target, and the 30% solver-evaluation saving threshold. The pools cover a
transient RC target-waveform problem and a 13.56 MHz pi-match problem. Each
candidate has one Scenario x Control solve, so candidate evaluations and
ngspice calls are identical in this benchmark.

Reproduce the complete pool evaluations, retrospective ordering, fixed random
baseline, and fresh selected-candidate verification with:

```powershell
uv run --frozen python -m pcd solver-diagnose --json
uv run --frozen python -m pcd validate-case bench/ml/ranking_rc_waveform.yaml --strict --json
uv run --frozen python -m pcd validate-case bench/ml/ranking_rf_match.yaml --strict --json
uv run --frozen python bench/ml/run_ranking_benchmark.py --run-root runs/p3_ranking_20260930
```

Use a new run root for each reproduction. The runner rejects an existing root
so the selected-candidate verification cannot be silently satisfied from the
raw-result cache.

The preregistered 2026-09-30 run did **not** pass. RC required 12 evaluations
versus a fixed-random median of 14 (14.3% saving); RF required 19 versus 27
(29.6%). All 162 pool solves and both fresh selected-candidate verification
solves succeeded. The pool, seeds, model, and threshold were not changed after
reading the outcomes. Therefore PCD retains retrospective ML evaluation but
does not expose sequential candidate proposal, Bayesian optimization, or a
saved-ngspice-call claim. `ranking_evaluation.json` and `ranking_trace.csv`
under the selected run root are the machine-readable evidence.
