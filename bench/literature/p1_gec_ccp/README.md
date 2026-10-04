# Hargis 1994 GEC CCP benchmark

This benchmark converts the 32 central rows in Tables III and IV of Hargis et
al. into fundamental-frequency `R+jX` loads at the powered-electrode surface.
The rows cover 13, 33, 66, and 133 Pa, four external drive levels, and two GEC
cell populations.

The table labels 24 MHz and 34 MHz identify the populations' empty-cell
resonances. They are not drive frequencies; all rows use 13.56 MHz.

## Data ownership

- [`sources.yaml`](sources.yaml) is the canonical citation, extraction,
  convention, and applicability manifest for the two primary references.
- `raw_tables_iii_iv.csv` is the single canonical publication transcription.
- `derived_impedance_all32.csv` is its immutable 32-row electrical view.
- `derived_impedance_66pa.csv` is an immutable eight-row subset view used by
  focused control-authority cases.
- `reported_spread_envelope.csv` is an immutable 32-row deterministic view of
  the reported 66 Pa group spread.
- `phase_systematic_*.csv` are two counterfactual common-mode sensitivity
  views, not measurements or confidence intervals.

`run_all32_benchmark.py` derives every view in memory, compares it with the
committed files, checks table completeness, units, reference plane, passivity,
power closure, and replays all 32 central points through NGSpice. Normal runs
never rewrite files under `bench/`.

Run the source and design checks from the repository root:

```powershell
uv run python bench/literature/p1_gec_ccp/run_all32_benchmark.py --require-ngspice --output runs/literature/hargis_source.json
uv run python bench/literature/p1_gec_ccp/run_design_cases.py --run-root runs/literature/hargis_control
uv run python bench/literature/p1_gec_ccp/run_phase_sensitivity.py --run-root runs/literature/hargis_phase
uv run python bench/literature/p1_gec_ccp/run_hardware_family_comparison.py --run-root runs/literature/hargis_hardware
```

## Electrical derivation

The published `Ve1` and `Ie1` values are fundamental peak amplitudes and the
phase is voltage relative to current:

```text
|Z| = Ve1 / Ie1
R   = |Z| cos(phi)
X   = |Z| sin(phi)
P1  = 0.5 Ve1 Ie1 cos(phi)
```

The reported dissipated power includes the first five harmonics. `P1` is a
consistency check, not a claim that all plasma power is fundamental. Across
the 32 rows its maximum difference from the reported value is 3.133%.

## Independent evidence families

The platform hardware comparison evaluates four families independently:

| family | rows | meaning |
|---|---:|---|
| `central_operating_conditions` | 32 | published central points over pressure and applied drive |
| `reported_apparatus_spread` | 32 | four deterministic 66 Pa boundaries from reported group variation |
| `phase_model_minus6` | 8 | one common-mode -6 degree model-form stress |
| `phase_model_plus6` | 8 | one common-mode +6 degree model-form stress |

The reported spread contains cell-to-cell and measurement effects; it is not
instrument-only uncertainty. Sobolewski motivates the phase sensitivity, but
does not establish that every Hargis row has an independent +/-6 degree error.

For the fixed-L shortlist and 4x4 C1/C2 controls, the result vector is:

| L1 [uH] | central | apparatus spread | phase -6 | phase +6 |
|---:|---:|---:|---:|---:|
| 1.4 | 17/32 | 26/32 | 5/8 | 7/8 |
| 1.5 | 22/32 | 30/32 | 7/8 | 8/8 |
| 1.6 | 26/32 | 20/32 | 4/8 | 8/8 |

No combined coverage, implicit 40/40/10/10 weighting, or global "best L1" is
reported. The central family favors 1.6 uH; the spread and negative phase
families favor 1.5 uH; the positive phase family ties 1.5 and 1.6 uH. A scalar
decision requires an actual apparatus and an explicitly authorized objective.

## What is and is not established

Established:

- publication transcription and `V/I/phase -> R+jX` conversion;
- 13.56 MHz drive and powered-electrode-surface reference plane;
- software decisions for explicitly declared fixed/limited/bounded controls;
- sensitivity of those platform fixtures to distinct evidence families.

Not established:

- a microscopic sheath model or plasma-state inference;
- a production chamber load distribution;
- Hargis's or Sobolewski's unpublished matcher settings;
- a universal 10% reflected-power threshold;
- statistical probabilities for spread or phase sensitivity rows.
