# Measured parallel RLC terminal benchmark

The source is `rlc_circuit.json` from *Impedance data for various
electrochemical and electrical systems*, version 1.0.0, by Michael A.
Danzer, Christian Plank, and Tom Rüther, with Christina Roth and Jonathan
Schneider as contributors. The original file is redistributed byte for byte
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The record
describes a custom parallel capacitor and lossy inductor measured using a
Biologic VMP300 with galvanostatic excitation. [Zenodo record and files](https://doi.org/10.5281/zenodo.10794584)

The source supplies 81 complex impedances over approximately 100 Hz–10 kHz.
The benchmark checks the original hash, ordered distinct frequencies, finite
positive resistance, and agreement of rectangular and polar values. The
reported low-frequency real part is about 9.49 Ω, while the dataset landing
page lists an inductor resistance of 9.7 mΩ. The fit leaves resistance free and
keeps both values traceable. The dataset does not establish the origin of the
difference, fixture de-embedding, or measurement uncertainty.

## What is checked

- Both models fit the same 12 frequency rows and predict the same five withheld
  contiguous rows near resonance. The other 64 rows remain visible in the
  comparison and have no role in the benchmark decision.
- The models have three free parameters each. One places resistance in series
  with the coil and puts that branch in parallel with the capacitor. The
  comparison model places R, L, and C in parallel.
- The source reports impedance. PCD takes complex V/I, so the adapter writes
  a synthetic 1 V reference and `I = 1/Z`. The derived current is a representation
  of the same impedance, independent of the instrument's excitation amplitude.
- Independent circuit equations must match fresh ngspice predictions on fit and
  holdout points within 1 µΩ. The series-loss model must meet 2 Ω fit RMS and
  8 Ω holdout RMS limits, and pass local sensitivity checks. The comparison
  model must be rejected with at least twice the holdout error. These are
  benchmark thresholds for this trace.

The split measures prediction at withheld frequencies of a single acquisition.
Independent acquisitions, confidence intervals, and apparatus qualification
require additional evidence.

## Reproduce

```powershell
uv run --frozen python bench/literature/p2_danzer2024_rlc/run.py --run-root runs/measured_rlc
```

The output includes the two executable YAML cases, their shared V/I table,
both identification records, `benchmark_result.json`, and a comparison CSV
and figure. The figure shows measured and predicted resistance, reactance,
and complex error against frequency, with the withheld band shaded. The run
directory must be new so the benchmark starts with fresh calculations.
