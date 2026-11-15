# Measured nanosecond discharge pulse and terminal-model boundary

This benchmark uses two source files from Nicolas Q. Minesi, Victorien
Blanchard, Erwan Pannier, Gabi-Daniel Stancu, and Christophe Laux, *Plasma-assisted
combustion with nanosecond discharges. Part I*, dataset version 2. The two files
in `source/` are redistributed byte for byte under the dataset's
[Etalab Open License 2.0](https://spdx.org/licenses/etalab-2.0.html). The executable
benchmark records both source SHA-256 digests. [Dataset record](https://doi.org/10.57745/8LZ7LU)

| File | Interpretation |
|---|---|
| `Fig3_current-voltage-energy.txt` | Published anode voltage, current, and cumulative energy for a nanosecond discharge in an atmospheric lean methane–air flame. The electrical trace is an average across at least 1000 measurements. |
| `Fig7_ne_vs_time.txt` | Separate optical electron-density measurements by position, with reported uncertainties. The figure uses the series at 2.25 mm from the cathode on its published time axis and preserves missing values. |

The electrical plane is the anode probe location in the published setup. The
repetition rate of the discharge is 20 kHz. The electrical file covers a
500 ns shot-averaged trace including reflected pulses. Its 1251 samples and the optical series have different
measurement meanings. [Experimental article](https://doi.org/10.1088/1361-6595/ac5cd4)

## Checks

1. Reintegrate signed terminal power `V * I` on the original electrical time
   grid and compare against the published energy trace. The maximum difference
   must stay below 1% of the published end value. This tolerance tests source
   processing and units. The article separately reports experimental uncertainty.
2. Prescribe the measured terminal voltage as the PWL source of a one-port SPICE
   circuit. Choose a constant positive resistance to match measured net energy
   on the first 30 ns after the first current sample above 1 A. The ngspice
   current and saved source energy must agree with `I = V / R` and the exact
   integral of `V²/R` over each linear segment of the prescribed voltage.
3. Require the model to reproduce energy in the chosen window within 1%, while
   the time-weighted current NRMSE exceeds 50%. NRMSE is the square root of
   the integral of squared current error divided by the integral of squared
   measured current. This negative control demonstrates the difference between
   a scalar fit and waveform prediction.

The electrical time axis is shifted by the observed 1 A crossing in the file.
The optical panel uses its own published time values. The benchmark evaluates an
electrical model under measured terminal voltage. Inferring electron density,
species energy, discharge dynamics, or calibration transfer requires a physical
model and separately validated measurements. The comparison is within the
published trace and uses no independent acquisition holdout.

## Reproduce

```powershell
uv run --frozen python bench/literature/p2_minesi2022_pulse/run.py --run-root runs/measured_pulse
```

The new run root contains the replayable deck and case, the PCD simulation and
analysis record, `electrical_comparison.csv`, `optical_evidence.csv`,
`pulse_comparison.png`, and `benchmark_result.json`.
