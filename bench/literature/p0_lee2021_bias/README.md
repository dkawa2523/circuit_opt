# P0 — Lee 2021 bias-path reference-plane benchmark

This P0 benchmark uses the exact Table-I values from Lee, Kwon, and Chung,
AIP Advances 11, 025027 (2021), DOI `10.1063/6.0000883`.

It asks three narrow questions:

1. Do the reported `Csh series [C0 parallel (Rp series Lp)]` values reconstruct
   the reported corrected plasma-terminal impedance at 13.56 MHz?
2. Does the current PCD/ngspice path reproduce an independently evaluated ideal
   pi match at the VI-probe/matcher-output reference plane?
3. How does the same frozen matcher-output fixture respond if impedances from
   two different downstream planes are substituted without their transforms?

Run it from the repository root:

```powershell
uv run python bench/literature/p0_lee2021_bias/run.py
```

Add `--run-root <path>` to retain generated netlists and solver artifacts, or
`--output <file.json>` to retain the compact result. By default all run
artifacts are temporary.

The 10% reflected-power limit is a platform decision criterion, not a claim
that Lee et al. specified a universal chamber limit. The benchmark consumes
the paper's already corrected values; it does not claim to reproduce the
unpublished distributed parameters of the 720 mm RF path.

The three impedances are alternate electrical descriptions at named planes,
not three operating scenarios. The two downstream substitutions are a
plane-misuse sensitivity only: failure of the frozen fixture does not prove
that either published impedance is physically wrong, nor that PCD can infer or
validate the missing de-embedding transform.
