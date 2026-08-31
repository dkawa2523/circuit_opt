# P1 — Colpo et al. (1999) ICP fixture benchmark

This case reproduces two pieces of tabulated/textual evidence from
<https://doi.org/10.1063/1.369268>:

- the two-section `L // C` conductive-dummy topology and its three resonance
  frequencies;
- the published graphite-dummy impedance `1.26 + j57 ohm` at 13.56 MHz and
  its comparison with the paper's boundary calculation `1.4 + j52 ohm`.

Run from the repository root:

```powershell
uv run python bench/literature/p1_colpo1999_icp/run.py
```

The rounded component values predict 7.047, 13.888, and 19.817 MHz.  The
middle value differs from the paper's 14.5 MHz label by about 4.2%, which is
within the declared tolerance for publication rounding.  The runner compares
the analytic circuit, generic PCD circuit realization, NGSpice response,
off-resonance points, passivity, and the known dummy load.

This validates fixture topology and a published known-load replay.  It does
not validate a plasma-on state model, planar production ICP, matching-network
range, density, uniformity, or process performance.  Figures 2 and 3 are not
silently converted into precise golden data.  The separate `digitized/`
challenge set carries two-reader marker pairing, axis calibration, exclusions,
and extraction uncertainty, and is intentionally not treated as a tight
experimental golden.
