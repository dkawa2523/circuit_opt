# Colpo 1999 plasma-on impedance challenge set

`qualified_impedance.csv` is the single canonical transcription of marker
centres in Figures 2 and 3 of Colpo, Ernst, and Rossi,
<https://doi.org/10.1063/1.369268>.  It uses the mean of two independent reads
and carries the reader disagreement as data.  It pairs `R_m` and `X_m` only
when the same pressure marker is present at the same reported power:

- 10, 100, and 1000 mTorr;
- 1, 2, 3, 4, and 5 kW;
- 15 paired points at 13.56 MHz.

The three 500 W resistance markers have no paired reactance marker and remain
explicitly unqualified in `excluded_points.csv`; no interpolation fills them.
`axis_calibration.yaml` preserves the transforms, raster hashes, marker
mapping, reference plane, and bounded reading uncertainty.  The raster itself
is not redistributed.  A conservative `+/-6 ohm` for R and `+/-7 ohm` for X
covers base pixel reading uncertainty plus the observed reader disagreement.
These bounds do not include RFZ60 measurement uncertainty.

Run the source-fidelity and one-port replay checks from the repository root:

```powershell
uv run python bench/literature/p1_colpo1999_icp/digitized/run.py
```

The runner reverses every pixel transform, checks pairing, exclusions,
passivity and reported monotone trends, then replays each central `R+jX` point
through the platform's `impedance_point` model and NGSpice. Passing accepts
this as a central-point global RFZ60-plane challenge set carrying
digitization-only bounds as metadata.  The bounds are not automatically
expanded by the normal `impedance_table` path, so this runner does not claim
uncertainty-propagated tuner coverage.  It is not a tight experimental golden,
a plasma-only impedance, an `icp_transformer` parameter fit, a 50-ohm
generator-plane reflection test, or a production planar ICP qualification.
The paper does not publish matcher settings, so no claim is made that a
platform matcher fixture reproduces the paper's less-than-1% reflected-power
operation.

## Explicit digitization-corner challenge

`uncertainty_corners.csv` materializes the four rectangular combinations of
`R_center +/- 6 ohm` and `X_center +/- 7 ohm` for every qualified centre. The
result is 60 derived outcomes belonging to 15 parent pressure-power
conditions. `uncertainty_challenge.yaml` is the machine-readable
derivation and claim boundary; the parent table remains unchanged.  This is a
conservative correlation-unknown envelope, not 60 independent observations or
a confidence interval.

Run the central replay plus the fixed-network and 16-state bounded-tuner
challenge from the repository root:

```powershell
uv run python bench/literature/p1_colpo1999_icp/digitized/run_uncertainty_challenge.py
```

The report emphasizes robust, mixed, and failed parent conditions; it does not
treat the 60 corners as independent operating observations. The two matching
networks and the 10% reflected-power criterion are platform
engineering fixtures.  They are deliberately not identified as the unpublished
Colpo matcher, and they establish neither RFZ60 measurement uncertainty nor
production tuner dynamics, component stress, loss, or hardware qualification.
