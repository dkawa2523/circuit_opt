# Reviewed outputs

This directory contains versioned products generated from reviewed benchmark
runs: the cross-disciplinary JSON report and the PDF figure pack. Their source
runs live under ignored `runs/` and can be reproduced with the commands in
`bench/reports/README.md` and `bench/figures/README.md`.

Do not use `output/` as a mutable simulation directory. Ordinary study results,
raw solver files, and reusable caches belong under `runs/` or an explicit
`pcd run --output` location.
