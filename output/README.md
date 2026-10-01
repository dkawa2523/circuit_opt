# Reviewed outputs

This directory contains versioned products generated from reviewed benchmark
runs: the concise Markdown/JSON/CSV benchmark report and the PDF figure pack. Their source
runs live under ignored `runs/` and can be reproduced with the commands in
`bench/reports/README.md` and `bench/figures/README.md`.

`pdf/dual-frequency-ccp-pulsed-bias-study.pdf` is the reviewed four-page
electrical conformance and bounded inverse-identification evidence for the
literature-grounded 40 MHz / 800 kHz rectangular-bias CCP problem. It names
the wafer/chuck and upper-reflection observables separately and does not claim
self-consistent plasma-state or process qualification.

`pdf/etch-ccp-dcs-dynamic-impedance-study.pdf` is the reviewed five-page
60 MHz / 2 MHz / pulsed-negative-DC etch-CCP circuit study. It includes the
external matching/feed/bias network, time-varying bulk resistance and
inductance, two time-varying lossy sheath capacitances, independent forward
conformance, dynamic impedance plots, and a complete 81-candidate bounded
identification with an unused reflected-voltage hold-out.

`pptx/PCD_RF_Circuit_Design_Engineering_Review_JA_v12_9slides.pptx` is the
single version-controlled review deck. Superseded tracked revisions are not
kept beside the current artifact; Git history remains the revision record.
Ignored local exports may coexist in this directory.

Do not use `output/` as a mutable simulation directory. Ordinary study results,
raw solver files, and reusable caches belong under `runs/` or an explicit
`pcd run --output` location.
