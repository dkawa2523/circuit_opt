# Changelog

## 0.16.0

- distinguish complete SPICE decks from title-less circuit fragments;
- replace imported sources by exact name rather than inferred node conflict;
- preserve unavailable current as missing data and enforce requested RF cycles;
- provide the stable `pcd.api` extension surface and isolate plugin inputs;
- add dataset identities and runtime fingerprints to `evaluations.csv`;
- centralize saved-run artifact resolution and make preparation failures reuse
  their original run directory;
- provide an explicit, dry-run-by-default retention command for old study
  generations without touching reusable caches;
- document module ownership, supported extension boundaries, and generated
  output responsibilities;
- add the repository-scoped `$pcd-runner` Codex skill for validated ngspice,
  study, benchmark, visualization, and result-inspection workflows.
