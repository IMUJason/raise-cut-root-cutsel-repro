# Release v0.1.0

Initial public reproducibility release for the Plan 5 / RAISE-Cut root cut selection study.

## Included in this release

- analysis-ready Python package and scripts;
- sanitized manifests with local absolute paths removed;
- merged JSONL results and archived summary CSVs needed for analysis-level reproduction;
- manuscript asset generation scripts;
- lightweight unit tests;
- citation and archival metadata for GitHub and Zenodo workflows.

## Reproduction scope

This release is analysis-first. Users can rebuild summary artifacts and manuscript tables/figures from the archived merged results without `PySCIPOpt`.

Optional solver reruns are exposed in the package but require:

- a local SCIP installation;
- `PySCIPOpt`;
- benchmark instances supplied by the user under valid local paths.

## Known limitations

- Raw benchmark instances are not redistributed.
- Two paper-facing summary CSV files are intentionally preserved from archived outputs because the full upstream tuning pipeline is not part of the public package and one rule-search ranking is sensitive to floating-point tie handling across environments.

## Suggested GitHub release text

`v0.1.0` is the first public reproducibility release for the RAISE-Cut root cut selection study. It contains the minimal code, sanitized manifests, merged results, archived summaries, and rebuild scripts needed to reproduce the analysis-level artifacts reported in the manuscript. Full solver reruns remain optional and require local SCIP/`PySCIPOpt` plus user-supplied benchmark instances.
