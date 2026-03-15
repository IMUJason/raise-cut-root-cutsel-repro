# RAISE-Cut Root Cut Selection Reproducibility Package

This repository is a minimal public reproducibility package extracted from the Plan 5 workspace for the root-level cut selection study centered on `RAISE-Cut` and its portfolio variants.

It is designed for two levels of reuse:

1. Analysis-level reproduction from archived merged result files.
2. Optional solver-side reruns if `SCIP` and `PySCIPOpt` are available locally.

## Included

- `src/plan5`: core selector, regime, logging, QUBO, and utility code.
- `scripts`: data-summary rebuild scripts, manuscript-asset generator, and optional solver rerun entry points.
- `tests`: lightweight unit tests for selector logic and route utilities.
- `data/manifests`: audited manifests used in the study.
- `data/results`: merged JSONL and CSV artifacts from completed experimental runs.
- `data/work`: archived summary CSVs used to generate manuscript figures and tables.

## Not Included

- Raw benchmark instance files.
- A bundled `SCIP` installation.
- Private local machine paths; release artifacts in this package are sanitized for public upload.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts\rebuild_analysis_artifacts.py
python scripts\generate_manuscript_assets.py
```

This writes rebuilt summaries to `outputs/recomputed_work/` and manuscript-ready figures and tables to `outputs/manuscript_assets/`.

Two locked paper artifacts are intentionally copied from archived summaries during rebuild:

- `plan5_weight_tuning_summary.csv`
- `external_unseen_dev120_raise_portfolio_ud_rule_search.csv`

They are preserved this way because the public package does not include the full upstream tuning pipeline, and the UD rule-search ranking is sensitive to floating-point tie handling across environments.

To generate manuscript assets from the rebuilt summaries instead of the archived CSVs:

```powershell
python scripts\generate_manuscript_assets.py --work-root outputs\recomputed_work --output-root outputs\recomputed_assets
```

## Optional Solver Reruns

Analysis-only reproduction does not require `PySCIPOpt`. Full solver reruns do.

```powershell
python -m pip install -e .[solver]
python scripts\run_root_cutsel_experiment.py --help
```

You must provide your own local instance collection and manifests with valid paths before attempting reruns.

## Suggested Public Repository Name

Recommended GitHub repository name: `raise-cut-root-cutsel-repro`

## Citation and Archival Metadata

- Repository citation metadata: `CITATION.cff`
- Zenodo deposition metadata: `.zenodo.json`
- Copy-ready availability statements: `docs/availability-statements.md`
- Draft GitHub release notes: `docs/release-v0.1.0.md`
