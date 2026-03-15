# Availability Statements

This document provides copy-paste-ready availability text for manuscript submission, revision, and final publication workflows.

## Code Availability

The reproducibility package for the RAISE-Cut root cut selection study is publicly available at:

`https://github.com/IMUJason/raise-cut-root-cutsel-repro`

The repository contains the selector implementation used for analysis, sanitized manifests, archived merged result files, archived summary tables, and scripts that regenerate the analysis-level figures and tables used in the manuscript. Analysis-level reproduction does not require `PySCIPOpt`.

## Data Availability

The repository includes sanitized experimental artifacts needed for analysis-level reproduction, including merged JSONL outputs and archived summary CSV files. Raw benchmark instance files are not redistributed in this public package because they originate from external benchmark collections with their own access conditions. Researchers who have local access to the corresponding instance sets can attach their own paths through the provided manifests and optionally rerun the solver-side pipeline.

## Reproducibility Scope

This public package supports:

- regeneration of analysis summaries from archived merged result files;
- regeneration of manuscript figures and tables from archived or recomputed summaries;
- inspection of the audited experimental data flow used in the study.

This public package does not, by itself, guarantee full end-to-end reruns of every original solver experiment, because such reruns require a local SCIP installation, `PySCIPOpt`, and user-supplied benchmark instances.

## Short Version for Journal Submission Systems

Code and sanitized analysis artifacts are publicly available at `https://github.com/IMUJason/raise-cut-root-cutsel-repro`. The repository supports analysis-level reproduction of the reported summaries and manuscript assets. Raw benchmark instances are not redistributed in the public package; solver-side reruns require locally installed SCIP/`PySCIPOpt` and user-provided benchmark files.
