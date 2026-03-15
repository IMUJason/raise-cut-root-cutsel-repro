from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
RESULTS_ROOT = REPO_ROOT / "data" / "results"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "recomputed_work"


def run(*args: str | Path) -> None:
    command = [sys.executable, *(str(arg) for arg in args)]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def merged_paths(dataset: str) -> tuple[Path, Path]:
    root = RESULTS_ROOT / dataset
    results = next(root.glob("*_results_merged.jsonl"))
    rounds = next(root.glob("*_rounds_merged.jsonl"))
    return results, rounds


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "data" / "work" / "plan5_weight_tuning_summary.csv", OUTPUT_ROOT / "plan5_weight_tuning_summary.csv")

    summarize = SCRIPTS_ROOT / "summarize_root_cutsel_results.py"
    analyze = SCRIPTS_ROOT / "analyze_root_cutsel_stats.py"
    route_audit = SCRIPTS_ROOT / "summarize_portfolio_routes.py"
    routed_cases = SCRIPTS_ROOT / "export_routed_cases.py"
    rule_search = SCRIPTS_ROOT / "search_raise_portfolio_ud_rules.py"
    combine = SCRIPTS_ROOT / "combine_results_jsonl.py"

    hold_results, hold_rounds = merged_paths("holdout_confirmatory_test140_v1")
    run(
        summarize,
        "--input-jsonl",
        hold_results,
        "--output-csv",
        OUTPUT_ROOT / "root_cutsel_holdout140_v1_summary.csv",
        "--output-md",
        OUTPUT_ROOT / "root_cutsel_holdout140_v1_summary.md",
        "--instance-csv",
        OUTPUT_ROOT / "root_cutsel_holdout140_v1_instances.csv",
        "--instance-md",
        OUTPUT_ROOT / "root_cutsel_holdout140_v1_instances.md",
        "--baseline-modes",
        "default",
        "efficacy",
    )
    run(
        analyze,
        "--results-jsonl",
        hold_results,
        "--rounds-jsonl",
        hold_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "root_cutsel_holdout140_raise_v1_stats",
        "--primary-mode",
        "raise_cut",
        "--baseline-modes",
        "plan5_adaptive",
        "efficacy",
        "default",
    )

    confirm_results, confirm_rounds = merged_paths("external_unseen_confirmatory140_resource_matched_v1")
    run(
        analyze,
        "--results-jsonl",
        confirm_results,
        "--rounds-jsonl",
        confirm_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats",
        "--primary-mode",
        "raise_portfolio",
        "--baseline-modes",
        "raise_cut",
        "scip_ensemble",
        "efficacy",
        "default",
    )
    run(
        route_audit,
        "--results-jsonl",
        confirm_results,
        "--portfolio-mode",
        "raise_portfolio",
        "--reference-mode",
        "raise_cut",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_route_stats.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_route_stats.md",
    )
    run(
        analyze,
        "--results-jsonl",
        confirm_results,
        "--rounds-jsonl",
        confirm_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats",
        "--primary-mode",
        "raise_portfolio_rc",
        "--baseline-modes",
        "raise_cut",
        "raise_portfolio",
        "scip_ensemble",
        "efficacy",
        "default",
    )
    run(
        route_audit,
        "--results-jsonl",
        confirm_results,
        "--portfolio-mode",
        "raise_portfolio_rc",
        "--reference-mode",
        "raise_cut",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_route_stats.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_route_stats.md",
    )

    dev_results, _ = merged_paths("external_unseen_dev120_raise_portfolio_ud_v1")
    dev_raise_cut_results, dev_raise_cut_rounds = merged_paths("external_unseen_dev120_raise_cut_baselines_v1")
    dev_scip_builtin_results, _ = merged_paths("external_unseen_dev120_scip_builtin_v1")
    dev_combined_results = OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_combined_results.jsonl"
    run(
        combine,
        "--input-jsonl",
        dev_results,
        dev_raise_cut_results,
        dev_scip_builtin_results,
        "--output-jsonl",
        dev_combined_results,
    )
    run(
        summarize,
        "--input-jsonl",
        dev_combined_results,
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_summary.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_summary.md",
        "--instance-csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_instances.csv",
        "--instance-md",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_instances.md",
        "--baseline-modes",
        "default",
        "efficacy",
        "raise_cut",
        "scip_dynamic",
    )
    run(
        analyze,
        "--results-jsonl",
        dev_combined_results,
        "--rounds-jsonl",
        dev_raise_cut_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_stats",
        "--primary-mode",
        "raise_cut",
        "--baseline-modes",
        "efficacy",
        "default",
        "scip_dynamic",
        "scip_hybrid",
        "raise_portfolio_ud",
    )
    run(
        route_audit,
        "--results-jsonl",
        dev_combined_results,
        "--portfolio-mode",
        "raise_portfolio_ud",
        "--reference-mode",
        "raise_cut",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_route_stats.md",
    )
    run(
        routed_cases,
        "--results-jsonl",
        dev_combined_results,
        "--portfolio-mode",
        "raise_portfolio_ud",
        "--reference-mode",
        "raise_cut",
        "--expert-mode",
        "scip_dynamic",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_routed_cases.csv",
    )
    run(
        rule_search,
        "--input-jsonl",
        dev_combined_results,
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_rule_search.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_rule_search.md",
    )
    shutil.copyfile(
        REPO_ROOT / "data" / "work" / "external_unseen_dev120_raise_portfolio_ud_rule_search.csv",
        OUTPUT_ROOT / "external_unseen_dev120_raise_portfolio_ud_rule_search.csv",
    )

    hold_ud_results, hold_ud_rounds = merged_paths("external_unseen_holdout120_v2_ud_confirmatory_v1")
    run(
        summarize,
        "--input-jsonl",
        hold_ud_results,
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_summary.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_summary.md",
        "--instance-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_instances.csv",
        "--instance-md",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_instances.md",
        "--baseline-modes",
        "default",
        "efficacy",
        "raise_cut",
        "scip_dynamic",
    )
    run(
        analyze,
        "--results-jsonl",
        hold_ud_results,
        "--rounds-jsonl",
        hold_ud_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_raise_cut_stats",
        "--primary-mode",
        "raise_cut",
        "--baseline-modes",
        "efficacy",
        "default",
        "scip_dynamic",
        "raise_portfolio_ud",
    )
    run(
        route_audit,
        "--results-jsonl",
        hold_ud_results,
        "--portfolio-mode",
        "raise_portfolio_ud",
        "--reference-mode",
        "raise_cut",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.md",
    )
    run(
        routed_cases,
        "--results-jsonl",
        hold_ud_results,
        "--portfolio-mode",
        "raise_portfolio_ud",
        "--reference-mode",
        "raise_cut",
        "--expert-mode",
        "scip_dynamic",
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_routed_cases.csv",
    )

    scip_hybrid_results, _ = merged_paths("external_unseen_holdout120_v2_scip_hybrid_v1")
    full_results = OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_results.jsonl"
    run(
        combine,
        "--input-jsonl",
        hold_ud_results,
        scip_hybrid_results,
        "--output-jsonl",
        full_results,
    )
    run(
        summarize,
        "--input-jsonl",
        full_results,
        "--output-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_summary.csv",
        "--output-md",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_summary.md",
        "--instance-csv",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_instances.csv",
        "--instance-md",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_instances.md",
        "--baseline-modes",
        "default",
        "efficacy",
        "scip_dynamic",
        "scip_hybrid",
        "raise_portfolio_ud",
    )
    run(
        analyze,
        "--results-jsonl",
        full_results,
        "--rounds-jsonl",
        hold_ud_rounds,
        "--output-prefix",
        OUTPUT_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats",
        "--primary-mode",
        "raise_cut",
        "--baseline-modes",
        "default",
        "efficacy",
        "scip_dynamic",
        "scip_hybrid",
        "raise_portfolio_ud",
    )

    print(f"rebuilt analysis artifacts under {OUTPUT_ROOT}")
