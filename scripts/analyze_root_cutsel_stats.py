from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from plan5.scip_cutsel import normalize_cut_family_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--rounds-jsonl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--primary-mode", default="plan5_adaptive")
    parser.add_argument("--baseline-modes", nargs="+", default=["efficacy", "default"])
    parser.add_argument("--dominance-threshold", type=float, default=0.55)
    parser.add_argument("--min-regime-size", type=int, default=3)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sign_test_pvalue(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(total, k) for k in range(0, tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_float_dtype(safe[column]):
            safe[column] = safe[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = [str(column) for column in safe.columns]
    rows = [headers]
    rows.extend([[str(value) for value in row] for row in safe.astype(object).fillna("").values.tolist()])
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    header_line = "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |" for row in rows[1:]]
    return "\n".join([header_line, divider, *body])


def paired_stats(frame: pd.DataFrame, mode_a: str, mode_b: str) -> dict[str, object]:
    a = frame[frame["mode"] == mode_a][["instance_id", "gap_eval", "solving_time", "n_cuts_applied", "dualbound"]]
    b = frame[frame["mode"] == mode_b][["instance_id", "gap_eval", "solving_time", "n_cuts_applied", "dualbound"]]
    merged = a.merge(b, on="instance_id", suffixes=(f"_{mode_a}", f"_{mode_b}"))
    finite = merged.dropna(subset=[f"gap_eval_{mode_a}", f"gap_eval_{mode_b}"]).copy()
    gap_delta = finite[f"gap_eval_{mode_a}"] - finite[f"gap_eval_{mode_b}"]
    log_gap_delta = np.log1p(finite[f"gap_eval_{mode_a}"]) - np.log1p(finite[f"gap_eval_{mode_b}"])
    wins = int((gap_delta < -1e-9).sum())
    losses = int((gap_delta > 1e-9).sum())
    ties = int(len(gap_delta) - wins - losses)

    def _wilcoxon(values: pd.Series) -> tuple[float | None, float | None]:
        nonzero = values[np.abs(values) > 1e-12]
        if len(nonzero) == 0:
            return None, None
        try:
            result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", correction=False, method="auto")
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return None, None

    gap_stat, gap_p = _wilcoxon(gap_delta)
    log_gap_stat, log_gap_p = _wilcoxon(log_gap_delta)
    return {
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n_pairs": int(len(merged)),
        "n_gap_pairs": int(len(finite)),
        "gap_wins": wins,
        "gap_losses": losses,
        "gap_ties": ties,
        "mean_gap_delta": None if finite.empty else float(gap_delta.mean()),
        "median_gap_delta": None if finite.empty else float(gap_delta.median()),
        "mean_log_gap_delta": None if finite.empty else float(log_gap_delta.mean()),
        "median_log_gap_delta": None if finite.empty else float(log_gap_delta.median()),
        "mean_time_delta": float((merged[f"solving_time_{mode_a}"] - merged[f"solving_time_{mode_b}"]).mean()),
        "mean_cut_delta": float((merged[f"n_cuts_applied_{mode_a}"] - merged[f"n_cuts_applied_{mode_b}"]).mean()),
        "mean_dualbound_delta": float((merged[f"dualbound_{mode_a}"] - merged[f"dualbound_{mode_b}"]).mean()),
        "sign_test_pvalue": sign_test_pvalue(wins, losses),
        "wilcoxon_gap_statistic": gap_stat,
        "wilcoxon_gap_pvalue": gap_p,
        "wilcoxon_log_gap_statistic": log_gap_stat,
        "wilcoxon_log_gap_pvalue": log_gap_p,
    }


def build_regimes(results_frame: pd.DataFrame, rounds_rows: list[dict], baseline_mode: str, dominance_threshold: float, min_regime_size: int) -> pd.DataFrame:
    lookup_frame = results_frame[["run_id", "instance_id", "mode"]].drop_duplicates()
    run_lookup = {
        (row["run_id"], row["instance_id"]): {"instance_id": row["instance_id"], "mode": row["mode"]}
        for _, row in lookup_frame.iterrows()
    }
    run_family_counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rounds_rows:
        key = (row["run_id"], row["instance_id"])
        if key not in run_lookup:
            continue
        if run_lookup[key]["mode"] != baseline_mode:
            continue
        counts = run_family_counts.setdefault(key, {})
        for cut_name in row.get("selected_cut_names", []):
            family = normalize_cut_family_name(cut_name)
            counts[family] = counts.get(family, 0) + 1

    records = []
    for (run_id, instance_id), meta in run_lookup.items():
        if meta["mode"] != baseline_mode:
            continue
        counts = run_family_counts.get((run_id, instance_id), {})
        total = sum(counts.values())
        if total == 0:
            dominant_family = "no_rootcuts"
            dominant_share = 0.0
        else:
            dominant_family = max(counts, key=counts.get)
            dominant_share = counts[dominant_family] / total
            if dominant_share < dominance_threshold:
                dominant_family = "mixed"
        records.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "baseline_mode": baseline_mode,
                "dominant_family_regime": dominant_family,
                "dominant_family_share": dominant_share,
                "family_count_total": total,
            }
        )
    regime_frame = pd.DataFrame(records)
    if regime_frame.empty:
        return regime_frame
    counts = regime_frame["dominant_family_regime"].value_counts()
    regime_frame["dominant_family_regime"] = regime_frame["dominant_family_regime"].map(
        lambda regime: regime if regime in {"mixed", "no_rootcuts"} or counts.get(regime, 0) >= min_regime_size else "other"
    )
    return regime_frame


def family_wise_table(results_frame: pd.DataFrame, regime_frame: pd.DataFrame, primary_mode: str, baseline_mode: str) -> pd.DataFrame:
    if regime_frame.empty:
        return pd.DataFrame()
    merged = results_frame.merge(regime_frame[["instance_id", "dominant_family_regime"]], on="instance_id", how="left")
    rows = []
    for regime, regime_frame_part in merged.groupby("dominant_family_regime"):
        stats = paired_stats(regime_frame_part, primary_mode, baseline_mode)
        rows.append({"dominant_family_regime": regime, **stats})
    return pd.DataFrame(rows).sort_values(["n_gap_pairs", "dominant_family_regime"], ascending=[False, True])


if __name__ == "__main__":
    args = parse_args()
    results_path = Path(args.results_jsonl)
    rounds_path = Path(args.rounds_jsonl)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    results_rows = load_jsonl(results_path)
    rounds_rows = load_jsonl(rounds_path)
    results_frame = pd.DataFrame(results_rows)
    results_frame["gap_eval"] = results_frame["gap"].where(results_frame["gap"] < 1e19)

    comparison_rows = []
    for baseline_mode in args.baseline_modes:
        comparison_rows.append(paired_stats(results_frame, args.primary_mode, baseline_mode))
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_csv = output_prefix.with_name(output_prefix.name + "_paired_stats.csv")
    comparison_md = output_prefix.with_name(output_prefix.name + "_paired_stats.md")
    comparison_frame.to_csv(comparison_csv, index=False)
    comparison_md.write_text(
        "\n".join(
            [
                "# Root-Level Paired Statistics",
                "",
                f"- results: `{results_path.resolve()}`",
                f"- rounds: `{rounds_path.resolve()}`",
                "",
                dataframe_to_markdown(comparison_frame),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    regime_frame = build_regimes(
        results_frame=results_frame,
        rounds_rows=rounds_rows,
        baseline_mode="efficacy",
        dominance_threshold=args.dominance_threshold,
        min_regime_size=args.min_regime_size,
    )
    regime_csv = output_prefix.with_name(output_prefix.name + "_family_regimes.csv")
    regime_md = output_prefix.with_name(output_prefix.name + "_family_regimes.md")
    if not regime_frame.empty:
        regime_frame.to_csv(regime_csv, index=False)
        regime_md.write_text(
            "\n".join(
                [
                    "# Root-Level Dominant-Family Regimes",
                    "",
                    "- Regime assignment is inferred from the `efficacy` run's selected cut-family mix.",
                    f"- dominance threshold: `{args.dominance_threshold}`",
                    "",
                    dataframe_to_markdown(regime_frame.sort_values('instance_id')),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    family_rows = []
    for baseline_mode in args.baseline_modes:
        frame = family_wise_table(results_frame, regime_frame, args.primary_mode, baseline_mode)
        if frame.empty:
            continue
        frame.insert(1, "baseline_mode", baseline_mode)
        family_rows.append(frame)
    family_frame = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    family_csv = output_prefix.with_name(output_prefix.name + "_family_stats.csv")
    family_md = output_prefix.with_name(output_prefix.name + "_family_stats.md")
    if not family_frame.empty:
        family_frame.to_csv(family_csv, index=False)
        family_md.write_text(
            "\n".join(
                [
                    "# Root-Level Family-Wise Statistics",
                    "",
                    "- Family strata are inferred from the dominant selected cut family in the `efficacy` baseline.",
                    "- This is an analysis regime, not a claim of official MIPLIB family taxonomy.",
                    "",
                    dataframe_to_markdown(family_frame),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
