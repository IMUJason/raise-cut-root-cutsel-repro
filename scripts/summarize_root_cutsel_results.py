from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--instance-csv")
    parser.add_argument("--instance-md")
    parser.add_argument("--baseline-modes", nargs="+", default=["default", "efficacy"])
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def sign_test_pvalue(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(total, k) for k in range(0, tail + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def paired_comparisons(frame: pd.DataFrame, baseline_mode: str) -> list[dict[str, object]]:
    baseline = frame[frame["mode"] == baseline_mode][["instance_id", "gap_eval", "dualbound", "n_cuts_applied", "solving_time"]]
    baseline = baseline.rename(
        columns={
            "gap_eval": "gap_baseline",
            "dualbound": "dualbound_baseline",
            "n_cuts_applied": "cuts_baseline",
            "solving_time": "time_baseline",
        }
    )
    rows: list[dict[str, object]] = []
    for mode in sorted(frame["mode"].unique()):
        if mode == baseline_mode:
            continue
        current = frame[frame["mode"] == mode][["instance_id", "gap_eval", "dualbound", "n_cuts_applied", "solving_time"]]
        merged = current.merge(baseline, on="instance_id", how="inner")
        if merged.empty:
            continue
        finite_gap = merged.dropna(subset=["gap_eval", "gap_baseline"]).copy()
        gap_delta = finite_gap["gap_eval"] - finite_gap["gap_baseline"]
        dual_delta = merged["dualbound"] - merged["dualbound_baseline"]
        wins = int((gap_delta < -1e-9).sum()) if not gap_delta.empty else 0
        losses = int((gap_delta > 1e-9).sum()) if not gap_delta.empty else 0
        ties = int(len(gap_delta) - wins - losses) if not gap_delta.empty else 0
        rows.append(
            {
                "baseline_mode": baseline_mode,
                "mode": mode,
                "n_pairs": int(len(merged)),
                "n_gap_pairs": int(len(gap_delta)),
                "mean_gap_delta": float(gap_delta.mean()) if not gap_delta.empty else None,
                "median_gap_delta": float(gap_delta.median()) if not gap_delta.empty else None,
                "mean_dualbound_delta": float(dual_delta.mean()),
                "mean_cut_delta": float((merged["n_cuts_applied"] - merged["cuts_baseline"]).mean()),
                "mean_time_delta": float((merged["solving_time"] - merged["time_baseline"]).mean()),
                "gap_wins": wins,
                "gap_losses": losses,
                "gap_ties": ties,
                "gap_sign_test_pvalue": sign_test_pvalue(wins, losses),
            }
        )
    return rows


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


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    frame = load_results(input_path)
    if frame.empty:
        raise SystemExit(f"No rows found in {input_path}")
    frame["gap_eval"] = frame["gap"].where(frame["gap"] < 1e19)

    summary = (
        frame.groupby("mode", as_index=False)
        .agg(
            n_instances=("instance_id", "nunique"),
            n_finite_gap=("gap_eval", lambda values: int(values.notna().sum())),
            mean_gap=("gap_eval", "mean"),
            median_gap=("gap_eval", "median"),
            mean_dualbound=("dualbound", "mean"),
            mean_cuts_applied=("n_cuts_applied", "mean"),
            mean_lp_iterations=("n_lp_iterations", "mean"),
            mean_time=("solving_time", "mean"),
            mean_root_candidates=("root_candidates_total", "mean"),
            mean_root_selected=("root_selected_total", "mean"),
        )
        .sort_values("mean_gap", ascending=True)
    )
    summary.to_csv(output_csv, index=False)

    comparisons: list[dict[str, object]] = []
    for baseline_mode in args.baseline_modes:
        comparisons.extend(paired_comparisons(frame, baseline_mode))
    comparison_frame = pd.DataFrame(comparisons)

    lines = [
        "# Root-Level Cut Selector Summary",
        "",
        f"- input results: `{input_path.resolve()}`",
        f"- output summary: `{output_csv.resolve()}`",
        "",
        "## Aggregate by mode",
        "",
        dataframe_to_markdown(summary),
    ]
    if not comparison_frame.empty:
        lines.extend(["", "## Paired comparisons", "", dataframe_to_markdown(comparison_frame)])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.instance_csv or args.instance_md:
        gap_pivot = frame.pivot(index="instance_id", columns="mode", values="gap_eval")
        dual_pivot = frame.pivot(index="instance_id", columns="mode", values="dualbound")
        time_pivot = frame.pivot(index="instance_id", columns="mode", values="solving_time")
        cuts_pivot = frame.pivot(index="instance_id", columns="mode", values="n_cuts_applied")
        instance_rows = []
        all_modes = sorted(frame["mode"].unique())
        for instance_id in sorted(frame["instance_id"].unique()):
            row: dict[str, object] = {"instance_id": instance_id}
            gap_candidates: dict[str, float] = {}
            for mode in all_modes:
                gap_value = gap_pivot.at[instance_id, mode] if mode in gap_pivot.columns else None
                dual_value = dual_pivot.at[instance_id, mode] if mode in dual_pivot.columns else None
                time_value = time_pivot.at[instance_id, mode] if mode in time_pivot.columns else None
                cuts_value = cuts_pivot.at[instance_id, mode] if mode in cuts_pivot.columns else None
                row[f"{mode}_gap"] = gap_value
                row[f"{mode}_dualbound"] = dual_value
                row[f"{mode}_time"] = time_value
                row[f"{mode}_cuts"] = cuts_value
                if pd.notna(gap_value):
                    gap_candidates[mode] = float(gap_value)
            row["best_gap_mode"] = min(gap_candidates, key=gap_candidates.get) if gap_candidates else None
            if "plan5" in gap_pivot.columns and "efficacy" in gap_pivot.columns:
                plan5_gap = gap_pivot.at[instance_id, "plan5"]
                efficacy_gap = gap_pivot.at[instance_id, "efficacy"]
                if pd.notna(plan5_gap) and pd.notna(efficacy_gap):
                    row["plan5_minus_efficacy_gap"] = float(plan5_gap - efficacy_gap)
                else:
                    row["plan5_minus_efficacy_gap"] = None
            if "plan5" in dual_pivot.columns and "efficacy" in dual_pivot.columns:
                row["plan5_minus_efficacy_dualbound"] = float(
                    dual_pivot.at[instance_id, "plan5"] - dual_pivot.at[instance_id, "efficacy"]
                )
            instance_rows.append(row)

        instance_frame = pd.DataFrame(instance_rows)
        if args.instance_csv:
            instance_csv = Path(args.instance_csv)
            instance_csv.parent.mkdir(parents=True, exist_ok=True)
            instance_frame.to_csv(instance_csv, index=False)
        if args.instance_md:
            instance_md = Path(args.instance_md)
            instance_md.parent.mkdir(parents=True, exist_ok=True)
            instance_lines = [
                "# Root-Level Per-Instance Table",
                "",
                f"- input results: `{input_path.resolve()}`",
                "",
                dataframe_to_markdown(instance_frame),
            ]
            instance_md.write_text("\n".join(instance_lines) + "\n", encoding="utf-8")
