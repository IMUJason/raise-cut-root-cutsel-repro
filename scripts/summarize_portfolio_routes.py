from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--portfolio-mode", required=True)
    parser.add_argument("--reference-mode", default="raise_cut")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No rows found in {path}")
    return frame


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
    results_path = Path(args.results_jsonl)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    frame = load_jsonl(results_path)
    portfolio = frame[frame["mode"] == args.portfolio_mode][
        [
            "instance_id",
            "gap",
            "portfolio_routed_mode",
            "portfolio_route_policy",
            "portfolio_route_reason",
        ]
    ].copy()
    reference = frame[frame["mode"] == args.reference_mode][["instance_id", "gap"]].rename(columns={"gap": "reference_gap"})
    merged = portfolio.merge(reference, on="instance_id", how="left")
    routed_to_reference = merged["portfolio_routed_mode"] == args.reference_mode
    merged.loc[routed_to_reference & merged["reference_gap"].isna(), "reference_gap"] = merged.loc[
        routed_to_reference & merged["reference_gap"].isna(), "gap"
    ]

    def outcome(row: pd.Series) -> str:
        gap = row["gap"]
        ref_gap = row["reference_gap"]
        if pd.isna(gap) or pd.isna(ref_gap) or gap >= 1e19 or ref_gap >= 1e19:
            return "unpaired"
        delta = float(gap) - float(ref_gap)
        if delta < -1e-9:
            return "win"
        if delta > 1e-9:
            return "loss"
        return "tie"

    merged["paired_outcome"] = merged.apply(outcome, axis=1)
    route_stats = (
        merged.groupby(["portfolio_routed_mode", "portfolio_route_reason"], as_index=False)
        .agg(
            count=("instance_id", "count"),
            win=("paired_outcome", lambda values: int((values == "win").sum())),
            loss=("paired_outcome", lambda values: int((values == "loss").sum())),
            tie=("paired_outcome", lambda values: int((values == "tie").sum())),
            unpaired=("paired_outcome", lambda values: int((values == "unpaired").sum())),
        )
        .sort_values(["count", "portfolio_routed_mode", "portfolio_route_reason"], ascending=[False, True, True])
    )
    route_stats.to_csv(output_csv, index=False)
    output_md.write_text(
        "\n".join(
            [
                f"# {args.portfolio_mode} Route Audit",
                "",
                f"- results: `{results_path.resolve()}`",
                f"- reference_mode: `{args.reference_mode}`",
                "",
                dataframe_to_markdown(route_stats),
                "",
            ]
        ),
        encoding="utf-8",
    )
