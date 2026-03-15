from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--portfolio-mode", required=True)
    parser.add_argument("--reference-mode", default="raise_cut")
    parser.add_argument("--expert-mode", default="scip_dynamic")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No rows found in {path}")
    frame = frame.sort_values(["instance_id", "mode"]).drop_duplicates(["instance_id", "mode"], keep="last").copy()
    frame["gap_eval"] = frame["gap"].where(frame["gap"] < 1e19)
    return frame


def outcome(delta: float | None) -> str:
    if delta is None or pd.isna(delta):
        return "unpaired"
    if delta < -1e-9:
        return "win"
    if delta > 1e-9:
        return "loss"
    return "tie"


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_frame(Path(args.results_jsonl))
    portfolio = frame[frame["mode"] == args.portfolio_mode][
        [
            "instance_id",
            "portfolio_routed_mode",
            "portfolio_route_reason",
            "portfolio_probe_candidate_count",
            "portfolio_probe_mean_obj_parallelism",
            "portfolio_probe_regime",
            "portfolio_probe_dominant_family",
            "portfolio_probe_dominant_share",
            "portfolio_probe_family_count",
            "gap_eval",
            "gap",
            "dualbound",
            "solving_time",
            "n_cuts_applied",
        ]
    ].copy()
    reference = frame[frame["mode"] == args.reference_mode][
        ["instance_id", "gap_eval", "gap", "dualbound", "solving_time", "n_cuts_applied"]
    ].rename(columns=lambda value: value if value == "instance_id" else f"{value}_{args.reference_mode}")
    expert = frame[frame["mode"] == args.expert_mode][
        ["instance_id", "gap_eval", "gap", "dualbound", "solving_time", "n_cuts_applied"]
    ].rename(columns=lambda value: value if value == "instance_id" else f"{value}_{args.expert_mode}")

    merged = portfolio.merge(reference, on="instance_id", how="left").merge(expert, on="instance_id", how="left")
    merged = merged[merged["portfolio_routed_mode"] == args.expert_mode].copy()
    ref_col = f"gap_eval_{args.reference_mode}"
    portfolio_gap = pd.to_numeric(merged["gap_eval"], errors="coerce")
    reference_gap = pd.to_numeric(merged[ref_col], errors="coerce")
    finite_mask = np.isfinite(portfolio_gap) & np.isfinite(reference_gap)
    merged["delta_gap_vs_raise_cut"] = np.where(finite_mask, portfolio_gap - reference_gap, np.nan)
    merged["outcome_vs_raise_cut"] = merged["delta_gap_vs_raise_cut"].map(outcome)
    merged.to_csv(output_path, index=False)
    print(f"wrote {len(merged)} rows to {output_path.resolve()}")
