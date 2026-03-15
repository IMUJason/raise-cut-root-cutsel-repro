from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--candidate-count-grid", nargs="+", type=int, default=[200, 250, 300, 350, 400])
    parser.add_argument("--dominant-share-max-grid", nargs="+", type=float, default=[0.70, 0.80, 0.90])
    parser.add_argument("--family-count-grid", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--min-routed", type=int, default=6)
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


def build_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["instance_id", "mode"]).drop_duplicates(["instance_id", "mode"], keep="last").copy()
    frame["gap_eval"] = frame["gap"].where(frame["gap"] < 1e19)
    probe = frame[frame["mode"] == "raise_portfolio_ud"][
        [
            "instance_id",
            "portfolio_probe_candidate_count",
            "portfolio_probe_mean_obj_parallelism",
            "portfolio_probe_regime",
            "portfolio_probe_dominant_family",
            "portfolio_probe_dominant_share",
            "portfolio_probe_family_count",
        ]
    ].copy()
    raise_cut = frame[frame["mode"] == "raise_cut"][
        ["instance_id", "gap_eval", "gap", "dualbound", "solving_time", "n_cuts_applied"]
    ].rename(columns=lambda value: value if value == "instance_id" else f"{value}_raise_cut")
    scip_dynamic = frame[frame["mode"] == "scip_dynamic"][
        ["instance_id", "gap_eval", "gap", "dualbound", "solving_time", "n_cuts_applied"]
    ].rename(columns=lambda value: value if value == "instance_id" else f"{value}_scip_dynamic")
    return probe.merge(raise_cut, on="instance_id", how="inner").merge(scip_dynamic, on="instance_id", how="inner")


def evaluate_rule(
    combined: pd.DataFrame,
    candidate_count_min: int,
    dominant_share_max: float,
    family_count_min: int,
    exclude_cmir_dominant: bool,
) -> dict[str, object]:
    use_dynamic = (
        (combined["portfolio_probe_regime"] == "mixed")
        & (combined["portfolio_probe_candidate_count"] >= candidate_count_min)
        & (combined["portfolio_probe_dominant_share"] <= dominant_share_max)
        & (combined["portfolio_probe_family_count"] >= family_count_min)
    )
    if exclude_cmir_dominant:
        use_dynamic &= combined["portfolio_probe_dominant_family"] != "cmir"

    chosen_gap = np.where(use_dynamic, combined["gap_eval_scip_dynamic"], combined["gap_eval_raise_cut"])
    raise_gap = combined["gap_eval_raise_cut"].to_numpy()
    finite_mask = np.isfinite(chosen_gap) & np.isfinite(raise_gap)
    gap_delta = chosen_gap[finite_mask] - raise_gap[finite_mask]
    log_gap_delta = np.log1p(chosen_gap[finite_mask]) - np.log1p(raise_gap[finite_mask]) if finite_mask.any() else np.array([])

    routed = combined.loc[use_dynamic].copy()
    raw_gap_delta_routed = (routed["gap_scip_dynamic"] - routed["gap_raise_cut"]).replace([np.inf, -np.inf], np.nan).dropna()
    dual_delta_routed = (
        routed["dualbound_scip_dynamic"] - routed["dualbound_raise_cut"]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    time_delta_routed = (
        routed["solving_time_scip_dynamic"] - routed["solving_time_raise_cut"]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    cut_delta_routed = (
        routed["n_cuts_applied_scip_dynamic"] - routed["n_cuts_applied_raise_cut"]
    ).replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "candidate_count_min": candidate_count_min,
        "dominant_share_max": dominant_share_max,
        "family_count_min": family_count_min,
        "exclude_cmir_dominant": exclude_cmir_dominant,
        "n_instances": int(combined["instance_id"].nunique()),
        "n_routed": int(use_dynamic.sum()),
        "policy_gap_pairs": int(len(gap_delta)),
        "policy_gap_wins": int((gap_delta < -1e-9).sum()),
        "policy_gap_losses": int((gap_delta > 1e-9).sum()),
        "policy_gap_ties": int(len(gap_delta) - (gap_delta < -1e-9).sum() - (gap_delta > 1e-9).sum()),
        "policy_mean_gap_delta": None if len(gap_delta) == 0 else float(np.mean(gap_delta)),
        "policy_mean_log_gap_delta": None if len(log_gap_delta) == 0 else float(np.mean(log_gap_delta)),
        "routed_raw_gap_pairs": int(len(raw_gap_delta_routed)),
        "routed_raw_gap_wins": int((raw_gap_delta_routed < -1e-9).sum()),
        "routed_raw_gap_losses": int((raw_gap_delta_routed > 1e-9).sum()),
        "routed_mean_raw_gap_delta": None if raw_gap_delta_routed.empty else float(raw_gap_delta_routed.mean()),
        "routed_dualbound_pairs": int(len(dual_delta_routed)),
        "routed_dualbound_wins": int((dual_delta_routed > 1e-9).sum()),
        "routed_dualbound_losses": int((dual_delta_routed < -1e-9).sum()),
        "routed_mean_dualbound_delta": None if dual_delta_routed.empty else float(dual_delta_routed.mean()),
        "routed_mean_time_delta": None if time_delta_routed.empty else float(time_delta_routed.mean()),
        "routed_mean_cut_delta": None if cut_delta_routed.empty else float(cut_delta_routed.mean()),
    }


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    frame = load_jsonl(input_path)
    combined = build_frame(frame)

    rows = []
    for candidate_count_min, dominant_share_max, family_count_min, exclude_cmir_dominant in itertools.product(
        args.candidate_count_grid,
        args.dominant_share_max_grid,
        args.family_count_grid,
        [False, True],
    ):
        rows.append(
            evaluate_rule(
                combined=combined,
                candidate_count_min=candidate_count_min,
                dominant_share_max=dominant_share_max,
                family_count_min=family_count_min,
                exclude_cmir_dominant=exclude_cmir_dominant,
            )
        )

    search_frame = pd.DataFrame(rows).sort_values(
        [
            "policy_gap_losses",
            "routed_raw_gap_losses",
            "routed_dualbound_losses",
            "policy_mean_log_gap_delta",
            "n_routed",
        ],
        ascending=[True, True, True, True, False],
    )
    search_frame.to_csv(output_csv, index=False)

    recommended = search_frame[
        (search_frame["n_routed"] >= args.min_routed)
        & (search_frame["policy_gap_losses"] == 0)
        & (search_frame["routed_raw_gap_losses"] == 0)
        & (search_frame["routed_dualbound_losses"] == 0)
    ].head(10)

    lines = [
        "# RAISE-Portfolio-UD Rule Search",
        "",
        f"- input: `{input_path.resolve()}`",
        f"- candidate_count grid: `{args.candidate_count_grid}`",
        f"- dominant_share_max grid: `{args.dominant_share_max_grid}`",
        f"- family_count grid: `{args.family_count_grid}`",
        f"- minimum routed instances for recommendation: `{args.min_routed}`",
        "",
        "## Recommended Rules",
        "",
        dataframe_to_markdown(recommended if not recommended.empty else search_frame.head(10)),
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
