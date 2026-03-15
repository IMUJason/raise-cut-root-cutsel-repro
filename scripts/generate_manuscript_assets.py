from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7A7A7A",
    "lightgrey": "#D9D9D9",
    "dark": "#1A1A1A",
}

MODE_LABELS = {
    "raise_cut": "RAISE-Cut",
    "plan5_adaptive": "Adaptive",
    "efficacy": "Efficacy",
    "default": "SCIP default",
    "raise_portfolio": "RAISE-Portfolio",
    "raise_portfolio_rc": "RAISE-Portfolio-RC",
    "raise_portfolio_ud": "RAISE-Portfolio-UD",
    "scip_dynamic": "SCIP dynamic",
    "scip_hybrid": "SCIP hybrid",
    "scip_ensemble": "SCIP ensemble",
}

ROUTE_REASON_LABELS = {
    "default_raise_cut": "Default abstain",
    "no_probe_summary_abstain_to_raise_cut": "No probe summary",
    "large_pool_ultra_low_obj_parallelism": "Large ultra-low parallelism",
    "high_dominance_clique": "High-dominance clique",
    "rc_abstain_to_raise_cut": "RC abstain",
    "rc_large_pool_low_obj_parallelism": "RC low-parallelism ensemble",
    "ud_abstain_to_raise_cut": "UD abstain",
    "ud_large_diffuse_mixed_noncmir": "UD dynamic route",
}

OUTCOME_COLORS = {
    "win": PALETTE["green"],
    "loss": PALETTE["vermillion"],
    "tie": PALETTE["blue"],
    "unpaired": PALETTE["grey"],
}

REGIME_ORDER = ["mixed", "gom", "cmir", "implbd", "clique", "flowcover", "other", "zerohalf", "no_rootcuts"]

CONFIG_LABELS = {
    "baseline": "Deployed (locked)",
    "diversity_push": "Diversity-heavy",
    "pair_penalty_strong": "Stronger pair penalty",
    "balanced_sparse": "Balanced sparse",
    "efficacy_heavy": "Efficacy-heavy",
    "parallelism_push": "Parallelism-heavy",
}

REGIME_LABELS = {
    "mixed": "Mixed",
    "gom": "Gomory",
    "cmir": "CMIR",
    "implbd": "ImplBd",
    "clique": "Clique",
    "flowcover": "FlowCover",
    "other": "Other",
    "zerohalf": "Zero-half",
    "no_rootcuts": "No root cuts",
}


@dataclass
class PairRecord:
    left: str
    right: str
    wins: int
    losses: int
    ties: int
    mean_log_gap_delta: float
    sign_test_pvalue: float | None
    mean_time_delta: float | None


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_WORK_ROOT = REPO_ROOT / "data" / "work"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "manuscript_assets"

WORK_ROOT = DEFAULT_WORK_ROOT
FIG_ROOT = DEFAULT_OUTPUT_ROOT / "figures"
TABLE_ROOT = DEFAULT_OUTPUT_ROOT / "tables"
LINEAGE_PATH = DEFAULT_OUTPUT_ROOT / "figure_table_lineage.csv"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def as_int(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode.replace("_", " "))


def regime_label(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " "))


def find_pair(rows: list[dict[str, str]], left: str, right: str) -> PairRecord:
    for row in rows:
        if row["mode_a"] == left and row["mode_b"] == right:
            return PairRecord(
                left=left,
                right=right,
                wins=as_int(row["gap_wins"]),
                losses=as_int(row["gap_losses"]),
                ties=as_int(row["gap_ties"]),
                mean_log_gap_delta=as_float(row["mean_log_gap_delta"]) or 0.0,
                sign_test_pvalue=as_float(row["sign_test_pvalue"]),
                mean_time_delta=as_float(row["mean_time_delta"]),
            )
    for row in rows:
        if row["mode_a"] == right and row["mode_b"] == left:
            return PairRecord(
                left=left,
                right=right,
                wins=as_int(row["gap_losses"]),
                losses=as_int(row["gap_wins"]),
                ties=as_int(row["gap_ties"]),
                mean_log_gap_delta=-(as_float(row["mean_log_gap_delta"]) or 0.0),
                sign_test_pvalue=as_float(row["sign_test_pvalue"]),
                mean_time_delta=-(as_float(row["mean_time_delta"]) or 0.0),
            )
    raise KeyError(f"Comparison not found: {left} vs {right}")


def color_for_delta(delta: float) -> str:
    return PALETTE["green"] if delta < 0 else PALETTE["vermillion"]


def add_panel_label(ax: plt.Axes, label: str, *, y: float = -0.20) -> None:
    ax.text(0.5, y, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="center", va="top")


def save(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, facecolor="white")


def write_table(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, *, facecolor: str, edgecolor: str = "#333333", fontsize: int = 9) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["dark"], wrap=True)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#444444") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=color)
    ax.add_patch(arrow)


def draw_poly_arrow(ax: plt.Axes, points: list[tuple[float, float]], *, color: str = "#444444", linewidth: float = 1.2) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)
    arrow = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14, linewidth=linewidth, color=color)
    ax.add_patch(arrow)


def draw_polyline(ax: plt.Axes, points: list[tuple[float, float]], *, color: str = "#444444", linewidth: float = 1.2) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth)


def generate_graphical_abstract(data: dict[str, PairRecord]) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.03, 0.22), (0.20, 0.54), "Audited instance chain\nbenchmark hold-out\nexternal unseen splits\nhashes and JSONL logs", facecolor="#EAF4FB", fontsize=9)
    draw_box(ax, (0.28, 0.22), (0.20, 0.54), "Root-pool probe\ncandidate count\nfamily mix and dominance\nsolver-side state summary", facecolor="#F7F7F7", fontsize=9)
    draw_box(ax, (0.53, 0.22), (0.20, 0.54), "RAISE decision layer\ndense expert for dominant pools\ninteraction expert for irregular pools\noptional dynamic delegation", facecolor="#E9F7F1", fontsize=9)
    draw_box(ax, (0.78, 0.22), (0.19, 0.54), "Observed outcome\n27/23/19 vs Adaptive\nrouting is route-sensitive\nclassical solver engineering", facecolor="#F6EFF8", fontsize=9)

    draw_arrow(ax, (0.23, 0.49), (0.28, 0.49))
    draw_arrow(ax, (0.48, 0.49), (0.53, 0.49))
    draw_arrow(ax, (0.73, 0.49), (0.78, 0.49))

    draw_box(ax, (0.28, 0.82), (0.18, 0.10), f"Benchmark h140\n27 / 23 / 19\nΔ = {data['bench_vs_adaptive'].mean_log_gap_delta:+.4f}", facecolor="#F7F7F7", fontsize=8.1)
    draw_box(ax, (0.55, 0.82), (0.16, 0.10), "Portfolio c140\n1 / 2 / 73\n1 / 5 / 71", facecolor="#F7F7F7", fontsize=8.1)
    draw_box(ax, (0.80, 0.82), (0.17, 0.10), "UD d120: 3 / 0 / 63\nUD h120: 4 / 2 / 54", facecolor="#F7F7F7", fontsize=8.1)
    ax.text(0.50, 0.07, "Quantum-inspired pairwise scoring is used only when the root-pool regime justifies it.", ha="center", va="center", fontsize=9.5, color=PALETTE["dark"])

    save(fig, FIG_ROOT / "graphical_abstract.pdf")
    save(fig, FIG_ROOT / "graphical_abstract.png")
    plt.close(fig)


def generate_framework_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.55
    h = 0.28
    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    texts = [
        "Audited data\nbenchmark and external splits\nlocked manifests\nhashes and logs",
        "Root-pool probe\ncandidate count\ndominant-family share\nfamily count",
        "Regime diagnosis\ndominant or mixed\nfamily-aware trigger\nsafe default behavior",
        "Expert choice\ndense ranking\ninteraction subset\noptional dynamic route",
        "Evaluation\npaired log-gap tests\ntransfer analysis\nroute audit",
    ]
    colors = ["#EAF4FB", "#F7F7F7", "#FFF4D6", "#E9F7F1", "#FDEFEA"]
    widths = [0.15, 0.15, 0.15, 0.15, 0.15]

    for x, w, text, color in zip(xs, widths, texts, colors):
        draw_box(ax, (x, y), (w, h), text, facecolor=color, fontsize=8.6)

    for left, width, right in zip(xs[:-1], widths[:-1], xs[1:]):
        draw_arrow(ax, (left + width, y + h / 2), (right, y + h / 2))

    draw_box(
        ax,
        (0.05, 0.11),
        (0.90, 0.19),
        "Scientific traceability layer\nDeterministic manifests  •  file hashes  •  JSONL root logs  •  route reasons  •  figure/table lineage  •  reference manifest",
        facecolor="#F6EFF8",
        fontsize=8.8,
    )

    save(fig, FIG_ROOT / "framework_overview.pdf")
    plt.close(fig)


def generate_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 9.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1)
    ax.axis("off")

    ax.text(0.06, 0.95, "RAISE-Cut mainline", ha="left", va="center", fontsize=9.6, color=PALETTE["dark"], fontweight="bold")
    draw_box(ax, (0.30, 0.84), (0.40, 0.09), "Root pool\nrows, efficacy, SCIP metadata", facecolor="#EAF4FB", fontsize=8.9)
    draw_box(ax, (0.30, 0.70), (0.40, 0.09), "Probe top min(60,$m_t$)\nshare $\\rho_t$, family count, dominant label", facecolor="#F7F7F7", fontsize=8.9)
    draw_box(ax, (0.24, 0.55), (0.52, 0.10), "RAISE-Cut gate\ninteraction for flowcover / other\notherwise dense efficacy expert", facecolor="#F6EFF8", fontsize=8.8)
    draw_box(ax, (0.07, 0.39), (0.26, 0.09), "Dense expert\nefficacy order\ncap 50", facecolor="#E9F7F1", fontsize=8.7)
    draw_box(ax, (0.67, 0.39), (0.26, 0.09), "Interaction expert\ngreedy + local exchange\ntruncated working set", facecolor="#FFF4D6", fontsize=8.7)
    draw_box(ax, (0.27, 0.145), (0.46, 0.09), "Applied cuts\nregime tag and structured root log", facecolor="#FDEFEA", fontsize=8.8)

    draw_arrow(ax, (0.50, 0.84), (0.50, 0.79))
    draw_arrow(ax, (0.50, 0.70), (0.50, 0.65))
    draw_poly_arrow(ax, [(0.36, 0.55), (0.36, 0.515), (0.20, 0.515), (0.20, 0.48)])
    draw_poly_arrow(ax, [(0.64, 0.55), (0.64, 0.515), (0.80, 0.515), (0.80, 0.48)])
    draw_polyline(ax, [(0.20, 0.39), (0.20, 0.305), (0.46, 0.305), (0.50, 0.305)])
    draw_polyline(ax, [(0.80, 0.39), (0.80, 0.305), (0.54, 0.305), (0.50, 0.305)])
    draw_arrow(ax, (0.50, 0.305), (0.50, 0.235))

    ax.plot([0.05, 0.95], [0.090, 0.090], color=PALETTE["lightgrey"], linewidth=1.0, linestyle="--")
    ax.text(0.06, 0.066, "Optional RAISE-Portfolio-UD layer", ha="left", va="center", fontsize=9.2, color=PALETTE["dark"], fontweight="bold")
    bottom_y = -0.060
    bottom_w = 0.27
    bottom_h = 0.085
    left_x, center_x, right_x = 0.04, 0.37, 0.70
    draw_box(ax, (left_x, bottom_y), (bottom_w, bottom_h), "Separate probe solve\nsame root-state summary", facecolor="#EAF4FB", fontsize=8.2)
    draw_box(ax, (center_x, bottom_y), (bottom_w, bottom_h), "UD gate\nmixed, $m_t\\geq 250$\n$\\rho_t\\leq 0.80$, families $\\geq 3$\ndominant family $\\neq$ cmir", facecolor="#F6EFF8", fontsize=7.8)
    draw_box(ax, (right_x, bottom_y), (bottom_w, bottom_h), "Final route\nabstain to RAISE-Cut\nor delegate to SCIP dynamic", facecolor="#E9F7F1", fontsize=8.1)

    connector_y = bottom_y + bottom_h / 2.0
    draw_arrow(ax, (left_x + bottom_w, connector_y), (center_x, connector_y))
    draw_arrow(ax, (center_x + bottom_w, connector_y), (right_x, connector_y))

    save(fig, FIG_ROOT / "method_overview.pdf")
    plt.close(fig)


def bar_panel(ax: plt.Axes, labels: list[str], deltas: list[float], annotations: list[str], subtitle: str) -> None:
    y = np.arange(len(labels))
    colors = [color_for_delta(v) for v in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(subtitle)
    span = max(max(abs(v) for v in deltas), 0.02)
    margin = max(0.018, 0.45 * span)
    ax.set_xlim(-span - margin, span + margin)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    x_right = ax.get_xlim()[1] - 0.06 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for yi, delta, ann in zip(y, deltas, annotations):
        ax.text(x_right, yi, ann, va="center", ha="right", fontsize=7, bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.80, "edgecolor": "none"})
    ax.spines[["top", "right"]].set_visible(False)


def generate_results_overview(data: dict[str, PairRecord]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.9))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, wspace=0.55)

    labels_a = ["Adaptive", "Efficacy", "Default"]
    recs_a = [data["bench_vs_adaptive"], data["bench_vs_efficacy"], data["bench_vs_default"]]
    bar_panel(
        axes[0],
        labels_a,
        [r.mean_log_gap_delta for r in recs_a],
        [f"{r.wins}/{r.losses}/{r.ties}" for r in recs_a],
        "Δ log-gap of RAISE-Cut vs comparator",
    )
    add_panel_label(axes[0], "(a) Benchmark hold-out")

    labels_b = ["Portfolio", "Portfolio-RC", "UD dev120", "UD hold-out120"]
    recs_b = [data["portfolio_vs_raise_cut"], data["portfolio_rc_vs_raise_cut"], data["ud_dev_vs_raise_cut"], data["ud_hold_vs_raise_cut"]]
    bar_panel(
        axes[1],
        labels_b,
        [r.mean_log_gap_delta for r in recs_b],
        [f"{r.wins}/{r.losses}/{r.ties}" for r in recs_b],
        "Δ log-gap of routed challenger vs RAISE-Cut",
    )
    add_panel_label(axes[1], "(b) External routing transfer")

    labels_c = ["Default", "Efficacy", "Dynamic", "Hybrid", "UD extension"]
    recs_c = [data["hold_vs_default"], data["hold_vs_efficacy"], data["hold_vs_dynamic"], data["hold_vs_hybrid"], data["hold_vs_ud"]]
    bar_panel(
        axes[2],
        labels_c,
        [r.mean_log_gap_delta for r in recs_c],
        [f"{r.wins}/{r.losses}/{r.ties}" for r in recs_c],
        "Δ log-gap of RAISE-Cut vs comparator",
    )
    add_panel_label(axes[2], "(c) Untouched external hold-out")

    save(fig, FIG_ROOT / "results_overview.pdf")
    plt.close(fig)


def build_heatmap_matrix(rows: list[dict[str, str]], baselines: list[str]) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    present_regimes = [r for r in REGIME_ORDER if any(row["dominant_family_regime"] == r for row in rows)]
    matrix = np.full((len(present_regimes), len(baselines)), np.nan)
    wins = np.full((len(present_regimes), len(baselines)), np.nan)
    losses = np.full((len(present_regimes), len(baselines)), np.nan)
    for i, regime in enumerate(present_regimes):
        for j, baseline in enumerate(baselines):
            for row in rows:
                if row["dominant_family_regime"] == regime and row["baseline_mode"] == baseline:
                    matrix[i, j] = as_float(row["mean_log_gap_delta"]) or 0.0
                    wins[i, j] = as_int(row["gap_wins"])
                    losses[i, j] = as_int(row["gap_losses"])
                    break
    return matrix, present_regimes, baselines, np.stack([wins, losses], axis=-1)


def heatmap_panel(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], annotations: np.ndarray) -> None:
    cmap = mpl.colormaps["RdYlGn_r"].copy()
    cmap.set_bad(color="#F2F2F2")
    vmax = max(0.11, float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 0.11
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), [mode_label(c) for c in col_labels], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            wins = int(annotations[i, j, 0])
            losses = int(annotations[i, j, 1])
            ax.text(j, i, f"{value:+.2f}\n{wins}-{losses}", ha="center", va="center", fontsize=6)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def generate_regime_heatmap() -> None:
    bench_rows = load_csv(WORK_ROOT / "root_cutsel_holdout140_raise_v1_stats_family_stats.csv")
    ext_rows = load_csv(WORK_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_family_stats.csv")

    bench_baselines = ["plan5_adaptive", "efficacy"]
    ext_baselines = ["default", "efficacy", "scip_dynamic", "scip_hybrid", "raise_portfolio_ud"]

    bench_m, bench_r, bench_c, bench_ann = build_heatmap_matrix(bench_rows, bench_baselines)
    ext_m, ext_r, ext_c, ext_ann = build_heatmap_matrix(ext_rows, ext_baselines)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1))
    fig.subplots_adjust(bottom=0.34, right=0.88, wspace=0.45)
    im = heatmap_panel(axes[0], bench_m, bench_r, bench_c, bench_ann)
    add_panel_label(axes[0], "(a) Benchmark hold-out", y=-0.26)
    heatmap_panel(axes[1], ext_m, ext_r, ext_c, ext_ann)
    add_panel_label(axes[1], "(b) Untouched external hold-out", y=-0.26)
    cbar = fig.colorbar(im, ax=axes, shrink=0.92)
    cbar.set_label("Mean log-gap delta")
    save(fig, FIG_ROOT / "regime_heatmap.pdf")
    plt.close(fig)


def scatter_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    for outcome in ["win", "loss", "tie", "unpaired"]:
        subset = [row for row in rows if row["outcome_vs_raise_cut"] == outcome]
        if not subset:
            continue
        x = [as_float(row["portfolio_probe_candidate_count"]) or 0.0 for row in subset]
        y = [as_float(row["portfolio_probe_dominant_share"]) or 0.0 for row in subset]
        size = [70 + 20 * as_int(row["portfolio_probe_family_count"]) for row in subset]
        ax.scatter(x, y, s=size, c=OUTCOME_COLORS[outcome], label=outcome, alpha=0.9, edgecolors="white", linewidths=0.8)
    ax.axvline(250, color="#444444", linestyle="--", linewidth=1.0)
    ax.axhline(0.80, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Probe candidate count")
    ax.set_ylabel("Dominant-family share")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(200, max(900, max(as_float(r["portfolio_probe_candidate_count"]) or 0 for r in rows) + 50))
    ax.set_ylim(0.30, 0.85)
    for row in rows:
        if row["outcome_vs_raise_cut"] == "loss":
            ax.annotate(
                row["instance_id"],
                (as_float(row["portfolio_probe_candidate_count"]) or 0.0, as_float(row["portfolio_probe_dominant_share"]) or 0.0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )


def generate_route_scatter() -> None:
    dev_rows = [row for row in load_csv(WORK_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_routed_cases.csv") if row["portfolio_routed_mode"] == "scip_dynamic"]
    hold_rows = [row for row in load_csv(WORK_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_routed_cases.csv") if row["portfolio_routed_mode"] == "scip_dynamic"]
    dev_route_stats = load_csv(WORK_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv")
    hold_route_stats = load_csv(WORK_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.csv")

    fig = plt.figure(figsize=(12.8, 5.2))
    fig.subplots_adjust(bottom=0.18, top=0.84)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    scatter_panel(ax1, dev_rows)
    add_panel_label(ax1, "(a) External development routes")
    scatter_panel(ax2, hold_rows)
    add_panel_label(ax2, "(b) Untouched external hold-out routes")

    dyn_dev = next(row for row in dev_route_stats if row["portfolio_routed_mode"] == "scip_dynamic")
    dyn_hold = next(row for row in hold_route_stats if row["portfolio_routed_mode"] == "scip_dynamic")
    labels = ["dev120", "holdout120"]
    bottoms = np.zeros(2)
    for outcome in ["win", "loss", "tie", "unpaired"]:
        values = [as_int(dyn_dev[outcome]), as_int(dyn_hold[outcome])]
        ax3.bar(labels, values, bottom=bottoms, color=OUTCOME_COLORS[outcome], label=outcome)
        bottoms += np.array(values)
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax3, "(c) Routed outcome counts")

    legend_order = ["win", "loss", "tie", "unpaired"]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=OUTCOME_COLORS[label],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        for label in legend_order
    ]
    fig.legend(handles, legend_order, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4, handletextpad=0.4, columnspacing=1.0)

    save(fig, FIG_ROOT / "route_scatter.pdf")
    plt.close(fig)


def generate_main_results_table(data: dict[str, PairRecord]) -> None:
    rows = [
        ("Benchmark h140", data["bench_vs_adaptive"]),
        ("Benchmark h140", data["bench_vs_efficacy"]),
        ("Benchmark h140", data["bench_vs_default"]),
        ("External c140", data["portfolio_vs_raise_cut"]),
        ("External c140", data["portfolio_rc_vs_raise_cut"]),
        ("External d120", data["ud_dev_vs_raise_cut"]),
        ("External h120", data["ud_hold_vs_raise_cut"]),
    ]
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}Xrrrrr}",
        r"\toprule",
        r"Dataset & Comparison & Wins & Losses & Ties & Mean log-gap $\Delta$ & Sign-test $p$ \\",
        r"\midrule",
    ]
    for dataset, record in rows:
        comparison = f"{mode_label(record.left)} vs {mode_label(record.right)}"
        pvalue = "--" if record.sign_test_pvalue is None else f"{record.sign_test_pvalue:.4f}"
        lines.append(
            f"{tex_escape(dataset)} & {tex_escape(comparison)} & {record.wins} & {record.losses} & {record.ties} & {record.mean_log_gap_delta:+.4f} & {pvalue} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table(TABLE_ROOT / "main_results.tex", lines)


def generate_baseline_table(data: dict[str, PairRecord]) -> None:
    rows = [data["hold_vs_default"], data["hold_vs_efficacy"], data["hold_vs_dynamic"], data["hold_vs_hybrid"], data["hold_vs_ud"]]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & Wins & Losses & Ties & Mean log-gap $\Delta$ & Sign-test $p$ \\",
        r"\midrule",
    ]
    for record in rows:
        comparison = f"{mode_label(record.left)} vs {mode_label(record.right)}"
        pvalue = "--" if record.sign_test_pvalue is None else f"{record.sign_test_pvalue:.4f}"
        lines.append(
            f"{tex_escape(comparison)} & {record.wins} & {record.losses} & {record.ties} & {record.mean_log_gap_delta:+.4f} & {pvalue} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLE_ROOT / "pilot_baselines.tex", lines)


def generate_route_audit_table() -> None:
    blocks = [
        ("Portfolio", "c140", load_csv(WORK_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_route_stats.csv")),
        ("Portfolio-RC", "c140", load_csv(WORK_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_route_stats.csv")),
        ("Portfolio-UD", "d120", load_csv(WORK_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv")),
        ("Portfolio-UD", "h120", load_csv(WORK_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.csv")),
    ]
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{0.9cm}>{\raggedright\arraybackslash}Xrrrrr}",
        r"\toprule",
        r"Policy & Split & Routed expert / reason & Count & Wins & Losses & Ties & Unpaired \\",
        r"\midrule",
    ]
    for policy, split, rows in blocks:
        for row in rows:
            expert = f"{mode_label(row['portfolio_routed_mode'])}; {ROUTE_REASON_LABELS.get(row['portfolio_route_reason'], row['portfolio_route_reason'])}"
            lines.append(
                f"{tex_escape(policy)} & {tex_escape(split)} & {tex_escape(expert)} & {as_int(row['count'])} & {as_int(row['win'])} & {as_int(row['loss'])} & {as_int(row['tie'])} & {as_int(row['unpaired'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table(TABLE_ROOT / "route_audit.tex", lines)


def generate_weight_tuning_table() -> None:
    rows = sorted(load_csv(WORK_ROOT / "plan5_weight_tuning_summary.csv"), key=lambda row: as_float(row["avg_gap"]) or float("inf"))
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Configuration & Avg. root gap & Avg. dual bound \\",
        r"\midrule",
    ]
    for row in rows:
        label = CONFIG_LABELS.get(row["config_name"], row["config_name"].replace("_", " "))
        avg_gap = as_float(row["avg_gap"]) or 0.0
        avg_dualbound = as_float(row["avg_dualbound"]) or 0.0
        if row["config_name"] == "baseline":
            label = rf"\textbf{{{tex_escape(label)}}}"
            lines.append(rf"{label} & \textbf{{{avg_gap:.4f}}} & \textbf{{{avg_dualbound:.1f}}} \\")
        else:
            lines.append(rf"{tex_escape(label)} & {avg_gap:.4f} & {avg_dualbound:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLE_ROOT / "weight_tuning.tex", lines)


def generate_ud_tradeoff_figure() -> None:
    rows = load_csv(WORK_ROOT / "external_unseen_dev120_raise_portfolio_ud_rule_search.csv")
    aggregated: dict[tuple[int, float], dict[str, float | int]] = {}
    for row in rows:
        x = as_int(row["n_routed"])
        y = round(as_float(row["policy_mean_log_gap_delta"]) or 0.0, 10)
        key = (x, y)
        bucket = aggregated.setdefault(key, {"count": 0, "x": x, "y": float(y)})
        bucket["count"] = int(bucket["count"]) + 1

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    xvals = [float(bucket["x"]) for bucket in aggregated.values()]
    yvals = [float(bucket["y"]) for bucket in aggregated.values()]
    sizes = [55 + 18 * int(bucket["count"]) for bucket in aggregated.values()]
    ax.scatter(
        xvals,
        yvals,
        s=sizes,
        color=PALETTE["sky"],
        edgecolors=PALETTE["blue"],
        linewidths=0.8,
        alpha=0.85,
        zorder=2,
    )
    for bucket in aggregated.values():
        count = int(bucket["count"])
        if count > 1:
            ax.text(
                float(bucket["x"]),
                float(bucket["y"]) + 0.0015,
                f"×{count}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=PALETTE["dark"],
            )

    deployed = next(
        row
        for row in rows
        if row["candidate_count_min"] == "250"
        and row["dominant_share_max"] == "0.8"
        and row["family_count_min"] == "3"
        and row["exclude_cmir_dominant"] == "True"
    )
    deployed_x = as_int(deployed["n_routed"])
    deployed_y = as_float(deployed["policy_mean_log_gap_delta"]) or 0.0
    ax.scatter(
        [deployed_x],
        [deployed_y],
        marker="*",
        s=220,
        color=PALETTE["orange"],
        edgecolors=PALETTE["dark"],
        linewidths=0.8,
        zorder=4,
    )
    ax.annotate(
        "Deployed",
        xy=(deployed_x, deployed_y),
        xytext=(deployed_x + 0.25, deployed_y + 0.006),
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": PALETTE["dark"]},
        fontsize=7,
        color=PALETTE["dark"],
    )
    ax.annotate(
        "Looser top rules",
        xy=(9, deployed_y),
        xytext=(8.55, deployed_y - 0.010),
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": PALETTE["grey"]},
        fontsize=7,
        color=PALETTE["grey"],
    )
    ax.axhline(0.0, color=PALETTE["grey"], linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlabel("Delegated cases on dev split")
    ax.set_ylabel(r"Mean log-gap $\Delta$")
    ax.set_xlim(0.5, max(xvals) + 0.8)
    ax.set_ylim(min(yvals) - 0.012, max(yvals) + 0.012)
    ax.grid(axis="y", color=PALETTE["lightgrey"], linewidth=0.6)
    ax.text(
        0.02,
        0.96,
        "Negative values favor delegation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=PALETTE["dark"],
    )
    fig.tight_layout()
    save(fig, FIG_ROOT / "ud_tradeoff.pdf")


def generate_regime_support_table() -> None:
    bench_rows = load_csv(WORK_ROOT / "root_cutsel_holdout140_raise_v1_stats_family_stats.csv")
    hold_rows = load_csv(WORK_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_family_stats.csv")

    bench_map = {row["dominant_family_regime"]: row for row in bench_rows if row["baseline_mode"] == "plan5_adaptive"}
    hold_map = {row["dominant_family_regime"]: row for row in hold_rows if row["baseline_mode"] == "scip_dynamic"}

    regimes = [
        regime
        for regime in REGIME_ORDER
        if regime not in {"no_rootcuts"}
        and (regime in bench_map or regime in hold_map)
        and regime != "scg"
    ]

    def fmt(row: dict[str, str] | None) -> str:
        if row is None or as_int(row.get("n_gap_pairs")) == 0:
            return "--"
        wins = as_int(row["gap_wins"])
        losses = as_int(row["gap_losses"])
        ties = as_int(row["gap_ties"])
        delta = as_float(row["mean_log_gap_delta"]) or 0.0
        return rf"{wins}-{losses}-{ties}; {delta:+.3f}"

    lines = [
        r"\begin{tabularx}{\textwidth}{lXX}",
        r"\toprule",
        r"Regime & Benchmark h140 vs Adaptive & External h120 vs SCIP dynamic \\",
        r"\midrule",
    ]
    for regime in regimes:
        lines.append(
            rf"{tex_escape(regime_label(regime))} & {fmt(bench_map.get(regime))} & {fmt(hold_map.get(regime))} \\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table(TABLE_ROOT / "regime_support.tex", lines)


def lineage_row(
    artifact_id: str,
    artifact_type: str,
    caption: str,
    source_result_files: str,
    source_manifest: str,
    output_path: str,
    generated_on: str,
    notes: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "caption": caption,
        "source_run_ids": "",
        "source_result_files": source_result_files,
        "source_manifest": source_manifest,
        "generation_script": "scripts/generate_manuscript_assets.py",
        "output_path": output_path,
        "generated_on": generated_on,
        "notes": notes,
    }


def write_lineage() -> None:
    generated_on = datetime.now().isoformat(timespec="seconds")
    rows = [
        lineage_row(
            "F1",
            "figure",
            "Graphical abstract summarizing the audited RAISE workflow and external validation chain.",
            "work/root_cutsel_holdout140_raise_v1_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats_paired_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_stats_paired_stats.csv;work/external_unseen_holdout120_v2_ud_confirmatory_v1_raise_cut_stats_paired_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_test140_resource_matched_valid_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/figures/graphical_abstract.pdf",
            generated_on,
            "Also exported to PNG for journal submission.",
        ),
        lineage_row(
            "F2",
            "figure",
            "Overall framework linking audited data construction, regime diagnosis, expert choice, and evaluation.",
            "work/root_cutsel_holdout140_raise_v1_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats_paired_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_stats_paired_stats.csv;work/external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_paired_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_test140_resource_matched_valid_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/figures/framework_overview.pdf",
            generated_on,
            "High-level logic figure for the audited method and evaluation chain.",
        ),
        lineage_row(
            "F3",
            "figure",
            "Method overview for RAISE-Cut and the optional RAISE-Portfolio-UD delegation layer.",
            "work/root_cutsel_holdout140_raise_v1_stats_paired_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv",
            "paper/submission/eswa_manuscript/figures/method_overview.pdf",
            generated_on,
            "Schematic reflects the deployed route rule and regime switch.",
        ),
        lineage_row(
            "F4",
            "figure",
            "Overview of benchmark confirmation, external routing transfer, and fresh strong baselines.",
            "work/root_cutsel_holdout140_raise_v1_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats_paired_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_stats_paired_stats.csv;work/external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_paired_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_test140_resource_matched_valid_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/figures/results_overview.pdf",
            generated_on,
            "Negative deltas favor the left-named method in each panel.",
        ),
        lineage_row(
            "F5",
            "figure",
            "Regime-wise heatmap of mean log-gap delta for RAISE-Cut.",
            "work/root_cutsel_holdout140_raise_v1_stats_family_stats.csv;work/external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_family_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/figures/regime_heatmap.pdf",
            generated_on,
            "Cell annotations show mean delta and win-loss counts.",
        ),
        lineage_row(
            "F6",
            "figure",
            "Routed-case scatter and outcome audit for the uncertainty-driven dynamic delegation rule.",
            "work/external_unseen_dev120_raise_portfolio_ud_v1_routed_cases.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv;work/external_unseen_holdout120_v2_ud_confirmatory_v1_routed_cases.csv;work/external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.csv",
            "work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/figures/route_scatter.pdf",
            generated_on,
            "Scatter uses the deployed candidate-count and dominance-share thresholds.",
        ),
        lineage_row(
            "F7",
            "figure",
            "Development-set coverage-regret view of the uncertainty-driven route search.",
            "work/external_unseen_dev120_raise_portfolio_ud_rule_search.csv",
            "work/manifests/external_unseen_dev120_manifest.csv",
            "paper/submission/eswa_manuscript/figures/ud_tradeoff.pdf",
            generated_on,
            "Deployed rule highlighted against looser top-performing alternatives.",
        ),
        lineage_row(
            "T2",
            "table",
            "Main paired results for the benchmark hold-out and the external portfolio transfer chain.",
            "work/root_cutsel_holdout140_raise_v1_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats_paired_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats_paired_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_stats_paired_stats.csv;work/external_unseen_holdout120_v2_ud_confirmatory_v1_raise_cut_stats_paired_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_test140_resource_matched_valid_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/tables/generated/main_results.tex",
            generated_on,
            "Comparison table used in the main manuscript.",
        ),
        lineage_row(
            "T3",
            "table",
            "Fresh external strong-baseline comparisons on the external hold-out.",
            "work/external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_paired_stats.csv",
            "work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/tables/generated/pilot_baselines.tex",
            generated_on,
            "Strong-baseline table used in the main manuscript.",
        ),
        lineage_row(
            "T4",
            "table",
            "Route audit across portfolio policies and splits.",
            "work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_route_stats.csv;work/external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_route_stats.csv;work/external_unseen_dev120_raise_portfolio_ud_v1_route_stats.csv;work/external_unseen_holdout120_v2_ud_confirmatory_v1_route_stats.csv",
            "work/manifests/external_unseen_test140_resource_matched_valid_manifest.csv;work/manifests/external_unseen_dev120_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/tables/generated/route_audit.tex",
            generated_on,
            "Counts and outcomes by routed expert.",
        ),
        lineage_row(
            "T5",
            "table",
            "Development-stage weight sweep used to freeze the deployed linear-plus-pairwise coefficients.",
            "work/plan5_weight_tuning_summary.csv",
            "work/manifests/dev_manifest.csv",
            "paper/submission/eswa_manuscript/tables/generated/weight_tuning.tex",
            generated_on,
            "Appendix calibration table for frozen coefficients.",
        ),
        lineage_row(
            "T6",
            "table",
            "Descriptive regime-wise support for the benchmark adaptive comparison and the external SCIP dynamic comparison.",
            "work/root_cutsel_holdout140_raise_v1_stats_family_stats.csv;work/external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_family_stats.csv",
            "work/manifests/test_manifest.csv;work/manifests/external_unseen_holdout120_v2_manifest.csv",
            "paper/submission/eswa_manuscript/tables/generated/regime_support.tex",
            generated_on,
            "Appendix regime table supporting the gating interpretation.",
        ),
    ]
    fieldnames = list(rows[0].keys())
    with LINEAGE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_data() -> dict[str, PairRecord]:
    holdout_rows = load_csv(WORK_ROOT / "root_cutsel_holdout140_raise_v1_stats_paired_stats.csv")
    portfolio_rows = load_csv(WORK_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_stats_paired_stats.csv")
    portfolio_rc_rows = load_csv(WORK_ROOT / "external_unseen_confirmatory140_resource_matched_v1_raise_portfolio_rc_stats_paired_stats.csv")
    dev_ud_rows = load_csv(WORK_ROOT / "external_unseen_dev120_raise_portfolio_ud_v1_stats_paired_stats.csv")
    holdout_ud_rows = load_csv(WORK_ROOT / "external_unseen_holdout120_v2_ud_confirmatory_v1_raise_cut_stats_paired_stats.csv")
    holdout_full_rows = load_csv(WORK_ROOT / "external_unseen_holdout120_v2_full_baselines_v1_raise_cut_stats_paired_stats.csv")
    return {
        "bench_vs_adaptive": find_pair(holdout_rows, "raise_cut", "plan5_adaptive"),
        "bench_vs_efficacy": find_pair(holdout_rows, "raise_cut", "efficacy"),
        "bench_vs_default": find_pair(holdout_rows, "raise_cut", "default"),
        "portfolio_vs_raise_cut": find_pair(portfolio_rows, "raise_portfolio", "raise_cut"),
        "portfolio_rc_vs_raise_cut": find_pair(portfolio_rc_rows, "raise_portfolio_rc", "raise_cut"),
        "ud_dev_vs_raise_cut": find_pair(dev_ud_rows, "raise_portfolio_ud", "raise_cut"),
        "ud_hold_vs_raise_cut": find_pair(holdout_ud_rows, "raise_portfolio_ud", "raise_cut"),
        "hold_vs_default": find_pair(holdout_full_rows, "raise_cut", "default"),
        "hold_vs_efficacy": find_pair(holdout_full_rows, "raise_cut", "efficacy"),
        "hold_vs_dynamic": find_pair(holdout_full_rows, "raise_cut", "scip_dynamic"),
        "hold_vs_hybrid": find_pair(holdout_full_rows, "raise_cut", "scip_hybrid"),
        "hold_vs_ud": find_pair(holdout_full_rows, "raise_cut", "raise_portfolio_ud"),
    }


def main() -> None:
    global WORK_ROOT, FIG_ROOT, TABLE_ROOT, LINEAGE_PATH
    args = parse_args()
    WORK_ROOT = Path(args.work_root).resolve()
    output_root = Path(args.output_root).resolve()
    FIG_ROOT = output_root / "figures"
    TABLE_ROOT = output_root / "tables"
    LINEAGE_PATH = output_root / "figure_table_lineage.csv"
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    pair_data = build_pair_data()
    generate_graphical_abstract(pair_data)
    generate_framework_overview()
    generate_method_overview()
    generate_results_overview(pair_data)
    generate_regime_heatmap()
    generate_route_scatter()
    generate_ud_tradeoff_figure()
    generate_main_results_table(pair_data)
    generate_baseline_table(pair_data)
    generate_route_audit_table()
    generate_weight_tuning_table()
    generate_regime_support_table()
    write_lineage()
    print(f"Wrote figures to {FIG_ROOT}")
    print(f"Wrote tables to {TABLE_ROOT}")
    print(f"Wrote lineage to {LINEAGE_PATH}")


if __name__ == "__main__":
    main()
