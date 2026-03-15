from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .schemas import CutCandidate


DEFAULT_MU_WEIGHTS = {
    "complementarity_proxy": 0.45,
    "family_diversity_bonus": 0.1,
    "cosine_similarity": -0.25,
    "support_overlap_ratio": -0.15,
    "same_family": -0.15,
}


def extract_pairwise_interactions(
    cuts: Iterable[CutCandidate],
    feature_records: list[dict],
    mu_weights: dict[str, float] | None = None,
    top_r_neighbors: int = 10,
    overlap_threshold: float = 0.2,
    complementarity_threshold: float = 0.1,
) -> tuple[list[dict], np.ndarray]:
    cuts = list(cuts)
    mu_weights = mu_weights or DEFAULT_MU_WEIGHTS
    n_cuts = len(cuts)
    retained_neighbors: dict[int, set[int]] = defaultdict(set)
    preliminary: list[tuple[int, int, dict]] = []

    for i in range(n_cuts):
        for j in range(i + 1, n_cuts):
            record = _pair_record(cuts[i], cuts[j], feature_records[i], feature_records[j], mu_weights)
            preliminary.append((i, j, record))

    for anchor in range(n_cuts):
        candidates = [
            (other, rec["cosine_similarity"])
            for i, j, rec in preliminary
            for other in ([j] if i == anchor else ([i] if j == anchor else []))
        ]
        for other, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[:top_r_neighbors]:
            retained_neighbors[anchor].add(other)

    quadratic = np.zeros((n_cuts, n_cuts), dtype=float)
    pair_records: list[dict] = []
    for i, j, record in preliminary:
        retain = (
            j in retained_neighbors[i]
            or i in retained_neighbors[j]
            or record["support_overlap_ratio"] >= overlap_threshold
            or record["complementarity_proxy"] >= complementarity_threshold
        )
        mu_ij = float(record["mu_ij"]) if retain else 0.0
        record["mu_ij"] = mu_ij
        record["is_retained_after_sparsification"] = bool(retain)
        pair_records.append(record)
        quadratic[i, j] = mu_ij
        quadratic[j, i] = mu_ij

    np.fill_diagonal(quadratic, 0.0)
    return pair_records, quadratic


def _pair_record(
    cut_i: CutCandidate,
    cut_j: CutCandidate,
    features_i: dict,
    features_j: dict,
    weights: dict[str, float],
) -> dict:
    coeff_i = cut_i.coefficients
    coeff_j = cut_j.coefficients
    cosine_similarity = _cosine_similarity(coeff_i, coeff_j)
    overlap = _support_overlap_ratio(coeff_i, coeff_j)
    same_family = 1 if cut_i.family == cut_j.family else 0
    diversity_bonus = 0 if same_family else 1
    support_disjointness = 1.0 - overlap
    complementarity = support_disjointness * min(
        float(features_i["efficacy_norm"]),
        float(features_j["efficacy_norm"]),
    )
    mu_ij = (
        weights.get("complementarity_proxy", 0.0) * complementarity
        + weights.get("family_diversity_bonus", 0.0) * diversity_bonus
        + weights.get("cosine_similarity", 0.0) * cosine_similarity
        + weights.get("support_overlap_ratio", 0.0) * overlap
        + weights.get("same_family", 0.0) * same_family
    )
    return {
        "cut_i_id": cut_i.cut_id,
        "cut_j_id": cut_j.cut_id,
        "cosine_similarity": float(cosine_similarity),
        "support_overlap_ratio": float(overlap),
        "same_family": int(same_family),
        "family_diversity_bonus": int(diversity_bonus),
        "support_disjointness": float(support_disjointness),
        "complementarity_proxy": float(complementarity),
        "violation_gap_abs": float(abs(features_i["violation_norm"] - features_j["violation_norm"])),
        "efficacy_gap_abs": float(abs(features_i["efficacy_norm"] - features_j["efficacy_norm"])),
        "mu_ij": float(mu_ij),
    }


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.abs(np.dot(v1, v2) / (n1 * n2)))


def _support_overlap_ratio(v1: np.ndarray, v2: np.ndarray) -> float:
    s1 = v1 != 0
    s2 = v2 != 0
    union = np.count_nonzero(s1 | s2)
    if union == 0:
        return 0.0
    inter = np.count_nonzero(s1 & s2)
    return float(inter / union)
