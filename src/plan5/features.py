from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .schemas import CutCandidate, NodeContext


DEFAULT_LAMBDA_WEIGHTS = {
    "violation_norm": 0.4,
    "efficacy_norm": 0.35,
    "objective_parallelism": 0.15,
    "support_density": -0.1,
}


def extract_single_cut_features(
    cuts: Iterable[CutCandidate],
    context: NodeContext,
    family_priors: dict[str, float] | None = None,
    lambda_weights: dict[str, float] | None = None,
) -> list[dict]:
    cuts = list(cuts)
    if not cuts:
        return []

    family_priors = family_priors or {}
    lambda_weights = lambda_weights or DEFAULT_LAMBDA_WEIGHTS
    violations = np.array([cut.violation_raw for cut in cuts], dtype=float)
    efficacies = np.array([cut.efficacy_raw for cut in cuts], dtype=float)
    violation_norms = _robust_scale(violations)
    efficacy_norms = _robust_scale(efficacies)

    records: list[dict] = []
    for idx, cut in enumerate(cuts):
        coeffs = cut.coefficients
        support_mask = coeffs != 0
        support_size = int(np.count_nonzero(support_mask))
        support_density = support_size / max(1, context.lp_cols)
        l1_norm = float(np.sum(np.abs(coeffs)))
        l2_norm = float(np.linalg.norm(coeffs))
        objective_parallelism = _objective_parallelism(coeffs, context.objective_vector)
        family_prior = float(family_priors.get(cut.family, 0.0))
        lambda_i = (
            lambda_weights.get("violation_norm", 0.0) * float(violation_norms[idx])
            + lambda_weights.get("efficacy_norm", 0.0) * float(efficacy_norms[idx])
            + lambda_weights.get("objective_parallelism", 0.0) * objective_parallelism
            + lambda_weights.get("support_density", 0.0) * support_density
            + family_prior
        )
        records.append(
            {
                "cut_id": cut.cut_id,
                "cut_family": cut.family,
                "generator_name": cut.generator_name,
                "source_round": int(cut.source_round),
                "violation_raw": float(cut.violation_raw),
                "violation_norm": float(violation_norms[idx]),
                "efficacy_raw": float(cut.efficacy_raw),
                "efficacy_norm": float(efficacy_norms[idx]),
                "objective_parallelism": objective_parallelism,
                "support_size": support_size,
                "support_density": float(support_density),
                "rhs_magnitude": float(abs(cut.rhs)),
                "l1_norm": l1_norm,
                "l2_norm": l2_norm,
                "candidate_pool_size": int(context.candidate_pool_size),
                "node_depth": int(context.node_depth),
                "lp_rows": int(context.lp_rows),
                "lp_cols": int(context.lp_cols),
                "lambda_i": float(lambda_i),
            }
        )
    return records


def _robust_scale(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    median = np.median(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if abs(iqr) < eps:
        centered = values - median
        max_abs = np.max(np.abs(centered))
        if max_abs < eps:
            return np.zeros_like(values)
        return centered / max_abs
    return (values - median) / (iqr + eps)


def _objective_parallelism(coefficients: np.ndarray, objective_vector: np.ndarray | None) -> float:
    if objective_vector is None:
        return 0.0
    if coefficients.shape[0] != objective_vector.shape[0]:
        return 0.0
    coeff_norm = np.linalg.norm(coefficients)
    obj_norm = np.linalg.norm(objective_vector)
    if coeff_norm == 0 or obj_norm == 0:
        return 0.0
    return float(np.abs(np.dot(coefficients, objective_vector) / (coeff_norm * obj_norm)))
