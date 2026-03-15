from __future__ import annotations

import numpy as np

from .schemas import QUBOModel


def build_qubo(
    feature_records: list[dict],
    pairwise_matrix: np.ndarray,
    budget_k: int,
    penalty_rho: float = 1.0,
) -> QUBOModel:
    linear = np.array([record["lambda_i"] for record in feature_records], dtype=float)
    quadratic = np.array(pairwise_matrix, dtype=float)
    linear = linear + penalty_rho * (2 * budget_k - 1)
    quadratic = quadratic + _pair_penalty_matrix(len(feature_records), penalty_rho)
    cut_ids = tuple(record["cut_id"] for record in feature_records)
    return QUBOModel(
        linear=linear,
        quadratic=quadratic,
        budget_k=budget_k,
        penalty_rho=penalty_rho,
        cut_ids=cut_ids,
        metadata={"objective": "maximize"},
    )


def evaluate_qubo(qubo: QUBOModel, bitvector: np.ndarray) -> float:
    x = np.asarray(bitvector, dtype=float)
    return float(x @ qubo.linear + 0.5 * x @ qubo.quadratic @ x)


def _pair_penalty_matrix(n_vars: int, penalty_rho: float) -> np.ndarray:
    matrix = np.zeros((n_vars, n_vars), dtype=float)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            matrix[i, j] = -2.0 * penalty_rho
            matrix[j, i] = -2.0 * penalty_rho
    return matrix
