from __future__ import annotations

from .features import extract_single_cut_features
from .interactions import extract_pairwise_interactions
from .qaoa_backend import select_qaoa_inspired, select_qaoa_unconstrained_baseline
from .qubo import build_qubo
from .schemas import CutCandidate, NodeContext
from .selectors import select_qubo_classical, select_topk_linear


def run_selection_pipeline(
    cuts: list[CutCandidate],
    context: NodeContext,
    budget_k: int,
    penalty_rho: float = 1.0,
) -> dict:
    feature_records = extract_single_cut_features(cuts, context)
    pair_records, pairwise_matrix = extract_pairwise_interactions(cuts, feature_records)
    qubo = build_qubo(feature_records, pairwise_matrix, budget_k=budget_k, penalty_rho=penalty_rho)
    linear_result = select_topk_linear(qubo)
    qubo_result = select_qubo_classical(qubo)
    qaoa_unconstrained = select_qaoa_unconstrained_baseline(qubo, depth_p=1, max_qubits=min(16, qubo.size))
    qaoa_result = select_qaoa_inspired(qubo, depth_p=1, max_qubits=min(20, qubo.size))
    return {
        "feature_records": feature_records,
        "pair_records": pair_records,
        "qubo": qubo,
        "results": {
            "linear": linear_result,
            "qubo": qubo_result,
            "qaoa_unconstrained": qaoa_unconstrained,
            "qaoa": qaoa_result,
        },
    }
