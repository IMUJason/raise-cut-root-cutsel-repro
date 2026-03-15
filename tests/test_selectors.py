import unittest

import numpy as np

from plan5.features import extract_single_cut_features
from plan5.interactions import extract_pairwise_interactions
from plan5.qubo import build_qubo
from plan5.schemas import CutCandidate, NodeContext
from plan5.selectors import select_qubo_classical, select_topk_linear


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cuts = [
            CutCandidate("c0", np.array([1.0, 0.0, 1.0, 0.0]), 0.1, "gomory", violation_raw=1.0, efficacy_raw=0.9),
            CutCandidate("c1", np.array([1.0, 0.0, 0.0, 1.0]), 0.2, "mir", violation_raw=0.8, efficacy_raw=0.7),
            CutCandidate("c2", np.array([0.0, 1.0, 1.0, 0.0]), 0.3, "gomory", violation_raw=0.7, efficacy_raw=0.6),
            CutCandidate("c3", np.array([0.0, 1.0, 0.0, 1.0]), 0.4, "mir", violation_raw=0.6, efficacy_raw=0.5),
        ]
        self.context = NodeContext(lp_cols=4, lp_rows=6, candidate_pool_size=4, objective_vector=np.array([1, 1, 1, 1]))

    def test_linear_and_qubo_select_exact_k(self) -> None:
        features = extract_single_cut_features(self.cuts, self.context)
        _, pairwise = extract_pairwise_interactions(self.cuts, features, top_r_neighbors=2)
        qubo = build_qubo(features, pairwise, budget_k=2, penalty_rho=1.0)
        linear_result = select_topk_linear(qubo)
        qubo_result = select_qubo_classical(qubo, strategy="greedy_local")
        self.assertEqual(len(linear_result.selected_indices), 2)
        self.assertEqual(len(qubo_result.selected_indices), 2)


if __name__ == "__main__":
    unittest.main()
