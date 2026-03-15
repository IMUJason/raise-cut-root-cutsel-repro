import unittest

import numpy as np

from plan5.qaoa_backend import select_qaoa_inspired, select_qaoa_unconstrained_baseline
from plan5.schemas import QUBOModel


class QAOABackendTests(unittest.TestCase):
    def test_qaoa_returns_exact_k_subset(self) -> None:
        linear = np.array([1.2, 1.1, 0.2, 0.1])
        quadratic = np.zeros((4, 4))
        qubo = QUBOModel(
            linear=linear,
            quadratic=quadratic,
            budget_k=2,
            penalty_rho=1.0,
            cut_ids=("c0", "c1", "c2", "c3"),
        )
        result = select_qaoa_inspired(qubo, depth_p=1, max_qubits=8)
        self.assertEqual(len(result.selected_indices), 2)
        self.assertEqual(result.selector_name, "P5-QAOA-C")
        self.assertEqual(result.metadata["feasible_probability"], 1.0)

    def test_constrained_qaoa_has_no_worse_feasible_mass_than_unconstrained(self) -> None:
        linear = np.array([0.8, 0.7, 0.1, 0.05])
        quadratic = np.zeros((4, 4))
        qubo = QUBOModel(
            linear=linear,
            quadratic=quadratic,
            budget_k=2,
            penalty_rho=1.0,
            cut_ids=("c0", "c1", "c2", "c3"),
        )
        constrained = select_qaoa_inspired(qubo, depth_p=1, max_qubits=8)
        unconstrained = select_qaoa_unconstrained_baseline(qubo, depth_p=1, max_qubits=8)
        self.assertGreaterEqual(constrained.metadata["feasible_probability"], unconstrained.metadata["feasible_probability"])


if __name__ == "__main__":
    unittest.main()
