import unittest

import numpy as np

from plan5.scip_cutsel import (
    compute_round_aware_dense_cap,
    compute_family_rarity,
    greedy_adaptive_select,
    greedy_regime_select,
    infer_candidate_regime,
    normalize_cut_family_name,
    reorder_dense_candidates_with_quota,
    route_probe_to_mode,
    route_probe_to_mode_with_policy,
)


class SCIPCutselUtilsTests(unittest.TestCase):
    def test_normalize_cut_family_name(self) -> None:
        self.assertEqual(normalize_cut_family_name("gom8_x55"), "gom")
        self.assertEqual(normalize_cut_family_name("flowcover12_row7"), "flowcover")
        self.assertEqual(normalize_cut_family_name("CNS..342"), "cns")
        self.assertEqual(normalize_cut_family_name("R0110_mcseq0"), "r")

    def test_compute_family_rarity_prefers_minor_families(self) -> None:
        rarity = compute_family_rarity(["gom", "gom", "gom", "cmir", "flowcover"])
        self.assertLess(rarity["gom"], rarity["cmir"])
        self.assertLess(rarity["gom"], rarity["flowcover"])

    def test_greedy_adaptive_select_stops_before_capacity(self) -> None:
        linear_scores = np.array([1.0, 0.7, 0.28, 0.12, 0.05])
        pair_matrix = np.zeros((5, 5))
        selected = greedy_adaptive_select(
            linear_scores=linear_scores,
            pair_matrix=pair_matrix,
            nselected_cap=5,
            min_selected=2,
            relative_marginal_threshold=0.3,
            absolute_marginal_threshold=0.02,
            max_nonpositive_streak=1,
        )
        self.assertEqual(selected, (0, 1))

    def test_greedy_adaptive_select_respects_minimum(self) -> None:
        linear_scores = np.array([0.4, 0.03, 0.02, 0.01])
        pair_matrix = np.zeros((4, 4))
        selected = greedy_adaptive_select(
            linear_scores=linear_scores,
            pair_matrix=pair_matrix,
            nselected_cap=4,
            min_selected=3,
            relative_marginal_threshold=0.9,
            absolute_marginal_threshold=0.05,
            max_nonpositive_streak=0,
        )
        self.assertEqual(len(selected), 3)

    def test_infer_candidate_regime_detects_dominant_family(self) -> None:
        feature_records = [
            {"cut_family": "gom", "efficacy": 0.9},
            {"cut_family": "gom", "efficacy": 0.8},
            {"cut_family": "gom", "efficacy": 0.7},
            {"cut_family": "cmir", "efficacy": 0.2},
        ]
        regime = infer_candidate_regime(feature_records, probe_cap=4, dominance_threshold=0.55)
        self.assertEqual(regime.regime_name, "gom")
        self.assertEqual(regime.dominant_family, "gom")
        self.assertGreater(regime.dominant_share, 0.55)

    def test_infer_candidate_regime_detects_mixed_pool(self) -> None:
        feature_records = [
            {"cut_family": "gom", "efficacy": 0.6},
            {"cut_family": "cmir", "efficacy": 0.5},
            {"cut_family": "implbd", "efficacy": 0.4},
            {"cut_family": "clique", "efficacy": 0.3},
        ]
        regime = infer_candidate_regime(feature_records, probe_cap=4, dominance_threshold=0.55)
        self.assertEqual(regime.regime_name, "mixed")

    def test_greedy_regime_select_respects_budget_floor_and_quota(self) -> None:
        linear_scores = np.array([0.90, 0.85, 0.40, 0.39, 0.38])
        pair_matrix = np.zeros((5, 5))
        selected = greedy_regime_select(
            linear_scores=linear_scores,
            pair_matrix=pair_matrix,
            families=["gom", "gom", "cmir", "cmir", "cmir"],
            nselected_cap=5,
            budget_floor=3,
            dominant_family="gom",
            dominant_quota=2,
            quota_bonus=0.1,
            relative_marginal_threshold=0.6,
            absolute_marginal_threshold=0.05,
            max_nonpositive_streak=0,
        )
        self.assertEqual(len(selected), 3)
        self.assertGreaterEqual(sum(1 for idx in selected if idx in {0, 1}), 2)

    def test_compute_round_aware_dense_cap_decays_with_rounds_and_history(self) -> None:
        early = compute_round_aware_dense_cap(
            regime_name="mixed",
            round_index=1,
            max_cap=50,
            maxnselectedcuts=50,
            candidate_count=120,
            dominant_share=0.82,
            dominant_history=0,
            dominant_streak=0,
            base_caps={"mixed": 48, "default": 40},
            floor_caps={"mixed": 20, "default": 16},
            decay_start_round=2,
            per_round_decay=3,
            strong_dominance_threshold=0.75,
            history_penalty_weight=0.03,
            max_history_penalty=8,
            streak_penalty=2,
        )
        late = compute_round_aware_dense_cap(
            regime_name="mixed",
            round_index=6,
            max_cap=50,
            maxnselectedcuts=50,
            candidate_count=120,
            dominant_share=0.82,
            dominant_history=120,
            dominant_streak=3,
            base_caps={"mixed": 48, "default": 40},
            floor_caps={"mixed": 20, "default": 16},
            decay_start_round=2,
            per_round_decay=3,
            strong_dominance_threshold=0.75,
            history_penalty_weight=0.03,
            max_history_penalty=8,
            streak_penalty=2,
        )
        self.assertEqual(early, 48)
        self.assertLess(late, early)
        self.assertGreaterEqual(late, 20)

    def test_reorder_dense_candidates_with_quota_defers_excess_dominant_family(self) -> None:
        reordered = reorder_dense_candidates_with_quota(
            ranked_indices=[0, 1, 2, 3, 4],
            families=["gom", "gom", "gom", "cmir", "clique"],
            dominant_family="gom",
            dominant_quota=2,
        )
        self.assertEqual(reordered[:4], [0, 1, 3, 4])
        self.assertEqual(reordered[4], 2)

    def test_route_probe_to_mode_uses_low_obj_large_pool_ensemble_rule(self) -> None:
        mode = route_probe_to_mode(
            {
                "candidate_count": 700,
                "mean_obj_parallelism": 0.0005,
                "regime_name": "mixed",
                "dominant_share": 0.40,
            }
        )
        self.assertEqual(mode, "scip_ensemble")

    def test_route_probe_to_mode_keeps_high_obj_gom_on_raise_cut(self) -> None:
        mode = route_probe_to_mode(
            {
                "candidate_count": 200,
                "mean_obj_parallelism": 0.12,
                "regime_name": "gom",
                "dominant_share": 0.95,
            }
        )
        self.assertEqual(mode, "raise_cut")

    def test_route_probe_to_mode_with_policy_rc_widens_safe_ensemble_region(self) -> None:
        mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 280,
                "mean_obj_parallelism": 0.018,
                "regime_name": "mixed",
                "dominant_share": 0.42,
            },
            policy_name="raise_portfolio_rc",
        )
        self.assertEqual(mode, "scip_ensemble")

    def test_route_probe_to_mode_with_policy_rc_abstains_on_clique_branch(self) -> None:
        mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 9,
                "mean_obj_parallelism": 0.13,
                "regime_name": "clique",
                "dominant_share": 0.82,
            },
            policy_name="raise_portfolio_rc",
        )
        self.assertEqual(mode, "raise_cut")

    def test_route_probe_to_mode_with_policy_ud_uses_dynamic_on_large_diffuse_mixed_pool(self) -> None:
        mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 757,
                "mean_obj_parallelism": 0.069,
                "regime_name": "mixed",
                "dominant_family": "implbd",
                "dominant_share": 0.45,
                "family_count": 4,
            },
            policy_name="raise_portfolio_ud",
        )
        self.assertEqual(mode, "scip_dynamic")

    def test_route_probe_to_mode_with_policy_ud_abstains_on_cmir_dominant_mixed_pool(self) -> None:
        mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 387,
                "mean_obj_parallelism": 0.055,
                "regime_name": "mixed",
                "dominant_family": "cmir",
                "dominant_share": 0.49,
                "family_count": 4,
            },
            policy_name="raise_portfolio_ud",
        )
        self.assertEqual(mode, "raise_cut")

    def test_route_probe_to_mode_with_policy_ud_abstains_on_small_or_concentrated_pool(self) -> None:
        small_mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 200,
                "mean_obj_parallelism": 0.01,
                "regime_name": "mixed",
                "dominant_family": "gom",
                "dominant_share": 0.40,
                "family_count": 5,
            },
            policy_name="raise_portfolio_ud",
        )
        concentrated_mode = route_probe_to_mode_with_policy(
            {
                "candidate_count": 500,
                "mean_obj_parallelism": 0.01,
                "regime_name": "mixed",
                "dominant_family": "gom",
                "dominant_share": 0.92,
                "family_count": 3,
            },
            policy_name="raise_portfolio_ud",
        )
        self.assertEqual(small_mode, "raise_cut")
        self.assertEqual(concentrated_mode, "raise_cut")


if __name__ == "__main__":
    unittest.main()
