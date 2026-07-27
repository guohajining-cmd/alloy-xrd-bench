from __future__ import annotations

import math
import unittest

from alloy_xrd_bench.scoring import (
    complete_capability,
    diagnostic_flags,
    normalized_sro_score,
    occupancy_score,
    reference_overall_score,
    require_complete_predictions,
    robustness_score,
    trustworthiness_score,
)


class ScoringProtocolTests(unittest.TestCase):
    def test_perfect_predictions_receive_full_scores(self):
        self.assertEqual(100.0, normalized_sro_score(model_mae=0.0, baseline_mae=0.2))
        self.assertEqual(
            100.0,
            occupancy_score(pair_accuracy=1.0, mrr=1.0, top_k_accuracy=1.0),
        )
        self.assertEqual(
            100.0,
            trustworthiness_score(
                ece=0.0,
                empirical_coverage=0.9,
                target_coverage=0.9,
                aurc=0.02,
                oracle_aurc=0.02,
                reference_aurc=0.20,
                identifiability_auroc=1.0,
            ),
        )

    def test_baseline_level_or_worse_sro_receives_zero(self):
        self.assertEqual(0.0, normalized_sro_score(model_mae=0.2, baseline_mae=0.2))
        self.assertEqual(0.0, normalized_sro_score(model_mae=0.3, baseline_mae=0.2))

    def test_missing_or_duplicate_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, "missing prediction"):
            require_complete_predictions(["a", "b"], ["a"])
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            require_complete_predictions(["a", "b"], ["a", "a", "b"])

    def test_robustness_requires_credible_clean_performance(self):
        self.assertEqual(
            0.0,
            robustness_score(
                clean_scores={"sro": 35.0, "occupancy": 35.0},
                perturbed_scores={"sro": 35.0, "occupancy": 35.0},
                clean_floor=40.0,
            ),
        )
        self.assertAlmostEqual(
            80.0,
            robustness_score(
                clean_scores={"sro": 80.0, "occupancy": 60.0},
                perturbed_scores={"sro": 64.0, "occupancy": 48.0},
                clean_floor=40.0,
            ),
        )

    def test_geometric_overall_exposes_a_failed_dimension(self):
        strong = reference_overall_score(
            {"sro": 80.0, "occupancy": 80.0, "robustness": 80.0, "ood": 80.0, "trust": 80.0}
        )
        failed = reference_overall_score(
            {"sro": 80.0, "occupancy": 5.0, "robustness": 80.0, "ood": 80.0, "trust": 80.0}
        )
        self.assertAlmostEqual(80.0, strong)
        self.assertAlmostEqual((80.0**4 * 5.0) ** (1.0 / 5.0), failed)
        self.assertLess(failed, 50.0)

    def test_complete_capability_requires_gate_and_every_core_dimension(self):
        scores = {"sro": 70.0, "occupancy": 65.0, "robustness": 60.0, "ood": 55.0, "trust": 50.0}
        self.assertTrue(complete_capability(0.92, scores))
        self.assertFalse(complete_capability(0.89, scores))
        scores["trust"] = 39.9
        self.assertFalse(complete_capability(0.95, scores))

    def test_diagnostics_explain_shortcut_and_identifiability_failures(self):
        flags = diagnostic_flags(
            family_macro_f1=0.95,
            xrd_occupancy_score=52.0,
            composition_only_occupancy_score=50.0,
            family_score=96.0,
            occupancy_score_value=35.0,
            clean_score=80.0,
            perturbed_score=50.0,
            id_score=75.0,
            ood_score=40.0,
            nonidentifiable_confidence=0.9,
        )
        self.assertIn("shortcut_risk", flags)
        self.assertIn("occupancy_blindness", flags)
        self.assertIn("perturbation_brittleness", flags)
        self.assertIn("ood_collapse", flags)
        self.assertIn("overconfidence_risk", flags)
        self.assertNotIn("basic_diffraction_fail", flags)
        self.assertTrue(math.isfinite(reference_overall_score({"a": 50.0})))


if __name__ == "__main__":
    unittest.main()
