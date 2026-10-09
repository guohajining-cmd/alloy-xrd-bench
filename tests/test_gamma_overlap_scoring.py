from __future__ import annotations

import unittest


class GammaOverlapScoringTests(unittest.TestCase):
    def setUp(self):
        self.truth = [
            {
                "sample_id": "sample_1",
                "component_id": "truth_gamma",
                "parent_pair_id": "pair_1",
                "phase": "gamma",
                "h": 2,
                "k": 0,
                "l": 0,
                "center_deg": 44.30,
                "fwhm_deg": 0.18,
                "integrated_intensity": 6.0,
                "eta": 0.3,
            },
            {
                "sample_id": "sample_1",
                "component_id": "truth_gamma_prime",
                "parent_pair_id": "pair_1",
                "phase": "gamma_prime",
                "h": 2,
                "k": 0,
                "l": 0,
                "center_deg": 44.42,
                "fwhm_deg": 0.20,
                "integrated_intensity": 4.0,
                "eta": 0.4,
            },
        ]

    def test_perfect_predictions_reach_metric_optima_despite_order(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        predictions = [
            {**self.truth[1], "component_id": "pred_2", "confidence": 1.0},
            {**self.truth[0], "component_id": "pred_1", "confidence": 1.0},
        ]
        result = score_component_predictions(self.truth, predictions)

        self.assertEqual(1.0, result["component_precision"])
        self.assertEqual(1.0, result["component_recall"])
        self.assertEqual(1.0, result["component_f1"])
        self.assertEqual(1.0, result["phase_hkl_accuracy"])
        self.assertEqual(0.0, result["center_mae_deg"])
        self.assertEqual(0.0, result["fwhm_mae_deg"])
        self.assertEqual(0.0, result["area_mae"])
        self.assertEqual(1.0, result["overlap_region_f1"])
        self.assertAlmostEqual(0.0, result["forward_reconstruction_l1"])
        self.assertEqual(0.0, result["confidence_ece"])
        self.assertEqual(1.0, result["coverage_at_90_confidence"])
        self.assertEqual(0.0, result["aurc"])
        self.assertEqual(0.0, result["risk_at_80_coverage"])

    def test_wrong_phase_and_extra_component_are_penalized(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        predictions = [
            {**self.truth[0], "component_id": "pred_1", "phase": "gamma_prime", "confidence": 0.8},
            {**self.truth[1], "component_id": "pred_2", "confidence": 0.9},
            {**self.truth[1], "component_id": "pred_extra", "center_deg": 55.0, "confidence": 0.2},
        ]
        result = score_component_predictions(self.truth, predictions)

        self.assertLess(result["component_precision"], 1.0)
        self.assertLess(result["component_recall"], 1.0)
        self.assertLess(result["phase_hkl_accuracy"], 1.0)
        self.assertGreater(result["confidence_ece"], 0.0)

    def test_empty_predictions_score_zero_recall(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        result = score_component_predictions(self.truth, [])

        self.assertEqual(0.0, result["component_recall"])
        self.assertEqual(0.0, result["component_f1"])

    def test_prediction_on_expected_empty_window_counts_as_false_positive(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        prediction = {**self.truth[0], "sample_id": "empty_window", "component_id": "pred", "confidence": 0.5}
        result = score_component_predictions([], [prediction], expected_sample_ids=["empty_window"])

        self.assertEqual(0.0, result["component_precision"])
        self.assertEqual(0.0, result["component_f1"])

    def test_empty_window_uses_declared_parent_for_parent_macro_oracle(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        prediction = {**self.truth[0], "component_id": "pred", "confidence": 1.0}
        result = score_component_predictions(
            [self.truth[0]],
            [prediction],
            expected_sample_ids=["sample_1", "empty_window"],
            expected_parent_ids={"sample_1": "pair_1", "empty_window": "pair_1"},
        )

        self.assertEqual(1.0, result["component_f1"])
        self.assertEqual(1.0, result["count_accuracy"])

    def test_prediction_confidence_is_required(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        prediction = {**self.truth[0], "component_id": "pred"}
        with self.assertRaisesRegex(ValueError, "confidence"):
            score_component_predictions(self.truth, [prediction])

    def test_tied_confidence_risk_is_invariant_to_component_ids(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        truth = [self.truth[0]]
        for true_id, false_id in (("a", "b"), ("b", "a")):
            predictions = [dict(truth[0], component_id=true_id, confidence=1.0),
                           dict(truth[0], component_id=false_id, center_deg=55.0, confidence=1.0)]
            result = score_component_predictions(truth, predictions)
            self.assertAlmostEqual(2 / 3, result["component_f1"])
            self.assertAlmostEqual(0.5, result["aurc"])
            self.assertAlmostEqual(0.5, result["risk_at_80_coverage"])

    def test_risk_uses_expected_prefix_within_partial_confidence_tie(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        truth = [self.truth[0]]
        predictions = [dict(truth[0], component_id="a", confidence=0.5),
                       dict(truth[0], component_id="b", center_deg=55.0, confidence=0.5),
                       dict(truth[0], component_id="c", center_deg=56.0, confidence=1.0)]
        result = score_component_predictions(truth, predictions)
        # The high-confidence error comes first; expected risks are 1, 3/4, 2/3.
        self.assertAlmostEqual(29 / 36, result["aurc"])
        self.assertAlmostEqual(2 / 3, result["risk_at_80_coverage"])

    def test_duplicate_geometry_matching_does_not_depend_on_component_id(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        results = []
        for high_id, low_id in (("a", "b"), ("b", "a")):
            predictions = [dict(self.truth[0], component_id=high_id, confidence=1.0),
                           dict(self.truth[0], component_id=low_id, confidence=0.1)]
            results.append(score_component_predictions([self.truth[0]], predictions))
        self.assertEqual(results[0]["aurc"], results[1]["aurc"])
        self.assertEqual(results[0]["confidence_ece"], results[1]["confidence_ece"])

    def test_target_difficulty_merges_degenerate_reflections_and_ignores_context(self):
        from alloy_xrd_bench.gamma_overlap_scoring import target_overlap_difficulty

        components = [dict(self.truth[1], component_id="221", h=2, k=2, l=1, center_deg=71.25),
                      dict(self.truth[1], component_id="300", h=3, k=0, l=0, center_deg=71.25),
                      dict(self.truth[0], component_id="context", center_deg=71.26, score_eligible=False)]
        result = target_overlap_difficulty(components)
        self.assertEqual("single", result["difficulty"])
        self.assertIsNone(result["minimum_separation_ratio"])
        self.assertEqual(1, result["scored_component_count"])
        self.assertEqual("empty", target_overlap_difficulty([components[-1]])["difficulty"])
        self.assertEqual("strong", target_overlap_difficulty(self.truth)["difficulty"])

    def test_overlap_region_metric_penalizes_wrong_location(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        predictions = [
            {**self.truth[0], "component_id": "pred_1", "center_deg": 50.00, "confidence": 0.8},
            {**self.truth[1], "component_id": "pred_2", "center_deg": 50.05, "confidence": 0.9},
        ]
        result = score_component_predictions(self.truth, predictions)

        self.assertEqual(0.0, result["overlap_region_precision"])
        self.assertEqual(0.0, result["overlap_region_recall"])
        self.assertEqual(0.0, result["overlap_region_f1"])

    def test_degenerate_gamma_prime_reflections_merge_into_one_scoring_component(self):
        from alloy_xrd_bench.gamma_overlap_scoring import build_scoring_components

        raw = [
            {
                **self.truth[1], "component_id": "gp_221", "h": 2, "k": 2, "l": 1,
                "center_deg": 71.25, "integrated_intensity": 3.0,
            },
            {
                **self.truth[1], "component_id": "gp_300", "h": 3, "k": 0, "l": 0,
                "center_deg": 71.25, "integrated_intensity": 5.0,
            },
        ]

        grouped = build_scoring_components(raw)

        self.assertEqual(2, len(raw))
        self.assertEqual(1, len(grouped))
        self.assertEqual("gamma_prime:221|300", grouped[0]["hkl_group_id"])
        self.assertEqual([[2, 2, 1], [3, 0, 0]], grouped[0]["allowed_hkls"])
        self.assertTrue(grouped[0]["degenerate_group"])
        self.assertEqual(8.0, grouped[0]["integrated_intensity"])
        self.assertEqual(8.0, grouped[0]["merged_integrated_intensity"])

    def test_hkl_group_id_controls_matching_for_degenerate_group(self):
        from alloy_xrd_bench.gamma_overlap_scoring import score_component_predictions

        truth = [{
            **self.truth[1], "h": 2, "k": 2, "l": 1,
            "hkl_group_id": "gamma_prime:221|300",
        }]
        prediction = [{
            **truth[0], "component_id": "prediction", "h": 3, "k": 0, "l": 0,
            "confidence": 1.0,
        }]

        result = score_component_predictions(truth, prediction)

        self.assertEqual(1.0, result["component_f1"])
        self.assertEqual(1.0, result["phase_hkl_accuracy"])


if __name__ == "__main__":
    unittest.main()
