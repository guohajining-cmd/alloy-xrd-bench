import unittest

import numpy as np

from alloy_xrd_bench.hea_validation_metrics import (
    family_classification_metrics,
    masked_parent_macro_sro,
    occupancy_ranking_metrics,
    prediction_completeness,
    robustness_retention,
)


class ValidationMetricTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]])
        self.mask = np.asarray([[True, True], [True, False], [True, True]])
        self.parents = ["p1", "p1", "p2"]
        self.shells = [1, 2]

    def test_oracle_sro_is_exact(self):
        result = masked_parent_macro_sro(
            self.truth, self.truth, self.mask, self.parents, self.shells
        )

        self.assertEqual(0.0, result["masked_parent_macro_mae"])
        self.assertEqual(0.0, result["rmse"])

    def test_masked_entries_do_not_affect_error(self):
        prediction = self.truth.copy()
        prediction[~self.mask] = 999.0

        result = masked_parent_macro_sro(
            self.truth, prediction, self.mask, self.parents, self.shells
        )

        self.assertEqual(0.0, result["masked_parent_macro_mae"])

    def test_parent_macro_weights_parents_equally(self):
        prediction = self.truth.copy()
        prediction[0, 0] = 2.0
        prediction[2, 0] = 3.5

        result = masked_parent_macro_sro(
            self.truth, prediction, self.mask, self.parents, self.shells
        )

        self.assertAlmostEqual(1.0, result["masked_parent_macro_mae"])

    def test_perfect_ranking(self):
        result = occupancy_ranking_metrics(
            [0.1, 0.9], [0.1, 0.9], ["g", "g"], identifiable=[True, True]
        )

        self.assertEqual(1.0, result["pair_accuracy"])
        self.assertEqual(1.0, result["mrr"])
        self.assertEqual(1.0, result["top3_accuracy"])

    def test_ranking_accepts_numpy_arrays(self):
        result = occupancy_ranking_metrics(
            np.asarray([0.1, 0.9]),
            np.asarray([0.1, 0.9]),
            ["g", "g"],
            identifiable=np.asarray([True, True]),
        )

        self.assertEqual(1.0, result["pair_accuracy"])

    def test_ranking_tie_scores_as_chance(self):
        result = occupancy_ranking_metrics(
            [0.1, 0.9], [0.5, 0.5], ["g", "g"], identifiable=[True, True]
        )

        self.assertEqual(0.5, result["pair_accuracy"])

    def test_low_identifiability_items_are_excluded_and_reported(self):
        result = occupancy_ranking_metrics(
            [0.1, 0.9, 0.2],
            [0.9, 0.1, 0.2],
            ["g", "g", "g"],
            identifiable=[True, True, False],
        )

        self.assertEqual(0.0, result["pair_accuracy"])
        self.assertAlmostEqual(2 / 3, result["identifiable_coverage"])

    def test_family_metrics_keep_unpredicted_classes(self):
        result = family_classification_metrics(
            ["fcc", "bcc", "hcp", "complex"],
            ["fcc", "fcc", "fcc", "fcc"],
            classes=["fcc", "bcc", "hcp", "complex"],
        )

        self.assertAlmostEqual(0.1, result["macro_f1"])
        self.assertAlmostEqual(0.25, result["balanced_accuracy"])

    def test_prediction_completeness_rejects_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "missing prediction sample IDs"):
            prediction_completeness(["a", "b"], ["a"])

    def test_robustness_retention_uses_matched_ids(self):
        result = robustness_retention(
            {"a": 1.0, "b": 0.5}, {"a": 0.8, "b": 0.25}
        )

        self.assertAlmostEqual(0.65, result["mean_retention"])
        self.assertEqual(2, result["matched_count"])


if __name__ == "__main__":
    unittest.main()
