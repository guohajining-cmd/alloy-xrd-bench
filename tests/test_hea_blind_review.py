import unittest

import pandas as pd

from alloy_xrd_bench.hea_family_adjudication import (
    AdjudicationValidationError,
    build_double_blind_review,
    finalize_double_blind_review,
    select_stratified_review_sample,
)


class HEADoubleBlindReviewTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for family in ["fcc_like", "bcc_like", "complex", "hcp_like"]:
            for index in range(60):
                rows.append(
                    {
                        "structure_state_id": f"{family}-{index}",
                        "structure_family": family,
                        "classification_confidence": 0.5 + index / 200,
                        "template_mapping_rms": index / 100,
                        "formula": "AlCoCrNi",
                        "chemical_system": "Al-Co-Cr-Ni",
                    }
                )
        self.structures = pd.DataFrame(rows)

    def test_selection_is_deterministic_and_covers_all_strata(self):
        first = select_stratified_review_sample(self.structures, sample_size=200, seed=2027)
        second = select_stratified_review_sample(self.structures, sample_size=200, seed=2027)
        self.assertEqual(first["structure_state_id"].tolist(), second["structure_state_id"].tolist())
        self.assertEqual(200, len(first))
        self.assertEqual(4, first["structure_family"].nunique())
        self.assertGreaterEqual(first.groupby("structure_family").size().min(), 40)

    def test_export_contains_two_blinded_rows_per_structure(self):
        selected = select_stratified_review_sample(self.structures, sample_size=20, seed=2027)
        review = build_double_blind_review(selected, reviewer_ids=["expert-a", "expert-b"])
        self.assertEqual(40, len(review))
        self.assertTrue(review.groupby("structure_state_id")["reviewer_id"].nunique().eq(2).all())
        self.assertNotIn("structure_family", review.columns)
        self.assertTrue(review["expert_label"].isna().all())

    def test_finalize_requires_two_independent_reviews_and_resolved_disagreements(self):
        selected = select_stratified_review_sample(self.structures, sample_size=20, seed=2027)
        review = build_double_blind_review(selected, reviewer_ids=["expert-a", "expert-b"])
        truth = selected.set_index("structure_state_id")["structure_family"]
        review["expert_label"] = review["structure_state_id"].map(truth)
        review["review_date"] = "2026-07-25"
        review["notes"] = "independent review"
        result = finalize_double_blind_review(selected, review)
        self.assertEqual(0, result["summary"]["pending_disagreement_count"])
        self.assertEqual(1.0, result["summary"]["cohen_kappa"])
        self.assertEqual(1.0, result["summary"]["overall_accuracy"])
        self.assertTrue(result["summary"]["review_gate_passed"])

        review.loc[review.index[0], "expert_label"] = "complex"
        with self.assertRaisesRegex(AdjudicationValidationError, "unresolved disagreement"):
            finalize_double_blind_review(selected, review)
        resolution = pd.DataFrame(
            [{"structure_state_id": review.loc[0, "structure_state_id"], "consensus_label": truth.loc[review.loc[0, "structure_state_id"]], "adjudicator_id": "expert-c", "adjudication_date": "2026-07-25", "notes": "resolved"}]
        )
        resolved = finalize_double_blind_review(selected, review, resolutions=resolution)
        self.assertEqual(0, resolved["summary"]["pending_disagreement_count"])

    def test_same_reviewer_cannot_fill_both_rows(self):
        selected = select_stratified_review_sample(self.structures, sample_size=4, seed=2027)
        review = build_double_blind_review(selected, reviewer_ids=["expert-a", "expert-b"])
        review["reviewer_id"] = "expert-a"
        review["expert_label"] = "fcc_like"
        review["review_date"] = "2026-07-25"
        with self.assertRaisesRegex(AdjudicationValidationError, "independent reviewers"):
            finalize_double_blind_review(selected, review)


if __name__ == "__main__":
    unittest.main()
