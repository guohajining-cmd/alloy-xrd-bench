import unittest

import pandas as pd

from alloy_xrd_bench.hea_family_adjudication import (
    ADJUDICATION_COLUMNS,
    AdjudicationValidationError,
    apply_adjudications,
    build_adjudication_export,
    validate_adjudications,
)


class HEAFamilyAdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.queue = pd.DataFrame(
            [
                {
                    "structure_state_id": "state-1",
                    "current_label": "fcc_like",
                    "proposed_label": "bcc_like",
                    "review_pass": False,
                },
                {
                    "structure_state_id": "state-2",
                    "current_label": "bcc_like",
                    "proposed_label": "complex",
                    "review_pass": False,
                },
            ]
        )
        self.evidence = pd.DataFrame(
            [
                {
                    "structure_state_id": "state-1",
                    "ptm_evidence": "bcc=0.91",
                    "cna_evidence": "bcc=0.88",
                },
                {
                    "structure_state_id": "state-2",
                    "ptm_evidence": "other=0.72",
                    "cna_evidence": "other=0.69",
                },
            ]
        )

    def completed_export(self):
        frame = build_adjudication_export(self.queue, self.evidence)
        frame["reviewer_id"] = ["expert-a", "expert-b"]
        frame["review_date"] = ["2026-07-25", "2026-07-25"]
        frame["decision"] = ["accept_proposed", "exclude"]
        frame["expert_label"] = ["bcc_like", pd.NA]
        frame["notes"] = ["Reviewed diffraction and structure", "Ambiguous"]
        return frame

    def test_export_has_frozen_columns_and_keeps_decisions_pending(self):
        exported = build_adjudication_export(self.queue, self.evidence)
        self.assertEqual(list(ADJUDICATION_COLUMNS), list(exported.columns))
        self.assertEqual(["pending", "pending"], exported["decision"].tolist())
        self.assertTrue(exported["reviewer_id"].isna().all())
        self.assertTrue(exported["expert_label"].isna().all())
        self.assertEqual("bcc=0.91", exported.loc[0, "ptm_evidence"])

    def test_valid_completed_adjudications_are_accepted(self):
        validated = validate_adjudications(self.completed_export())
        self.assertEqual(2, len(validated))

    def test_empty_reviewer_is_rejected(self):
        frame = self.completed_export()
        frame.loc[0, "reviewer_id"] = "  "
        with self.assertRaisesRegex(AdjudicationValidationError, "reviewer_id"):
            validate_adjudications(frame)

    def test_invalid_label_is_rejected(self):
        frame = self.completed_export()
        frame.loc[0, "expert_label"] = "made_up_phase"
        with self.assertRaisesRegex(AdjudicationValidationError, "label"):
            validate_adjudications(frame)

    def test_duplicate_structure_id_is_rejected(self):
        frame = self.completed_export()
        frame.loc[1, "structure_state_id"] = "state-1"
        with self.assertRaisesRegex(AdjudicationValidationError, "duplicate"):
            validate_adjudications(frame)

    def test_pending_decision_is_rejected(self):
        frame = self.completed_export()
        frame.loc[0, "decision"] = "pending"
        with self.assertRaisesRegex(AdjudicationValidationError, "pending"):
            validate_adjudications(frame)

    def test_proposed_label_is_not_automatically_a_human_label(self):
        frame = self.completed_export()
        frame.loc[0, "expert_label"] = pd.NA
        with self.assertRaisesRegex(AdjudicationValidationError, "expert_label"):
            validate_adjudications(frame)

    def test_apply_updates_a_copy_and_preserves_provenance(self):
        manifest = pd.DataFrame(
            [
                {"structure_state_id": "state-1", "family_label": "fcc_like"},
                {"structure_state_id": "state-2", "family_label": "bcc_like"},
            ]
        )
        updated, provenance = apply_adjudications(
            manifest, self.completed_export(), label_column="family_label"
        )
        self.assertEqual("fcc_like", manifest.loc[0, "family_label"])
        self.assertEqual("bcc_like", updated.loc[0, "family_label"])
        self.assertEqual(1, len(updated))
        self.assertEqual(2, len(provenance))


if __name__ == "__main__":
    unittest.main()
