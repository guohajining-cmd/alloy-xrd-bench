from pathlib import Path
import json
import math
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.gamma_overlap_readiness import (
    _comparison_artifact_issues,
    _score_artifact_issues,
    _training_artifact_issues,
    assess_gamma_readiness,
)


class GammaOverlapReadinessTests(unittest.TestCase):
    def test_training_audit_requires_summary_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tempdir:
            run = Path(tempdir)
            self.assertEqual(
                ["missing_best_model", "missing_training_summary"],
                _training_artifact_issues(run),
            )
            (run / "best_model.pt").write_bytes(b"checkpoint")
            (run / "training_summary.json").write_text(json.dumps({"status": "pass"}))
            self.assertEqual([], _training_artifact_issues(run))

    def test_comparison_audit_requires_frozen_bootstrap_contract(self):
        with tempfile.TemporaryDirectory() as tempdir:
            comparison = Path(tempdir)
            (comparison / "comparison.json").write_text(json.dumps({
                "bootstrap_replicates": 2000,
                "bootstrap_seed": 2027,
                "five_point_gate_passed": True,
            }))
            pd.DataFrame({"replicate": range(2000), "improvement": [0.1] * 2000}).to_parquet(
                comparison / "bootstrap_samples.parquet", index=False
            )
            pd.DataFrame({
                "parent_pair_id": ["p"], "reference": [0.2], "candidate": [0.3],
            }).to_parquet(comparison / "paired_parent_metrics.parquet", index=False)

            self.assertEqual([], _comparison_artifact_issues(comparison))
            (comparison / "bootstrap_samples.parquet").unlink()
            self.assertIn("missing_bootstrap_samples", _comparison_artifact_issues(comparison))

    def test_score_artifact_audit_requires_metrics_and_frozen_slices(self):
        with tempfile.TemporaryDirectory() as tempdir:
            score = Path(tempdir)
            metrics = {
                key: 0.5 for key in (
                    "component_f1", "phase_hkl_accuracy", "count_accuracy",
                    "center_mae_deg", "fwhm_mae_deg", "area_mae",
                    "forward_reconstruction_l1", "confidence_ece",
                    "coverage_at_90_confidence", "aurc", "risk_at_80_coverage",
                )
            }
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            pd.DataFrame({"parent_pair_id": ["p"], "component_f1": [0.5]}).to_parquet(
                score / "per_parent_pair.parquet", index=False
            )
            pd.DataFrame({
                "instrument_slice": ["instrument_id", "instrument_ood"],
                "component_f1": [0.6, 0.4],
            }).to_parquet(score / "per_instrument_slice.parquet", index=False)
            pd.DataFrame({
                "difficulty": ["single", "resolved", "partial", "strong", "severe"],
                "component_f1": [0.5] * 5,
            }).to_parquet(score / "per_difficulty.parquet", index=False)

            self.assertEqual([], _score_artifact_issues(score, task="window"))
            metrics["component_f1"] = math.nan
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            self.assertIn("nonfinite_metrics:component_f1", _score_artifact_issues(score, task="window"))
            metrics["component_f1"] = 0.5
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            pd.DataFrame({
                "instrument_slice": ["instrument_id", "instrument_ood"],
                "component_f1": [math.nan, 0.4],
            }).to_parquet(score / "per_instrument_slice.parquet", index=False)
            self.assertIn(
                "nonfinite_instrument_component_f1",
                _score_artifact_issues(score, task="window"),
            )
            (score / "per_instrument_slice.parquet").unlink()
            self.assertIn("missing_per_instrument_slice", _score_artifact_issues(score, task="window"))

    def test_score_artifact_allows_undefined_conditional_metrics_for_abstention(self):
        with tempfile.TemporaryDirectory() as tempdir:
            score = Path(tempdir)
            metrics = {
                "component_f1": 0.0,
                "phase_hkl_accuracy": 0.0,
                "count_accuracy": 0.0,
                "center_mae_deg": None,
                "fwhm_mae_deg": None,
                "area_mae": None,
                "forward_reconstruction_l1": 1.0,
                "confidence_ece": None,
                "coverage_at_90_confidence": 0.0,
                "aurc": None,
                "risk_at_80_coverage": None,
            }
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            (score / "submission_validation.json").write_text(json.dumps({
                "status": "pass", "component_count": 0,
            }))
            pd.DataFrame({"parent_pair_id": ["p"], "component_f1": [0.0]}).to_parquet(
                score / "per_parent_pair.parquet", index=False
            )
            pd.DataFrame({
                "instrument_slice": ["instrument_id", "instrument_ood"],
                "component_f1": [0.0, 0.0],
            }).to_parquet(score / "per_instrument_slice.parquet", index=False)
            pd.DataFrame({
                "difficulty": ["single", "resolved", "partial", "strong", "severe"],
                "component_f1": [0.0] * 5,
            }).to_parquet(score / "per_difficulty.parquet", index=False)

            self.assertEqual([], _score_artifact_issues(score, task="window"))

            metrics["center_mae_deg"] = 0.1
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            self.assertIn(
                "partially_undefined_matched_metrics",
                _score_artifact_issues(score, task="window"),
            )
            metrics["center_mae_deg"] = None

            metrics["confidence_ece"] = 0.0
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            self.assertIn(
                "partially_undefined_calibration_metrics",
                _score_artifact_issues(score, task="window"),
            )
            metrics["confidence_ece"] = None

            metrics["component_f1"] = 0.5
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            self.assertIn(
                "undefined_matched_metrics_without_abstention",
                _score_artifact_issues(score, task="window"),
            )

            metrics["component_f1"] = 0.0
            (score / "scorecard.json").write_text(json.dumps({"metrics": metrics}))
            (score / "submission_validation.json").write_text(json.dumps({
                "status": "pass", "component_count": 3,
            }))
            self.assertIn(
                "undefined_calibration_with_predictions",
                _score_artifact_issues(score, task="window"),
            )

    def test_incomplete_artifacts_are_reported_not_treated_as_ready(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            release = root / "release"
            run = root / "run"
            release.mkdir()
            run.mkdir()

            report = assess_gamma_readiness(release, run, root / "readiness.json")

            self.assertFalse(report["gamma_benchmark_ready"])
            self.assertIn("RELEASE_AUDIT_FAILED", report["failure_codes"])
            self.assertTrue(report["missing_artifacts"])


if __name__ == "__main__":
    unittest.main()
