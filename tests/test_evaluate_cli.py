from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from alloy_xrd_bench.evaluate import (
    compare_scores,
    expected_calibration_error,
    paired_parent_bootstrap,
    selective_risk_metrics,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_ece_is_zero_for_empirically_calibrated_bins(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(0.0, expected_calibration_error(labels, probabilities, bins=2))

    def test_selective_risk_prefers_confident_correct_predictions(self):
        result = selective_risk_metrics(
            np.asarray([0, 1, 1, 0]),
            np.asarray([0.1, 0.9, 0.6, 0.4]),
        )
        self.assertLessEqual(result["aurc"], result["reference_aurc"])
        self.assertGreaterEqual(result["identifiability_auroc"], 0.5)

    def test_parent_bootstrap_is_deterministic_and_positive(self):
        reference = {"p1": 0.5, "p2": 0.4, "p3": 0.6}
        candidate = {"p1": 0.3, "p2": 0.2, "p3": 0.4}
        first = paired_parent_bootstrap(reference, candidate, replicates=200, seed=2027)
        second = paired_parent_bootstrap(reference, candidate, replicates=200, seed=2027)
        self.assertEqual(first, second)
        self.assertGreater(first["skill_points"], 5.0)
        self.assertGreater(first["ci95_low"], 0.0)
        self.assertAlmostEqual(0.5, first["reference_mae"])
        self.assertAlmostEqual(0.3, first["candidate_mae"])
        self.assertAlmostEqual(0.2, first["mae_difference"])
        self.assertLess(first["mae_difference_ci95_low"], first["mae_difference_ci95_high"])

    def test_compare_aggregates_multiple_seed_score_directories_by_parent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for model, values in {
                "reference": [(0.5, 0.6), (0.4, 0.5)],
                "candidate": [(0.2, 0.3), (0.3, 0.2)],
            }.items():
                for seed, errors in zip([2027, 2028], values):
                    path = root / model / str(seed) / "id_split_test"
                    path.mkdir(parents=True)
                    pd.DataFrame(
                        {
                            "task": ["sro_parent", "sro_parent"],
                            "group_id": ["p1", "p2"],
                            "metric": ["mae", "mae"],
                            "value": list(errors),
                        }
                    ).to_parquet(path / "metrics_by_slice.parquet")

            result = compare_scores(
                root / "reference", root / "candidate", replicates=100, seed=2027
            )

            self.assertEqual(2, result["reference_run_count"])
            self.assertEqual(2, result["candidate_run_count"])
            self.assertEqual("pass", result["status"])
            self.assertEqual(100, result["replicates"])
            self.assertEqual(2027, result["seed"])
            self.assertTrue(result["passes_five_point_gate"])


class EvaluationCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "dataset"
        self.predictions = self.root / "predictions" / "oracle" / "2027"
        (self.dataset / "manifests").mkdir(parents=True)
        self.predictions.mkdir(parents=True)
        targets = json.loads(Path("configs/hea_validation_v1.json").read_text())["sro_targets"]
        shell1 = {"Al->Co": 0.2}
        shell2 = {"Co->Al": -0.1}
        rows = []
        for index, family in enumerate(["fcc_like", "bcc_like"]):
            rows.append(
                {
                    "structure_state_id": f"state-{index}",
                    "parent_group_id": f"parent-{index}",
                    "structure_family": family,
                    "sro_shell1_json": json.dumps(shell1),
                    "sro_shell2_json": json.dumps(shell2),
                    "id_split": "test",
                    "composition_ood_split": "test",
                    "chemistry_ood_split": "test",
                    "sro_ood_split": "test",
                }
            )
        pd.DataFrame(rows).to_parquet(self.dataset / "manifests" / "task_samples.parquet")
        pairs = pd.DataFrame(
            [
                {
                    "pair_id": "pair-0-a",
                    "candidate_structure_state_id": "candidate-a",
                    "parent_group_id": "parent-0",
                    "sro_distance": 0.2,
                    "id_split": "test",
                    "noise_identifiable": True,
                },
                {
                    "pair_id": "pair-0-b",
                    "candidate_structure_state_id": "candidate-b",
                    "parent_group_id": "parent-0",
                    "sro_distance": 0.8,
                    "id_split": "test",
                    "noise_identifiable": True,
                },
            ]
        )
        pairs.to_parquet(self.dataset / "manifests" / "occupancy_pairs_calibrated.parquet")
        metadata = {
            "schema_version": "sitexrd-predictions-v1",
            "model_id": "oracle",
            "seed": 2027,
            "split_column": "id_split",
            "split_value": "test",
            "tasks": ["family", "sro", "occupancy", "uncertainty"],
        }
        (self.predictions / "metadata.json").write_text(json.dumps(metadata))
        pd.DataFrame(
            [
                {"sample_id": "state-0", "parent_group_id": "parent-0", "model_id": "oracle", "seed": 2027, "predicted_label": "fcc", "probabilities_json": json.dumps({"fcc": 1.0, "bcc": 0.0, "hcp": 0.0, "complex": 0.0})},
                {"sample_id": "state-1", "parent_group_id": "parent-1", "model_id": "oracle", "seed": 2027, "predicted_label": "bcc", "probabilities_json": json.dumps({"fcc": 0.0, "bcc": 1.0, "hcp": 0.0, "complex": 0.0})},
            ]
        ).to_parquet(self.predictions / "family.parquet")
        values = np.zeros(len(targets), dtype=float)
        lower = np.full(len(targets), -1.0)
        upper = np.full(len(targets), 1.0)
        values[targets.index("shell1:Al->Co")] = 0.2
        values[targets.index("shell2:Co->Al")] = -0.1
        pd.DataFrame(
            [
                {"sample_id": f"state-{i}", "parent_group_id": f"parent-{i}", "model_id": "oracle", "seed": 2027, "sro_predictions_json": json.dumps(values.tolist()), "sro_lower_json": json.dumps(lower.tolist()), "sro_upper_json": json.dumps(upper.tolist())}
                for i in range(2)
            ]
        ).to_parquet(self.predictions / "sro.parquet")
        pd.DataFrame(
            [
                {"sample_id": "candidate-a", "parent_group_id": "parent-0", "model_id": "oracle", "seed": 2027, "predicted_score": 0.2, "confidence": 0.9},
                {"sample_id": "candidate-b", "parent_group_id": "parent-0", "model_id": "oracle", "seed": 2027, "predicted_score": 0.8, "confidence": 0.9},
            ]
        ).to_parquet(self.predictions / "occupancy.parquet")
        pd.DataFrame(
            [
                {"sample_id": "pair-0-a", "parent_group_id": "parent-0", "model_id": "oracle", "seed": 2027, "identifiable_probability": 1.0, "abstain": False},
                {"sample_id": "pair-0-b", "parent_group_id": "parent-0", "model_id": "oracle", "seed": 2027, "identifiable_probability": 1.0, "abstain": False},
            ]
        ).to_parquet(self.predictions / "uncertainty.parquet")

    def tearDown(self):
        self.tempdir.cleanup()

    def run_cli(self, *args: str):
        return subprocess.run(
            [sys.executable, "-m", "alloy_xrd_bench.evaluate", *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_validate_and_score_oracle_submission(self):
        validated = self.run_cli(
            "validate", "--dataset", str(self.dataset), "--predictions", str(self.predictions)
        )
        self.assertEqual(0, validated.returncode, validated.stderr)
        self.assertEqual("pass", json.loads(validated.stdout)["status"])
        output = self.root / "scores"
        scored = self.run_cli(
            "score", "--dataset", str(self.dataset), "--predictions", str(self.predictions), "--output", str(output), "--bootstrap-replicates", "50"
        )
        self.assertEqual(0, scored.returncode, scored.stderr)
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(1.0, metrics["family"]["macro_f1"])
        self.assertEqual(0.0, metrics["sro"]["masked_parent_macro_mae"])
        self.assertEqual(1.0, metrics["occupancy"]["pair_accuracy"])
        for name in ["metrics_by_slice.parquet", "bootstrap_samples.parquet", "diagnostic_flags.json", "scorecard.md"]:
            self.assertTrue((output / name).exists(), name)

    def test_validate_rejects_missing_and_nonfinite_predictions(self):
        frame = pd.read_parquet(self.predictions / "sro.parquet").iloc[:1]
        frame.to_parquet(self.predictions / "sro.parquet", index=False)
        result = self.run_cli(
            "validate", "--dataset", str(self.dataset), "--predictions", str(self.predictions)
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing prediction", result.stderr)

    def test_sro_only_ood_does_not_require_occupancy_split_columns(self):
        metadata_path = self.predictions / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["split_column"] = "composition_ood_split"
        metadata["tasks"] = ["sro"]
        metadata_path.write_text(json.dumps(metadata))

        result = self.run_cli(
            "validate", "--dataset", str(self.dataset), "--predictions", str(self.predictions)
        )

        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
