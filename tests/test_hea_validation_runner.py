from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_validation import materialize_formal_submission


class ValidationRunnerTests(unittest.TestCase):
    def test_materializes_a_valid_submission_from_completed_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset = root / "dataset"
            (dataset / "manifests").mkdir(parents=True)
            pd.DataFrame([
                {
                    "structure_state_id": "state-1",
                    "xrd_sample_id": "xrd-1",
                    "parent_group_id": "parent-1",
                    "id_split": "test",
                }
            ]).to_parquet(dataset / "manifests" / "task_samples.parquet")
            prediction = root / "physics.parquet"
            pd.DataFrame([
                {
                    "model": "physics-transformer",
                    "seed": 2027,
                    "split_column": "id_split",
                    "xrd_sample_id": "xrd-1",
                    "parent_group_id": "parent-1",
                    "sro_predictions": [0.1, -0.2],
                }
            ]).to_parquet(prediction)

            submission = materialize_formal_submission(
                dataset, prediction, root / "predictions",
                model_name="physics-transformer", seed=2027, split_column="id_split",
            )

            frame = pd.read_parquet(submission / "sro.parquet")
            self.assertEqual(["state-1"], frame["sample_id"].tolist())
            self.assertEqual(["parent-1"], frame["parent_group_id"].tolist())
            self.assertTrue((submission / "metadata.json").is_file())

    def test_training_parser_exposes_explicit_smoke_sample_limit(self):
        from alloy_xrd_bench.hea_validation import build_parser

        args = build_parser().parse_args([
            "train",
            "--model", "physics-transformer",
            "--dataset", "dataset",
            "--config", "config.json",
            "--output", "output",
            "--max-samples-per-split", "8",
        ])

        self.assertEqual(8, args.max_samples_per_split)

    def test_runner_orders_required_gates_and_models(self):
        text = Path("scripts/workstation/run_hea_validation.sh").read_text()

        self.assertIn("set -euo pipefail", text)
        self.assertIn("flock -n", text)
        self.assertIn("run_snapshot.sha256", text)
        self.assertIn("sha256sum --check", text)
        self.assertLess(text.index(" audit "), text.index(" controls "))
        self.assertLess(text.index(" controls "), text.index("composition-ridge"))
        self.assertIn("peak-xgboost", text)
        self.assertNotIn("composition-extratrees", text)
        self.assertIn("resnet1d", text)
        self.assertIn("transformer", text)
        self.assertIn("xrd_models=(peak-xgboost resnet1d transformer)", text)
        self.assertIn("2027 2028 2029 2030 2031", text)
        self.assertIn("id_split composition_ood_split chemistry_ood_split sro_ood_split", text)
        self.assertIn("alloy_xrd_bench.evaluate score", text)
        self.assertIn("alloy_xrd_bench.evaluate compare", text)
        self.assertIn("alloy_xrd_bench.hea_benchmark_readiness", text)
        self.assertIn("alloy_xrd_bench.hea_prediction_audit", text)
        self.assertIn("alloy_xrd_bench.hea_release_package", text)
        self.assertIn("--output-dataset \"$output_dataset\"", text)
        self.assertIn("--minimum-seeds 5", text)
        self.assertIn(
            "--required-models peak-xgboost resnet1d transformer physics-transformer",
            text,
        )
        self.assertIn("--bootstrap-replicates 2000 --bootstrap-seed 2027", text)
        self.assertLess(text.index("hea_release_package"), text.index("observations-freeze"))
        self.assertLess(text.index("bootstrap_summary.json"), text.rindex("hea_release_package"))
        self.assertIn("CUDA_VISIBLE_DEVICES=0", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", text)
        self.assertIn("run_is_complete", text)
        self.assertIn("score_is_complete", text)
        self.assertIn(" report ", text)

        contract = json.loads(Path("configs/hea_benchmark_v1_1.json").read_text())
        self.assertEqual(
            [
                "composition_ridge",
                "peak_xgboost",
                "resnet1d",
                "transformer",
                "physics_transformer",
            ],
            list(contract["core_release_gates"]["required_models"]),
        )


if __name__ == "__main__":
    unittest.main()
