import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd


ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"]


class ValidationCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "manifests").mkdir()
        (self.root / "grids").mkdir()
        rows = []
        features = []
        for index, split in enumerate(
            ["train", "validation", "test", "train", "validation", "test"]
        ):
            state_id = f"state-{index}"
            rows.append(
                {
                    "structure_state_id": state_id,
                    "parent_group_id": f"parent-{index}",
                    "structure_family": "fcc_like" if index == 0 else "bcc_like",
                    "xrd_sample_id": f"sample-{index}",
                    "grid_shard_path": "grids/grid.h5",
                    "grid_row_index": index,
                    "sro_shell1_json": json.dumps({"Al->Co": float(index)}),
                    "sro_shell2_json": json.dumps({"Co->Al": float(-index)}),
                    "id_split": split,
                    "composition_ood_split": split,
                    "chemistry_ood_split": split,
                    "sro_ood_split": split,
                }
            )
            feature = {
                "structure_state_id": state_id,
                "parent_group_id": f"parent-{index}",
            }
            feature.update({f"fraction_{element}": 0.125 for element in ELEMENTS})
            features.append(feature)
        pd.DataFrame(rows).to_parquet(self.root / "manifests" / "task_samples.parquet")
        pd.DataFrame(features).to_parquet(
            self.root / "manifests" / "composition_only_features.parquet"
        )
        two_theta = np.linspace(10.0, 90.0, 4001, dtype=np.float32)
        intensity = np.stack(
            [
                np.exp(-0.5 * ((two_theta - (25.0 + 5.0 * index)) / 0.1) ** 2)
                for index in range(6)
            ]
        ).astype(np.float32)
        with h5py.File(self.root / "grids" / "grid.h5", "w") as handle:
            handle.create_dataset("two_theta", data=two_theta)
            handle.create_dataset("intensity", data=intensity)
            handle.create_dataset(
                "xrd_sample_id", data=np.asarray([f"sample-{i}".encode() for i in range(6)])
            )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_audit_reports_zero_parent_leakage(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "audit",
                "--dataset",
                str(self.root),
                "--config",
                "configs/hea_validation_v1.json",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(0, payload["parent_group_leakage_count"])
        self.assertEqual(6, payload["row_count"])
        self.assertEqual("pass", payload["status"])
        self.assertEqual(6, payload["canonical_count"])
        self.assertEqual("pre-release", payload["result_qualification"])

    def test_controls_separate_oracle_from_shuffled_labels(self):
        pd.DataFrame(
            {
                "parent_group_id": ["occupancy-parent", "occupancy-parent"],
                "sro_distance": [0.1, 0.2],
                "low_identifiability": [False, False],
                "id_split": ["test", "test"],
            }
        ).to_parquet(self.root / "manifests" / "occupancy_pairs.parquet", index=False)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "controls",
                "--dataset",
                str(self.root),
                "--config",
                "configs/hea_validation_v1.json",
                "--split",
                "all",
                "--seed",
                "2027",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(0.0, payload["sro"]["oracle"]["masked_parent_macro_mae"])
        self.assertGreater(
            payload["sro"]["shuffled"]["masked_parent_macro_mae"], 0.0
        )
        self.assertNotIn("occupancy_ranking", payload)

    def test_train_composition_ridge_writes_complete_predictions(self):
        output = self.root / "run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "train",
                "--model",
                "composition-ridge",
                "--dataset",
                str(self.root),
                "--config",
                "configs/hea_validation_v1.json",
                "--output",
                str(output),
                "--seed",
                "2027",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        predictions = pd.read_parquet(payload["prediction_path"])
        self.assertEqual(2, len(predictions))
        self.assertEqual({"sample-2", "sample-5"}, set(predictions["xrd_sample_id"]))
        self.assertTrue(Path(payload["metrics_path"]).exists())
        metadata = json.loads((Path(payload["submission_path"]) / "metadata.json").read_text())
        self.assertEqual(["sro"], metadata["tasks"])

    def test_train_composition_extratrees_writes_formal_submission_for_selected_split(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())
        config["model_budgets"]["composition_extratrees"] = {
            "n_estimators": 4,
            "min_samples_leaf": 1,
            "n_jobs": 1,
        }
        config_path = self.root / "extratrees-config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "extratrees-run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "train",
                "--model",
                "composition-extratrees",
                "--dataset",
                str(self.root),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--split-column",
                "chemistry_ood_split",
                "--seed",
                "2027",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        submission = Path(payload["submission_path"])
        metadata = json.loads((submission / "metadata.json").read_text())
        self.assertEqual("composition-extratrees", metadata["model_id"])
        self.assertEqual("chemistry_ood_split", metadata["split_column"])
        self.assertEqual(["sro"], metadata["tasks"])
        sro = pd.read_parquet(submission / "sro.parquet")
        family = pd.read_parquet(submission / "family.parquet")
        self.assertEqual({"state-2", "state-5"}, set(sro["sample_id"]))
        self.assertEqual(set(sro["sample_id"]), set(family["sample_id"]))

    def test_train_peak_xgboost_writes_complete_predictions(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())
        config["model_budgets"]["peak_xgboost"]["n_estimators"] = 2
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "peak-run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "train",
                "--model",
                "peak-xgboost",
                "--dataset",
                str(self.root),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--seed",
                "2027",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        checkpoint = Path(payload["checkpoint_path"])
        self.assertTrue(checkpoint.exists())
        self.assertEqual(".joblib", checkpoint.suffix)
        predictions = pd.read_parquet(payload["prediction_path"])
        self.assertEqual(2, len(predictions))
        self.assertEqual(128, len(predictions.iloc[0]["sro_predictions"]))
        submission = Path(payload["submission_path"])
        self.assertTrue((submission / "family.parquet").exists())
        self.assertTrue((submission / "sro.parquet").exists())
        metadata = json.loads((submission / "metadata.json").read_text())
        self.assertEqual(["sro"], metadata["tasks"])

    def test_train_resnet_smoke_writes_checkpoint_and_predictions(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())
        config["model_budgets"]["resnet1d"].update(
            {"max_epochs": 1, "batch_size": 2, "patience": 1}
        )
        config_path = self.root / "neural-config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "resnet-run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "train",
                "--model",
                "resnet1d",
                "--dataset",
                str(self.root),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--seed",
                "2027",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(Path(payload["checkpoint_path"]).exists())
        self.assertEqual(2, len(pd.read_parquet(payload["prediction_path"])))
        submission = Path(payload["submission_path"])
        metadata = json.loads((submission / "metadata.json").read_text())
        self.assertEqual(["sro"], metadata["tasks"])

    def test_report_marks_pre_release_and_compares_xrd_baseline(self):
        run_root = self.root / "report-run"
        predictions = run_root / "predictions"
        predictions.mkdir(parents=True)
        evidence = run_root / "evidence"
        evidence.mkdir()
        (run_root / "audit.json").write_text(
            json.dumps({"parent_group_leakage_count": 0, "row_count": 6})
        )
        (run_root / "controls.json").write_text(
            json.dumps(
                {
                    "control_gate_passed": True,
                    "occupancy_ranking": {
                        "oracle": {"identifiable_coverage": 0.00725}
                    },
                }
            )
        )
        (evidence / "identifiability.json").write_text(
            json.dumps(
                {
                    "pair_count": 4000,
                    "identifiable_pair_count": 0,
                    "identifiable_pair_fraction": 0.0,
                    "status": "diagnostic",
                }
            )
        )
        for model, mae, family_f1 in [
            ("composition-ridge", 0.45, 0.24),
            ("peak-xgboost", 0.36, 0.46),
        ]:
            (predictions / f"{model}.metrics.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "seed": 2027,
                        "sro": {"masked_parent_macro_mae": mae, "rmse": mae + 0.1},
                        "family": {"macro_f1": family_f1},
                    }
                )
            )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alloy_xrd_bench.hea_validation",
                "report",
                "--run-root",
                str(run_root),
                "--config",
                "configs/hea_validation_v1.json",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        result_payload = json.loads(result.stdout)
        report = Path(result_payload["report_path"]).read_text()
        summary = json.loads(Path(result_payload["summary_path"]).read_text())
        self.assertIn("SRO Core", report)
        self.assertIn("XRD beats composition-only: YES", report)
        self.assertIn("Occupancy identifiability diagnostic", report)
        self.assertIn("Family diagnostic", report)
        self.assertNotIn("Family ground truth validated", report)
        self.assertNotIn("experimentally calibrated noise", report)
        self.assertNotIn("Overall score", report)
        self.assertIn("peak-xgboost", summary["model_aggregates"])
        self.assertEqual(
            0.36,
            summary["model_aggregates"]["peak-xgboost"]["sro_parent_macro_mae_mean"],
        )
        self.assertEqual(0.0, summary["occupancy_identifiable_coverage"])


if __name__ == "__main__":
    unittest.main()
