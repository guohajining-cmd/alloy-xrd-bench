import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_prediction_audit import audit_prediction_matrix


SPLITS = [
    "id_split",
    "composition_ood_split",
    "chemistry_ood_split",
    "sro_ood_split",
]


class HeaPredictionAuditTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        predictions = root / "predictions"
        (dataset / "manifests").mkdir(parents=True)
        rows = []
        for index, value in enumerate(["train", "test"]):
            row = {
                "structure_state_id": f"state-{index}",
                "xrd_sample_id": f"xrd-{index}",
                "parent_group_id": f"parent-{index}",
            }
            row.update({split: value for split in SPLITS})
            rows.append(row)
        pd.DataFrame(rows).to_parquet(dataset / "manifests" / "task_samples.parquet")
        return dataset, predictions

    def _write_run(self, root: Path, model: str, seed: int, split: str, ids=None) -> None:
        path = root / "submissions" / model / str(seed) / f"{split}_test"
        path.mkdir(parents=True)
        ids = ["state-1"] if ids is None else ids
        pd.DataFrame(
            {
                "sample_id": ids,
                "xrd_sample_id": ["xrd-1"] * len(ids),
                "parent_group_id": ["parent-1"] * len(ids),
                "model": [model] * len(ids),
                "model_id": [model] * len(ids),
                "seed": [seed] * len(ids),
                "split_column": [split] * len(ids),
                "sro_predictions": [[0.0]] * len(ids),
                "sro_predictions_json": ["[0.0]"] * len(ids),
            }
        ).to_parquet(path / "sro.parquet", index=False)
        (path / "metadata.json").write_text(
            json.dumps({"model_id": model, "seed": seed, "split_column": split})
        )

    def test_requires_exact_frozen_model_seed_split_matrix(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset, predictions = self._fixture(Path(tempdir))
            for split in SPLITS:
                self._write_run(predictions, "composition-ridge", 2027, split)
                for model in ["peak-xgboost", "resnet1d", "transformer", "physics-transformer"]:
                    for seed in range(2027, 2032):
                        self._write_run(predictions, model, seed, split)

            report = audit_prediction_matrix(dataset, predictions)

            self.assertEqual("pass", report["status"])
            self.assertEqual(84, report["required_run_count"])
            self.assertEqual(0, report["missing_count"])
            self.assertEqual(0, report["duplicate_count"])
            self.assertEqual(0, report["unexpected_count"])

    def test_base_only_matrix_is_rejected_when_physics_runs_are_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset, predictions = self._fixture(Path(tempdir))
            for split in SPLITS:
                self._write_run(predictions, "composition-ridge", 2027, split)
                for model in ["peak-xgboost", "resnet1d", "transformer"]:
                    for seed in range(2027, 2032):
                        self._write_run(predictions, model, seed, split)
            report = audit_prediction_matrix(dataset, predictions)

            self.assertEqual("fail", report["status"])
            self.assertEqual(84, report["required_run_count"])
            self.assertEqual(20, report["missing_count"])
            self.assertEqual(0, report["unexpected_count"])

    def test_missing_seed_and_duplicate_prediction_ids_fail(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset, predictions = self._fixture(Path(tempdir))
            self._write_run(
                predictions,
                "composition-ridge",
                2027,
                "id_split",
                ids=["state-1", "state-1"],
            )

            report = audit_prediction_matrix(dataset, predictions)

            self.assertEqual("fail", report["status"])
            self.assertGreater(report["missing_count"], 0)
            self.assertEqual(1, report["duplicate_count"])


if __name__ == "__main__":
    unittest.main()
