import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from alloy_xrd_bench.hea_validation_data import (
    ValidationRow,
    audit_parent_group_leakage,
    load_validation_rows,
    read_grid_row,
    vectorize_sro,
)


class ValidationContractTests(unittest.TestCase):
    def test_contract_has_fixed_vocab_and_seeds(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())

        self.assertEqual(
            ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"],
            config["elements"],
        )
        self.assertEqual([2027, 2028, 2029, 2030, 2031], config["seeds"])
        self.assertEqual(128, len(config["sro_targets"]))
        self.assertEqual("pre-release", config["result_qualification"])


class ValidationDataTests(unittest.TestCase):
    def setUp(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())
        self.targets = config["sro_targets"]
        self.tempdir = tempfile.TemporaryDirectory()
        self.h5_path = Path(self.tempdir.name) / "grid.h5"
        with h5py.File(self.h5_path, "w") as handle:
            handle.create_dataset("intensity", data=np.ones((2, 4001), dtype=np.float32))
            handle.create_dataset("two_theta", data=np.linspace(10, 90, 4001, dtype=np.float32))
            handle.create_dataset("xrd_sample_id", data=np.asarray([b"sample-a", b"sample-b"]))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_vectorize_sro_preserves_mask(self):
        values, mask = vectorize_sro('{"Al->Co": -0.25}', "{}", self.targets)

        shell1 = self.targets.index("shell1:Al->Co")
        shell2 = self.targets.index("shell2:Al->Co")
        self.assertEqual(-0.25, values[shell1])
        self.assertTrue(mask[shell1])
        self.assertFalse(mask[shell2])

    def test_grid_reader_rejects_wrong_sample_id(self):
        with self.assertRaisesRegex(ValueError, "sample ID mismatch"):
            read_grid_row(self.h5_path, 0, expected_sample_id="wrong")

    def test_grid_reader_returns_valid_row(self):
        row = read_grid_row(self.h5_path, 1, expected_sample_id="sample-b")

        self.assertEqual((4001,), row.shape)
        self.assertTrue(np.isfinite(row).all())

    def test_parent_group_leakage_is_reported(self):
        rows = [
            self._row("sample-a", "parent-a", "train"),
            self._row("sample-b", "parent-a", "test"),
            self._row("sample-c", "parent-b", "test"),
        ]

        self.assertEqual(["parent-a"], audit_parent_group_leakage(rows, "id_split"))

    def test_manifest_loader_joins_composition_and_maps_family(self):
        root = Path(self.tempdir.name)
        (root / "manifests").mkdir()
        (root / "grids").mkdir()
        pd.DataFrame(
            [{
                "structure_state_id": "state-a",
                "parent_group_id": "parent-a",
                "structure_family": "l12_like",
                "xrd_sample_id": "sample-a",
                "grid_shard_path": "grids/grid.h5",
                "grid_row_index": 0,
                "sro_shell1_json": '{"Al->Co": -0.25}',
                "sro_shell2_json": "{}",
                "id_split": "train",
                "composition_ood_split": "test",
                "chemistry_ood_split": "train",
                "sro_ood_split": "validation",
            }]
        ).to_parquet(root / "manifests" / "task_samples.parquet")
        composition = {
            "structure_state_id": "state-a",
            "parent_group_id": "parent-a",
        }
        composition.update({f"fraction_{element}": 0.125 for element in ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"]})
        pd.DataFrame([composition]).to_parquet(
            root / "manifests" / "composition_only_features.parquet"
        )
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())

        rows = load_validation_rows(root, config)

        self.assertEqual(1, len(rows))
        self.assertEqual("fcc", rows[0].family)
        np.testing.assert_allclose(np.full(8, 0.125), rows[0].composition)
        self.assertTrue(Path(rows[0].grid_shard_path).is_absolute())

    def test_manifest_loader_rejects_grid_path_escape(self):
        root = Path(self.tempdir.name)
        (root / "manifests").mkdir()
        task = {
            "structure_state_id": "state-a",
            "parent_group_id": "parent-a",
            "structure_family": "fcc_like",
            "xrd_sample_id": "sample-a",
            "grid_shard_path": "../outside.h5",
            "grid_row_index": 0,
            "sro_shell1_json": "{}",
            "sro_shell2_json": "{}",
            "id_split": "train",
            "composition_ood_split": "train",
            "chemistry_ood_split": "train",
            "sro_ood_split": "train",
        }
        pd.DataFrame([task]).to_parquet(root / "manifests" / "task_samples.parquet")
        composition = {
            "structure_state_id": "state-a",
            "parent_group_id": "parent-a",
        }
        composition.update({f"fraction_{element}": 0.125 for element in ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"]})
        pd.DataFrame([composition]).to_parquet(
            root / "manifests" / "composition_only_features.parquet"
        )
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())

        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            load_validation_rows(root, config)

    def _row(self, sample_id, parent_id, split):
        return ValidationRow(
            structure_state_id=sample_id,
            parent_group_id=parent_id,
            xrd_sample_id=sample_id,
            grid_shard_path=str(self.h5_path),
            grid_row_index=0,
            family="fcc",
            composition=np.zeros(8, dtype=np.float32),
            sro_values=np.zeros(128, dtype=np.float32),
            sro_mask=np.zeros(128, dtype=bool),
            splits={"id_split": split},
        )


if __name__ == "__main__":
    unittest.main()
