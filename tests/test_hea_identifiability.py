import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from alloy_xrd_bench.hea_identifiability import (
    add_noise_distances,
    assign_identifiability_classes,
    calibrate_parent_noise_floors,
    label_pair_identifiability,
    nearest_prototype_accuracy,
    audit_parent_noise_coverage,
    compute_observation_distances,
    weak_peak_weighted_l1,
)


class HEAIdentifiabilityTests(unittest.TestCase):
    def test_noise_floor_is_parent_level_95th_percentile(self):
        distances = pd.DataFrame(
            {
                "parent_group_id": ["p1"] * 4 + ["p2"] * 4,
                "noise_l1_distance": [0.01, 0.02, 0.03, 0.04, 0.10, 0.20, 0.30, 0.40],
                "noise_cosine_distance": [0.001, 0.002, 0.003, 0.004, 0.01, 0.02, 0.03, 0.04],
            }
        )
        floors = calibrate_parent_noise_floors(distances, quantile=0.95)
        self.assertAlmostEqual(0.0385, floors.loc["p1", "noise_l1_q95"])
        self.assertAlmostEqual(0.385, floors.loc["p2", "noise_l1_q95"])
        self.assertIn("global_noise_l1_q95", floors.columns)

    def test_pair_is_identifiable_only_above_its_parent_noise_floor(self):
        pairs = pd.DataFrame(
            {
                "pair_id": ["above", "below"],
                "parent_group_id": ["p1", "p2"],
                "xrd_l1_distance": [0.05, 0.20],
            }
        )
        floors = pd.DataFrame(
            {
                "noise_l1_q95": [0.04, 0.30],
                "global_noise_l1_q95": [0.20, 0.20],
            },
            index=pd.Index(["p1", "p2"], name="parent_group_id"),
        )
        labelled = label_pair_identifiability(pairs, floors)
        self.assertAlmostEqual(1.25, labelled.loc[0, "identifiability_ratio"])
        self.assertEqual([True, False], labelled["noise_identifiable"].tolist())
        self.assertEqual(
            ["parent_specific", "parent_specific"],
            labelled["noise_floor_source"].tolist(),
        )

    def test_noise_coverage_requires_parent_specific_variants_for_every_parent(self):
        pairs = pd.DataFrame(
            {
                "parent_group_id": ["covered", "covered", "missing", "missing"],
                "pair_id": ["a", "b", "c", "d"],
            }
        )
        floors = pd.DataFrame(
            {
                "parent_group_id": ["covered"],
                "noise_observation_count": [5],
            }
        ).set_index("parent_group_id")
        report = audit_parent_noise_coverage(pairs, floors, minimum_variants=5)
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(2, report["occupancy_parent_group_count"])
        self.assertEqual(1, report["parent_specific_calibration_group_count"])
        self.assertEqual(["missing"], report["missing_parent_group_ids"])

    def test_distances_match_frozen_xrd_scale(self):
        clean = np.asarray([[0.0, 100.0], [100.0, 0.0]])
        observed = np.asarray([[10.0, 90.0], [80.0, 20.0]])
        result = add_noise_distances(clean, observed)
        np.testing.assert_allclose(result["noise_l1_distance"], [0.1, 0.2])
        self.assertTrue((result["noise_cosine_distance"] >= 0).all())

    def test_cross_root_calibration_reads_clean_from_v1_and_observation_from_v1_1(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "benchmark_v1"
            output = root / "benchmark_v1_1"
            clean_relative = Path("xrd/runs/clean/grids/grids_0000.h5")
            observation_relative = Path("xrd/runs/obs/grids/grids_0000.h5")
            (source / clean_relative).parent.mkdir(parents=True)
            (output / observation_relative).parent.mkdir(parents=True)
            with h5py.File(source / clean_relative, "w") as handle:
                handle.create_dataset("intensity", data=np.asarray([[0.0, 100.0]]))
            with h5py.File(output / observation_relative, "w") as handle:
                handle.create_dataset("intensity", data=np.asarray([[10.0, 90.0]]))
            manifest_dir = source / "xrd" / "runs" / "clean"
            pd.DataFrame(
                [{"xrd_sample_id": "clean-1", "grid_shard_path": clean_relative.as_posix(), "grid_row_index": 0}]
            ).to_parquet(manifest_dir / "xrd_samples.parquet", index=False)
            observations = pd.DataFrame(
                [{
                    "xrd_sample_id": "obs-1",
                    "source_xrd_sample_id": "clean-1",
                    "structure_state_id": "state-1",
                    "parent_group_id": "parent-1",
                    "grid_shard_path": observation_relative.as_posix(),
                    "grid_row_index": 0,
                }]
            )
            distances = compute_observation_distances(
                source, observations, observation_dataset=output
            )
            self.assertAlmostEqual(0.1, distances.loc[0, "noise_l1_distance"])

    def test_nearest_prototype_oracle_and_three_level_labels(self):
        prototype_a = np.asarray([100.0, 0.0])
        prototype_b = np.asarray([0.0, 100.0])
        views_a = np.asarray([[90.0, 10.0], [80.0, 20.0]])
        views_b = np.asarray([[10.0, 90.0], [20.0, 80.0]])
        self.assertEqual(
            1.0,
            nearest_prototype_accuracy(prototype_a, prototype_b, views_a, views_b),
        )
        pairs = pd.DataFrame(
            {
                "identifiability_ratio": [1.2, 0.7, 0.4, 2.0],
                "oracle_accuracy": [0.9, 0.7, 0.9, np.nan],
            }
        )
        labelled = assign_identifiability_classes(pairs)
        self.assertEqual(
            ["identifiable", "weakly_identifiable", "non_identifiable", "non_identifiable"],
            labelled["identifiability_class"].tolist(),
        )
        self.assertEqual("missing_views", labelled.loc[3, "identifiability_status"])

    def test_weak_peak_distance_emphasizes_low_intensity_regions(self):
        first = np.asarray([100.0, 1.0, 0.0])
        second = np.asarray([100.0, 2.0, 0.0])
        ordinary = np.mean(np.abs(first - second)) / 100.0
        self.assertGreater(weak_peak_weighted_l1(first, second), ordinary)


if __name__ == "__main__":
    unittest.main()
