from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


class GammaOverlapBuilderTests(unittest.TestCase):
    def test_default_config_has_frozen_grid_and_noise_ranges(self):
        from alloy_xrd_bench.gamma_overlap_builder import load_overlap_config

        config = load_overlap_config(Path("configs/nickel_gamma_overlap_v1.json"))

        self.assertEqual("nickel_gamma_overlap_v1", config["config_id"])
        self.assertEqual(20.0, config["two_theta_min"])
        self.assertEqual(100.0, config["two_theta_max"])
        self.assertEqual(0.02, config["step_deg"])
        self.assertEqual(3.0, config["window_width_deg"])

    def test_synthesis_is_deterministic_and_clean_is_exact_component_sum(self):
        from alloy_xrd_bench.gamma_overlap_builder import load_overlap_config, synthesize_mixture

        config = load_overlap_config(Path("configs/nickel_gamma_overlap_v1.json"))
        first = synthesize_mixture(
            sample_id="sample_1",
            parent_pair_id="pair_1",
            gamma_reflections=[_reflection("gamma", (2, 0, 0), 44.30, 8.0)],
            gamma_prime_reflections=[_reflection("gamma_prime", (2, 0, 0), 44.36, 6.0)],
            sample_seed=41,
            config=config,
        )
        second = synthesize_mixture(
            sample_id="sample_1",
            parent_pair_id="pair_1",
            gamma_reflections=[_reflection("gamma", (2, 0, 0), 44.30, 8.0)],
            gamma_prime_reflections=[_reflection("gamma_prime", (2, 0, 0), 44.36, 6.0)],
            sample_seed=41,
            config=config,
        )

        self.assertEqual((4001,), first["two_theta"].shape)
        np.testing.assert_array_equal(first["observed"], second["observed"])
        np.testing.assert_array_equal(first["clean"], first["component_curves"].sum(axis=0))
        self.assertLess(float(np.max(np.abs(first["clean"] - first["component_curves"].sum(axis=0)))), 1e-12)
        self.assertEqual(first["component_rows"], second["component_rows"])

    def test_window_extraction_writes_151_point_overlap_and_single_windows(self):
        from alloy_xrd_bench.gamma_overlap_builder import extract_windows, load_overlap_config, synthesize_mixture

        config = load_overlap_config(Path("configs/nickel_gamma_overlap_v1.json"))
        sample = synthesize_mixture(
            sample_id="sample_2",
            parent_pair_id="pair_2",
            gamma_reflections=[
                _reflection("gamma", (1, 1, 1), 35.00, 2.0),
                _reflection("gamma", (2, 0, 0), 44.30, 8.0),
            ],
            gamma_prime_reflections=[_reflection("gamma_prime", (2, 0, 0), 44.36, 6.0)],
            sample_seed=42,
            config=config,
        )
        windows = extract_windows(sample, config=config)

        self.assertTrue(windows)
        self.assertTrue(all(window["two_theta"].shape == (151,) for window in windows))
        self.assertTrue(any(window["component_count"] >= 2 for window in windows))
        self.assertTrue(any(window["difficulty"] == "single" for window in windows))

    def test_split_assignment_keeps_every_parent_pair_together(self):
        from alloy_xrd_bench.gamma_overlap_builder import assign_pair_splits

        rows = [
            {"sample_id": f"sample_{pair}_{variant}", "parent_pair_id": f"pair_{pair}"}
            for pair in range(20)
            for variant in range(3)
        ]
        assigned = assign_pair_splits(rows, seed=2027)
        splits_by_pair = {}
        for row in assigned:
            splits_by_pair.setdefault(row["parent_pair_id"], set()).add(row["split"])

        self.assertTrue(all(len(splits) == 1 for splits in splits_by_pair.values()))
        self.assertEqual({"train", "validation", "private_test"}, {row["split"] for row in assigned})

    def test_cross_split_cosine_reports_nearest_pair(self):
        from alloy_xrd_bench.gamma_overlap_builder import maximum_cross_split_cosine

        result = maximum_cross_split_cosine(
            np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
            ["train", "validation", "train"],
            ["a", "b", "c"],
        )

        self.assertAlmostEqual(0.8, result["maximum_cosine_similarity"])
        self.assertEqual(["a", "b"], result["record_ids"])

    def test_window_selection_is_deterministic_and_balanced(self):
        from alloy_xrd_bench.gamma_overlap_builder import select_balanced_windows

        rows = []
        for difficulty, count in {"single": 20, "resolved": 10, "partial": 8, "strong": 2, "severe": 7}.items():
            rows.extend(
                {"window_id": f"{difficulty}_{index}", "difficulty": difficulty, "split": "train"}
                for index in range(count)
            )
        first = select_balanced_windows(rows, seed=7, single_fraction=0.20, maximum_overlap_class_ratio=3.0)
        second = select_balanced_windows(rows, seed=7, single_fraction=0.20, maximum_overlap_class_ratio=3.0)

        self.assertEqual([row["window_id"] for row in first], [row["window_id"] for row in second])
        counts = {name: sum(row["difficulty"] == name for row in first) for name in ("resolved", "partial", "strong", "severe")}
        self.assertLessEqual(max(counts.values()) / min(counts.values()), 3.0)
        self.assertGreaterEqual(sum(row["difficulty"] == "single" for row in first) / len(first), 0.20)

    def test_source_run_split_balances_record_counts(self):
        from alloy_xrd_bench.gamma_overlap import _make_parent_pairs

        run_sizes = [20, 15, 10, 10, 10, 10, 5, 5, 5, 5, 5]
        gamma_prime = []
        for run_index, size in enumerate(run_sizes):
            gamma_prime.extend(
                {
                    "parent_structure_id": f"gp_{run_index}_{row_index}",
                    "source_run_id": f"run_{run_index}",
                }
                for row_index in range(size)
            )
        gamma = [{"parent_structure_id": f"gamma_{index}"} for index in range(9)]
        pairs = _make_parent_pairs(
            gamma_prime,
            gamma,
            seed=2027,
            split_ratios={"train": 0.70, "validation": 0.15, "private_test": 0.15},
        )

        counts = {split: sum(row["split"] == split for row in pairs) for split in ("train", "validation", "private_test")}
        self.assertEqual({"train": 70, "validation": 15, "private_test": 15}, counts)

    def test_fingerprint_components_cannot_cross_splits(self):
        from alloy_xrd_bench.gamma_overlap_builder import (
            isolate_pair_split_components,
            maximum_cross_split_cosine,
        )

        rows = [
            {
                "parent_pair_id": f"pair_{index}",
                "gamma_parent_id": f"gamma_{index}",
                "source_run_id": f"run_{index}",
            }
            for index in range(6)
        ]
        fingerprints = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0.7, 0.7, 0], [0.7, 0, 0.7]],
            dtype=float,
        )
        assigned = isolate_pair_split_components(
            rows,
            fingerprints,
            seed=2027,
            ratios={"train": 0.50, "validation": 0.25, "private_test": 0.25},
            cosine_threshold=0.999,
        )
        split_by_pair = {row["parent_pair_id"]: row["split"] for row in assigned}

        self.assertEqual(split_by_pair["pair_0"], split_by_pair["pair_3"])
        check = maximum_cross_split_cosine(
            fingerprints,
            [row["split"] for row in assigned],
            [row["parent_pair_id"] for row in assigned],
        )
        self.assertLess(check["maximum_cosine_similarity"], 0.999)


def _reflection(phase: str, hkl: tuple[int, int, int], center: float, intensity: float) -> dict:
    return {
        "phase": phase,
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        "center_deg": center,
        "raw_integrated_intensity": intensity,
        "reflection_kind": "parent_bragg",
        "scored": True,
    }


if __name__ == "__main__":
    unittest.main()
