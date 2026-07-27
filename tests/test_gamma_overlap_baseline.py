from __future__ import annotations

import unittest

import numpy as np

from alloy_xrd_bench.gamma_overlap_physics import ComponentSpec, render_components


class GammaOverlapBaselineTests(unittest.TestCase):
    def test_catalog_uses_training_parents_only(self):
        from alloy_xrd_bench.gamma_overlap_baseline import build_training_candidate_catalog

        phase_rows = [
            _phase_row("g_train", "gamma", 44.30),
            _phase_row("gp_train", "gamma_prime", 44.40),
            _phase_row("gp_test", "gamma_prime", 45.50),
        ]
        splits = [
            {"gamma_parent_id": "g_train", "gamma_prime_parent_id": "gp_train", "split": "train"},
            {"gamma_parent_id": "g_train", "gamma_prime_parent_id": "gp_test", "split": "private_test"},
        ]

        catalog = build_training_candidate_catalog(phase_rows, splits)

        self.assertEqual({("gamma", 2, 0, 0), ("gamma_prime", 2, 0, 0)}, {(r["phase"], r["h"], r["k"], r["l"]) for r in catalog})
        gamma_prime = next(row for row in catalog if row["phase"] == "gamma_prime")
        self.assertEqual(44.40, gamma_prime["center_deg"])

    def test_two_component_window_fit_recovers_both_phases(self):
        from alloy_xrd_bench.gamma_overlap_baseline import fit_window_baseline

        grid = np.arange(43.0, 46.0 + 0.01, 0.02)
        truth = [
            ComponentSpec("gamma", (2, 0, 0), 44.30, 0.16, 6.0, 0.4),
            ComponentSpec("gamma_prime", (2, 0, 0), 44.42, 0.18, 4.0, 0.4),
        ]
        observed, _ = render_components(grid, truth)
        predictions = fit_window_baseline(
            window_id="window_1",
            two_theta=grid,
            observed=observed,
            candidate_catalog=[
                {"phase": "gamma", "h": 2, "k": 0, "l": 0, "center_deg": 44.30},
                {"phase": "gamma_prime", "h": 2, "k": 0, "l": 0, "center_deg": 44.42},
            ],
            max_components=2,
        )

        self.assertEqual({"gamma", "gamma_prime"}, {row["phase"] for row in predictions})
        self.assertEqual(2, len(predictions))
        self.assertTrue(all(abs(row["center_deg"] - target) < 0.03 for row, target in zip(sorted(predictions, key=lambda r: r["center_deg"]), [44.30, 44.42])))


def _phase_row(parent: str, phase: str, center: float) -> dict:
    return {
        "parent_structure_id": parent,
        "phase": phase,
        "h": 2,
        "k": 0,
        "l": 0,
        "center_deg": center,
        "raw_integrated_intensity": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
