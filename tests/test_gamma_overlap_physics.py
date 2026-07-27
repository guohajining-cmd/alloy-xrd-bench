from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class GammaOverlapPhysicsTests(unittest.TestCase):
    def test_parent_hkl_mapping_and_cubic_canonicalization(self):
        from alloy_xrd_bench.gamma_overlap_physics import (
            canonical_cubic_hkl,
            parent_hkl_from_supercell,
        )

        self.assertEqual((2, 0, 0), canonical_cubic_hkl((0, -2, 0)))
        self.assertEqual((2, 1, 1), canonical_cubic_hkl((-1, 2, 1)))
        self.assertEqual((2, 0, 0), parent_hkl_from_supercell((6, 0, 0), (3, 3, 3)))
        self.assertIsNone(parent_hkl_from_supercell((2, 1, 1), (3, 3, 3)))

    def test_discrete_pseudo_voigt_has_requested_area_and_fwhm(self):
        from alloy_xrd_bench.gamma_overlap_physics import pseudo_voigt_profile

        grid = np.arange(40.0, 50.0 + 0.001, 0.001)
        profile = pseudo_voigt_profile(
            grid,
            center_deg=45.0,
            fwhm_deg=0.2,
            integrated_intensity=7.5,
            eta=0.35,
        )

        self.assertAlmostEqual(7.5, float(np.trapezoid(profile, grid)), places=10)
        half = float(profile.max()) / 2.0
        support = grid[profile >= half]
        self.assertAlmostEqual(0.2, float(support[-1] - support[0]), delta=0.002)

    def test_component_sum_reconstructs_clean_pattern_exactly(self):
        from alloy_xrd_bench.gamma_overlap_physics import (
            ComponentSpec,
            render_components,
        )

        grid = np.arange(20.0, 100.0 + 0.01, 0.02)
        components = [
            ComponentSpec("gamma", (2, 0, 0), 44.30, 0.18, 6.0, 0.3),
            ComponentSpec("gamma_prime", (2, 0, 0), 44.42, 0.20, 4.0, 0.4),
        ]

        clean, curves = render_components(grid, components)

        self.assertEqual((2, len(grid)), curves.shape)
        self.assertLess(float(np.max(np.abs(clean - curves.sum(axis=0)))), 1e-12)

    def test_overlap_difficulty_boundaries(self):
        from alloy_xrd_bench.gamma_overlap_physics import overlap_difficulty

        self.assertEqual("single", overlap_difficulty([]))
        self.assertEqual("resolved", overlap_difficulty([2.0]))
        self.assertEqual("partial", overlap_difficulty([1.0]))
        self.assertEqual("strong", overlap_difficulty([0.5]))
        self.assertEqual("severe", overlap_difficulty([0.49]))

    def test_gamma_library_is_deterministic_unique_and_ni97al11(self):
        from alloy_xrd_bench.gamma_overlap import generate_gamma_library

        with tempfile.TemporaryDirectory() as tmp:
            first = generate_gamma_library(
                output_dir=Path(tmp) / "first",
                count=8,
                seed=2027,
                candidate_count=4,
            )
            second = generate_gamma_library(
                output_dir=Path(tmp) / "second",
                count=8,
                seed=2027,
                candidate_count=4,
            )

        self.assertEqual(8, len(first))
        self.assertEqual(8, len({row["structure_sha256"] for row in first}))
        self.assertEqual([row["structure_sha256"] for row in first], [row["structure_sha256"] for row in second])
        self.assertTrue(all(row["composition"] == {"Al": 11, "Ni": 97} for row in first))

    def test_gamma_library_uses_frozen_audited_lattice_samples(self):
        from alloy_xrd_bench.gamma_overlap import generate_gamma_library

        with tempfile.TemporaryDirectory() as tmp:
            rows = generate_gamma_library(
                output_dir=Path(tmp),
                count=4,
                seed=17,
                candidate_count=4,
                lattice_a_values=(3.52, 3.58),
                lattice_source="test_audited_relaxations",
            )

        self.assertEqual([3.52, 3.58, 3.52, 3.58], [row["lattice_a_angstrom"] for row in rows])
        self.assertTrue(all(row["lattice_parameter_source"] == "test_audited_relaxations" for row in rows))


if __name__ == "__main__":
    unittest.main()
