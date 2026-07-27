from __future__ import annotations

import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from alloy_xrd_bench.structures import _structure_from_recipe


class GammaOverlapSourceTests(unittest.TestCase):
    def test_screen_classifies_core_vacancy_and_quarantine_sizes(self):
        from alloy_xrd_bench.gamma_overlap_sources import classify_darus_structure

        bulk = _l12_supercell()
        vacancy = bulk.copy()
        vacancy.remove_sites([0])
        defect_96 = Structure(bulk.lattice, [site.specie for site in bulk[:96]], [site.frac_coords for site in bulk[:96]])
        extended = bulk.copy()
        extended.make_supercell([1, 1, 2])
        defect_144 = Structure(extended.lattice, [site.specie for site in extended[:144]], [site.frac_coords for site in extended[:144]])

        self.assertEqual("accepted_core", classify_darus_structure(bulk)["eligibility"])
        self.assertEqual("accepted_core", classify_darus_structure(vacancy)["eligibility"])
        self.assertEqual("ood_quarantine", classify_darus_structure(defect_96)["eligibility"])
        self.assertEqual("ood_quarantine", classify_darus_structure(defect_144)["eligibility"])

    def test_parent_reflections_are_integer_canonical_and_satellites_are_separate(self):
        from alloy_xrd_bench.gamma_overlap_sources import extract_parent_reflections

        structure = _l12_supercell()
        structure.translate_sites([1], [0.03, 0.0, 0.0], frac_coords=True, to_unit_cell=True)
        result = extract_parent_reflections(
            structure,
            phase="gamma_prime",
            supercell=(3, 3, 3),
        )

        self.assertGreater(len(result["parent_reflections"]), 0)
        self.assertGreater(len(result["satellite_reflections"]), 0)
        self.assertTrue(all(row["h"] >= row["k"] >= row["l"] >= 0 for row in result["parent_reflections"]))
        self.assertTrue(all(row["reflection_kind"] == "parent_bragg" for row in result["parent_reflections"]))
        self.assertTrue(all(row["scored"] for row in result["parent_reflections"]))
        self.assertTrue(all(not row["scored"] for row in result["satellite_reflections"]))

    def test_gamma_reflections_obey_fcc_extinction(self):
        from alloy_xrd_bench.gamma_overlap_sources import extract_parent_reflections

        gamma, _, _ = _structure_from_recipe(
            {
                "alloy_system": "Ni-Al",
                "phase": "fcc",
                "composition": {"Ni": 97, "Al": 11},
                "supercell": [3, 3, 3],
                "lattice_a": 3.57,
                "seed": 2027,
                "generation_method": "pair_balance_occupancy",
                "candidate_count": 2,
            }
        )
        result = extract_parent_reflections(gamma, phase="gamma", supercell=(3, 3, 3))

        for row in result["parent_reflections"]:
            parity = {row["h"] % 2, row["k"] % 2, row["l"] % 2}
            self.assertEqual(1, len(parity))


def _l12_supercell() -> Structure:
    structure, _, _ = _structure_from_recipe(
        {
            "alloy_system": "Ni-Al",
            "phase": "l12",
            "composition": {"Ni": 3, "Al": 1},
            "sublattice_composition": {"corner": {"Al": 1}, "face": {"Ni": 1}},
            "supercell": [3, 3, 3],
            "lattice_a": 3.57,
            "seed": 1,
            "generation_method": "l12_sublattice_occupancy",
        }
    )
    return structure


if __name__ == "__main__":
    unittest.main()
