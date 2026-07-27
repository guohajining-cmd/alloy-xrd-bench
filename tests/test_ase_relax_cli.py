import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from alloy_xrd_bench.potentials import register_potential
from alloy_xrd_bench.relax import ase_relax_initial_structures
from alloy_xrd_bench.structures import generate_initial_structures


class AseRelaxationTests(unittest.TestCase):
    def test_ase_relaxation_writes_per_potential_manifests(self):
        class FakeAseAdapter:
            def load(self, root_path, potential):
                from ase.calculators.lj import LennardJones

                self._calculator = LennardJones()
                return {"ok": True, "adapter": "fake_ase"}

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "Ni",
                                "phase": "fcc",
                                "composition": {"Ni": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.52,
                                "seed": 1,
                                "generation_method": "deterministic_random_occupancy",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            generate_initial_structures(root=root, recipes_path=recipe_path)
            register_potential(
                root=root,
                potential_id="pot_fake_ase",
                family="Fake ASE",
                engine="fake_ase",
                elements=["Ni"],
                phases=["fcc"],
                source_url="https://example.org/fake",
                doi="10.0000/fake",
                license_name="test",
                redistribution_policy="metadata_only",
                citation_keys=["fake"],
                file_sha256="0" * 64,
            )

            with patch("alloy_xrd_bench.relax.adapter_for_engine", return_value=FakeAseAdapter()):
                rows = ase_relax_initial_structures(
                    root=root,
                    potential_id="pot_fake_ase",
                    fmax=10.0,
                    steps=2,
                    relax_cell=False,
                )

            manifest_dir = root / "data" / "potentials" / "pot_fake_ase" / "generated" / "manifests"
            relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
            relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")
            edges = pd.read_parquet(manifest_dir / "provenance_edges.parquet")
            json_path = root / relaxed.iloc[0]["artifact_path_json"]
            cif_path = root / relaxed.iloc[0]["artifact_path_cif"]
            json_exists = json_path.exists()
            cif_exists = cif_path.exists()

        self.assertEqual(1, len(rows))
        self.assertEqual("fake_ase", relaxations.iloc[0]["engine"])
        self.assertTrue(json_exists)
        self.assertTrue(cif_exists)
        self.assertEqual(1, len(relaxed))
        self.assertIn("produced", set(edges["relation_type"]))

    def test_ase_relaxation_skips_initial_structures_outside_potential_elements(self):
        class FakeAseAdapter:
            def load(self, root_path, potential):
                from ase.calculators.lj import LennardJones

                self._calculator = LennardJones()
                return {"ok": True, "adapter": "fake_ase"}

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "Ni",
                                "phase": "fcc",
                                "composition": {"Ni": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.52,
                                "seed": 1,
                                "generation_method": "deterministic_random_occupancy",
                            },
                            {
                                "alloy_system": "Al",
                                "phase": "fcc",
                                "composition": {"Al": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 4.05,
                                "seed": 2,
                                "generation_method": "deterministic_random_occupancy",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            generate_initial_structures(root=root, recipes_path=recipe_path)
            register_potential(
                root=root,
                potential_id="pot_fake_ase",
                family="Fake ASE",
                engine="fake_ase",
                elements=["Ni"],
                phases=["fcc"],
                source_url="https://example.org/fake",
                doi="10.0000/fake",
                license_name="test",
                redistribution_policy="metadata_only",
                citation_keys=["fake"],
                file_sha256="0" * 64,
            )

            with patch("alloy_xrd_bench.relax.adapter_for_engine", return_value=FakeAseAdapter()):
                rows = ase_relax_initial_structures(
                    root=root,
                    potential_id="pot_fake_ase",
                    fmax=10.0,
                    steps=2,
                    relax_cell=False,
                )

            manifest_dir = root / "data" / "potentials" / "pot_fake_ase" / "generated" / "manifests"
            relaxations = pd.read_parquet(manifest_dir / "relaxations.parquet")
            relaxed = pd.read_parquet(manifest_dir / "relaxed_structures.parquet")

        self.assertEqual(1, len(rows))
        self.assertEqual(1, len(relaxations))
        self.assertEqual(1, len(relaxed))
        self.assertEqual("init_Ni_fcc_Ni1_seed000001", relaxations.iloc[0]["initial_structure_id"])


if __name__ == "__main__":
    unittest.main()
