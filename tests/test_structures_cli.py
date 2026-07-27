import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from alloy_xrd_bench.contract import canonical_structure_hash


class StructuresCliTests(unittest.TestCase):
    def test_default_recipe_catalog_contains_v0_1_systems(self):
        recipe_path = Path("configs/alloy_recipes_v0_1.yml")
        payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        systems = {
            (recipe["alloy_system"], recipe["phase"], recipe.get("phase_role", ""))
            for recipe in payload["recipes"]
        }

        self.assertTrue(
            {
                ("CrCoNi", "fcc", ""),
                ("CoCrFeMnNi", "fcc", ""),
                ("TiTaVW", "bcc", ""),
                ("MoNbTaVW", "bcc", ""),
                ("WTaCrV", "bcc", ""),
                ("Ni-Al", "fcc", "gamma"),
                ("Ni-Al", "l12", "gamma_prime"),
                ("Ni-Al-Ti", "fcc", "gamma"),
                ("Ni-Al-Ti", "l12", "gamma_prime"),
                ("Ni-Al-Ta", "fcc", "gamma"),
                ("Ni-Al-Ta", "l12", "gamma_prime"),
                ("Ni-Al-Ti-Ta", "fcc", "gamma"),
                ("Ni-Al-Ti-Ta", "l12", "gamma_prime"),
            }
            <= systems
        )

    def test_default_recipe_catalog_has_multiple_seeds_and_composition_buckets(self):
        recipe_path = Path("configs/alloy_recipes_v0_1.yml")
        payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        recipes = payload["recipes"]
        seeds_by_system = {}
        compositions_by_system = {}
        for recipe in recipes:
            system = recipe["alloy_system"]
            seeds_by_system.setdefault(system, set()).add(recipe["seed"])
            composition_label = tuple(sorted(recipe["composition"].items()))
            compositions_by_system.setdefault(system, set()).add(composition_label)

        self.assertTrue(all(len(seeds) >= 2 for seeds in seeds_by_system.values()))
        self.assertTrue(any(len(compositions) >= 2 for compositions in compositions_by_system.values()))
        self.assertTrue(
            {"pair_balance_occupancy", "l12_sublattice_occupancy"}
            <= {recipe["generation_method"] for recipe in recipes}
        )

    def test_generate_initial_structures_writes_shared_manifest_and_stable_hashes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "CrCoNi",
                                "phase": "fcc",
                                "composition": {"Cr": 1, "Co": 1, "Ni": 1},
                                "supercell": [2, 2, 2],
                                "lattice_a": 3.58,
                                "seed": 7,
                                "generation_method": "pair_balance_occupancy",
                                "candidate_count": 8,
                            },
                            {
                                "alloy_system": "TiTaVW",
                                "phase": "bcc",
                                "composition": {"Ti": 1, "Ta": 1, "V": 1, "W": 1},
                                "supercell": [2, 2, 2],
                                "lattice_a": 3.2,
                                "seed": 11,
                                "generation_method": "pair_balance_occupancy",
                                "candidate_count": 8,
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.structures",
                    "generate-initial",
                    "--root",
                    str(root),
                    "--recipes",
                    str(recipe_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            second = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.structures",
                    "generate-initial",
                    "--root",
                    str(root),
                    "--recipes",
                    str(recipe_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            manifest_path = root / "data" / "shared" / "manifests" / "initial_structures.parquet"
            rows = pd.read_parquet(manifest_path).to_dict(orient="records")
            first_hashes = [row["structure_sha256"] for row in rows]
            second_rows = pd.read_parquet(manifest_path).to_dict(orient="records")
            artifact_path = root / rows[0]["artifact_path"]
            structure_doc = json.loads(artifact_path.read_text(encoding="utf-8"))
            actual_artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("", second.stderr)
        self.assertEqual(0, second.returncode)
        self.assertEqual(2, len(rows))
        self.assertEqual(first_hashes, [row["structure_sha256"] for row in second_rows])
        self.assertEqual(2, len({row["initial_structure_id"] for row in rows}))
        self.assertTrue(rows[0]["artifact_path"].startswith("data/shared/structures_initial/"))
        self.assertEqual(canonical_structure_hash(structure_doc), rows[0]["structure_sha256"])
        self.assertEqual(actual_artifact_hash, rows[0]["artifact_file_sha256"])
        self.assertIn("generation_score", rows[0])
        self.assertIn("candidate_count", rows[0])
        self.assertGreaterEqual(rows[0]["generation_score"], 0)
        self.assertEqual(8, rows[0]["candidate_count"])

    def test_l12_sublattice_occupancy_keeps_gamma_prime_ordering_and_pair_manifest(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            recipe_path = root / "recipes.yml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {
                        "recipes": [
                            {
                                "alloy_system": "Ni-Al",
                                "phase": "fcc",
                                "phase_role": "gamma",
                                "pair_id": "pair_NiAl_seed000001",
                                "nominal_alloy_system": "Ni-Al",
                                "partitioning_rule": "Ni-rich gamma / Ni3Al gamma-prime",
                                "composition": {"Ni": 9, "Al": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.57,
                                "seed": 1,
                                "generation_method": "deterministic_random_occupancy",
                            },
                            {
                                "alloy_system": "Ni-Al",
                                "phase": "l12",
                                "phase_role": "gamma_prime",
                                "pair_id": "pair_NiAl_seed000001",
                                "nominal_alloy_system": "Ni-Al",
                                "partitioning_rule": "Ni-rich gamma / Ni3Al gamma-prime",
                                "composition": {"Ni": 3, "Al": 1},
                                "sublattice_composition": {
                                    "corner": {"Al": 1},
                                    "face": {"Ni": 1},
                                },
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.57,
                                "seed": 2,
                                "generation_method": "l12_sublattice_occupancy",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.structures",
                    "generate-initial",
                    "--root",
                    str(root),
                    "--recipes",
                    str(recipe_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            initial_rows = pd.read_parquet(
                root / "data" / "shared" / "manifests" / "initial_structures.parquet"
            )
            pair_rows = pd.read_parquet(
                root / "data" / "shared" / "manifests" / "gamma_prime_pairs.parquet"
            ).to_dict(orient="records")
            gamma_prime_row = initial_rows[initial_rows["phase_role"] == "gamma_prime"].iloc[0].to_dict()
            structure_doc = json.loads((root / gamma_prime_row["artifact_path"]).read_text(encoding="utf-8"))
            sites = structure_doc["sites"]
            species_by_coord = {
                tuple(round(float(value), 6) for value in site["abc"]): site["species"][0]["element"]
                for site in sites
            }

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("Al", species_by_coord[(0.0, 0.0, 0.0)])
        self.assertEqual({"Ni"}, {species_by_coord[coord] for coord in species_by_coord if coord != (0.0, 0.0, 0.0)})
        self.assertEqual(1, len(pair_rows))
        self.assertEqual("pair_NiAl_seed000001", pair_rows[0]["pair_id"])
        self.assertTrue(pair_rows[0]["gamma_initial_structure_id"].startswith("init_Ni-Al_fcc_"))
        self.assertTrue(pair_rows[0]["gamma_prime_initial_structure_id"].startswith("init_Ni-Al_l12_"))
