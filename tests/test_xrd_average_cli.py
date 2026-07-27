import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


class XrdAverageCliTests(unittest.TestCase):
    def test_average_groups_xrd_by_composition_and_writes_manifest(self):
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
                                "alloy_system": "Ni",
                                "phase": "fcc",
                                "composition": {"Ni": 1},
                                "supercell": [1, 1, 1],
                                "lattice_a": 3.52,
                                "seed": 2,
                                "generation_method": "deterministic_random_occupancy",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            commands = [
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
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "register",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_smoke",
                    "--family",
                    "identity",
                    "--engine",
                    "identity_smoke",
                    "--elements",
                    "Ni",
                    "--phases",
                    "fcc",
                    "--source-url",
                    "local-smoke",
                    "--doi",
                    "not-applicable",
                    "--license",
                    "not-applicable",
                    "--redistribution-policy",
                    "metadata_only",
                    "--citation-key",
                    "smoke",
                    "--file-sha256",
                    "0" * 64,
                ],
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.relax",
                    "smoke",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_smoke",
                    "--quality-tier",
                    "pass",
                ],
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.xrd",
                    "simulate",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_smoke",
                ],
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.xrd",
                    "average",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_smoke",
                ],
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            manifest = pd.read_parquet(
                root
                / "data"
                / "potentials"
                / "pot_smoke"
                / "generated"
                / "manifests"
                / "xrd_averages.parquet"
            )
            row = manifest.iloc[0]
            grid = pd.read_parquet(root / row["grid_path"])
            peaks = pd.read_parquet(root / row["peak_path"])

        self.assertEqual([0, 0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(1, len(manifest))
        self.assertEqual(2, len(row["source_xrd_sample_ids"]))
        self.assertGreater(len(grid), 0)
        self.assertGreater(len(peaks), 0)
        self.assertAlmostEqual(100.0, float(grid["intensity"].max()), places=6)
