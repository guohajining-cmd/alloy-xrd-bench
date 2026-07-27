import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from alloy_xrd_bench.validate import validate_benchmark_root


class PipelineSmokeTests(unittest.TestCase):
    def test_recipe_to_smoke_relax_to_xrd_pipeline_validates(self):
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
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            report = validate_benchmark_root(root)

        self.assertEqual([], [result.stderr for result in results if result.stderr])
        self.assertEqual([0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(0, report["issue_count"])
        self.assertEqual(1, report["manifest_counts"]["initial_structures"])
        self.assertEqual(1, report["manifest_counts"]["relaxations"])
        self.assertEqual(1, report["manifest_counts"]["relaxed_structures"])
        self.assertEqual(1, report["manifest_counts"]["xrd_samples"])
