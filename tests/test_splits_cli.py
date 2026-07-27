import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from alloy_xrd_bench.validate import validate_benchmark_root


class SplitsCliTests(unittest.TestCase):
    def test_apply_splits_updates_xrd_manifest_from_alloy_system_rules(self):
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
            split_path = root / "splits.yml"
            split_path.write_text(
                yaml.safe_dump(
                    {"splits": {"private_test": {"alloy_systems": ["Ni"]}}},
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
                    "alloy_xrd_bench.splits",
                    "apply",
                    "--root",
                    str(root),
                    "--config",
                    str(split_path),
                ],
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            xrd = pd.read_parquet(
                root
                / "data"
                / "potentials"
                / "pot_smoke"
                / "generated"
                / "manifests"
                / "xrd_samples.parquet"
            )
            report = validate_benchmark_root(root)

        self.assertEqual([0, 0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(["private_test"], xrd["split"].tolist())
        self.assertEqual(0, report["issue_count"])
