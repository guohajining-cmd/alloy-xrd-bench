import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


class QualityCliTests(unittest.TestCase):
    def test_quality_audit_writes_physics_sanity_outputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            output_dir = root / "audits"
            recipe_path = Path(tempdir) / "recipes.yml"
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
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.quality",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            summary = json.loads((output_dir / "quality_summary.json").read_text(encoding="utf-8"))
            rows = pd.read_csv(output_dir / "physics_sanity.csv")

        self.assertEqual([0, 0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(0, summary["issue_count"])
        self.assertGreater(len(rows), 0)
        self.assertEqual({"pass"}, set(rows["status"]))
