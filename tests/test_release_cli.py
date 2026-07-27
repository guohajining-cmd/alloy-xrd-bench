import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd
import yaml


class ReleaseCliTests(unittest.TestCase):
    def test_release_export_hides_private_xrd_and_relaxed_artifacts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
            recipes = Path(tempdir) / "recipes.yml"
            splits = Path(tempdir) / "splits.yml"
            recipes.write_text(
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
            splits.write_text(
                yaml.safe_dump(
                    {"splits": {"private_test": {"alloy_systems": ["Al"]}}},
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
                    str(recipes),
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
                    "Al",
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
                    str(splits),
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
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release",
                    str(root),
                    "--output-dir",
                    str(release_root),
                    "--release-version",
                    "test",
                ],
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            public_xrd = pd.read_parquet(release_root / "manifests" / "xrd_samples.parquet")
            public_avg = pd.read_parquet(release_root / "manifests" / "xrd_averages.parquet")
            private_inputs = pd.read_parquet(release_root / "manifests" / "private_inputs.parquet")
            copied_xrd_files = list((release_root / "data" / "potentials").glob("**/xrd/**/*.parquet"))
            summary = json.loads((release_root / "release_summary.json").read_text(encoding="utf-8"))
            dataset_card_exists = (release_root / "DATASET_CARD.md").exists()
            citation_exists = (release_root / "CITATION.cff").exists()
            croissant_exists = (release_root / "croissant.jsonld").exists()
            schema_exists = (release_root / "schemas" / "xrd_samples.schema.json").exists()
            config_exists = (release_root / "configs" / "xrd_config_default.json").exists()
            completion_audit_exists = (release_root / "docs" / "COMPLETION_AUDIT.md").exists()

        self.assertEqual([0, 0, 0, 0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(["id"], public_xrd["split"].tolist())
        self.assertEqual(["id"], public_avg["split"].tolist())
        self.assertEqual(["private_test"], private_inputs["split"].tolist())
        self.assertEqual({"Al"}, set(private_inputs["alloy_system"]))
        self.assertEqual({"Al": 1.0}, private_inputs.iloc[0]["composition"])
        self.assertEqual(4, len(copied_xrd_files))
        self.assertTrue(all("Al" not in str(path) for path in copied_xrd_files))
        self.assertTrue(dataset_card_exists)
        self.assertTrue(citation_exists)
        self.assertTrue(croissant_exists)
        self.assertTrue(schema_exists)
        self.assertTrue(config_exists)
        self.assertTrue(completion_audit_exists)
        self.assertEqual(1, summary["public_xrd_samples"])
        self.assertEqual(1, summary["private_inputs"])
