import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from alloy_xrd_bench.validate import validate_benchmark_root


def chgnet_available() -> bool:
    try:
        import chgnet  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(chgnet_available(), "CHGNet is not installed")
class ChgnetCliTests(unittest.TestCase):
    def test_chgnet_relaxation_writes_valid_manifests(self):
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
                    "pot_chgnet_0p3p0",
                    "--family",
                    "CHGNet",
                    "--engine",
                    "chgnet",
                    "--elements",
                    "Ni",
                    "--phases",
                    "fcc",
                    "--source-url",
                    "https://github.com/CederGroupHub/chgnet",
                    "--doi",
                    "10.1038/s42256-023-00716-3",
                    "--license",
                    "BSD-3-Clause",
                    "--redistribution-policy",
                    "metadata_only",
                    "--citation-key",
                    "chgnet",
                    "--file-sha256",
                    "0" * 64,
                ],
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.relax",
                    "chgnet",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_chgnet_0p3p0",
                    "--steps",
                    "1",
                    "--fmax",
                    "1.0",
                    "--no-relax-cell",
                    "--use-device",
                    "cpu",
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
                    "pot_chgnet_0p3p0",
                ],
            ]
            results = [
                subprocess.run(command, check=False, capture_output=True, text=True)
                for command in commands
            ]
            report = validate_benchmark_root(root)

        self.assertEqual([0, 0, 0, 0], [result.returncode for result in results])
        self.assertEqual(0, report["issue_count"])
        self.assertEqual(1, report["manifest_counts"]["relaxations"])
        self.assertEqual(1, report["manifest_counts"]["relaxed_structures"])
        self.assertEqual(1, report["manifest_counts"]["xrd_samples"])
