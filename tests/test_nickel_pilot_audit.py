from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from alloy_xrd_bench.structures import generate_initial_structures


class NickelPilotAuditTests(unittest.TestCase):
    def test_audit_writes_reports_for_valid_gamma_pair(self):
        from alloy_xrd_bench.nickel_audit import run_nickel_pilot_audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = self._write_recipes(root, pair_count=1)
            generate_initial_structures(root=root, recipes_path=recipe_path)

            report = run_nickel_pilot_audit(
                root=root,
                output_dir=root / "audits" / "nickel_pilot",
                include_relaxed=False,
            )

            self.assertEqual(1, report["initial_pairs"]["pair_count"])
            self.assertEqual(1, report["initial_pairs"]["unique_gamma_prime_structures"])
            self.assertEqual(0, report["checks"]["fail"])
            self.assertTrue((root / "audits" / "nickel_pilot" / "checks.csv").exists())
            self.assertTrue((root / "audits" / "nickel_pilot" / "report.md").exists())
            saved = json.loads((root / "audits" / "nickel_pilot" / "summary.json").read_text())
            self.assertEqual(report["initial_pairs"], saved["initial_pairs"])

    def test_audit_reports_duplicate_gamma_prime_seed_structures(self):
        from alloy_xrd_bench.nickel_audit import run_nickel_pilot_audit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe_path = self._write_recipes(root, pair_count=2)
            generate_initial_structures(root=root, recipes_path=recipe_path)

            report = run_nickel_pilot_audit(
                root=root,
                output_dir=root / "audits" / "nickel_pilot",
                include_relaxed=False,
            )

            self.assertEqual(2, report["initial_pairs"]["pair_count"])
            self.assertEqual(1, report["initial_pairs"]["unique_gamma_prime_structures"])
            self.assertEqual(1, report["initial_pairs"]["duplicate_gamma_prime_count"])
            self.assertGreater(report["checks"]["warn"], 0)

    @staticmethod
    def _write_recipes(root: Path, *, pair_count: int) -> Path:
        recipes = []
        for offset in range(pair_count):
            seed = 23 + offset
            pair_id = f"pair_Ni-Al_seed{seed:06d}"
            recipes.extend(
                [
                    {
                        "alloy_system": "Ni-Al",
                        "phase": "fcc",
                        "phase_role": "gamma",
                        "structure_family": "gamma_gamma_prime",
                        "pair_id": pair_id,
                        "nominal_alloy_system": "Ni-Al",
                        "partitioning_rule": "test pair",
                        "composition": {"Ni": 9, "Al": 1},
                        "supercell": [2, 2, 2],
                        "lattice_a": 3.57,
                        "seed": seed,
                        "generation_method": "pair_balance_occupancy",
                        "candidate_count": 4,
                    },
                    {
                        "alloy_system": "Ni-Al",
                        "phase": "l12",
                        "phase_role": "gamma_prime",
                        "structure_family": "gamma_gamma_prime",
                        "pair_id": pair_id,
                        "nominal_alloy_system": "Ni-Al",
                        "partitioning_rule": "test pair",
                        "composition": {"Ni": 3, "Al": 1},
                        "sublattice_composition": {
                            "corner": {"Al": 1},
                            "face": {"Ni": 1},
                        },
                        "supercell": [2, 2, 2],
                        "lattice_a": 3.57,
                        "seed": seed,
                        "generation_method": "l12_sublattice_occupancy",
                    },
                ]
            )
        path = root / "recipes.yml"
        path.write_text(yaml.safe_dump({"recipes": recipes}, sort_keys=False), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
