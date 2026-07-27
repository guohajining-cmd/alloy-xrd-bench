import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure


REPO_ROOT = Path(__file__).resolve().parents[1]


class HierarchicalMiniCliTests(unittest.TestCase):
    def test_build_writes_hierarchical_tables_patterns_and_smoke_metrics(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            self._write_source(
                root,
                potential_id="pot_fcc",
                initial_id="init_CrCoNi_fcc_fixture",
                alloy_system="CrCoNi",
                phase="fcc",
                structure=Structure(
                    Lattice.cubic(3.6),
                    ["Cr", "Co", "Ni", "Ni"],
                    [
                        [0, 0, 0],
                        [0, 0.5, 0.5],
                        [0.5, 0, 0.5],
                        [0.5, 0.5, 0],
                    ],
                ),
            )
            self._write_source(
                root,
                potential_id="pot_bcc",
                initial_id="init_TiTaVW_bcc_fixture",
                alloy_system="TiTaVW",
                phase="bcc",
                structure=Structure(
                    Lattice.cubic(3.2),
                    ["Ti", "Ta"],
                    [[0, 0, 0], [0.5, 0.5, 0.5]],
                ),
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alloy_xrd_bench.hierarchical_mini",
                    "build",
                    "--root",
                    str(root),
                    "--output",
                    "data/v0.3_hierarchical_mini",
                    "--source",
                    "pot_fcc:CrCoNi:fcc",
                    "--source",
                    "pot_bcc:TiTaVW:bcc",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            output = root / "data" / "v0.3_hierarchical_mini"
            expected = {
                "xrd_samples.csv",
                "site_occupancy.csv",
                "component_peaks.csv",
                "tasks.csv",
                "provenance_edges.csv",
                "summary.json",
            }
            self.assertTrue(expected <= {path.name for path in output.iterdir()})

            xrd = pd.read_csv(output / "xrd_samples.csv")
            sites = pd.read_csv(output / "site_occupancy.csv")
            components = pd.read_csv(output / "component_peaks.csv")
            tasks = pd.read_csv(output / "tasks.csv")
            edges = pd.read_csv(output / "provenance_edges.csv")
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(4, len(xrd))
            self.assertEqual({"clean", "broadened"}, set(xrd["observation_kind"]))
            self.assertEqual({4001}, set(xrd["point_count"]))
            self.assertEqual(6, len(sites))
            self.assertTrue((sites["occupancy"] == 1.0).all())
            self.assertGreater(len(components), 0)
            self.assertEqual(
                {"phase_classification", "composition_regression", "lattice_parameter_regression"},
                set(tasks["task_name"]),
            )
            self.assertEqual(12, len(tasks))
            self.assertTrue(set(xrd["xrd_sample_id"]) <= set(edges["target_id"]))

            for pattern_path in xrd["pattern_path"]:
                with np.load(root / pattern_path) as pattern:
                    self.assertEqual((4001,), pattern["two_theta"].shape)
                    self.assertEqual((4001,), pattern["intensity"].shape)
                    self.assertTrue(np.isfinite(pattern["intensity"]).all())
                    self.assertGreaterEqual(float(pattern["intensity"].min()), 0.0)
                    self.assertAlmostEqual(100.0, float(pattern["intensity"].max()), places=5)

            self.assertEqual(2, summary["structure_count"])
            self.assertEqual(4, summary["xrd_sample_count"])
            self.assertFalse(summary["baseline"]["scientific_result"])
            self.assertEqual("engineering_smoke_only", summary["release_status"])
            self.assertFalse(
                summary["target_diagnostics"]["lattice_parameter_regression_nontrivial"]
            )
            self.assertEqual(
                {"CrCoNi/fcc": 1, "TiTaVW/bcc": 1},
                summary["target_diagnostics"]["lattice_parameter_unique_by_alloy_phase"],
            )

    @staticmethod
    def _write_source(
        root: Path,
        *,
        potential_id: str,
        initial_id: str,
        alloy_system: str,
        phase: str,
        structure: Structure,
    ) -> None:
        shared = root / "data" / "shared" / "manifests"
        shared.mkdir(parents=True, exist_ok=True)
        initial_path = shared / "initial_structures.parquet"
        initial_row = {
            "initial_structure_id": initial_id,
            "alloy_system": alloy_system,
            "phase": phase,
            "composition": structure.composition.get_el_amt_dict(),
        }
        if initial_path.exists():
            initial = pd.read_parquet(initial_path)
            initial = pd.concat([initial, pd.DataFrame([initial_row])], ignore_index=True)
        else:
            initial = pd.DataFrame([initial_row])
        initial.to_parquet(initial_path, index=False)

        generated = root / "data" / "potentials" / potential_id / "generated"
        manifests = generated / "manifests"
        structures = generated / "structures_relaxed"
        manifests.mkdir(parents=True, exist_ok=True)
        structures.mkdir(parents=True, exist_ok=True)
        relaxed_id = f"rstr_{potential_id}_{alloy_system}_{phase}_fixture"
        relaxation_id = f"relax_{potential_id}_{alloy_system}_{phase}_fixture"
        artifact = structures / f"{relaxed_id}.json"
        artifact.write_text(json.dumps(structure.as_dict()), encoding="utf-8")
        relative_artifact = artifact.relative_to(root).as_posix()

        pd.DataFrame(
            [
                {
                    "relaxation_run_id": relaxation_id,
                    "initial_structure_id": initial_id,
                    "potential_id": potential_id,
                    "engine": "fixture",
                    "converged": True,
                    "fmax": 0.01,
                }
            ]
        ).to_parquet(manifests / "relaxations.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "relaxed_structure_id": relaxed_id,
                    "relaxation_run_id": relaxation_id,
                    "structure_sha256": "a" * 64,
                    "artifact_path_json": relative_artifact,
                    "quality_tier": "pass",
                }
            ]
        ).to_parquet(manifests / "relaxed_structures.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
