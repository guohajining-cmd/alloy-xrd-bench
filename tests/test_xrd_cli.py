import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml
from pymatgen.core import Lattice, Structure

from alloy_xrd_bench.contract import canonical_structure_hash
from alloy_xrd_bench.validate import validate_benchmark_root


class XrdCliTests(unittest.TestCase):
    def test_simulate_writes_grid_peaks_manifest_and_provenance(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            build_relaxed_workspace(root)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.xrd",
                    "simulate",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_test",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            manifest_path = (
                root
                / "data"
                / "potentials"
                / "pot_test"
                / "generated"
                / "manifests"
                / "xrd_samples.parquet"
            )
            xrd_rows = pd.read_parquet(manifest_path).to_dict(orient="records")
            grid_path = root / xrd_rows[0]["grid_path"]
            peak_path = root / xrd_rows[0]["peak_path"]
            grid = pd.read_parquet(grid_path)
            peaks = pd.read_parquet(peak_path)
            report = validate_benchmark_root(root)
            grid_exists = grid_path.exists()
            peak_exists = peak_path.exists()
            grid_len = len(grid)
            peaks_len = len(peaks)
            grid_max = float(grid["intensity"].max())

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual(1, len(xrd_rows))
        self.assertTrue(grid_exists)
        self.assertTrue(peak_exists)
        self.assertGreater(grid_len, 0)
        self.assertGreater(peaks_len, 0)
        self.assertAlmostEqual(100.0, grid_max, places=6)
        self.assertEqual(0, report["issue_count"])


def build_relaxed_workspace(root: Path) -> None:
    potential_id = "pot_test"
    paths = {
        "registry": root / "data" / "registry",
        "shared_manifests": root / "data" / "shared" / "manifests",
        "shared_structures": root / "data" / "shared" / "structures_initial",
        "potential": root / "data" / "potentials" / potential_id / "potential",
        "manifests": root / "data" / "potentials" / potential_id / "generated" / "manifests",
        "logs": root / "data" / "potentials" / potential_id / "generated" / "relaxation_logs",
        "relaxed": root / "data" / "potentials" / potential_id / "generated" / "structures_relaxed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    structure = Structure(
        Lattice.cubic(3.52),
        ["Ni", "Ni", "Ni", "Ni"],
        [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    )
    structure_doc = structure.as_dict()
    artifact_bytes = json.dumps(structure_doc, sort_keys=True, indent=2).encode("utf-8")
    cif_bytes = b"# cif\n"
    (paths["shared_structures"] / "init.json").write_bytes(artifact_bytes)
    (paths["relaxed"] / "rstr.json").write_bytes(artifact_bytes)
    (paths["relaxed"] / "rstr.cif").write_bytes(cif_bytes)
    (paths["logs"] / "relax.jsonl").write_text('{"step":1}\n', encoding="utf-8")
    (paths["potential"] / "metadata.yml").write_text("potential_id: pot_test\n", encoding="utf-8")

    catalog = {
        "potentials": [
            {
                "potential_id": potential_id,
                "family": "test",
                "engine": "identity",
                "elements": ["Ni"],
                "phases": ["fcc"],
                "source_url": "https://example.org/test",
                "doi": "10.0000/test",
                "license": "MIT",
                "file_sha256": "a" * 64,
                "redistribution_policy": "metadata_only",
                "citation_keys": ["test"],
                "potential_dir": f"data/potentials/{potential_id}/potential",
                "generated_data_dir": f"data/potentials/{potential_id}/generated",
            }
        ]
    }
    (paths["registry"] / "potentials_catalog.yml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "initial_structure_id": "init_Ni_fcc",
                "alloy_system": "Ni",
                "phase": "fcc",
                "composition": {"Ni": 1.0},
                "supercell": "1x1x1",
                "generation_method": "test",
                "seed": 1,
                "structure_sha256": canonical_structure_hash(structure_doc),
                "artifact_file_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "artifact_path": "data/shared/structures_initial/init.json",
            }
        ]
    ).to_parquet(paths["shared_manifests"] / "initial_structures.parquet")
    pd.DataFrame(
        [
            {
                "relaxation_run_id": "relax_Ni_fcc",
                "initial_structure_id": "init_Ni_fcc",
                "potential_id": potential_id,
                "engine": "identity",
                "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "converged": True,
                "fmax": 0.0,
                "stress": {"xx": 0.0},
                "volume_change": 0.0,
                "steps": 1,
                "log_path": f"data/potentials/{potential_id}/generated/relaxation_logs/relax.jsonl",
            }
        ]
    ).to_parquet(paths["manifests"] / "relaxations.parquet")
    pd.DataFrame(
        [
            {
                "relaxed_structure_id": "rstr_Ni_fcc",
                "relaxation_run_id": "relax_Ni_fcc",
                "structure_sha256": canonical_structure_hash(structure_doc),
                "artifact_json_file_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": f"data/potentials/{potential_id}/generated/structures_relaxed/rstr.json",
                "artifact_path_cif": f"data/potentials/{potential_id}/generated/structures_relaxed/rstr.cif",
                "quality_tier": "pass",
            }
        ]
    ).to_parquet(paths["manifests"] / "relaxed_structures.parquet")
    pd.DataFrame(
        [
            {
                "source_id": "init_Ni_fcc",
                "source_type": "initial_structure",
                "target_id": "relax_Ni_fcc",
                "target_type": "relaxation_run",
                "relation_type": "input_to",
            },
            {
                "source_id": potential_id,
                "source_type": "potential_file",
                "target_id": "relax_Ni_fcc",
                "target_type": "relaxation_run",
                "relation_type": "used_by",
            },
            {
                "source_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "source_type": "relax_config",
                "target_id": "relax_Ni_fcc",
                "target_type": "relaxation_run",
                "relation_type": "configured_by",
            },
            {
                "source_id": "relax_Ni_fcc",
                "source_type": "relaxation_run",
                "target_id": "rstr_Ni_fcc",
                "target_type": "relaxed_structure",
                "relation_type": "produced",
            },
        ]
    ).to_parquet(paths["manifests"] / "provenance_edges.parquet")
