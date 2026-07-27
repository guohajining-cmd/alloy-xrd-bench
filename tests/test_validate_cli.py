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


class ValidateCliTests(unittest.TestCase):
    def test_cli_validates_minimal_potential_scoped_workspace(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            build_workspace(root)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.validate",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual(0, payload["issue_count"])
        self.assertEqual(1, payload["manifest_counts"]["potentials_catalog"])
        self.assertEqual(1, payload["manifest_counts"]["xrd_samples"])
        self.assertEqual(6, payload["manifest_counts"]["provenance_edges"])

    def test_cli_returns_nonzero_for_potential_scope_violation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            build_workspace(root, grid_path="data/potentials/pot_other/generated/xrd/grid/xrd.parquet")

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.validate",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(1, result.returncode)
        payload = json.loads(result.stdout)
        self.assertIn(
            "artifact_outside_potential_scope",
            {issue["code"] for issue in payload["issues"]},
        )

    def test_cli_returns_nonzero_for_unregistered_potential_directory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            build_workspace(root)
            (
                root
                / "data"
                / "potentials"
                / "pot_unregistered"
                / "generated"
                / "manifests"
            ).mkdir(parents=True)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.validate",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(1, result.returncode)
        payload = json.loads(result.stdout)
        self.assertIn(
            "unregistered_potential_directory",
            {issue["code"] for issue in payload["issues"]},
        )


def build_workspace(root: Path, grid_path: str | None = None) -> None:
    potential_id = "pot_mace_mp_0"
    paths = {
        "registry": root / "data" / "registry",
        "shared_manifests": root / "data" / "shared" / "manifests",
        "shared_structures": root / "data" / "shared" / "structures_initial",
        "potential_manifests": root
        / "data"
        / "potentials"
        / potential_id
        / "generated"
        / "manifests",
        "relax_logs": root / "data" / "potentials" / potential_id / "generated" / "relaxation_logs",
        "relaxed": root / "data" / "potentials" / potential_id / "generated" / "structures_relaxed",
        "xrd_grid": root / "data" / "potentials" / potential_id / "generated" / "xrd" / "grid",
        "xrd_peaks": root / "data" / "potentials" / potential_id / "generated" / "xrd" / "peaks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    initial_doc = {"initial": True}
    relaxed_doc = {"relaxed": True}
    initial_bytes = json.dumps(initial_doc, sort_keys=True, indent=2).encode("utf-8")
    relaxed_bytes = json.dumps(relaxed_doc, sort_keys=True, indent=2).encode("utf-8")
    cif_bytes = b"# cif\n"
    (paths["shared_structures"] / "init.json").write_bytes(initial_bytes)
    (paths["relaxed"] / "rstr.json").write_bytes(relaxed_bytes)
    (paths["relaxed"] / "rstr.cif").write_bytes(cif_bytes)
    (paths["relax_logs"] / "relax.jsonl").write_text('{"step":1}\n', encoding="utf-8")
    pd.DataFrame({"two_theta": [20.0], "intensity": [100.0]}).to_parquet(paths["xrd_grid"] / "xrd.parquet")
    pd.DataFrame({"two_theta": [20.0], "intensity": [100.0]}).to_parquet(paths["xrd_peaks"] / "peaks.parquet")

    catalog = {
        "potentials": [
            {
                "potential_id": potential_id,
                "family": "MACE",
                "engine": "mace",
                "elements": ["Cr", "Co", "Ni"],
                "phases": ["fcc"],
                "source_url": "https://example.org/mace",
                "doi": "10.0000/example",
                "license": "MIT",
                "file_sha256": "a" * 64,
                "redistribution_policy": "metadata_only",
                "citation_keys": ["mace"],
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
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "alloy_system": "CrCoNi",
                "phase": "fcc",
                "composition": {"Cr": 0.33, "Co": 0.33, "Ni": 0.34},
                "supercell": "4x4x4",
                "generation_method": "sqs_like",
                "seed": 7,
                "structure_sha256": canonical_structure_hash(initial_doc),
                "artifact_file_sha256": hashlib.sha256(initial_bytes).hexdigest(),
                "artifact_path": "data/shared/structures_initial/init.json",
            }
        ]
    ).to_parquet(paths["shared_manifests"] / "initial_structures.parquet")
    pd.DataFrame(
        [
            {
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "initial_structure_id": "init_CrCoNi_fcc_000001",
                "potential_id": potential_id,
                "engine": "mace",
                "relax_config_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "converged": True,
                "fmax": 0.03,
                "stress": {"xx": 0.0, "yy": 0.0, "zz": 0.0},
                "volume_change": 0.04,
                "steps": 45,
                "log_path": f"data/potentials/{potential_id}/generated/relaxation_logs/relax.jsonl",
            }
        ]
    ).to_parquet(paths["potential_manifests"] / "relaxations.parquet")
    pd.DataFrame(
        [
            {
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "relaxation_run_id": "relax_CrCoNi_fcc_000001",
                "structure_sha256": canonical_structure_hash(relaxed_doc),
                "artifact_json_file_sha256": hashlib.sha256(relaxed_bytes).hexdigest(),
                "artifact_cif_file_sha256": hashlib.sha256(cif_bytes).hexdigest(),
                "artifact_path_json": f"data/potentials/{potential_id}/generated/structures_relaxed/rstr.json",
                "artifact_path_cif": f"data/potentials/{potential_id}/generated/structures_relaxed/rstr.cif",
                "quality_tier": "pass",
            }
        ]
    ).to_parquet(paths["potential_manifests"] / "relaxed_structures.parquet")
    pd.DataFrame(
        [
            {
                "xrd_sample_id": "xrd_CrCoNi_fcc_000001",
                "relaxed_structure_id": "rstr_CrCoNi_fcc_000001",
                "xrd_config_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "two_theta_min": 20.0,
                "two_theta_max": 100.0,
                "step": 0.02,
                "wavelength": "Cu Kalpha",
                "broadening": "gaussian_fwhm_0p1",
                "grid_path": grid_path
                or f"data/potentials/{potential_id}/generated/xrd/grid/xrd.parquet",
                "peak_path": f"data/potentials/{potential_id}/generated/xrd/peaks/peaks.parquet",
                "split": "train",
            }
        ]
    ).to_parquet(paths["potential_manifests"] / "xrd_samples.parquet")
    pd.DataFrame(
        [
            {
                "source_id": "init_CrCoNi_fcc_000001",
                "source_type": "initial_structure",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "input_to",
            },
            {
                "source_id": potential_id,
                "source_type": "potential_file",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "used_by",
            },
            {
                "source_id": "relax_fmax_0p05_maxsteps_1000_v1",
                "source_type": "relax_config",
                "target_id": "relax_CrCoNi_fcc_000001",
                "target_type": "relaxation_run",
                "relation_type": "configured_by",
            },
            {
                "source_id": "relax_CrCoNi_fcc_000001",
                "source_type": "relaxation_run",
                "target_id": "rstr_CrCoNi_fcc_000001",
                "target_type": "relaxed_structure",
                "relation_type": "produced",
            },
            {
                "source_id": "rstr_CrCoNi_fcc_000001",
                "source_type": "relaxed_structure",
                "target_id": "xrd_CrCoNi_fcc_000001",
                "target_type": "xrd_sample",
                "relation_type": "simulated_as",
            },
            {
                "source_id": "xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1",
                "source_type": "xrd_config",
                "target_id": "xrd_CrCoNi_fcc_000001",
                "target_type": "xrd_sample",
                "relation_type": "configured_by",
            },
        ]
    ).to_parquet(paths["potential_manifests"] / "provenance_edges.parquet")
