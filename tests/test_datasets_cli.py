import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


class DatasetCliTests(unittest.TestCase):
    def test_cfg_section_detection_accepts_header_columns(self):
        from alloy_xrd_bench.datasets import _cfg_block_has_section

        block = [
            " Energy",
            " PlusStress:  xx yy zz yz xz xy",
        ]

        self.assertTrue(_cfg_block_has_section(block, "Energy"))
        self.assertTrue(_cfg_block_has_section(block, "PlusStress"))

    def test_dataset_audit_validates_required_fields_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            registry = root / "data" / "registry"
            registry.mkdir(parents=True)
            (registry / "mlip_datasets.yml").write_text(
                yaml.safe_dump(
                    {
                        "datasets": [
                            _dataset_row("ds_one"),
                            {**_dataset_row("ds_one"), "source_url": ""},
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.datasets",
                    "audit",
                    "--root",
                    str(root),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(0, result.returncode)
            payload = json.loads(result.stdout)
            self.assertEqual(2, payload["dataset_count"])
            codes = {issue["code"] for issue in payload["issues"]}
            self.assertIn("duplicate_dataset_id", codes)
            self.assertIn("missing_required_field", codes)

    def test_download_dataset_uses_safe_storage_policy_by_license(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source_dir = root / "sources"
            source_dir.mkdir()
            source_file = source_dir / "sample.extxyz"
            source_file.write_text(_extxyz_text(), encoding="utf-8")
            _write_dataset_registry(
                root,
                [
                    _dataset_row(
                        "ds_open",
                        license="CC-BY-4.0",
                        redistribution_policy="redistributable_with_attribution",
                        raw_files=[
                            {
                                "filename": "sample.extxyz",
                                "url": source_file.as_uri(),
                                "role": "training",
                            }
                        ],
                    ),
                    _dataset_row(
                        "ds_unknown",
                        license="unknown",
                        redistribution_policy="metadata_only_until_license_resolved",
                        raw_files=[
                            {
                                "filename": "sample.extxyz",
                                "url": source_file.as_uri(),
                                "role": "training",
                            }
                        ],
                    ),
                ],
            )

            open_result = _run_dataset_cli(root, "download", "--dataset-id", "ds_open")
            unknown_result = _run_dataset_cli(root, "download", "--dataset-id", "ds_unknown")

            self.assertEqual("pass", open_result["status"])
            self.assertEqual("pass", unknown_result["status"])
            self.assertTrue((root / "data" / "mlip_datasets" / "ds_open" / "raw" / "sample.extxyz").exists())
            self.assertTrue((root / "local_dataset_cache" / "ds_unknown" / "raw" / "sample.extxyz").exists())
            self.assertFalse((root / "data" / "mlip_datasets" / "ds_unknown" / "raw").exists())

    def test_ingest_extxyz_writes_structure_manifest_with_optional_labels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset_raw = root / "data" / "mlip_datasets" / "ds_extxyz" / "raw"
            dataset_raw.mkdir(parents=True)
            (dataset_raw / "sample.extxyz").write_text(_extxyz_text(), encoding="utf-8")
            _write_dataset_registry(
                root,
                [
                    _dataset_row(
                        "ds_extxyz",
                        raw_files=[{"filename": "sample.extxyz", "role": "training"}],
                    )
                ],
            )

            report = _run_dataset_cli(root, "ingest", "--dataset-id", "ds_extxyz")

            self.assertEqual("pass", report["status"])
            structures = pd.read_parquet(
                root
                / "data"
                / "mlip_datasets"
                / "ds_extxyz"
                / "manifests"
                / "structures.parquet"
            )
            self.assertEqual(1, len(structures))
            row = structures.iloc[0].to_dict()
            self.assertEqual("ds_extxyz", row["dataset_id"])
            self.assertEqual(2, row["natoms"])
            self.assertTrue(row["xrd_eligible"])
            self.assertTrue(row["has_energy"])
            self.assertTrue((root / row["artifact_path_json"]).exists())

    def test_ingest_cfg_requires_type_map_and_records_parse_failure(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset_raw = root / "data" / "mlip_datasets" / "ds_cfg" / "raw"
            dataset_raw.mkdir(parents=True)
            (dataset_raw / "train.cfg").write_text(_minimal_cfg_text(), encoding="utf-8")
            _write_dataset_registry(
                root,
                [_dataset_row("ds_cfg", raw_files=[{"filename": "train.cfg", "role": "training"}])],
            )

            report = _run_dataset_cli(root, "ingest", "--dataset-id", "ds_cfg")

            self.assertEqual("partial", report["status"])
            parse_report = json.loads(
                (
                    root
                    / "data"
                    / "mlip_datasets"
                    / "ds_cfg"
                    / "manifests"
                    / "parse_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("missing_type_map", parse_report["files"][0]["code"])
            structures = pd.read_parquet(
                root / "data" / "mlip_datasets" / "ds_cfg" / "manifests" / "structures.parquet"
            )
            self.assertEqual(0, len(structures))

    def test_ingest_lightpfp_h5_extracts_structure_and_optional_labels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            dataset_raw = root / "data" / "mlip_datasets" / "ds_h5" / "raw"
            dataset_raw.mkdir(parents=True)
            _write_lightpfp_h5(dataset_raw / "sample.h5")
            _write_dataset_registry(
                root,
                [_dataset_row("ds_h5", raw_files=[{"filename": "sample.h5", "role": "training"}])],
            )

            report = _run_dataset_cli(root, "ingest", "--dataset-id", "ds_h5")

            self.assertEqual("pass", report["status"])
            structures = pd.read_parquet(
                root / "data" / "mlip_datasets" / "ds_h5" / "manifests" / "structures.parquet"
            )
            self.assertEqual(1, len(structures))
            row = structures.iloc[0].to_dict()
            self.assertEqual({"Al", "Ni"}, set(row["elements"]))
            self.assertTrue(row["has_energy"])
            self.assertTrue(row["has_forces"])
            self.assertTrue(row["has_stress"])
            self.assertTrue(row["xrd_eligible"])


def _run_dataset_cli(root: Path, *args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "alloy_xrd_bench.datasets", *args, "--root", str(root), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"CLI failed unexpectedly\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout)


def _dataset_row(
    dataset_id: str,
    *,
    license: str = "CC-BY-4.0",
    redistribution_policy: str = "redistributable_with_attribution",
    raw_files: list[dict] | None = None,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "source_url": "https://example.org/dataset",
        "doi": "10.0000/example",
        "license": license,
        "redistribution_policy": redistribution_policy,
        "target_systems": ["Al-Ni"],
        "formats": ["extxyz"],
        "local_paths": {},
        "related_potential_ids": [],
        "status": "ready",
        "raw_files": raw_files or [],
    }


def _write_dataset_registry(root: Path, rows: list[dict]) -> None:
    registry = root / "data" / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "mlip_datasets.yml").write_text(
        yaml.safe_dump({"datasets": rows}, sort_keys=False),
        encoding="utf-8",
    )


def _extxyz_text() -> str:
    return (
        "2\n"
        "Lattice=\"3.0 0 0 0 3.0 0 0 0 3.0\" "
        "Properties=species:S:1:pos:R:3 energy=-1.5 pbc=\"T T T\"\n"
        "Al 0.0 0.0 0.0\n"
        "Ni 1.5 1.5 1.5\n"
    )


def _minimal_cfg_text() -> str:
    return """BEGIN_CFG
 Size
    1
 Supercell
    3.0 0.0 0.0
    0.0 3.0 0.0
    0.0 0.0 3.0
 AtomData:  id type cartes_x cartes_y cartes_z fx fy fz
    1 0 0.0 0.0 0.0 0.0 0.0 0.0
 Energy
    -1.0
END_CFG
"""


def _write_lightpfp_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("0")
        group.create_dataset("cell", data=np.eye(3) * 3.0)
        group.create_dataset("positions", data=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]))
        group.create_dataset("numbers", data=np.array([13, 28]))
        group.create_dataset("forces", data=np.zeros((2, 3)))
        group.create_dataset("potential_energy", data=np.array(-1.5))
        group.create_dataset("stress", data=np.zeros(6))


if __name__ == "__main__":
    unittest.main()
