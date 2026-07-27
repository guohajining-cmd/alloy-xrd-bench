import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


class PackageCliTests(unittest.TestCase):
    def test_package_release_writes_manifest_checksums_and_zip(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
            archive_path = Path(tempdir) / "release.zip"
            make_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            release_result = subprocess.run(
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
                check=False,
                capture_output=True,
                text=True,
            )
            (release_root / "publication_record.yml").write_text(
                "release_archive_sha256: placeholder\n",
                encoding="utf-8",
            )
            package_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.package",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            manifest = pd.read_parquet(release_root / "file_manifest.parquet")
            checksums = (release_root / "checksums.sha256").read_text(encoding="utf-8")
            first_row = manifest.iloc[0]
            first_file = release_root / first_row["path"]
            actual_sha = _sha256(first_file)
            zip_names = set(zipfile.ZipFile(archive_path).namelist())

        self.assertEqual(0, make_result.returncode)
        self.assertEqual(0, release_result.returncode)
        self.assertEqual(0, package_result.returncode)
        self.assertGreater(len(manifest), 0)
        self.assertEqual(actual_sha, first_row["sha256"])
        self.assertIn(f"{actual_sha}  {first_row['path']}", checksums)
        self.assertIn("release_summary.json", zip_names)
        self.assertIn("checksums.sha256", zip_names)
        self.assertIn("file_manifest.parquet", zip_names)
        self.assertNotIn("publication_record.yml", zip_names)
        self.assertNotIn("publication_record.yml", set(manifest["path"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
