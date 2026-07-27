import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MetadataCliTests(unittest.TestCase):
    def test_croissant_metadata_includes_release_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
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
            metadata_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.metadata",
                    "croissant",
                    str(release_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads((release_root / "croissant.jsonld").read_text(encoding="utf-8"))
            file_ids = {item["@id"] for item in payload["fileObjects"]}

        self.assertEqual([0, 0, 0], [make_result.returncode, release_result.returncode, metadata_result.returncode])
        self.assertEqual("test", payload["version"])
        self.assertIn("manifests/xrd_samples.parquet", file_ids)
        self.assertIn("manifests/private_inputs.parquet", file_ids)
        self.assertNotIn("croissant.jsonld", file_ids)
        self.assertNotIn("file_manifest.parquet", file_ids)
        self.assertNotIn("checksums.sha256", file_ids)
        self.assertNotIn("publication_record.yml", file_ids)
