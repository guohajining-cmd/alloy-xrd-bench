import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class PotentialsCliTests(unittest.TestCase):
    def test_register_copies_file_and_updates_catalog_idempotently(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source-model.pt"
            payload = b"example model bytes"
            source.write_bytes(payload)
            expected_sha = hashlib.sha256(payload).hexdigest()

            command = [
                sys.executable,
                "-B",
                "-m",
                "alloy_xrd_bench.potentials",
                "register",
                "--root",
                str(root),
                "--potential-id",
                "pot_test",
                "--family",
                "MACE",
                "--engine",
                "mace",
                "--elements",
                "Cr",
                "Co",
                "Ni",
                "--phases",
                "fcc",
                "--source-url",
                "https://example.org/model",
                "--doi",
                "10.0000/test",
                "--license",
                "MIT",
                "--redistribution-policy",
                "redistributable",
                "--citation-key",
                "test_mlip",
                "--file",
                str(source),
            ]

            first = subprocess.run(command, check=False, capture_output=True, text=True)
            second = subprocess.run(command, check=False, capture_output=True, text=True)

            catalog = yaml.safe_load(
                (root / "data" / "registry" / "potentials_catalog.yml").read_text(encoding="utf-8")
            )
            copied = (
                root
                / "data"
                / "potentials"
                / "pot_test"
                / "potential"
                / "files"
                / "source-model.pt"
            )
            checksums = (
                root
                / "data"
                / "potentials"
                / "pot_test"
                / "potential"
                / "checksums.sha256"
            ).read_text(encoding="utf-8")
            copied_exists = copied.exists()

        self.assertEqual("", first.stderr)
        self.assertEqual(0, first.returncode)
        self.assertEqual("", second.stderr)
        self.assertEqual(0, second.returncode)
        self.assertEqual(1, len(catalog["potentials"]))
        self.assertEqual(expected_sha, catalog["potentials"][0]["file_sha256"])
        self.assertTrue(copied_exists)
        self.assertIn(expected_sha, checksums)
        self.assertIn("files/source-model.pt", checksums)

    def test_register_metadata_only_potential_without_copying_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "register",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_external",
                    "--family",
                    "CHGNet",
                    "--engine",
                    "chgnet",
                    "--elements",
                    "Ti",
                    "Ta",
                    "V",
                    "W",
                    "--phases",
                    "bcc",
                    "--source-url",
                    "https://example.org/external",
                    "--doi",
                    "10.0000/external",
                    "--license",
                    "custom",
                    "--redistribution-policy",
                    "metadata_only",
                    "--citation-key",
                    "external_mlip",
                    "--file-sha256",
                    "b" * 64,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            catalog = yaml.safe_load(
                (root / "data" / "registry" / "potentials_catalog.yml").read_text(encoding="utf-8")
            )
            files_dir = root / "data" / "potentials" / "pot_external" / "potential" / "files"
            install_text = (
                root
                / "data"
                / "potentials"
                / "pot_external"
                / "potential"
                / "INSTALL.md"
            ).read_text(encoding="utf-8")

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual("metadata_only", catalog["potentials"][0]["redistribution_policy"])
        self.assertEqual("b" * 64, catalog["potentials"][0]["file_sha256"])
        self.assertFalse(files_dir.exists())
        self.assertIn("https://example.org/external", install_text)
        self.assertIn("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", install_text)

    def test_register_copies_attribution_redistributable_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "nep.txt"
            payload = b"example redistributable model"
            source.write_bytes(payload)
            expected_sha = hashlib.sha256(payload).hexdigest()

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.potentials",
                    "register",
                    "--root",
                    str(root),
                    "--potential-id",
                    "pot_ccby",
                    "--family",
                    "NEP",
                    "--engine",
                    "nep",
                    "--elements",
                    "Cr",
                    "Ta",
                    "V",
                    "W",
                    "--phases",
                    "bcc",
                    "--source-url",
                    "https://example.org/ccby",
                    "--doi",
                    "10.0000/ccby",
                    "--license",
                    "CC-BY-4.0",
                    "--redistribution-policy",
                    "redistributable_with_attribution",
                    "--citation-key",
                    "ccby_mlip",
                    "--file",
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            catalog = yaml.safe_load(
                (root / "data" / "registry" / "potentials_catalog.yml").read_text(encoding="utf-8")
            )
            copied = root / "data" / "potentials" / "pot_ccby" / "potential" / "files" / "nep.txt"
            checksums = (
                root / "data" / "potentials" / "pot_ccby" / "potential" / "checksums.sha256"
            ).read_text(encoding="utf-8")
            copied_exists = copied.exists()

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertTrue(copied_exists)
        self.assertEqual(expected_sha, catalog["potentials"][0]["file_sha256"])
        self.assertEqual("redistributable_with_attribution", catalog["potentials"][0]["redistribution_policy"])
        self.assertIn("files/nep.txt", checksums)
