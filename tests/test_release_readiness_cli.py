import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class ReleaseReadinessCliTests(unittest.TestCase):
    def test_release_readiness_passes_packaged_release(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
            archive_path = Path(tempdir) / "release.zip"
            output_path = Path(tempdir) / "readiness.json"
            commands = [
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
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
            (release_root / "audits").mkdir(parents=True, exist_ok=True)
            (release_root / "audits" / "quality_summary.json").write_text(
                json.dumps(
                    {
                        "root": str(release_root),
                        "issue_count": 0,
                        "load_issue_count": 0,
                        "check_count": 1,
                        "status_counts": {"pass": 1},
                    },
                    sort_keys=True,
                ),
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
            readiness = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release_readiness",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                    "--json",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(readiness.stdout)
            written_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([0, 0], [result.returncode for result in results])
        self.assertEqual(0, package_result.returncode)
        self.assertEqual("", readiness.stderr)
        self.assertEqual(0, readiness.returncode)
        self.assertEqual(payload, written_payload)
        self.assertEqual("pass", payload["status"])
        self.assertEqual([], payload["issues"])
        self.assertGreater(payload["archive"]["zip_entries"], 0)
        self.assertEqual(0, payload["validation"]["issue_count"])
        self.assertEqual(0, payload["quality"]["issue_count"])

    def test_release_readiness_fails_when_archive_is_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
            archive_path = Path(tempdir) / "missing.zip"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
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
                check=True,
                capture_output=True,
                text=True,
            )
            readiness = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release_readiness",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(readiness.stdout)

        self.assertEqual(1, readiness.returncode)
        self.assertEqual("fail", payload["status"])
        self.assertIn("missing_archive", {issue["code"] for issue in payload["issues"]})

    def test_release_readiness_fails_for_metadata_only_potential_without_install_notes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            release_root = Path(tempdir) / "release"
            archive_path = Path(tempdir) / "release.zip"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/make_minimal_workspace.py",
                    "--root",
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
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
                check=True,
                capture_output=True,
                text=True,
            )
            (release_root / "audits" / "quality_summary.json").write_text(
                json.dumps({"issue_count": 0, "check_count": 1}),
                encoding="utf-8",
            )
            install_path = (
                release_root
                / "data"
                / "potentials"
                / "pot_mace_mp_0"
                / "potential"
                / "INSTALL.md"
            )
            if install_path.exists():
                install_path.unlink()
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.package",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            readiness = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release_readiness",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(readiness.stdout)

        self.assertEqual(1, readiness.returncode)
        self.assertIn(
            "missing_metadata_only_install_instructions",
            {issue["code"] for issue in payload["issues"]},
        )

    def test_release_readiness_requires_publication_record_when_requested(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root, release_root, archive_path = build_packaged_release(Path(tempdir))
            publication_record = release_root / "publication_record.yml"
            publication_record.write_text(
                yaml.safe_dump(
                    {
                        "archive": "zenodo",
                        "doi": "",
                        "dataset_url": "",
                        "release_date": "",
                        "code_repository_url": "",
                        "release_archive_sha256": _sha256(archive_path),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            readiness = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release_readiness",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                    "--publication-record",
                    str(publication_record),
                    "--require-publication",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(readiness.stdout)

        self.assertEqual(1, readiness.returncode)
        self.assertIn("missing_publication_field", {issue["code"] for issue in payload["issues"]})

    def test_release_readiness_accepts_complete_publication_record(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, release_root, archive_path = build_packaged_release(Path(tempdir))
            publication_record = release_root / "publication_record.yml"
            publication_record.write_text(
                yaml.safe_dump(
                    {
                        "archive": "zenodo",
                        "doi": "10.5281/zenodo.1234567",
                        "dataset_url": "https://zenodo.org/records/1234567",
                        "release_date": "2026-07-02",
                        "code_repository_url": "https://github.com/example/alloy-xrd-bench",
                        "release_archive_sha256": _sha256(archive_path),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            readiness = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.release_readiness",
                    str(release_root),
                    "--archive",
                    str(archive_path),
                    "--publication-record",
                    str(publication_record),
                    "--require-publication",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(readiness.stdout)

        self.assertEqual(0, readiness.returncode)
        self.assertEqual("pass", payload["status"])
        self.assertEqual("zenodo", payload["publication"]["archive"])
        self.assertTrue(payload["publication"]["archive_sha256_matches"])


if __name__ == "__main__":
    unittest.main()


def build_packaged_release(tempdir: Path) -> tuple[Path, Path, Path]:
    root = tempdir / "workspace"
    release_root = tempdir / "release"
    archive_path = tempdir / "release.zip"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/make_minimal_workspace.py",
            "--root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    (release_root / "audits").mkdir(parents=True, exist_ok=True)
    (release_root / "audits" / "quality_summary.json").write_text(
        json.dumps({"root": str(release_root), "issue_count": 0, "check_count": 1}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "alloy_xrd_bench.package",
            str(release_root),
            "--archive",
            str(archive_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return root, release_root, archive_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
