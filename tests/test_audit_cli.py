import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class AuditCliTests(unittest.TestCase):
    def test_audit_writes_license_and_leakage_csvs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            output_dir = root / "audits"
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
            audit_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.audit",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            license_rows = pd.read_csv(output_dir / "license_audit.csv").to_dict(orient="records")
            leakage_rows = pd.read_csv(output_dir / "leakage_audit.csv").to_dict(orient="records")

        self.assertEqual("", make_result.stderr)
        self.assertEqual(0, make_result.returncode)
        self.assertEqual("", audit_result.stderr)
        self.assertEqual(0, audit_result.returncode)
        self.assertEqual("pot_mace_mp_0", license_rows[0]["potential_id"])
        self.assertEqual("metadata_only", license_rows[0]["release_action"])
        self.assertEqual("pass", leakage_rows[0]["status"])
