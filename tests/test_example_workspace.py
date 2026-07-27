import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from alloy_xrd_bench.validate import validate_benchmark_root


class ExampleWorkspaceTests(unittest.TestCase):
    def test_minimal_workspace_script_generates_valid_workspace(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "minimal_workspace"
            result = subprocess.run(
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

            report = validate_benchmark_root(root)

        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual(0, report["issue_count"])
        self.assertEqual(1, report["manifest_counts"]["potentials_catalog"])
        self.assertEqual(1, report["manifest_counts"]["xrd_samples"])
        self.assertEqual(6, report["manifest_counts"]["provenance_edges"])
