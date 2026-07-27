import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class StatusCliTests(unittest.TestCase):
    def test_status_cli_reports_validation_storage_environment_and_potentials(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "workspace"
            output_path = Path(tempdir) / "status.json"
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
            status_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "alloy_xrd_bench.status",
                    str(root),
                    "--json",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(status_result.stdout)
            written_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, make_result.returncode)
        self.assertEqual("", status_result.stderr)
        self.assertEqual(0, status_result.returncode)
        self.assertEqual(payload, written_payload)
        self.assertEqual(0, payload["validation"]["issue_count"])
        self.assertEqual(["pot_mace_mp_0"], payload["storage"]["potential_directories"])
        self.assertEqual([], payload["storage"]["unregistered_potential_directories"])
        self.assertEqual(1, len(payload["potentials"]))
        self.assertEqual("pot_mace_mp_0", payload["potentials"][0]["potential_id"])
        self.assertEqual(1, payload["potentials"][0]["manifest_counts"]["xrd_samples"])
        self.assertIn("pandas", payload["environment"]["packages"])
        self.assertFalse(payload["git"]["is_repository"])
        self.assertIsNone(payload["git"]["head"])


if __name__ == "__main__":
    unittest.main()
