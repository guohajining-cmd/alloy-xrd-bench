from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ReleaseSmokeTests(unittest.TestCase):
    def test_workstation_runner_imports_from_configured_project_root(self):
        script = Path("scripts/workstation/run_release_smoke.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('cd "$project_root"', script)

    @patch("alloy_xrd_bench.release_smoke.run_public_no_spectrum_prior")
    @patch("alloy_xrd_bench.release_smoke.score_submission")
    @patch("alloy_xrd_bench.release_smoke.train_composition_ridge")
    @patch("alloy_xrd_bench.release_smoke.validate_public_release")
    def test_smoke_requires_both_public_evaluators(
        self, validate_gamma, train_hea, score_hea, score_gamma
    ):
        from alloy_xrd_bench.release_smoke import run_release_smoke

        validate_gamma.return_value = {"status": "pass"}
        train_hea.return_value = {"submission_path": "submission"}
        score_hea.return_value = {"status": "pass"}
        score_gamma.return_value = {"status": "pass"}
        with tempfile.TemporaryDirectory() as tempdir:
            report = run_release_smoke("hea", "config", "gamma", Path(tempdir))

            self.assertEqual("pass", report["status"])
            self.assertEqual("engineering-smoke", report["qualification"])
            self.assertEqual("pass", report["hea"]["score_status"])
            self.assertEqual("pass", report["gamma_decomp"]["score_status"])
            self.assertTrue((Path(tempdir) / "release_smoke.json").exists())
            score_gamma.assert_called_once_with(
                release_dir=Path("gamma").resolve(),
                output_dir=Path(tempdir).resolve() / "gamma_no_spectrum",
                task="window",
                split="validation",
            )


if __name__ == "__main__":
    unittest.main()
