from pathlib import Path
import unittest


class KddSubmissionRunnerTests(unittest.TestCase):
    def test_runner_requires_all_submission_evidence(self):
        text = Path("scripts/workstation/audit_kdd_submission.sh").read_text()

        for argument in (
            "--hea-readiness", "--gamma-readiness", "--paper",
            "--hea-paper-manifest", "--gamma-paper-manifest",
            "--release-smoke", "--environment-lock", "--external-sanity",
            "--publication-record", "--require-submission",
        ):
            self.assertIn(argument, text)


if __name__ == "__main__":
    unittest.main()
