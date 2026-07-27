import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class HeaBenchmarkReadinessTests(unittest.TestCase):
    def test_missing_scientific_evidence_returns_fixed_blocker_codes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output = root / "readiness.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alloy_xrd_bench.hea_benchmark_readiness",
                    "--evidence-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(2, result.returncode, result.stderr)
            report = json.loads(output.read_text())
            self.assertFalse(report["core_benchmark_ready"])
            self.assertIn("EVALUATOR_NOT_VALIDATED", report["failure_codes"])
            self.assertIn("PREDICTION_MATRIX_INCOMPLETE", report["failure_codes"])
            self.assertIn("OOD_EVALUATION_MISSING", report["failure_codes"])
            self.assertIn("BOOTSTRAP_EVIDENCE_MISSING", report["failure_codes"])
            self.assertIn("RELEASE_PACKAGE_INCOMPLETE", report["failure_codes"])

    def test_sro_core_release_is_not_blocked_by_diagnostic_tracks_or_model_gain(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            evidence = {
                "evaluator_smoke.json": {"status": "pass"},
                "controls.json": {"control_gate_passed": True},
                "prediction_completeness.json": {"status": "pass", "required_run_count": 84, "validated_run_count": 84, "missing_count": 0, "duplicate_count": 0, "unexpected_count": 0},
                "family_support.json": {"status": "diagnostic", "pending_review_count": 80},
                "identifiability.json": {"status": "diagnostic", "identifiable_parent_group_count": 0, "experimental_noise_calibrated": False},
                "model_comparison.json": {"passes_five_point_gate": False},
                "ood_summary.json": {"status": "pass"},
                "perturbation_summary.json": {"status": "pass", "seed_count": 5},
                "bootstrap_summary.json": {"status": "pass", "replicates": 2000, "seed": 2027},
                "release_package.json": {"status": "pass", "checksum_errors": 0, "license_errors": 0},
                "canonical_audit.json": {"status": "pass", "canonical_count": 84019},
            }
            for name, payload in evidence.items():
                (root / name).write_text(json.dumps(payload))
            output = root / "readiness.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alloy_xrd_bench.hea_benchmark_readiness",
                    "--evidence-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["core_benchmark_ready"])
            self.assertEqual("diagnostic_pending_review", report["diagnostic_family_status"])
            self.assertEqual("diagnostic_non_identifiable", report["diagnostic_occupancy_status"])
            self.assertFalse(report["experimental_noise_calibrated"])
            self.assertEqual([], report["failure_codes"])

    def test_missing_ood_blocks_core_release_with_fixed_code(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            evidence = {
                "canonical_audit.json": {"status": "pass", "canonical_count": 84019},
                "evaluator_smoke.json": {"status": "pass"},
                "controls.json": {"control_gate_passed": True},
                "prediction_completeness.json": {"status": "pass", "required_run_count": 84, "validated_run_count": 84, "missing_count": 0, "duplicate_count": 0, "unexpected_count": 0},
                "perturbation_summary.json": {"status": "pass", "seed_count": 5},
                "bootstrap_summary.json": {"status": "pass", "replicates": 2000, "seed": 2027},
                "release_package.json": {"status": "pass", "checksum_errors": 0, "license_errors": 0},
            }
            for name, payload in evidence.items():
                (root / name).write_text(json.dumps(payload))
            output = root / "readiness.json"
            result = subprocess.run(
                [sys.executable, "-m", "alloy_xrd_bench.hea_benchmark_readiness", "--evidence-root", str(root), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text())
            self.assertEqual(2, result.returncode)
            self.assertEqual(["OOD_EVALUATION_MISSING"], report["failure_codes"])

    def test_legacy_64_run_prediction_matrix_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            report = {
                "status": "pass", "required_run_count": 64, "validated_run_count": 64,
                "missing_count": 0, "duplicate_count": 0, "unexpected_count": 0,
            }
            (root / "prediction_completeness.json").write_text(json.dumps(report))

            from alloy_xrd_bench.hea_benchmark_readiness import assess_readiness
            readiness = assess_readiness(root)

            self.assertFalse(readiness["gates"]["prediction_matrix"])
            self.assertIn("PREDICTION_MATRIX_INCOMPLETE", readiness["failure_codes"])


if __name__ == "__main__":
    unittest.main()
