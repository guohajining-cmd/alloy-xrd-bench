import json
from pathlib import Path
import tempfile
import unittest


class BenchmarkReadinessTests(unittest.TestCase):
    def test_workstation_finalizer_uses_both_track_readiness_files(self):
        script = Path("scripts/workstation/finalize_sitexrd_v1_1.sh")

        self.assertTrue(script.exists())
        text = script.read_text(encoding="utf-8")
        self.assertIn("hea_benchmark_readiness.json", text)
        self.assertIn("gamma_benchmark_readiness.json", text)
        self.assertIn("alloy_xrd_bench.benchmark_readiness", text)

    def test_complete_benchmark_requires_both_independent_tracks(self):
        from alloy_xrd_bench.benchmark_readiness import assess_benchmark_readiness

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            hea = root / "hea.json"
            gamma = root / "gamma.json"
            hea.write_text(json.dumps({
                "core_benchmark_ready": True,
                "failure_codes": [],
                "diagnostic_family_status": "diagnostic_pending_review",
                "diagnostic_occupancy_status": "diagnostic_non_identifiable",
            }), encoding="utf-8")
            gamma.write_text(json.dumps({
                "gamma_benchmark_ready": False,
                "failure_codes": ["INCOMPLETE_EXPERIMENT_MATRIX"],
            }), encoding="utf-8")

            blocked = assess_benchmark_readiness(hea, gamma, root / "blocked.json")
            self.assertFalse(blocked["benchmark_ready"])
            self.assertEqual(["GAMMA_TRACK_NOT_READY"], blocked["failure_codes"])
            self.assertFalse(blocked["tracks"]["gamma_decomp"]["ready"])
            self.assertNotIn("composite_score", blocked)
            self.assertFalse(blocked["kdd_submission_ready"])

            gamma.write_text(json.dumps({
                "gamma_benchmark_ready": True,
                "failure_codes": [],
            }), encoding="utf-8")
            ready = assess_benchmark_readiness(hea, gamma, root / "ready.json")
            self.assertTrue(ready["benchmark_ready"])
            self.assertEqual("dual_track", ready["release_scope"])
            self.assertEqual([], ready["failure_codes"])
            self.assertEqual(
                "diagnostic_non_identifiable",
                ready["tracks"]["hea_sro"]["diagnostic_occupancy_status"],
            )
            self.assertFalse(ready["kdd_submission_ready"])
            self.assertIn("PAPER_FROZEN_NOT_READY", ready["submission_failure_codes"])

    def test_submission_readiness_requires_frozen_release_evidence(self):
        from alloy_xrd_bench.benchmark_readiness import assess_benchmark_readiness

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            hea = root / "hea.json"
            gamma = root / "gamma.json"
            hea.write_text(json.dumps({"core_benchmark_ready": True, "failure_codes": []}))
            gamma.write_text(json.dumps({"gamma_benchmark_ready": True, "failure_codes": []}))
            paper = root / "paper.tex"
            paper.write_text("Frozen two-task manuscript")
            hea_artifacts = root / "hea_artifacts.json"
            hea_artifacts.write_text(json.dumps({"artifacts": [{"path": "table"}]}))
            gamma_artifacts = root / "gamma_artifacts.json"
            gamma_artifacts.write_text(json.dumps({
                "artifacts": [{"path": "table"}], "complete_experiment_matrix": True,
                "missing_scorecards": [],
            }))
            smoke = root / "smoke.json"
            smoke.write_text(json.dumps({"status": "pass"}))
            lock = root / "environment.lock"
            lock.write_text("python=3.11")
            sanity = root / "sanity.json"
            sanity.write_text(json.dumps({
                "status": "pass", "exact_latent_truth_claimed": False,
            }))
            publication = root / "publication.json"
            publication.write_text(json.dumps({
                "dataset_doi": "10.0000/example", "code_tag": "v1.1.0",
            }))

            report = assess_benchmark_readiness(
                hea, gamma, root / "readiness.json",
                paper_path=paper,
                hea_paper_manifest_path=hea_artifacts,
                gamma_paper_manifest_path=gamma_artifacts,
                release_smoke_path=smoke,
                environment_lock_path=lock,
                external_sanity_path=sanity,
                publication_record_path=publication,
            )

            self.assertTrue(report["benchmark_ready"])
            self.assertTrue(report["kdd_submission_ready"])
            self.assertEqual([], report["submission_failure_codes"])


if __name__ == "__main__":
    unittest.main()
