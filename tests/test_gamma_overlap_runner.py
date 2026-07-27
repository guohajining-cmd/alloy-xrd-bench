from pathlib import Path
import unittest


class GammaOverlapRunnerTests(unittest.TestCase):
    def test_runner_covers_frozen_baseline_matrix_and_resume(self):
        text = Path("scripts/workstation/run_gamma_decomp_v1_1.sh").read_text()

        self.assertIn("seeds=(2027 2028 2029)", text)
        self.assertIn("tasks=(full window)", text)
        for model in (
            "no-spectrum",
            "peak-feature",
            "bic",
            "resnet-set",
            "transformer-set",
            "physics-constrained",
            "reconstruction-only",
            "weak_peaks",
            "no_perturbation_training",
            "random_split",
        ):
            self.assertIn(model, text)
        self.assertIn("validate-release", text)
        self.assertIn("make-oracle", text)
        self.assertIn("run_is_complete", text)
        self.assertIn("score_is_complete", text)
        self.assertIn("per_instrument_slice.parquet", text)
        self.assertIn("per_parent_pair.parquet", text)
        self.assertIn("per_difficulty.parquet", text)
        self.assertIn("rescore_existing_submission", text)
        self.assertIn("rescore_neural_submissions", text)
        self.assertIn("comparison_is_complete", text)
        self.assertIn("bootstrap_samples.parquet", text)
        self.assertIn("paired_parent_metrics.parquet", text)
        self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", text)
        self.assertIn("flock -n", text)
        self.assertIn("run_snapshot.sha256", text)
        self.assertIn("sha256sum --check", text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$gpu"', text)
        self.assertIn("run_neural 0", text)
        self.assertIn("run_neural 1", text)
        self.assertIn("training_pids", text)
        self.assertIn("--split test", text)
        self.assertIn("gamma_overlap_paper_artifacts", text)
        self.assertIn("--require-complete", text)
        self.assertIn("run_neural 0", text)
        self.assertIn("run_neural 1", text)
        self.assertIn("wait_training", text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$gpu"', text)

if __name__ == "__main__":
    unittest.main()
