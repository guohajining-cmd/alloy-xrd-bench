from pathlib import Path
import unittest


class HeaPhysicsRunnerTests(unittest.TestCase):
    def test_runner_covers_two_gpu_resumable_matrix_and_final_gates(self):
        text = Path("scripts/workstation/run_hea_physics_matrix.sh").read_text()

        self.assertIn("splits=(id_split composition_ood_split chemistry_ood_split sro_ood_split)", text)
        self.assertIn("seeds=(2027 2028 2029 2030 2031)", text)
        self.assertIn("run_worker 0", text)
        self.assertIn("run_worker 1", text)
        self.assertIn("run_is_complete", text)
        self.assertIn("score_is_complete", text)
        self.assertIn("materialize-submission", text)
        self.assertIn("physics-transformer", text)
        self.assertIn("physics-transformer_seed${seed}", text)
        self.assertIn("--required-models physics-transformer", text)
        self.assertNotIn("hea_prediction_audit", text)
        self.assertNotIn("hea_benchmark_readiness", text)
        self.assertIn("sitexrd_hea_validation_v1_1_physics_appendix", text)


if __name__ == "__main__":
    unittest.main()
