from pathlib import Path
import unittest


class DatasetVisualRunnerTests(unittest.TestCase):
    def test_workstation_runner_uses_formal_release_roots_and_frozen_seed(self):
        text = Path("scripts/workstation/generate_dataset_paper_visuals.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("/home/GJM/data/hea/benchmark_v1_1", text)
        self.assertIn("/home/GJM/data/releases/gamma-decomp-v1_1", text)
        self.assertIn("/home/GJM/runs/sitexrd_benchmark_v1_1/paper_visuals", text)
        self.assertIn("alloy_xrd_bench.dataset_paper_visuals", text)
        self.assertIn("--seed 2027", text)
        self.assertNotIn("mini", text.lower())
        self.assertNotIn("smoke", text.lower())


if __name__ == "__main__":
    unittest.main()
