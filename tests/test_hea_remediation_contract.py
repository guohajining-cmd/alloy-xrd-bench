import json
import unittest
from pathlib import Path, PurePosixPath


CONFIG_PATH = Path("configs/hea_benchmark_v1_1.json")


class HEARemediationContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_v1_1_separates_dataset_and_model_gates(self):
        self.assertEqual(
            "/home/GJM/data/hea/benchmark_v1",
            self.config["source_dataset_root"],
        )
        self.assertEqual(
            "/home/GJM/data/hea/benchmark_v1_1",
            self.config["output_dataset_root"],
        )
        self.assertEqual("sro_core", self.config["release_scope"])
        self.assertEqual(["sro"], self.config["primary_tasks"])
        self.assertEqual(
            ["family", "occupancy_identifiability"],
            self.config["diagnostic_tasks"],
        )
        self.assertFalse(self.config["experimental_noise_calibrated"])

    def test_source_and_output_roots_are_distinct(self):
        source = PurePosixPath(self.config["source_dataset_root"])
        output = PurePosixPath(self.config["output_dataset_root"])
        self.assertNotEqual(source, output)
        self.assertNotIn(source, output.parents)
        self.assertEqual("benchmark_v1", source.name)
        self.assertEqual("benchmark_v1_1", output.name)

    def test_frozen_reproducibility_and_split_contract(self):
        self.assertEqual([2027, 2028, 2029, 2030, 2031], self.config["seeds"])
        self.assertEqual(
            [
                "id_split",
                "composition_ood_split",
                "chemistry_ood_split",
                "sro_ood_split",
                "perturbation_evaluation",
            ],
            self.config["evaluation_slices"],
        )
        self.assertEqual(0.95, self.config["noise_calibration"]["quantile"])

    def test_sro_core_gates_exclude_family_and_occupancy_outcomes(self):
        gates = self.config["core_release_gates"]
        rendered = json.dumps(gates, sort_keys=True)
        self.assertNotIn("family", rendered)
        self.assertNotIn("identifiable_occupancy", rendered)
        self.assertNotIn("model_gain", rendered)
        self.assertEqual(5, gates["minimum_perturbation_seeds"])
        self.assertEqual(
            {
                "composition_ridge",
                "peak_xgboost",
                "resnet1d",
                "transformer",
                "physics_transformer",
            },
            set(gates["required_models"]),
        )
        self.assertEqual({"replicates": 2000, "seed": 2027}, gates["parent_bootstrap"])

        diagnostics = self.config["diagnostic_protocols"]
        self.assertFalse(diagnostics["family"]["release_gate"])
        self.assertEqual(80, diagnostics["family"]["pending_review_count"])
        self.assertFalse(diagnostics["occupancy_identifiability"]["release_gate"])
        self.assertEqual(1000, diagnostics["occupancy_identifiability"]["required_parent_count"])
        self.assertEqual(5, diagnostics["occupancy_identifiability"]["minimum_views_per_parent"])

    def test_paper_contract_has_two_primary_tasks_and_no_hea_rank_leaderboard(self):
        paths = [
            Path("docs/superpowers/specs/2026-07-22-sitexrd-bench-paper-design.md"),
            Path("docs/paper/sitexrd_bench_kdd2027_draft.md"),
            Path("docs/paper/non_data_workplan.md"),
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("HEA-Rank", combined)
        self.assertNotIn("three leaderboard tasks", combined)
        self.assertIn("two primary tasks", combined)


if __name__ == "__main__":
    unittest.main()
