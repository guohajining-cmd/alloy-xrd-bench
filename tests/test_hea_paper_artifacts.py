import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_paper_artifacts import generate_paper_artifacts


class HeaPaperArtifactTests(unittest.TestCase):
    def test_paper_consumes_generated_v1_1_results(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{generated/compact_primary_results.tex}", paper)
        self.assertIn("generated/hea_sro_results.tex", paper)
        self.assertIn("generated/hea_occupancy_signal_noise.png", paper)
        self.assertIn("generated/hea_perturbation_severity.png", paper)

    def test_generates_tables_and_figures_from_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            summary = root / "validation_summary.json"
            summary.write_text(json.dumps({
                "split_model_aggregates": {
                    "id_split": {
                        "composition-ridge": {"seed_count": 1, "sro_parent_macro_mae_mean": 0.45, "sro_parent_macro_mae_std": 0.0},
                        "transformer": {"seed_count": 5, "sro_parent_macro_mae_mean": 0.34, "sro_parent_macro_mae_std": 0.01},
                        "physics-transformer": {"seed_count": 5, "sro_parent_macro_mae_mean": 0.32, "sro_parent_macro_mae_std": 0.02},
                    }
                }
            }))
            pairs = root / "pairs.parquet"
            pd.DataFrame({
                "xrd_l1_distance": [0.01, 0.02],
                "noise_l1_floor": [0.08, 0.09],
                "identifiability_ratio": [0.125, 0.222],
                "noise_identifiable": [False, False],
            }).to_parquet(pairs)
            perturb = root / "perturbation" / "transformer" / "2027"
            perturb.mkdir(parents=True)
            (perturb / "perturbation_summary.json").write_text(json.dumps({
                "status": "pass", "model": "transformer", "seed": 2027,
                "severity_curve": [
                    {"severity_min": 0.0, "severity_max": 0.25, "parent_macro_mae": 0.35},
                    {"severity_min": 0.75, "severity_max": 1.0, "parent_macro_mae": 0.5},
                ],
            }))

            result = generate_paper_artifacts(summary, pairs, root / "perturbation", root / "paper")

            for key in ["results_csv", "results_tex", "occupancy_figure", "perturbation_figure", "manifest"]:
                self.assertGreater(Path(result[key]).stat().st_size, 20)
            self.assertIn("0.3400", Path(result["results_tex"]).read_text())
            self.assertIn("Physics-Constrained Transformer", Path(result["results_tex"]).read_text())


if __name__ == "__main__":
    unittest.main()
