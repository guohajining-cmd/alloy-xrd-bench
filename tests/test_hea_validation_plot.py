import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_validation_plot import create_validation_plots


class ValidationPlotTests(unittest.TestCase):
    def test_plotter_writes_scores_and_nonempty_images(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            summary = {
                "result_qualification": "pre-release",
                "publication_ready": False,
                "data_leakage_passed": True,
                "control_gate_passed": True,
                "xrd_beats_composition_only": True,
                "occupancy_identifiable_coverage": 0.00725,
                "occupancy_statistical_support": False,
                "ood_validation_status": "not_run",
                "perturbation_validation_status": "not_run",
                "model_aggregates": {
                    "composition-ridge": {
                        "seed_count": 1,
                        "sro_parent_macro_mae_mean": 0.45,
                        "sro_parent_macro_mae_std": 0.0,
                        "sro_rmse_mean": 0.67,
                        "sro_rmse_std": 0.0,
                        "family_macro_f1_mean": 0.24,
                    },
                    "transformer": {
                        "seed_count": 3,
                        "sro_parent_macro_mae_mean": 0.3375,
                        "sro_parent_macro_mae_std": 0.002,
                        "sro_rmse_mean": 0.50,
                        "sro_rmse_std": 0.003,
                        "family_macro_f1_mean": None,
                    },
                },
                "models": [
                    {"model": "composition-ridge", "seed": 2027, "sro_parent_macro_mae": 0.45},
                    {"model": "transformer", "seed": 2027, "sro_parent_macro_mae": 0.34},
                    {"model": "transformer", "seed": 2028, "sro_parent_macro_mae": 0.337},
                    {"model": "transformer", "seed": 2029, "sro_parent_macro_mae": 0.3355},
                ],
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary))

            artifacts = create_validation_plots(summary_path, root)

            scores = pd.read_csv(artifacts["scores_csv"])
            transformer = scores[scores["model"] == "transformer"].iloc[0]
            self.assertAlmostEqual(25.0, transformer["sro_skill_score"])
            for key in ["comparison_png", "stability_png", "gates_png"]:
                self.assertGreater(Path(artifacts[key]).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
