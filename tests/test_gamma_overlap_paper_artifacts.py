from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.gamma_overlap_paper_artifacts import generate_gamma_paper_artifacts


class GammaOverlapPaperArtifactTests(unittest.TestCase):
    def test_generates_table_and_slice_figures_from_frozen_scores(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            for model in ("no-spectrum", "peak-feature"):
                seeds = (None,) if model == "no-spectrum" else (2027, 2028, 2029)
                for task in ("full", "window"):
                    for seed in seeds:
                        if seed is None:
                            score = root / model / task / "test" / "score"
                        else:
                            score = root / model / task / str(seed) / "test" / "score"
                        score.mkdir(parents=True)
                        value = 0.2 if model == "no-spectrum" else 0.7 + (seed - 2027) * 0.01
                        (score / "scorecard.json").write_text(json.dumps({
                            "metrics": {
                                "component_f1": value,
                                "phase_hkl_accuracy": value - 0.05,
                                "count_accuracy": value - 0.1,
                                "center_mae_deg": 0.1,
                                "fwhm_mae_deg": 0.02,
                                "area_mae": 0.3,
                                "confidence_ece": 0.1,
                                "aurc": 0.2,
                            }
                        }))
                        pd.DataFrame({
                            "instrument_slice": ["instrument_id", "instrument_ood"],
                            "component_f1": [value, value - 0.1],
                        }).to_parquet(score / "per_instrument_slice.parquet", index=False)
                        pd.DataFrame({
                            "difficulty": ["resolved", "severe_overlap"],
                            "component_f1": [value, value - 0.2],
                        }).to_parquet(score / "per_difficulty.parquet", index=False)

            result = generate_gamma_paper_artifacts(root, root / "paper")

            for key in ("results_csv", "results_tex", "instrument_figure", "overlap_figure", "manifest"):
                self.assertGreater(Path(result[key]).stat().st_size, 20)
            table = Path(result["results_tex"]).read_text()
            self.assertIn("Peak Feature", table)
            self.assertIn("0.710", table)


if __name__ == "__main__":
    unittest.main()
