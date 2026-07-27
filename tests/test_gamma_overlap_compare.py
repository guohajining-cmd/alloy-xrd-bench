from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.gamma_overlap_compare import compare_parent_bootstrap


class GammaOverlapCompareTests(unittest.TestCase):
    def test_parent_bootstrap_is_paired_deterministic_and_reports_gate(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            pd.DataFrame({
                "parent_pair_id": ["a", "b", "c", "d"],
                "component_f1": [0.1, 0.2, 0.1, 0.2],
            }).to_parquet(reference / "per_parent_pair.parquet", index=False)
            pd.DataFrame({
                "parent_pair_id": ["a", "b", "c", "d"],
                "component_f1": [0.4, 0.5, 0.4, 0.5],
            }).to_parquet(candidate / "per_parent_pair.parquet", index=False)

            first = compare_parent_bootstrap(reference, candidate, root / "first", replicates=200, seed=2027)
            second = compare_parent_bootstrap(reference, candidate, root / "second", replicates=200, seed=2027)

            self.assertEqual(first["bootstrap_ci_95"], second["bootstrap_ci_95"])
            self.assertAlmostEqual(0.3, first["absolute_improvement"])
            self.assertTrue(first["five_point_gate_passed"])
            self.assertTrue((root / "first/bootstrap_samples.parquet").exists())
            self.assertEqual(first, json.loads((root / "first/comparison.json").read_text()))


if __name__ == "__main__":
    unittest.main()
