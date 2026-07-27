from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.gamma_overlap_ablation import build_random_split_view


class GammaOverlapAblationTests(unittest.TestCase):
    def test_random_split_view_reuses_patterns_and_rewrites_label_splits(self):
        from test_gamma_overlap_public import _write_internal_release
        from alloy_xrd_bench.gamma_overlap_public import export_public_release

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source"
            _write_internal_release(source)
            core = Path(export_public_release(source, root / "release")["core_dir"])

            report = build_random_split_view(core, root / "random", seed=2027)

            self.assertEqual("pass", report["status"])
            for task in ("full", "window"):
                samples = pd.read_parquet(root / "random" / task / "samples.parquet")
                components = pd.read_parquet(root / "random" / task / "components.parquet")
                expected = dict(zip(samples["sample_id"].astype(str), samples["split"].astype(str)))
                self.assertEqual(
                    set(components["split"].astype(str)),
                    set(samples["split"].astype(str)),
                )
                self.assertTrue(all(
                    expected[str(row.sample_id)] == str(row.split)
                    for row in components.itertuples()
                ))
                first = samples.iloc[0]
                self.assertTrue((root / "random" / first["pattern_path"]).exists())

                instrument_slices = pd.read_parquet(
                    root / "random" / "slices" / "instrument_slices.parquet"
                )
                task_slices = instrument_slices.loc[
                    instrument_slices["sample_id"].astype(str).isin(expected)
                ]
                self.assertEqual(set(expected), set(task_slices["sample_id"].astype(str)))
                self.assertTrue(all(
                    expected[str(row.sample_id)] == str(row.split)
                    for row in task_slices.itertuples()
                ))
                expected_validation = {
                    sample_id for sample_id, split in expected.items()
                    if split == "validation"
                }
                actual_validation = set(
                    task_slices.loc[
                        task_slices["split"].astype(str).eq("validation"),
                        "sample_id",
                    ].astype(str)
                )
                self.assertEqual(expected_validation, actual_validation)
                self.assertFalse(task_slices["sample_id"].astype(str).duplicated().any())


if __name__ == "__main__":
    unittest.main()
