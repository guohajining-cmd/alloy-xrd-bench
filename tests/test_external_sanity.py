import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


class ExternalSanityTests(unittest.TestCase):
    def test_hea_patterns_are_traceable_experimental_views_without_latent_truth(self):
        root = Path("data/external_sanity/hea_v1")
        manifest = pd.read_parquet(root / "manifest.parquet")

        self.assertEqual(2, len(manifest))
        self.assertTrue(manifest["source_doi"].eq("10.5281/zenodo.4623002").all())
        self.assertTrue(manifest["license"].eq("CC-BY-4.0").all())
        self.assertFalse(manifest["exact_latent_truth_available"].any())
        for row in manifest.itertuples():
            with np.load(Path(row.pattern_path)) as pattern:
                self.assertEqual({"two_theta", "intensity"}, set(pattern.files))
                self.assertEqual(row.point_count, len(pattern["two_theta"]))
                self.assertTrue(np.isfinite(pattern["intensity"]).all())
                self.assertTrue((pattern["intensity"] >= 0).all())

    def test_gamma_patterns_are_traceable_but_have_no_component_truth(self):
        root = Path("data/external_sanity/gamma_v1")
        manifest = pd.read_parquet(root / "manifest.parquet")

        self.assertEqual(2, len(manifest))
        self.assertTrue(manifest["source_doi"].eq("10.17863/CAM.87406").all())
        self.assertTrue(manifest["license"].eq("CC-BY-ND-4.0").all())
        self.assertFalse(manifest["exact_latent_truth_available"].any())
        for row in manifest.itertuples():
            lines = Path(row.pattern_path).read_text(encoding="utf-8-sig").splitlines()
            pattern = np.loadtxt(lines[1:])
            self.assertEqual((row.point_count, 2), pattern.shape)
            self.assertTrue(np.isfinite(pattern).all())
            self.assertTrue((pattern[:, 1] >= 0).all())
            self.assertTrue((np.diff(pattern[:, 0]) > 0).all())

    def test_combined_status_requires_traceable_sets_without_latent_claims(self):
        status = json.loads(Path("data/external_sanity/status.json").read_text())

        self.assertEqual("pass", status["status"])
        self.assertEqual("pass", status["tracks"]["hea_sro"]["status"])
        self.assertEqual("pass", status["tracks"]["gamma_decomp"]["status"])
        self.assertFalse(status["exact_latent_truth_claimed"])


if __name__ == "__main__":
    unittest.main()
