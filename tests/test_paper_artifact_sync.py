from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class PaperArtifactSyncTests(unittest.TestCase):
    def test_local_manifests_cover_and_verify_all_frozen_artifacts(self):
        root = Path("docs/paper/generated")
        manifests = (
            root / "hea_paper_artifacts.json",
            root / "gamma_paper_artifacts.json",
        )
        expected = {
            "hea_sro_results.csv",
            "hea_sro_results.tex",
            "hea_occupancy_signal_noise.png",
            "hea_perturbation_severity.png",
            "gamma_decomp_results.csv",
            "gamma_decomp_results.tex",
            "gamma_instrument_slices.png",
            "gamma_overlap_difficulty.png",
        }
        observed: set[str] = set()
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest.get("complete_experiment_matrix", True))
            self.assertEqual([], manifest.get("missing_scorecards", []))
            for record in manifest["artifacts"]:
                name = Path(record["path"]).name
                self.assertNotIn(name, observed)
                observed.add(name)
                artifact = root / name
                self.assertTrue(artifact.is_file(), name)
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                self.assertEqual(record["sha256"], digest, name)

        self.assertEqual(expected, observed)
        self.assertFalse((root / "paper_artifacts.json").exists())


if __name__ == "__main__":
    unittest.main()
