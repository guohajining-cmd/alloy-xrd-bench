import json
from pathlib import Path
import tempfile
import unittest

from alloy_xrd_bench.hea_ood import audit_ood_scores


class HeaOodTests(unittest.TestCase):
    def test_audit_requires_every_seed_and_ood_slice(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            splits = ["composition_ood_split", "chemistry_ood_split", "sro_ood_split"]
            for seed in [2027, 2028, 2029]:
                for split in splits:
                    path = root / "transformer" / str(seed) / f"{split}_test"
                    path.mkdir(parents=True)
                    (path / "metrics.json").write_text(
                        json.dumps({"sro": {"masked_parent_macro_mae": 0.3}})
                    )

            passed = audit_ood_scores(root, "transformer", [2027, 2028, 2029], splits)
            self.assertEqual("pass", passed["status"])
            self.assertEqual(9, passed["evaluated_run_count"])

            (root / "transformer" / "2029" / "sro_ood_split_test" / "metrics.json").unlink()
            failed = audit_ood_scores(root, "transformer", [2027, 2028, 2029], splits)
            self.assertEqual("fail", failed["status"])
            self.assertEqual(1, len(failed["missing_runs"]))


if __name__ == "__main__":
    unittest.main()
