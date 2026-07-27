import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_release_package import freeze_release_package


class HeaReleasePackageTests(unittest.TestCase):
    def test_freezes_manifests_config_provenance_and_checksums_in_v1_1(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "benchmark_v1"
            output = root / "benchmark_v1_1"
            (source / "manifests").mkdir(parents=True)
            pd.DataFrame([{"structure_state_id": "s", "parent_group_id": "p"}]).to_parquet(source / "manifests" / "task_samples.parquet")
            pd.DataFrame([{"xrd_sample_id": "x", "parent_group_id": "p"}]).to_parquet(source / "manifests" / "xrd_samples.parquet")
            config = root / "config.json"
            config.write_text(json.dumps({"release_scope": "sro_core", "experimental_noise_calibrated": False}))

            report = freeze_release_package(
                source,
                output,
                config,
                source_license="CC-BY-4.0",
                required_manifests=("task_samples.parquet", "xrd_samples.parquet"),
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(0, report["checksum_errors"])
            self.assertEqual(0, report["license_errors"])
            self.assertTrue((output / "manifests" / "task_samples.parquet").exists())
            labels = pd.read_parquet(output / "manifests" / "automated_family_labels.parquet")
            self.assertEqual("automated_unadjudicated", labels["ground_truth_status"].iloc[0])
            self.assertTrue((output / "release" / "checksums.sha256").exists())
            self.assertTrue((output / "release" / "source_provenance.json").exists())
            snapshot = json.loads((output / "release" / "hea_benchmark_v1_1.config.snapshot.json").read_text())
            self.assertEqual("sro_core", snapshot["release_scope"])

    def test_refuses_any_write_to_frozen_v1_root(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "home" / "GJM" / "data" / "hea" / "benchmark_v1"
            (source / "manifests").mkdir(parents=True)
            config = Path(tempdir) / "config.json"
            config.write_text("{}")

            with self.assertRaisesRegex(ValueError, "immutable"):
                freeze_release_package(source, source, config, source_license="CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()
