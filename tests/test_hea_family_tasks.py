import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.hea_family_tasks import (
    add_family_task_labels,
    audit_family_support,
    lattice_family,
    ordering_prototype,
    rare_open_set,
)


class HEAFamilyTaskTests(unittest.TestCase):
    def test_family_heads_preserve_lattice_ordering_and_open_set_meaning(self):
        self.assertEqual("fcc", lattice_family("l12_like"))
        self.assertEqual("bcc", lattice_family("b2_like"))
        self.assertEqual("l12", ordering_prototype("l12_like"))
        self.assertEqual("b2", ordering_prototype("b2_like"))
        self.assertEqual("disordered", ordering_prototype("fcc_like"))
        self.assertIsNone(ordering_prototype("hcp_like"))
        self.assertEqual("hcp", rare_open_set("hcp_like"))
        self.assertIsNone(rare_open_set("fcc_like"))

    def test_support_gate_counts_parent_groups_not_observations(self):
        rows = []
        for split, count in [("train", 2), ("validation", 1), ("test", 1)]:
            for family in ["fcc_like", "bcc_like"]:
                for parent in range(count):
                    for view in range(3):
                        rows.append({"parent_group_id": f"{split}-{family}-{parent}", "structure_family": family, "id_split": split, "view": view})
        labelled = add_family_task_labels(pd.DataFrame(rows))
        self.assertEqual(
            labelled["structure_family"].tolist(),
            labelled["automated_family_label"].tolist(),
        )
        report = audit_family_support(
            labelled,
            split_column="id_split",
            minimums={"train": 2, "validation": 1, "test": 1},
        )
        self.assertTrue(report["lattice_family_gate_passed"])
        self.assertEqual("diagnostic", report["status"])
        self.assertFalse(report["release_gate"])
        self.assertEqual(2, report["counts"]["train"]["fcc"])
        failed = audit_family_support(
            labelled,
            split_column="id_split",
            minimums={"train": 3, "validation": 1, "test": 1},
        )
        self.assertFalse(failed["lattice_family_gate_passed"])

    def test_cli_writes_machine_readable_support_evidence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = [
                {"structure_family": family, "parent_group_id": f"{split}-{family}", "id_split": split}
                for split in ["train", "validation", "test"]
                for family in ["fcc_like", "bcc_like"]
            ]
            source = root / "labels.parquet"
            output = root / "family_support.json"
            pd.DataFrame(rows).to_parquet(source)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "alloy_xrd_bench.hea_family_tasks",
                    "--labels",
                    str(source),
                    "--output",
                    str(output),
                    "--train-min",
                    "1",
                    "--validation-min",
                    "1",
                    "--test-min",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["lattice_family_gate_passed"])
            self.assertEqual(80, report["pending_review_count"])
            self.assertEqual("automated_unadjudicated", report["ground_truth_status"])


if __name__ == "__main__":
    unittest.main()
