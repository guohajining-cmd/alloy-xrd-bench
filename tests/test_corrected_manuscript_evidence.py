import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path("docs/paper/generated/revision_2026_10_06")


class CorrectedManuscriptEvidenceTests(unittest.TestCase):
    def test_corrected_table_is_phase_c_with_complete_split_model_grid(self):
        with (ROOT / "corrected_hea_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(20, len(rows))
        self.assertEqual({"C"}, {r["phase"] for r in rows})
        self.assertEqual(20, len({(r["model"], r["split_column"]) for r in rows}))
        primary = (ROOT / "hea_primary.tex").read_text()
        extended = (ROOT / "hea_extended.tex").read_text()
        for row in rows:
            self.assertIn(f"{float(row['mae_mean']):.4f}", extended)
            if row["model"] != "physics-transformer":
                self.assertIn(f"{float(row['mae_mean']):.4f}", primary)
        self.assertEqual(5, primary.count(r"\textbf{"))
        self.assertEqual(5, primary.count(r"\underline{"))
        self.assertIn(r"\textbf{Spectrum Transformer}", primary)
        self.assertIn(r"\underline{Peak XGBoost}", primary)
        self.assertNotIn("0.3400", primary)

    def test_extended_table_highlights_requested_exploratory_maes(self):
        table = (ROOT / "hea_extended.tex").read_text()
        with (ROOT / "corrected_hea_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for split, label, value in (
            ("chemistry_ood_split", "Chemistry OOD", "0.4069"),
            ("sro_ood_split", "Legacy SRO OOD", "0.4847"),
        ):
            self.assertIn(
                label + r" & Physics Transformer (exploratory) & \textbf{" + value + "}",
                table,
            )
            subset = [r for r in rows if r["split_column"] == split]
            best = min(subset, key=lambda r: float(r["mae_mean"]))
            self.assertEqual("physics-transformer", best["model"])
            self.assertEqual(value, f"{float(best['mae_mean']):.4f}")
        self.assertEqual(2, table.count(r"\textbf{"))
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        self.assertIn("no statistical significance is implied", paper)

    def test_signal_table_is_matched_same_learner_contrast(self):
        rows = json.loads((ROOT / "signal_controls.json").read_text())
        self.assertEqual(4, len(rows))
        table = (ROOT / "signal_controls.tex").read_text()
        for row in rows:
            self.assertEqual("control-xgboost-CX", row["candidate"])
            self.assertEqual("control-xgboost-C", row["reference"])
            self.assertEqual(1, row["fitted_seed_count"])
            self.assertEqual(10000, row["repeats"])
            self.assertAlmostEqual(row["candidate_mae"] - row["reference_mae"], row["delta_mae"])
            lo, hi = row["delta_ci95_parent"]
            self.assertIn(f"[{lo:.4f}, {hi:.4f}]", table)

    def test_corrected_figures_retain_pair_and_explicit_undefined_targets(self):
        manifest = json.loads((ROOT / "evidence_manifest.json").read_text())
        fig = manifest["figures"]
        self.assertEqual("occstr_gra62126339", fig["pair"]["left_structure_state_id"])
        self.assertEqual("occstr_gra62557950", fig["pair"]["right_structure_state_id"])
        self.assertEqual(32, fig["common_defined_targets"])
        self.assertAlmostEqual(80 / 3, fig["corrected_sro_l1_distance"])
        self.assertIn("not zero-filled", fig["masked_targets"])
        for name, expected in manifest["products"].items():
            self.assertEqual(expected, hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name)

    def test_paper_retains_scientific_boundaries_and_current_values(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        self.assertIn("26.67", paper)
        self.assertNotIn("13.33", paper)
        self.assertNotIn("0.4496", paper)
        self.assertIn("complete corrected result matrix remains pending", paper)
        self.assertIn("relative to a zero-change predictor by only 4.84--7.44", paper)
        self.assertIn("does not represent five distinct", paper)
        self.assertIn("releases/tag/v1.1.2", paper)
        self.assertIn("original v1.1.1 archive", paper)
        self.assertIn("does not establish completion of the outstanding validation", paper)


if __name__ == "__main__":
    unittest.main()
