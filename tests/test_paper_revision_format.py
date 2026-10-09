from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alloy_xrd_bench.dataset_paper_visuals import _write_compact_results


class PaperResultHighlightTests(unittest.TestCase):
    def test_user_confirmed_corresponding_authors_share_one_note(self):
        tex = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        author_block = tex.split(r"\author{Anonymous Author(s)}", 1)[1].split(r"\fi", 1)[0]
        anonymous, named = author_block.split(r"\else", 1)
        self.assertNotIn(r"\authornote", anonymous)
        self.assertNotIn(r"\email", anonymous)
        self.assertIn(
            "\\author{Jie Xiong}\n\\authornote{Corresponding authors: Jie Xiong and Tongyi Zhang.}",
            named,
        )
        self.assertIn("\\author{Tongyi Zhang}\n\\authornotemark[1]", named)
        xiong, zhang = named.split(r"\author{Jie Xiong}", 1)[1].split(r"\author{Tongyi Zhang}", 1)
        self.assertIn(r"\email{xiongjie@shu.edu.cn}", xiong)
        self.assertIn(r"\email{zhangty@shu.edu.cn}", zhang)
        self.assertEqual(2, named.count(r"\email{"))
        self.assertEqual(1, named.count(r"\authornote{"))
        self.assertNotIn(r"\authornote", named.split(r"\author{Jie Xiong}", 1)[0])

    def test_author_entry_shares_revised_manuscript(self):
        entry = Path("docs/paper/sitexrd_bench_kdd2027_author_version.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{sitexrd_bench_kdd2027.tex}", entry)
        self.assertNotIn(r"\begin{abstract}", entry)
        self.assertNotIn(r"\cite", entry)

    def test_rank_is_local_to_split_or_view_and_uses_metric_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([
                {"split_column": split, "model": model, "parent_macro_mae_mean": value, "random_seed_std": 0.01}
                for split, model, value in [
                    ("id_split", "composition-ridge", .4),
                    ("id_split", "resnet1d", .3),
                    ("id_split", "transformer", .2),
                    ("id_split", "physics-transformer", .1),
                    ("sro_ood_split", "composition-ridge", .1),
                    ("sro_ood_split", "transformer", .4),
                ]
            ]).to_csv(root / "hea.csv", index=False)
            pd.DataFrame([
                {"task": task, "model": model, "model_label": model, "component_f1_mean": value}
                for task, model, value in [
                    ("full", "prior", .3), ("window", "prior", .6),
                    ("full", "peak", .8), ("window", "peak", .4),
                ]
            ]).to_csv(root / "gamma.csv", index=False)
            _write_compact_results(root / "hea.csv", root / "gamma.csv", root / "table.tex")
            table = (root / "table.tex").read_text()
            self.assertIn(r"\textbf{Spectrum Transformer} & \textbf{0.2000}", table)
            self.assertIn(r"ResNet1D & \underline{0.3000}", table)
            self.assertIn(r"\textbf{Composition Ridge} & \textbf{0.1000}", table)
            self.assertIn(r"Full & \textbf{peak} & \textbf{0.800}", table)
            self.assertIn(r"Window & \textbf{prior} & \textbf{0.600}", table)
            self.assertNotIn("Physics", table)

    def test_displayed_ties_share_highlight_and_missing_metrics_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([
                {"split_column": "id_split", "model": model, "parent_macro_mae_mean": value, "random_seed_std": 0.0}
                for model, value in [("composition-ridge", .4), ("resnet1d", .30001), ("transformer", .30002)]
            ]).to_csv(root / "hea.csv", index=False)
            pd.DataFrame([
                {"task": "full", "model": model, "component_f1_mean": value}
                for model, value in [("missing", float("nan")), ("valid", .5)]
            ]).to_csv(root / "gamma.csv", index=False)
            _write_compact_results(root / "hea.csv", root / "gamma.csv", root / "table.tex")
            table = (root / "table.tex").read_text()
            self.assertEqual(2, table.count(r"\textbf{0.3000}"))
            self.assertIn(r"\underline{0.4000}", table)
            self.assertIn("missing & -- & --", table)
            self.assertNotIn("nan", table.lower())


if __name__ == "__main__":
    unittest.main()
