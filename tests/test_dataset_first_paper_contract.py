from pathlib import Path
import unittest


class DatasetFirstPaperContractTests(unittest.TestCase):
    def test_kdd_review_layout_uses_official_order_without_forced_transition_pages(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")

        self.assertIn(r"\documentclass[sigconf,review,anonymous]{acmart}", paper)
        bibliography = paper.index(r"\bibliography{sitexrd_bench_references}")
        appendix = paper.index(r"\appendix")
        self.assertLess(bibliography, appendix)
        transition = paper[paper.index(r"\section{Availability, Limitations, and Conclusion}"):appendix]
        self.assertNotIn(r"\clearpage", transition)

    def test_main_paper_defines_complementary_subproblems_for_future_refinement(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")

        self.assertIn("a first-stage benchmark toward alloy powder-XRD refinement", paper)
        self.assertIn("two complementary subproblems", paper)
        self.assertIn("we do not present a completed joint refinement algorithm", paper)
        self.assertNotIn("two independent task tracks", paper)
        self.assertIn("Gamma-Decomp (gamma/gamma-prime decomposition)", paper)
        self.assertIn("No cross-track composite score is defined", paper)
        self.assertNotIn("two independent benchmarks", paper)

    def test_experimental_scope_and_user_supplied_funding_are_explicit(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        normalized = " ".join(paper.split())
        self.assertIn("Among the resources considered in this study", normalized)
        self.assertIn("sample-matched SRO targets are rare", normalized)
        self.assertIn(r"\cite{he2024apt}", paper)
        expected = (
            "This work was financially supported by the Advanced Materials-National "
            "Science and Technology Major Project (Grant No. 2025ZD0620100), National "
            "Natural Science Foundation of China (Grant No. 52401015), and Shanghai "
            "Artificial Intelligence Open Source Award Project."
        )
        self.assertIn(expected, normalized)
        self.assertEqual(1, paper.count(r"\begin{acks}"))
        self.assertLess(paper.index(r"\end{acks}"), paper.index(r"\bibliography{"))
        self.assertIn("\\ifdefined\\anonymouscopy\n\\else\n\\begin{acks}", paper)

    def test_main_paper_is_data_first_and_uses_three_generated_data_figures_plus_coverage(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        main, appendix = paper.split(r"\appendix", maxsplit=1)

        section_order = [
            r"\section{Introduction}",
            r"\section{Related Work and Positioning}",
            r"\section{Dataset Overview}",
            r"\section{HEA-SRO Data}",
            r"\section{Gamma/Gamma-Prime Decomposition Data}",
            r"\section{Coverage, Tasks, and Splits}",
            r"\section{Benchmark Validation}",
            r"\section{Availability, Limitations, and Conclusion}",
        ]
        positions = [main.index(section) for section in section_order]
        self.assertEqual(sorted(positions), positions)
        for name in (
            "revision_2026_10_06/dataset_overview.png",
            "revision_2026_10_06/hea_sro_example.png",
            "gamma_decomp_example.png",
            "dataset_coverage.png",
        ):
            self.assertIn(f"generated/{name}", main)
        self.assertNotIn(r"\fbox{\parbox", main)
        self.assertEqual(4, main.count(r"\begin{table"))
        self.assertIn(r"\input{generated/revision_2026_10_06/hea_primary.tex}", main)
        self.assertIn(r"\input{generated/revision_2026_10_06/signal_controls.tex}", main)
        self.assertNotIn(r"\input{generated/compact_primary_results.tex}", main)

        for secondary in (
            "hea_perturbation_severity.png",
            "gamma_instrument_slices.png",
            "gamma_overlap_difficulty.png",
        ):
            self.assertNotIn(secondary, main)
            self.assertNotIn(secondary, appendix)
        self.assertIn("matched reevaluation", appendix)
        self.assertIn("No corrected numerical", appendix)

    def test_formal_hea_main_matrix_excludes_physics_constrained_experiment(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        main, appendix = paper.split(r"\appendix", maxsplit=1)
        table = Path("docs/paper/generated/revision_2026_10_06/hea_primary.tex").read_text()
        for model in ("Composition Ridge", "Peak XGBoost", "ResNet1D", "Spectrum Transformer"):
            self.assertIn(model, table)
        self.assertNotIn("Physics", table)
        self.assertNotIn("physics-constrained Transformer has the lowest ID", main)
        self.assertIn("physics variant", appendix)


if __name__ == "__main__":
    unittest.main()
