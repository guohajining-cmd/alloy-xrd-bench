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

    def test_main_paper_defines_one_benchmark_with_two_independent_tracks(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")

        self.assertIn("one benchmark with two independent task tracks", paper)
        self.assertIn("Gamma-Decomp (gamma/gamma-prime decomposition)", paper)
        self.assertIn("No cross-track composite score is defined", paper)
        self.assertNotIn("two independent benchmarks", paper)

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
            "dataset_overview.png",
            "hea_sro_example.png",
            "gamma_decomp_example.png",
            "dataset_coverage.png",
        ):
            self.assertIn(f"generated/{name}", main)
        self.assertNotIn(r"\fbox{\parbox", main)
        self.assertEqual(3, main.count(r"\begin{table"))
        self.assertIn(r"\input{generated/compact_primary_results.tex}", main)

        for secondary in (
            "hea_perturbation_severity.png",
            "gamma_instrument_slices.png",
            "gamma_overlap_difficulty.png",
        ):
            self.assertNotIn(secondary, main)
            self.assertIn(secondary, appendix)

    def test_formal_hea_main_matrix_excludes_physics_constrained_experiment(self):
        paper = Path("docs/paper/sitexrd_bench_kdd2027.tex").read_text(encoding="utf-8")
        main, appendix = paper.split(r"\appendix", maxsplit=1)
        self.assertIn("Composition Ridge, Peak XGBoost, ResNet1D, and Spectrum Transformer", main)
        self.assertNotIn("physics-constrained Transformer has the lowest ID", main)
        self.assertIn("physics-constrained HEA", appendix)


if __name__ == "__main__":
    unittest.main()
