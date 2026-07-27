from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "sitexrd_bench_kdd2027_eight_page_content_preview.pdf"

PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
INK = colors.HexColor("#172126")
MUTED = colors.HexColor("#526168")
TEAL = colors.HexColor("#087F8C")
CORAL = colors.HexColor("#D05A47")
PALE_TEAL = colors.HexColor("#E8F3F4")
PALE_CORAL = colors.HexColor("#F8ECE8")
RULE = colors.HexColor("#C9D3D6")
PAPER = colors.HexColor("#FCFDFD")

BODY = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=8.4,
    leading=11.1,
    textColor=INK,
    alignment=TA_LEFT,
    spaceAfter=4,
)
SMALL = ParagraphStyle(
    "small",
    parent=BODY,
    fontSize=7.2,
    leading=9.2,
    textColor=MUTED,
)
ABSTRACT = ParagraphStyle(
    "abstract",
    parent=BODY,
    fontName="Helvetica-Oblique",
    fontSize=8.0,
    leading=10.5,
)
SECTION = ParagraphStyle(
    "section",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=12.5,
    leading=14.5,
    textColor=INK,
    spaceAfter=5,
)
SUBSECTION = ParagraphStyle(
    "subsection",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=11.5,
    textColor=TEAL,
    spaceAfter=3,
)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER)


def draw_paragraph(canvas: Canvas, text: str, x: float, y: float, width: float, style=BODY) -> float:
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, PAGE_H)
    paragraph.drawOn(canvas, x, y - height)
    return y - height


def draw_header(canvas: Canvas, page: int, label: str) -> None:
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 0.12 * inch, PAGE_W, 0.12 * inch, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawString(MARGIN, PAGE_H - 0.33 * inch, "SITEXRD-BENCH | KDD WORKING CONTENT PREVIEW")
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.33 * inch, label.upper())
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 0.38 * inch, PAGE_W - MARGIN, 0.38 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.8)
    canvas.drawString(MARGIN, 0.22 * inch, "Working draft: planned experiments are explicitly marked and contain no fabricated results.")
    canvas.drawRightString(PAGE_W - MARGIN, 0.22 * inch, str(page))


def draw_section_title(canvas: Canvas, number: str, title: str, y: float) -> float:
    canvas.setFillColor(CORAL)
    canvas.roundRect(MARGIN, y - 18, 24, 18, 2, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawCentredString(MARGIN + 12, y - 12.5, number)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(MARGIN + 34, y - 14, title)
    return y - 29


def draw_callout(canvas: Canvas, title: str, text: str, x: float, y: float, width: float, accent=TEAL) -> float:
    p = Paragraph(text, SMALL)
    _, h = p.wrap(width - 22, PAGE_H)
    box_h = h + 31
    canvas.setFillColor(PALE_TEAL if accent == TEAL else PALE_CORAL)
    canvas.setStrokeColor(accent)
    canvas.roundRect(x, y - box_h, width, box_h, 3, fill=1, stroke=1)
    canvas.setFillColor(accent)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(x + 10, y - 14, title.upper())
    p.drawOn(canvas, x + 11, y - box_h + 10)
    return y - box_h


def draw_two_columns(canvas: Canvas, left: list[tuple[str, str]], right: list[tuple[str, str]], y: float) -> None:
    gap = 0.28 * inch
    width = (CONTENT_W - gap) / 2
    for x, blocks in ((MARGIN, left), (MARGIN + width + gap, right)):
        cy = y
        for heading, text in blocks:
            cy = draw_paragraph(canvas, heading, x, cy, width, SUBSECTION) - 2
            cy = draw_paragraph(canvas, text, x, cy, width, BODY) - 9


def draw_table(canvas: Canvas, data: list[list[str]], x: float, y: float, widths: list[float], font_size: float = 7.0) -> float:
    rows = [[Paragraph(cell, SMALL if row_index else ParagraphStyle("th", parent=SMALL, fontName="Helvetica-Bold", textColor=colors.white)) for cell in row] for row_index, row in enumerate(data)]
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]))
    _, h = table.wrap(sum(widths), PAGE_H)
    table.drawOn(canvas, x, y - h)
    return y - h


def page_one(canvas: Canvas) -> None:
    draw_header(canvas, 1, "Title and abstract")
    y = PAGE_H - 0.72 * inch
    y = draw_paragraph(canvas, "SiteXRD-Bench", MARGIN, y, CONTENT_W, ParagraphStyle(
        "title", parent=SECTION, fontSize=27, leading=29, textColor=INK, alignment=TA_CENTER
    )) - 2
    y = draw_paragraph(canvas, "A Hierarchical Benchmark for Chemical-Order Inference and Overlapping-Peak Decomposition in Structural Alloys", MARGIN + 0.35 * inch, y, CONTENT_W - 0.7 * inch, ParagraphStyle(
        "subtitle", parent=BODY, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL, alignment=TA_CENTER
    )) - 12
    y = draw_paragraph(canvas, "KDD Datasets and Benchmarks Track - eight-page content preview", MARGIN, y, CONTENT_W, ParagraphStyle(
        "meta", parent=SMALL, fontSize=8, alignment=TA_CENTER
    )) - 16
    y = draw_callout(canvas, "Central question", "Can models recover physically meaningful chemical order and overlapping diffraction components from powder XRD, rather than only reconstructing a visually similar pattern?", MARGIN + 0.45 * inch, y, CONTENT_W - 0.9 * inch, CORAL) - 16
    y = draw_paragraph(canvas, "Abstract", MARGIN, y, CONTENT_W, SECTION) - 4
    abstract = (
        "Atomic site occupancy, chemical ordering, and phase-resolved lattice parameters are central latent variables in multicomponent structural alloys, yet recovering them from powder X-ray diffraction (XRD) is often non-identifiable. Distinct atomic arrangements can produce nearly indistinguishable patterns, while overlapping reflections and acquisition perturbations can hide weak occupancy-sensitive evidence. SiteXRD-Bench organizes two complementary inverse problems: first- and second-shell chemical-order inference in BCC/FCC high-entropy alloys, and permutation-invariant gamma/gamma-prime component decomposition in Ni-based superalloys. Each sample retains provenance from structure and occupancy labels to indexed reflections, component profiles, and observed patterns. Evaluation separates whole-pattern reconstruction, component recovery, physical-parameter inference, and chemical-order recovery, with controlled composition, occupancy, instrument, and simulation-to-experiment shifts. Identifiability sets and forward-consistency checks expose models that fit spectra while producing structurally incorrect interpretations."
    )
    y = draw_paragraph(canvas, abstract, MARGIN + 0.18 * inch, y, CONTENT_W - 0.36 * inch, ABSTRACT) - 14
    draw_callout(canvas, "Current evidence status", "HEA inventory and pilot XRD exist. Gamma/gamma-prime component truth, frozen splits, baselines, and final result tables remain under construction.", MARGIN, y, CONTENT_W, TEAL)


def page_two(canvas: Canvas) -> None:
    draw_header(canvas, 2, "Motivation and contributions")
    y = draw_section_title(canvas, "1", "Introduction", PAGE_H - 0.68 * inch)
    left = [
        ("Why powder XRD is not enough by itself", "Powder XRD compresses a three-dimensional, multicomponent crystal into a one-dimensional pattern. Different latent structures can therefore remain compatible with almost the same observation. A low spectral reconstruction error does not establish that the correct occupancy, phase component, or lattice misfit was recovered."),
        ("HEA failure mode", "For a fixed composition and nominal BCC or FCC parent lattice, atom arrangements can differ in short-range order. Dominant Bragg peaks may remain similar while occupancy-sensitive evidence appears only as subtle intensity changes or weak reflections."),
        ("Superalloy failure mode", "The FCC gamma matrix and L1_2 gamma-prime precipitate have related lattices. Their reflections can overlap, making phase-specific lattice parameters, phase fraction, and lattice misfit difficult to identify under broadening and strain."),
    ]
    right = [
        ("What current evaluations miss", "Phase classification, peak detection, and whole-pattern reconstruction collapse distinct failure modes. They do not reveal whether a model used composition shortcuts, assigned components to the wrong phase, or returned an overconfident answer for an ambiguous pattern."),
        ("Four contributions", "<b>1.</b> Hierarchical truth from structure to observed XRD.<br/><b>2.</b> Two complementary alloy inverse-problem tracks.<br/><b>3.</b> Separate scoring of spectra, components, physical parameters, and chemical order.<br/><b>4.</b> Controlled OOD and identifiability evaluation."),
        ("Scope boundary", "The benchmark does not predict strength or creep life. Mechanical relevance motivates why SRO and gamma/gamma-prime parameters matter, but downstream property prediction is outside the benchmark task contract."),
    ]
    draw_two_columns(canvas, left, right, y)


def page_three(canvas: Canvas) -> None:
    draw_header(canvas, 3, "Formal benchmark")
    y = draw_section_title(canvas, "2", "Hierarchical Benchmark Formulation", PAGE_H - 0.68 * inch)
    y = draw_callout(canvas, "Forward hierarchy", "latent structure/order truth  ->  indexed reflections  ->  phase/HKL component profiles  ->  background, broadening, and noise  ->  observed XRD", MARGIN, y, CONTENT_W, TEAL) - 12
    data = [
        ["Level", "Stored truth", "Why it matters"],
        ["Structure", "Elements, fractional coordinates, cell, provenance", "Deterministic computational ground truth"],
        ["Order", "SRO matrices, sublattice distributions, phase parameters", "Primary physically meaningful targets"],
        ["Reflection", "2-theta, intensity, HKL, multiplicity, phase", "Connects structure and individual peaks"],
        ["Component", "Center, area, width, profile, phase/HKL", "Enables overlap decomposition evaluation"],
        ["Observation", "4,001-point clean or perturbed XRD", "What a benchmark model receives"],
    ]
    y = draw_table(canvas, data, MARGIN, y, [0.85 * inch, 2.55 * inch, CONTENT_W - 3.4 * inch]) - 15
    draw_two_columns(canvas, [
        ("Truth versus routing metadata", "Final relaxed structures define the computational truth. Derived family names such as bcc_like and fcc_like route samples into eligible tasks; they are not a substitute for site coordinates, SRO, or sublattice truth."),
        ("Permutation invariance", "Equivalent-site permutations must not create different labels. HEA tasks predict SRO or order descriptors rather than absolute atom indices. Gamma component sets are matched optimally instead of using arbitrary output order."),
    ], [
        ("Forward consistency", "Predicted latent parameters can be sent through the declared forward model to regenerate XRD. This detects internally inconsistent predictions, but a good regenerated spectrum never replaces latent-target accuracy."),
        ("Identifiability", "Pairs with low spectral distance but high latent distance form weakly identifiable challenge sets. Models must report calibrated uncertainty on these samples rather than only point estimates."),
    ], y)


def page_four(canvas: Canvas) -> None:
    draw_header(canvas, 4, "Dataset construction")
    y = draw_section_title(canvas, "3", "Dataset Construction and Current Inventory", PAGE_H - 0.68 * inch)
    counts = [
        ["Source rows", "Accepted", "Rejected", "Pilot structures", "Pilot XRD"],
        ["84,024", "84,019", "5", "1,000", "2,000"],
    ]
    y = draw_table(canvas, counts, MARGIN, y, [CONTENT_W / 5] * 5, 8) - 13
    families = [
        ["Final family", "Count", "Role"],
        ["fcc_like", "51,271", "HEA parent-topology candidate"],
        ["bcc_like", "32,512", "HEA parent-topology candidate"],
        ["l12_like", "53", "Ordered prototype / gamma-prime seed"],
        ["b2_like", "18", "Ordered prototype"],
        ["hcp_like + complex", "165", "Generalization and exclusion analysis"],
    ]
    y = draw_table(canvas, families, MARGIN, y, [1.3 * inch, 0.85 * inch, CONTENT_W - 2.15 * inch]) - 12
    draw_two_columns(canvas, [
        ("HEA construction", "Every accepted relaxed structure receives a stable ID, canonical hash, composition, quality status, parent-topology mapping, site labels, neighbor shells, and first/second-shell SRO. Controlled same-composition occupancy variants are still required to prevent composition shortcuts."),
        ("XRD reference configuration", "Clean pilot patterns use Cu K-alpha, 20-100 degrees, 0.02-degree spacing, and 4,001 points. Reflection tables retain peak position, intensity, HKL, and multiplicity. Perturbations are stored separately with their parameters."),
    ], [
        ("Gamma/gamma-prime construction", "The design pairs FCC gamma and L1_2 gamma-prime structures and varies phase fraction, phase-specific lattice parameters, misfit, sublattice disorder, broadening, strain, and texture. Component truth is stored before summation."),
        ("Current limitation", "The gamma/gamma-prime generator has interfaces and pairing manifests, but its frozen release, admitted relaxation model, component truth, and parameter ranges are not complete. It is therefore not yet a benchmark-ready track."),
    ], y)


def page_five(canvas: Canvas) -> None:
    draw_header(canvas, 5, "Tasks and metrics")
    y = draw_section_title(canvas, "4", "Leaderboard Tasks and Evaluation", PAGE_H - 0.68 * inch)
    tasks = [
        ["Task", "Input setting", "Target", "Primary metric"],
        ["HEA-SRO", "XRD + composition + parent", "Shell-1/2 Warren-Cowley matrices", "Masked macro-MAE; Spearman"],
        ["HEA-Rank", "Same-composition XRD group", "Distance from random mixing", "Pair accuracy; NDCG"],
        ["Gamma-Decomp", "Observed mixed XRD", "Unordered phase/HKL components", "Matched P/R; center/width error"],
        ["Gamma-Physics", "XRD + declared metadata", "Fraction, lattice constants, misfit, disorder", "Target-specific masked MAE"],
    ]
    y = draw_table(canvas, tasks, MARGIN, y, [1.0 * inch, 1.45 * inch, 2.15 * inch, CONTENT_W - 4.6 * inch]) - 14
    draw_two_columns(canvas, [
        ("Declared information settings", "HEA-known gives composition and parent topology to isolate occupancy evidence. HEA-blind removes these inputs and is an auxiliary harder setting. Every target mask is explicit; review metadata and routing labels cannot leak into model input."),
        ("Component matching", "Gamma components use deterministic bipartite matching. Invalid phase/HKL assignments receive prohibitive cost; valid matches combine normalized center, width, and profile distances."),
        ("Separate score levels", "The leaderboard reports pattern reconstruction, component recovery, physical parameters, and chemical order separately. No weighted aggregate may hide a physically incorrect solution behind a good spectral fit."),
    ], [
        ("OOD slices", "The main evaluation includes ID, composition OOD, occupancy/order OOD, instrument OOD, and simulation-to-experiment where public labels support the claimed target level."),
        ("Leakage controls", "Canonical structures, near duplicates, mixtures, occupancy variants, and all augmentations are grouped before splitting. Each release includes deterministic split configuration and an audit report."),
        ("Uncertainty", "Probabilistic entries report a proper scoring rule, interval or prediction-set coverage, and calibration error on both identifiable and ambiguous subsets."),
    ], y)


def page_six(canvas: Canvas) -> None:
    draw_header(canvas, 6, "Baselines")
    y = draw_section_title(canvas, "5", "Baseline Suite and Submission Interface", PAGE_H - 0.68 * inch)
    baseline_data = [
        ["Baseline", "Purpose", "Failure mode exposed"],
        ["Metadata-only", "Composition/parent/acquisition without XRD", "Dataset and composition shortcuts"],
        ["Peak features", "Detected positions and intensities + shallow model", "Dependence on engineered peak extraction"],
        ["1D ResNet", "Standard full-pattern neural baseline", "Pattern learning without explicit physics"],
        ["Sequence model", "Transformer or state-space encoder", "Long-range spectral interactions"],
        ["Physics-constrained", "Predict latent truth and regenerate XRD", "Benefit of forward consistency"],
        ["Pseudo-Voigt fit", "Constrained conventional component fitting", "Neural versus established decomposition"],
    ]
    y = draw_table(canvas, baseline_data, MARGIN, y, [1.25 * inch, 2.45 * inch, CONTENT_W - 3.7 * inch]) - 13
    draw_two_columns(canvas, [
        ("Common prediction contract", "Every baseline writes stable sample IDs, task name, point or distribution predictions, target masks, and optional reconstructed pattern references. One reference evaluator consumes all submissions."),
        ("Training policy", "Learned baselines use identical frozen splits and information settings. At least three random seeds are reported, together with compute budget and the hyperparameter search space."),
    ], [
        ("Sanity checks", "Perfect synthetic predictions must attain the exact metric optimum. Deliberately permuted component order must not change a valid set score. Deliberately wrong latent predictions must fail even when reconstructed XRD remains close."),
        ("Release interface", "A small smoke benchmark verifies installation and schema compatibility. Full generation and training environments are version locked, while tables are regenerated from frozen prediction artifacts."),
    ], y)


def page_seven(canvas: Canvas) -> None:
    draw_header(canvas, 7, "Planned experiments")
    y = draw_section_title(canvas, "6", "Experiments, Ablations, and Evidence Plan", PAGE_H - 0.68 * inch)
    y = draw_callout(canvas, "Status: experiments not yet run", "This page specifies the evidence required for submission. It intentionally contains no model scores, rankings, or claimed improvements.", MARGIN, y, CONTENT_W, CORAL) - 13
    exp = [
        ["Question", "Required comparison", "Paper output"],
        ["Does XRD add SRO information?", "XRD model vs metadata-only", "HEA-SRO ID/OOD table"],
        ["Do random splits inflate scores?", "Grouped OOD vs ordinary random split", "Shortcut inflation ablation"],
        ["Does hierarchy help?", "Latent + component supervision vs spectral loss", "Latent/spectral trade-off"],
        ["Can overlap be decomposed?", "Neural set predictor vs pseudo-Voigt", "Gamma component metrics"],
        ["Is uncertainty responsive?", "Identifiable vs ambiguous subsets", "Coverage and calibration plot"],
        ["What transfers to experiment?", "Synthetic training vs public experimental subset", "Supported-label gap only"],
    ]
    y = draw_table(canvas, exp, MARGIN, y, [1.65 * inch, 2.45 * inch, CONTENT_W - 4.1 * inch]) - 13
    draw_two_columns(canvas, [
        ("Required ablations", "Remove weak/superlattice peaks; remove composition metadata; remove component supervision; remove perturbation randomization; replace controlled grouping with random splits; remove latent losses and optimize only spectral reconstruction."),
        ("Reporting discipline", "Null and negative results remain. All learned models report multiple seeds or bootstrap intervals. Thresholds and normalizers are selected on training/validation data, never test scores."),
    ], [
        ("Main result layout", "Table 3 contains ID and primary OOD results for all four tasks. Figure 4 visualizes OOD degradation. Figure 5 plots spectral distance against latent distance and overlays uncertainty calibration."),
        ("Claim gate", "The final abstract receives numerical claims only after manifests, predictions, metric code, and generated tables are frozen and reproducible from the release tag."),
    ], y)


def page_eight(canvas: Canvas) -> None:
    draw_header(canvas, 8, "Limitations and readiness")
    y = draw_section_title(canvas, "7", "Limitations, Release, and Current Readiness", PAGE_H - 0.68 * inch)
    draw_two_columns(canvas, [
        ("What the benchmark can establish", "It can test whether a model recovers computational structure-derived SRO, declared component profiles, and phase parameters under controlled forward models and OOD perturbations."),
        ("What it cannot establish", "It cannot prove that conventional experimental powder XRD uniquely determines atomic occupancy. Periodic supercells may encode weak reflections differently from real diffuse scattering."),
        ("Experimental ambiguity", "Similar scattering factors, finite resolution, texture, strain, nanoscale broadening, and unmodeled phases can make inversion fundamentally ambiguous. Low spectral error is therefore never interpreted as unique structural recovery."),
    ], [
        ("Responsible release", "Every source requires license, DOI, checksum, and provenance. Reviewers must access the data and code without personal request. Dataset and benchmark cards state biases, intended uses, and unsupported claims."),
        ("KDD relevance", "The benchmark targets structured latent prediction, constrained inverse problems, data-centric OOD evaluation, calibration, and physical non-identifiability. Materials context defines the challenge; evaluation methodology provides the data-mining contribution."),
        ("Final paper package", "The submission will contain five main figures, three main tables, an audited related-work matrix, fixed schemas and splits, reference metrics, baseline code, and an ACM-formatted self-contained body."),
    ], y)
    status_y = 2.35 * inch
    status = [
        ["Workstream", "Current state", "Next gate"],
        ["HEA source + labels", "84,019 accepted; engineering audit passes", "Finish pilot review and controls"],
        ["HEA XRD", "1,000 clean + 1,000 augmented pilot patterns", "Freeze protocol; run full shards"],
        ["Gamma/gamma-prime", "Interfaces and design exist", "Freeze validated component dataset"],
        ["Evaluation", "Task and metric design exists", "Implement schemas, metrics, splits"],
        ["Paper", "Substantive working draft", "Add citations, figures, and frozen results"],
    ]
    draw_table(canvas, status, MARGIN, status_y, [1.35 * inch, 2.6 * inch, CONTENT_W - 3.95 * inch])


PAGES = [page_one, page_two, page_three, page_four, page_five, page_six, page_seven, page_eight]


def render(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=letter, pageCompression=1)
    canvas.setTitle("SiteXRD-Bench KDD 2027 Eight-Page Content Preview")
    canvas.setAuthor("SiteXRD-Bench working group")
    for index, draw_page in enumerate(PAGES):
        draw_page(canvas)
        if index < len(PAGES) - 1:
            canvas.showPage()
    canvas.save()
    return output


if __name__ == "__main__":
    path = render()
    print(path)
