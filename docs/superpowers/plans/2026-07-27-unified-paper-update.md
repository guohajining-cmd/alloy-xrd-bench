# Unified SiteXRD-Bench Paper Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present SiteXRD-Bench consistently as one versioned benchmark with independent HEA-SRO and gamma/gamma-prime decomposition tracks.

**Architecture:** Keep all frozen data, metrics, tables, and figures unchanged. Update only manuscript terminology and add contract tests that prevent the two tracks from being described as separate benchmarks or Gamma-Decomp as a gamma-only dataset.

**Tech Stack:** LaTeX (`acmart`), Python `unittest`/pytest contract tests, BibTeX, Poppler PDF rendering.

---

### Task 1: Freeze the unified terminology contract

**Files:**
- Modify: `tests/test_dataset_first_paper_contract.py`

- [ ] **Step 1: Add failing assertions**

Add a test that reads the main paper and requires these exact concepts:

```python
self.assertIn("one benchmark with two independent task tracks", paper)
self.assertIn("Gamma-Decomp (gamma/gamma-prime decomposition)", paper)
self.assertIn("No cross-track composite score is defined", paper)
self.assertNotIn("two independent benchmarks", paper)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```powershell
python -m pytest tests\test_dataset_first_paper_contract.py -q
```

Expected: the new terminology test fails before the manuscript edit.

### Task 2: Update the formal manuscript

**Files:**
- Modify: `docs/paper/sitexrd_bench_kdd2027.tex`

- [ ] **Step 1: Unify the abstract**

Replace the first benchmark definition with wording that contains:

```text
SiteXRD-Bench is one benchmark with two independent task tracks.
```

Define the second track on first use as:

```text
Gamma-Decomp (gamma/gamma-prime decomposition)
```

Retain the frozen counts of 84,019 HEA structure-XRD pairs, 10,000 Gamma
mixtures, 21,116 windows, and 4,000 occupancy diagnostics.

- [ ] **Step 2: Align section and table terminology**

Use `Gamma/Gamma-Prime Decomposition Data` as the section's descriptive title,
while retaining `Gamma-Decomp` as the stable task ID. Update the dataset/task
contract caption to state that both rows belong to one release and are scored
independently.

- [ ] **Step 3: Align conclusion and limitations**

State that no cross-track composite score exists because SRO matrix regression
and unordered component recovery have different target spaces and metrics.
Do not add claims of exact experimental occupancy recovery, experimental
component truth, mechanical-property prediction, or creep prediction.

- [ ] **Step 4: Run focused contract tests**

Run:

```powershell
python -m pytest tests\test_dataset_first_paper_contract.py tests\test_benchmark_readiness.py -q
```

Expected: all tests pass.

### Task 3: Rebuild and inspect the anonymous PDF

**Files:**
- Regenerate: `docs/paper/sitexrd_bench_kdd2027.pdf`

- [ ] **Step 1: Compile the manuscript**

Run from `docs/paper`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error sitexrd_bench_kdd2027.tex
bibtex sitexrd_bench_kdd2027
pdflatex -interaction=nonstopmode -halt-on-error sitexrd_bench_kdd2027.tex
pdflatex -interaction=nonstopmode -halt-on-error sitexrd_bench_kdd2027.tex
```

Expected: compilation succeeds with no undefined references or horizontal
overfull boxes.

- [ ] **Step 2: Scan generated text**

Use `pdftotext` to verify the PDF contains the unified definition and contains
none of `TODO`, `Working Draft`, fake DOI placeholders, or gamma-only wording.

- [ ] **Step 3: Render every page**

Run:

```powershell
pdftoppm -png -r 130 sitexrd_bench_kdd2027.pdf ..\..\tmp\pdfs\unified\page
```

Inspect all pages for clipped tables, overlapping figures, unreadable captions,
and broken anonymous-review headers.

- [ ] **Step 4: Run the full repository test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests and subtests pass.
