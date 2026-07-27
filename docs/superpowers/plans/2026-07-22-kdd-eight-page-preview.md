# KDD Eight-Page Content Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an exactly eight-page, clearly labeled working PDF that shows the current SiteXRD-Bench paper content without fabricating unavailable results.

**Architecture:** A small ReportLab renderer creates a deterministic US-letter PDF from frozen page-level content. The output separates established dataset facts from planned benchmark experiments, then PDF text extraction and page rendering verify count, legibility, and status labeling.

**Tech Stack:** Python 3, ReportLab, pypdf, Poppler `pdftoppm`.

---

### Task 1: Add the deterministic preview renderer

**Files:**
- Create: `scripts/render_kdd_content_preview.py`

- [x] Define page primitives for headers, footers, section titles, paragraphs, metric cards, and status tables.
- [x] Define eight page payloads matching the approved paper design.
- [x] Mark missing baseline results and the unfinished gamma/gamma-prime dataset as pending work rather than inserting values.
- [x] Write the final PDF to `output/pdf/sitexrd_bench_kdd2027_eight_page_content_preview.pdf`.

### Task 2: Generate and validate the PDF

**Files:**
- Create: `output/pdf/sitexrd_bench_kdd2027_eight_page_content_preview.pdf`

- [x] Run `python scripts/render_kdd_content_preview.py` and require exit code 0.
- [x] Use `pypdf` to assert exactly eight pages and verify the title plus the working-draft warning are extractable.
- [x] Render all pages to PNG with `pdftoppm` under `tmp/pdfs/sitexrd_preview/`.

### Task 3: Visual review

**Files:**
- Inspect: `tmp/pdfs/sitexrd_preview/page-1.png` through `page-8.png`

- [x] Inspect a contact sheet and representative full-resolution pages.
- [x] Confirm there is no clipping, overlap, broken glyph, or accidental claim that pending experiments are complete.
- [x] Report the PDF path and summarize the content of all eight pages.
