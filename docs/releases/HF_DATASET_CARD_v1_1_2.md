---
language:
- en
license: cc-by-4.0
pretty_name: Alloy-XRD-Bench
tags:
- materials-science
- x-ray-diffraction
- benchmark
- short-range-order
- scientific-machine-learning
configs:
- config_name: hea_sro_corrected_labels_v1_1_2
  default: true
  data_files:
  - split: labels
    path: corrections/v1.1.2/labels.parquet
---

# Alloy-XRD-Bench v1.1.2

Alloy-XRD-Bench is a first-stage benchmark toward alloy powder-XRD refinement.
HEA-SRO and Gamma-Decomp are complementary local-order and component-decomposition
subproblems with separate metrics. A complete joint refinement pipeline is a
future objective, not a capability demonstrated by this release.

## October 2026 correction

**Use the corrected SRO-label overlay for new evaluations.** Version 1.1.2 fixes
zero-count target handling, while preserving the original observations, state
identities, and frozen split membership. A defined shell with zero occurrences
of a neighbor species has alpha=1 for that pair; undefined central-species
shells remain masked. Historical and corrected target masks are not interchangeable.

The overlay contains 88,019 states (84,019 source structures plus 4,000 fixed-cell
counterfactual states), with 669,370 added defined target entries. Independent
coordinate-based checks covered 1,819 states. This is arithmetic verification
under the declared shell convention, not experimental SRO validation or
finite-size convergence.

## Downloads

1. Download the existing [v1.1 base archive](sitexrd_benchmark_data_v1_1.tar) and
   its [checksum](sitexrd_benchmark_data_v1_1.tar.sha256). These legacy files are
   retained unchanged.
2. Download [the v1.1.2 correction archive](corrections/v1.1.2/alloy_xrd_bench_v1.1.2_corrections.zip)
   or the files in [corrections/v1.1.2](tree/main/corrections/v1.1.2).
3. Check [SHA256SUMS.txt](corrections/v1.1.2/SHA256SUMS.txt), then use
   `labels.parquet` and `revision.json` together through the corrected code.

The correction directory includes label-change and legacy-split diagnostics,
corrected-result summaries, paired comparisons, a revised manuscript, and a
portable release manifest. It does not contain a new collection of XRD spectra.
The root base archive and historical individual-file exports retain their old
labels; apply the overlay rather than assuming they were silently replaced.

The viewer's `labels` split is an all-states sidecar, **not a training partition**.
Use the original task manifest for train/validation/test and parent-group
membership. The legacy SRO OOD membership remains frozen despite the label fix;
it is not revalidated as a newly defined label-shift split.

## Usage and links

- [Versioned code](https://github.com/guohajining-cmd/alloy-xrd-bench/tree/v1.1.2)
- [Correction instructions and limitations](https://github.com/guohajining-cmd/alloy-xrd-bench/blob/v1.1.2/docs/releases/v1.1.2.md)
- [Versioned dataset snapshot](https://huggingface.co/datasets/jimi7771/alloy-xrd-bench/tree/v1.1.2)
- [Revised manuscript](corrections/v1.1.2/paper/Alloy-XRD-Bench.pdf)
- [Original v1.1.1 archive DOI](https://doi.org/10.5281/zenodo.21620650)

Set `sro_label_revision` in the HEA validation configuration to your local
correction directory. `alloy_xrd_bench.sro_revision.apply_label_revision`
verifies the base manifest and label hashes, target order, identity fields,
and full state coverage before applying changes. It does not overwrite the
base manifest.

## Evaluation and limitations

Corrected HEA results and paired comparisons are under `results/` in the
correction directory. The HEA primary metric remains masked parent-macro MAE;
RMSE, shell correlations, and the relative SRO score are reported alongside it.
The same-learner composition versus composition-plus-peak contrast must be
distinguished from the model leaderboard. Parent intervals condition on fitted
models and are not estimates of experimental label uncertainty.

Gamma-Decomp retains 10,000 synthetic mixtures and 21,116 windows in the base
release. The v1.1.2 code fixes evaluation/public-input handling; a complete
corrected Gamma matrix remains pending. Historical Gamma files and scores must
not be described as a newly completed corrected evaluation.

The old perturbation renderer confounded width changes with relative integrated
peak area. A replacement renderer is implemented, but matched reevaluation
remains incomplete. Neither the old negative result nor this upload establishes
measured-noise calibration or an experimental identifiability limit.

Experimental SRO research exists. Benchmark-ready powder-XRD observations paired
with independently quantified, sample-matched SRO labels are rare among the
resources considered here. This is a scoped resource assessment, not a census.
The current benchmark does not supply such experimental latent targets.
Existing measured patterns support phase/whole-pattern checks. Traditional
Rietveld/full-pattern and privileged-information baselines, finite-size and
configuration-sampling studies, independent experimental target validation,
measured-noise calibration, and diffuse-scattering validation remain open.

## Provenance and licenses

The source HEA dataset is *DFT dataset for high entropy materials*, Kangming Li
and Jason Hattrick-Simpers, DOI `10.5281/zenodo.10854500` (CC BY 4.0). New
benchmark correction artifacts are CC BY 4.0. Retain the original source
attribution and per-source license notices when using the base release.

Legacy experimental files retain their own licenses. The AlCoCrFeNi source is
`10.5281/zenodo.4623002`; the IN718/AM718R source is `10.17863/CAM.87406`.
The latter has a documented BY-ND version discrepancy between registry and
repository metadata; this correction does not resolve that discrepancy or
grant broader rights. No additional raw experimental data are redistributed
in the correction package.

## Acknowledgments

This work was financially supported by the Advanced Materials-National Science
and Technology Major Project (Grant No. 2025ZD0620100), National Natural Science
Foundation of China (Grant No. 52401015), and Shanghai Artificial Intelligence
Open Source Award Project.
