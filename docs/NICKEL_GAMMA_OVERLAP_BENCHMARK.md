# Ni-Al Gamma/Gamma-Prime Overlap Benchmark

## Scientific Task

The benchmark asks a model to explain an observed Ni-Al powder XRD pattern as
an unordered set of physically named reflections. Each scored component has a
phase (`gamma` or `gamma_prime`), a parent-lattice HKL, center, FWHM,
integrated intensity, and pseudo-Voigt mixing parameter. The full-pattern task
must locate and explain overlap regions; the window task receives a fixed
3-degree crop and must decide whether it contains one or multiple components.

Element occupancy, phase-specific lattice constants, and lattice misfit are not
prediction targets for this release. Mixture weights and acquisition
perturbations are generation metadata only.

## Sources

Gamma-prime parents come from DaRUS-4480
(`https://doi.org/10.18419/DARUS-4480`, CC BY 4.0). The local CFG contains 1,162
DFT configurations with energy, force, and stress labels. The v1 core admits
the 107- and 108-atom, parent-indexable configurations. The 96- and 144-atom
complex supercells are retained in the OOD quarantine manifest.

Gamma parents are deterministic 3x3x3 FCC Ni97Al11 structures generated with
the repository's pair-balance occupancy method. They are synthetic benchmark
assets, not experimentally refined commercial-superalloy structures.
Their parent lattice parameters are deterministically sampled from the frozen
values 3.52427477 and 3.57 angstrom. These values summarize repository-audited
Ni-Al FCC relaxations from CHGNet 0.3.0, MACE-MP-0b3, and
M3GNet-MatPES-PBE-2025.2; they are computational priors, not experimental
measurements. Every gamma parent records this source string in
`parent_structures.parquet`.

## Parent-HKL Projection

Raw finite DFT snapshots repeat as low-symmetry supercells and can produce
artificial fractional-supercell reflections. The release maps supercell HKLs
back to the 3x3x3 FCC/L12 parent lattice. Integer parent reflections form the
scored component truth. Peaks without an integer parent mapping are retained in
`satellite_reflections.parquet` with `scored=false` and are not silently
presented as parent Bragg peaks.

## Build

```powershell
python -m alloy_xrd_bench.gamma_overlap pilot `
  --root . `
  --cfg local_potential_cache/pot_ni3al_mtp_darus_yield_anomaly_2024/downloads/train.cfg `
  --output data/releases/nickel_gamma_overlap_v1_pilot `
  --config configs/nickel_gamma_overlap_v1.json
```

Use `generate` instead of `pilot` for the 400-pair, 25-view production target.
Run `audit` after any copied or modified release. The audit reconstructs every
clean pattern from component parameters and checks arrays, parent HKLs,
artifact SHA-256 values, hidden private labels, window balance, and parent-pair
split isolation. The audited 288-pattern pilot is at
`data/releases/nickel_gamma_overlap_v1_pilot_20260725_v3`. The audited formal
release is at `data/releases/nickel_gamma_overlap_v1_production_20260725_v3`.

The formal release contains 400 unique parent pairs and 10,000 complete
patterns (7,050 train, 1,475 validation, and 1,475 private test), with 175,150
component rows and 21,116 selected windows. It uses 128 unique gamma parents
and 400 unique gamma-prime parents. The independent audit records 432,542
passing checks, zero failures, zero reconstruction error, and maximum
cross-split parent-pair cosine similarity 0.9983869631 under the frozen 0.999
threshold.

The window manifest is a deterministic balanced subset of all natural support-
connected windows. Single-component negatives occupy at least 20%, and within
each split the largest non-single difficulty class is at most three times the smallest
represented class. Complete patterns and their component truth are never
downsampled by this policy.

Source-run membership, shared gamma parents, and near-duplicate representative
clean patterns are converted to connected components before split assignment.
This constraint gives 282/59/59 parent pairs across train/validation/private
test; the slight deviation from 70/15/15 is required to keep connected
components intact.

## Evaluation

Predicted components are matched with deterministic Hungarian assignment.
Phase/HKL mismatches cannot match. Primary outputs are component
precision/recall/F1, phase/HKL accuracy, count accuracy, and matched errors for
center, FWHM, and integrated intensity. Window multi-component classification
F1 and full-pattern overlap-region F1 are separate metrics. Forward
reconstruction error is supplementary and does not replace correct component
identity.

## Public Track B Package

The KDD package maps internal `private_test` to public `test` and publishes all
three label splits. Test is reserved for final reporting and must not be used
for tuning. The release is split into:

- `gamma-decomp-core-v1`: model inputs, full/window labels, split manifests,
  schemas, baseline priors, audits, and checksums.
- `gamma-decomp-provenance-v1`: parent structures, phase reflections,
  provenance edges, and the source physical audit.

The core release contains 10,000 complete patterns and 21,116 windows. Window
labels contain 38,045 physical components: 25,652 score-eligible components
and 12,393 weak context components. Only `sample_id`, `task_id`, two-theta, and
observed intensity are allowed model inputs; provenance and generation fields
are excluded.

```powershell
python -m alloy_xrd_bench.gamma_overlap make-submission-template `
  --release data/releases/gamma-decomp-v1/gamma-decomp-core-v1 `
  --task window --split validation --output output/submission
python -m alloy_xrd_bench.gamma_overlap evaluate `
  --release data/releases/gamma-decomp-v1/gamma-decomp-core-v1 `
  --task window --split validation `
  --submission output/submission --output output/score
python -m alloy_xrd_bench.gamma_overlap train-baseline `
  --release data/releases/gamma-decomp-v1/gamma-decomp-core-v1 `
  --task window --model resnet-set --output output/window_resnet
```
