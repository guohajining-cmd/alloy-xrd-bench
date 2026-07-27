# Alloy-XRD-Bench

Alloy-XRD-Bench is a two-line benchmark contract for simulated alloy XRD data.
The structure line produces auditable relaxed structures, and the XRD line uses
those relaxed structures as the direct source for standardized diffraction
samples.

The public wording is:

```text
xrd_sample <- relaxed_structure_id
```

## HEA Benchmark v1.1

The HEA v1.1 release scope is `sro_core`. HEA-SRO is the sole leaderboard task;
Family and occupancy identifiability are disclosed diagnostics. The release
does not publish an overall composite score and does not claim experimental
noise calibration.

Run the complete workstation validation with:

```bash
bash scripts/workstation/run_hea_validation.sh
```

Generate the dataset-first paper figures only from the formal HEA and Gamma
v1.1 releases (the command fails if either formal manifest is unavailable):

```bash
bash scripts/workstation/generate_dataset_paper_visuals.sh
```

The runner writes four figures, deterministic sample selections, an input/output
checksum manifest, and the compact primary-results table under
`/home/GJM/runs/sitexrd_benchmark_v1_1/paper_visuals`, then synchronizes them to
`docs/paper/generated`.

Acceptance is the machine-readable result
`core_benchmark_ready=true` from `alloy_xrd_bench.hea_benchmark_readiness`.
The command requires exact predictions for four splits and the frozen model
matrix, matched perturbation evaluation, parent bootstrap evidence, and a
checksummed CC-BY-4.0 release package. See
`docs/HEA_BENCHMARK_CARD_V1_1.md` for the normative task boundary.
Physics-constrained HEA runs are appendix experiments and use a separate run
root. They can be generated with:

```bash
bash scripts/workstation/run_hea_physics_matrix.sh
```

The appendix runner defaults to `sitexrd_hea_validation_v1_1_physics_appendix`
and does not update the formal HEA prediction audit or readiness evidence.

## Gamma-Decomp v1.1

Gamma-Decomp is the second primary task. It evaluates unordered
gamma/gamma-prime phase/HKL-group components, component count, center, FWHM,
integrated intensity, calibration, and supplementary forward reconstruction.
Raw gamma-prime `(221)` and `(300)` reflections remain in provenance but form a
single degenerate scoring group. Model inputs contain only sample ID, task ID,
two-theta, and observed intensity; overlap labels and parent groups are separate
evaluation manifests. A clean spectrum is available only for train and only for
the perturbation-training ablation.

Run the resumable workstation matrix with:

```bash
bash scripts/workstation/run_gamma_decomp_v1_1.sh
```

The matrix includes no-spectrum, peak-feature, pseudo-Voigt, ResNet set,
Transformer set, physics-constrained, reconstruction-only, weak-peak,
clean-training, and deliberately leaky random-split controls. Readiness is the
machine-readable `gamma_benchmark_ready` field emitted by
`alloy_xrd_bench.gamma_overlap_readiness`.

## Complete Benchmark v1.1

Alloy-XRD-Bench v1.1 is ready only when both independent tracks pass. Run:

```bash
bash scripts/workstation/finalize_sitexrd_v1_1.sh
```

The final report must contain `benchmark_ready=true`. HEA-SRO and
Gamma-Decomp retain separate metrics and leaderboards; no cross-task composite
score is published. `kdd_submission_ready` is a stricter, separate gate for
frozen paper and publication materials. See
`docs/SITEXRD_BENCHMARK_CARD_V1_1.md` for the combined contract.

MLIP choice, initial structure, random seed, relaxation parameters, and logs are
recorded as provenance. They are not described as the direct source of an XRD
label.

## Repository Layout

```text
alloy_xrd_bench/       Runtime contract and validation helpers
configs/               Default XRD and relaxation configurations
docs/                  Benchmark contract and release policy
schemas/               Manifest schema files
templates/manifests/   Minimal manifest row examples
tests/                 Contract tests
audits/                Release audit CSV skeletons
MODEL_CARDS/           Model card skeletons for benchmark submissions
data/                  Shared inputs and per-potential data workspaces
```

The data layout is potential-scoped:

```text
data/potentials/{potential_id}/potential/   Potential files or metadata
data/potentials/{potential_id}/generated/   Data produced using that potential
```

Initial structures live under `data/shared/` because they are benchmark inputs,
not outputs of a specific potential.

## Verification

Run the contract test suite:

```powershell
python -m unittest discover -s tests
```

The tests cover required manifest fields, schema file synchronization, default
simulation thresholds, canonical structure hashing, provenance reachability,
private split isolation, and potential redistribution policy.

Validate a workspace or release:

```powershell
python -m alloy_xrd_bench.validate data/examples/minimal_workspace --json
```

Write a local status report with validation counts, per-potential manifest
counts, package versions, and detected local MLIP files:

```powershell
python -m alloy_xrd_bench.status . --json --output audits/workspace_status.json
```

Generate the minimal example workspace:

```powershell
python scripts/make_minimal_workspace.py --root data/examples/minimal_workspace
```

Register a potential workspace:

```powershell
python -m alloy_xrd_bench.potentials register --help
```

Generate the v0.1 shared initial structures:

```powershell
python -m alloy_xrd_bench.structures generate-initial --root . --recipes configs/alloy_recipes_v0_1.yml
```

Simulate XRD after a potential has produced relaxed structures:

```powershell
python -m alloy_xrd_bench.xrd simulate --root . --potential-id pot_example
```

Build and audit the Ni-Al gamma/gamma-prime overlapping-peak pilot:

```powershell
python -m alloy_xrd_bench.gamma_overlap pilot `
  --root . `
  --cfg local_potential_cache/pot_ni3al_mtp_darus_yield_anomaly_2024/downloads/train.cfg `
  --output data/releases/nickel_gamma_overlap_v1_pilot_20260725_v3
python -m alloy_xrd_bench.gamma_overlap audit `
  --output data/releases/nickel_gamma_overlap_v1_pilot_20260725_v3
```

See [NICKEL_GAMMA_OVERLAP_BENCHMARK.md](docs/NICKEL_GAMMA_OVERLAP_BENCHMARK.md)
for the source policy, parent-HKL projection, release tables, and scoring task.

Export and validate the public KDD Track B package:

```powershell
python -m alloy_xrd_bench.gamma_overlap export-public `
  --source data/releases/nickel_gamma_overlap_v1_production_20260725_v3 `
  --output data/releases/gamma-decomp-v1 `
  --include-provenance
python -m alloy_xrd_bench.gamma_overlap validate-release `
  --release data/releases/gamma-decomp-v1/gamma-decomp-core-v1
```

Create/evaluate submissions or train the registered physics and set baselines
with `make-submission-template`, `evaluate`, and `train-baseline`. Track B is
reported independently and does not enter the HEA Track A aggregate score.

Generate per-composition averaged XRD:

```powershell
python -m alloy_xrd_bench.xrd average --root . --potential-id pot_example
```

Build a staged occupancy-aware benchmark from a relaxed-structure CSV:

```powershell
python -m alloy_xrd_bench.occupancy_xrd inventory --csv <source.csv> --output <dataset>
python -m alloy_xrd_bench.occupancy_xrd label --dataset <dataset> --pilot-size 1000
python -m alloy_xrd_bench.occupancy_xrd split --dataset <dataset>
python -m alloy_xrd_bench.occupancy_xrd simulate --dataset <dataset> --subset pilot --augmentations 1
python -m alloy_xrd_bench.occupancy_xrd review-export --dataset <dataset> --output <review-dir> --batch-size 50
python -m alloy_xrd_bench.occupancy_xrd review-apply --dataset <dataset> --input <review.csv>
python -m alloy_xrd_bench.occupancy_xrd audit --dataset <dataset>
```

This path produces accounted structure manifests, BCC/FCC/L12/B2 occupancy
labels, Warren-Cowley SRO values, leakage-resistant split schemes, HDF5 XRD
grid shards, Parquet peak shards, and a benchmark-readiness report. Engineering
audit success does not bypass the required manual review of the balanced pilot.
The review export contains editable CSV files, CIF structures, XRD previews, and
an HTML index. Apply each edited CSV independently; unknown IDs and invalid
review states are rejected before the canonical review queue is changed.

Apply v0.1 split labels:

```powershell
python -m alloy_xrd_bench.splits apply --root . --config configs/splits_v0_1.yml
```

Run the engineering smoke relaxation path:

```powershell
python -m alloy_xrd_bench.relax smoke --root . --potential-id pot_example
```

Run CHGNet relaxation after registering `pot_chgnet_0p3p0`:

```powershell
python -m alloy_xrd_bench.relax chgnet --root . --potential-id pot_chgnet_0p3p0 --use-device cpu
```

Generate release audit CSVs:

```powershell
python -m alloy_xrd_bench.audit . --output-dir audits --release-version v0.1
```

Run physics sanity audits:

```powershell
python -m alloy_xrd_bench.quality . --output-dir audits
```

Refresh release-specific Croissant metadata:

```powershell
python -m alloy_xrd_bench.metadata croissant data/releases/v0.1-working
```

Package a release directory:

```powershell
python -m alloy_xrd_bench.package data/releases/v0.1-working --archive data/releases/v0.1-working.zip
```

Check whether a packaged release is ready to hand off to an external archive:

```powershell
python -m alloy_xrd_bench.release_readiness data/releases/v0.1-working `
  --archive data/releases/v0.1-working.zip `
  --json `
  --output audits/release_readiness_v0.1-working.json
```

After uploading the archive and filling `publication_record.yml`, verify the
final DOI/URL handoff metadata:

```powershell
python -m alloy_xrd_bench.release_readiness data/releases/v0.1-working `
  --archive data/releases/v0.1-working.zip `
  --publication-record data/releases/v0.1-working/publication_record.yml `
  --require-publication `
  --json
```

See [EXECUTION_WORKFLOW.md](docs/EXECUTION_WORKFLOW.md) for the potential
registration and generated-data workflow.
