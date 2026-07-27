# Alloy-XRD-Bench Benchmark Contract

## Scope

Alloy-XRD-Bench treats structures as first-class, traceable data assets. The
benchmark is still an XRD benchmark: every XRD sample is directly derived from a
`relaxed_structure_id`. MLIP files, initial structures, random seeds, relaxation
parameters, and logs are recorded in provenance so the structure line can be
audited without describing XRD labels as `xrd_from_mlip`.

## Data Products

- Structure Corpus: initial structures, relaxation runs, relaxed structures,
  logs, convergence status, and structure quality checks.
- XRD Benchmark: standardized continuous grids, peak tables, split metadata,
  labels, and task metadata.
- MLIP Registry: potential source, DOI or URL, license, elements, phases,
  engine, file hash, and redistribution policy.
- Provenance Graph: edge table linking initial structure, relaxation run,
  relaxed structure, XRD sample, potential, relaxation config, and XRD config.

## Default Phase 1 Parameters

The default XRD configuration is stored in
`configs/xrd_config_default.json`:

- Radiation: Cu Kalpha.
- Two-theta range: 20.0 to 100.0 degrees.
- Step: 0.02 degrees.
- Broadening: Gaussian, 0.1 degree FWHM.
- Intensity normalization: max intensity equals 100.0.

The default relaxation configuration is stored in
`configs/relax_config_default.json`:

- Force threshold: `fmax_eV_per_A <= 0.05`.
- Max steps: 1000.
- Abnormal volume change threshold: absolute fractional change above 0.20.

## Manifest Interfaces

The public manifests are:

- `potentials_catalog.yml`
- `initial_structures.parquet`
- `relaxations.parquet`
- `relaxed_structures.parquet`
- `xrd_samples.parquet`
- `provenance_edges.parquet`

The canonical schema definitions live under `schemas/`. JSONL examples live
under `templates/manifests/` for easy inspection and test fixture creation.

## Storage Layout

Every potential has an isolated workspace:

```text
data/potentials/{potential_id}/
  potential/
    metadata.yml
    checksums.sha256
    files/
  generated/
    manifests/
    relaxation_logs/
    structures_relaxed/
    xrd/
      grid/
      peaks/
  audits/
```

The `potentials_catalog.yml` row for a potential must declare:

- `potential_dir`: `data/potentials/{potential_id}/potential`
- `generated_data_dir`: `data/potentials/{potential_id}/generated`

Every directory under `data/potentials/` must correspond to exactly one
`potentials_catalog.yml` row. Unregistered potential directories are invalid,
even when they contain only scaffold files, because they make the provenance
inventory ambiguous.

Any artifact produced by that potential must live under its
`generated_data_dir`. This includes relaxation logs, relaxed structure JSON/CIF
files, XRD grids, and XRD peak tables.

Shared initial structures live under:

```text
data/shared/structures_initial/
```

Initial structures are shared inputs and are not considered data produced by a
specific potential. A relaxation run records which shared initial structure and
which potential produced each relaxed structure.

## ID Rules

IDs are stable strings with human-readable prefixes:

- `pot_`: potential entry.
- `init_`: initial structure.
- `relax_`: relaxation run.
- `rstr_`: relaxed structure.
- `xrd_`: XRD sample.

Config IDs are versioned strings, for example
`xrd_cu_kalpha_20_100_step_0p02_gauss_0p1_v1`.

## Structure Hash Rule

The canonical structure hash is a SHA-256 digest over compact, sorted,
UTF-8-encoded pymatgen JSON bytes. CIF files are convenience exchange artifacts
only and are not used as canonical hashes because CIF serialization can vary
without changing the structure.

## Provenance Rule

Every `xrd_sample_id` must trace to exactly one chain:

```text
initial_structure -> relaxation_run -> relaxed_structure -> xrd_sample
```

The same chain must also connect the relaxation run to `potential_file` and
`relax_config`, and the XRD sample to `xrd_config`.

The required provenance edges are:

```text
initial_structure -> relaxation_run      relation_type=input_to
potential_file -> relaxation_run         relation_type=used_by
relax_config -> relaxation_run           relation_type=configured_by
relaxation_run -> relaxed_structure      relation_type=produced
relaxed_structure -> xrd_sample          relation_type=simulated_as
xrd_config -> xrd_sample                 relation_type=configured_by
```

## Split and Leakage Rule

Supported split names are:

- `id`
- `composition_ood`
- `chemistry_ood`
- `phase_ood`
- `private_test`

During a competition, private test releases must not include relaxed structures
or XRD labels. They may include only input recipe, composition, phase, and task
metadata. After the public reveal, hidden provenance can be released for audit.

Private test rows must not share `relaxed_structure_id` with released rows.

## Evaluation and Leaderboard Rule

The normative HEA v1.1 scoring definition is `docs/SCORING_PROTOCOL.md`.
The release scope is `sro_core`: SRO recovery is the only leaderboard task.
ID, composition OOD, chemistry OOD, SRO OOD, and matched perturbation results
are reported without an overall composite score. Family and occupancy
identifiability are nonblocking diagnostics.

The composition-only baseline is mandatory. Parent structures, their
counterfactual occupancies, and all observation views are one statistical and
resampling group. Missing, duplicate, or unexpected prediction IDs invalidate a
submission; the evaluator never imputes predictions.

Non-identifiable occupancy pairs are not scored as ranking negatives. They
remain for abstention and identifiability analysis under the fixed 0.95
simulated-noise quantile. Real repeated-measurement calibration is deferred to
v1.2.

`classification_confidence` is dataset-admission metadata only. It is not a
scientific ground-truth target, model confidence, or leaderboard metric.

## HEA Versioned Remediation Rule

The released HEA v1 dataset at `/home/GJM/data/hea/benchmark_v1` is an
immutable, read-only source. Remediation commands may read it but must never
create, modify, rename, or delete any path beneath that root.

Every generated HEA v1.1 dataset artifact must resolve beneath
`/home/GJM/data/hea/benchmark_v1_1`. Run outputs may use a separate versioned
run directory, but they must not be written into the v1 source dataset. A
remediation command must fail before writing when its resolved output path is
the v1 root, is beneath the v1 root, or is outside its explicitly configured
v1.1 output root.

Dataset readiness and model capability are independent decisions. The
machine-readable SRO-core gates in `configs/hea_benchmark_v1_1.json` determine
whether v1.1 is publishable. Family review, occupancy identifiability count, and
model gain over Ridge cannot mark the SRO dataset ready or unready.

## Release Policy

Potential files are packaged only when their license explicitly allows
redistribution. Otherwise the release includes metadata, source URL or DOI,
checksum, and installation instructions. The audit file
`audits/license_audit.csv` records that decision for each potential.
