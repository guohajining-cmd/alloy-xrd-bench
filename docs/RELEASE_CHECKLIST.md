# Release Checklist

## HEA v1.1 SRO Core

- Confirm `/home/GJM/data/hea/benchmark_v1` is unchanged and all new dataset
  artifacts resolve under `/home/GJM/data/hea/benchmark_v1_1`.
- Verify zero `parent_group_id` leakage for all four split columns.
- Verify Composition Ridge once per split and Peak XGBoost, ResNet1D,
  and Transformer for seeds 2027-2031 per split (64 runs total).
- Reject missing, duplicate, unexpected, or identity-mismatched prediction IDs.
- Verify matched perturbation reports for all three XRD models and five seeds.
- Verify occupancy noise coverage is parent-specific for all 1,000 parents and
  ambiguous pairs are absent from ranking metrics.
- Verify parent bootstrap uses 2,000 replicates and seed 2027.
- Verify reports describe Family and occupancy as diagnostics and simulated
  perturbations as not calibrated to repeated experimental noise.
- Verify Zenodo DOI `10.5281/zenodo.10854500`, CC BY 4.0, source checksum,
  manifests, file list, and SHA-256 audit.
- Run `python -m alloy_xrd_bench.hea_benchmark_readiness` and require
  `core_benchmark_ready=true`.

## Gamma-Decomp v1.1

- Validate the public core release and require zero audit issues.
- Verify full-pattern and window tasks use parent-pair-isolated splits.
- Verify exact oracle metrics for both tasks.
- Verify no-spectrum, peak-feature, ResNet set, Transformer set, and
  physics-constrained results for seeds 2027-2029 as applicable.
- Verify pseudo-Voigt BIC on the window task.
- Verify reconstruction-only, weak-peak, clean-training, and random-split
  ablations are complete.
- Verify all paired comparisons use 2,000 parent-pair bootstrap replicates and
  seed 2027.
- Verify deterministic CuBLAS and attention kernels are enabled for neural
  training.
- Run `python -m alloy_xrd_bench.gamma_overlap_readiness` and require
  `gamma_benchmark_ready=true`.

## Combined v1.1

- Generate HEA and Gamma paper tables and figures from frozen score artifacts.
- Run `bash scripts/workstation/finalize_sitexrd_v1_1.sh` and require
  `benchmark_ready=true`.
- Report the two track metrics independently; do not publish a cross-task
  composite score.
- Keep `benchmark_ready` separate from the stricter `kdd_submission_ready`
  publication-material gate.

## General Release

- Run schema tests for every manifest.
- Verify ID uniqueness in every manifest with an ID field.
- Verify all foreign keys resolve.
- Verify every `xrd_sample_id` traces to one relaxed structure, relaxation run,
  potential, and initial structure.
- Verify canonical structure hashes match JSON artifacts.
- Verify CIF files are not used as canonical hash inputs.
- Verify relaxed structures have no element loss, no abnormal nearest-neighbor
  distance, and no abnormal volume change unless explicitly flagged.
- Verify XRD grids are non-empty, peaks are inside the two-theta range, and
  intensities are normalized to 100.
- Verify private test rows do not share `relaxed_structure_id` with released
  train, validation, or public test rows.
- Verify non-redistributable potentials are not packaged in release artifacts.
- Verify every directory under `data/potentials/` is declared in
  `potentials_catalog.yml`.
- Verify `CITATION.cff`, `croissant.jsonld`, `license_audit.csv`, and
  `leakage_audit.csv` are updated for the release version.
- Run `python -m alloy_xrd_bench.validate <release-root> --json`.
- Run `python -m alloy_xrd_bench.audit <release-root> --output-dir audits --release-version <version>`.
- Run `python -m alloy_xrd_bench.quality <release-root> --output-dir audits`.
- Run `python -m alloy_xrd_bench.metadata croissant <release-root>`.
- Run `python -m alloy_xrd_bench.package <release-root> --archive <release-root>.zip`.
- Run `python -m alloy_xrd_bench.release_readiness <release-root> --archive <release-root>.zip --json`.
- After external upload, fill `publication_record.yml` and run
  `python -m alloy_xrd_bench.release_readiness <release-root> --archive <release-root>.zip --publication-record <release-root>/publication_record.yml --require-publication --json`.
- Verify `file_manifest.parquet`, `checksums.sha256`, and the zip archive exist.
