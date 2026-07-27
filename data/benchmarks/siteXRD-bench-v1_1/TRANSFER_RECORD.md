# SiteXRD-Bench v1.1 Local Transfer Record

Transferred from `GJM@58.199.170.56:7895` on 2026-07-26.

## Dataset Roots

- `hea/benchmark_v1`: canonical HEA structures and clean XRD.
- `hea/benchmark_v1_1`: HEA v1.1 labels, splits, observations, and release metadata.
- `gamma-decomp-v1_1`: Gamma-Decomp core and provenance releases.

## Transfer Verification

- Transfer bundle SHA256: `4a05d8d2d2a8496100d2475e238ab16b6e6babc56a169062e04c21c70efb3ff2`.
- HEA v1 file count: 1,946.
- HEA v1.1 file count: 83.
- Gamma-Decomp v1.1 file count: 31,848.
- Internal checksum manifests: 10.
- Internal files checked: 33,763.
- Missing files: 0.
- Hash failures: 0.

During transfer audit, `gamma-decomp-core-v1.1/audits/summary.json` exposed a
pre-existing stale checksum caused by validation rewriting line endings. The
validator was changed to be read-only, the Gamma core checksum manifest was
regenerated on the workstation, and the corrected manifest was synchronized
locally before the final zero-failure audit.

The original transfer archive is retained under `_archives/` as a byte-for-byte
source snapshot. Use the extracted dataset roots above for the corrected formal
benchmark release.
