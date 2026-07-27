# Completion Audit

This audit maps the Alloy-XRD-Bench v0.1 working plan to current local
evidence. It separates the verified local handoff package from work that still
requires an external archive, repository remote, or additional MLIP choices.

## Verified Local Handoff

- Workspace root: `E:\project\xrd-benchmark`
- Release root: `data/releases/v0.1-working`
- Release archive: `data/releases/v0.1-working.zip`
- Release archive SHA-256: recorded in the sidecar
  `data/releases/v0.1-working/publication_record.yml`
- Handoff readiness report: `audits/release_readiness_v0.1-working.json`
- Publication readiness report: `audits/publication_readiness_v0.1-working.json`

The local handoff readiness status is `pass`. Final publication readiness is
expected to remain `fail` until `publication_record.yml` is filled with the
external archive, DOI, dataset URL, release date, and code repository URL.

## Phase Coverage

| Plan Item | Local Status | Evidence |
| --- | --- | --- |
| Benchmark contract | Complete for v0.1 working release | `docs/BENCHMARK_CONTRACT.md`, `schemas/*.schema.json`, `alloy_xrd_bench/contract.py` |
| Per-potential storage layout | Complete for registered potentials | `data/registry/potentials_catalog.yml`, `data/potentials/pot_chgnet_0p3p0/`, validator issue count 0 |
| Structure line | Complete for v0.1 CHGNet baseline | `data/shared/manifests/initial_structures.parquet`, 14 initial structures, `pair_balance_occupancy`, `candidate_count=32` |
| MLIP registry | Complete for CHGNet baseline | `pot_chgnet_0p3p0`, metadata-only policy, `INSTALL.md`, expected CHGNet 0.3.0 SHA-256 |
| Relaxation line | Complete for CHGNet baseline | 14 `relaxations`, 14 `relaxed_structures`, per-potential logs and artifacts |
| XRD line | Complete for CHGNet baseline | 14 individual XRD samples, 8 averaged XRD rows in the workspace |
| Splits and leakage control | Complete for v0.1 working split rules | split counts in `audits/workspace_status.json`, release private inputs = 3, leakage audit pass |
| Public/private release export | Complete for local handoff | public XRD = 11, public averages = 6, private inputs = 3 |
| Release metadata | Complete for local handoff | `DATASET_CARD.md`, `CITATION.cff`, `croissant.jsonld`, `license_audit.csv`, `leakage_audit.csv` |
| Release packaging | Complete for local handoff | 116 archive entries, file manifest/checksums/zip match, readiness status pass |
| Final public archive publication | Not complete | blocked on archive choice/upload and DOI/URL/date/repository fields |
| Additional MLIPs | Not complete | MACE, MatGL, UNEP, and alloy-specific potentials are not installed or run |
| Strict external SQS generation | Not complete | v0.1 uses pair-balance occupancy, not ATAT/mcsqs |
| Git publication | Partially complete | local Git repo exists on `main`; commit identity, initial commit, and remote are not configured |

## Test Plan Coverage

| Test Area | Status | Evidence |
| --- | --- | --- |
| Schema tests | Pass | `python -B -m unittest discover -s tests -v` |
| Provenance tests | Pass | validator issue count 0 for workspace and release |
| Hash tests | Pass | canonical structure hash tests, artifact hash validation, release checksums |
| Physics sanity tests | Pass | workspace quality issue count 0, release quality issue count 0 |
| Split tests | Pass | private relaxed-structure isolation and release hiding tests |
| License tests | Pass | metadata-only CHGNet weights are not packaged; `INSTALL.md` is packaged |
| Release readiness tests | Pass for local handoff | `alloy_xrd_bench.release_readiness` status `pass` |
| Final publication tests | Waiting for external metadata | `--require-publication` fails only on missing external fields |

## Remaining External Actions

The verified local handoff package is ready for external upload. The project is
not fully published until:

- `data/releases/v0.1-working.zip` is uploaded to the chosen archive.
- `data/releases/v0.1-working/publication_record.yml` is filled.
- `alloy_xrd_bench.release_readiness --require-publication` returns `pass`.
- Git identity is configured, an initial commit is created, and a remote is
  added for the code release.
