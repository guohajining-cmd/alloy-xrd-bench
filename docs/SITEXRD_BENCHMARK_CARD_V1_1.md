# SiteXRD-Bench v1.1 Benchmark Card

## Scope

SiteXRD-Bench v1.1 contains two independent simulated powder-XRD tasks:

- HEA-SRO predicts masked first- and second-shell Warren-Cowley short-range
  order under ID, composition OOD, chemistry OOD, SRO OOD, and matched
  simulated perturbations.
- Gamma-Decomp predicts unordered gamma/gamma-prime phase/HKL-group components
  for complete patterns and fixed overlap windows.

The benchmark publishes no cross-task composite score. A model may enter one
or both leaderboards, and each track retains its own metric definitions.

## Release Gates

HEA-SRO is ready only when `core_benchmark_ready=true`. Gamma-Decomp is ready
only when `gamma_benchmark_ready=true`. The combined release is ready only when
`alloy_xrd_bench.benchmark_readiness` reports `benchmark_ready=true`.

HEA Family and occupancy are diagnostics. Family labels are automated and
unadjudicated. Occupancy reports identifiability and abstention under simulated
perturbations and does not produce a ranking score.

## Data Boundary

Both tracks use computational structure and diffraction truth. Neither track
claims that conventional experimental powder XRD uniquely determines atomic
occupancy or exact phase components in real samples. Gamma parents include
deterministic synthetic Ni-Al structures, and Gamma-Decomp mixtures are
synthetic benchmark observations rather than experimentally refined
microstructures.

## Reproducibility

HEA neural models use seeds 2027-2031. Gamma learned models use seeds
2027-2029. Statistical comparisons use 2,000 group-aware bootstrap replicates
with seed 2027. Neural runners fix CuBLAS workspace behavior and disable
non-deterministic attention kernels.

Every released table and figure must be generated from frozen JSON, CSV,
Parquet, checkpoint, and score artifacts. Hand-copied result values are not
release evidence.

The external sanity set contains two HEA and two Ni-based experimental
patterns with stable DOIs and source hashes. These patterns have no exact SRO
or component truth and are excluded from leaderboard scoring.
