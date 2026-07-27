# HEA Benchmark Card v1.1

## Release Contract

- Scope: `sro_core`.
- Primary task: masked first- and second-shell SRO prediction.
- Diagnostics: automated Family support and occupancy identifiability/abstention.
- Required splits: ID, composition OOD, chemistry OOD, and SRO OOD.
- Statistical unit: `parent_group_id`.
- Experimental noise calibrated: false.

## Required Baselines

Composition Ridge runs once per split. Peak XGBoost, ResNet1D, Spectrum
Transformer, and Physics-Constrained Transformer run with seeds 2027-2031 and
train independently on each split. This is an 84-run matrix. ExtraTrees is
optional appendix evidence.

Every prediction artifact must contain model, seed, split, XRD sample ID,
parent-group ID, and the SRO prediction vector. Missing, duplicate, or extra IDs
invalidate a run.

## Metrics And Uncertainty

The primary metric is masked parent-macro MAE. RMSE and shell-level Spearman
correlations remain visible. Model gain over Ridge is a scientific result, not
a dataset readiness gate. Parent-paired uncertainty uses 2,000 bootstrap
replicates with seed 2027. Random-seed standard deviation and bootstrap 95% CI
must be labeled separately.

Matched perturbation evaluation reloads the ID-test checkpoint, aggregates
observation errors within `structure_state_id`, then across `parent_group_id`,
and reports clean MAE, perturbed MAE, retention, severity curves, and missing
joins. Composition Ridge is an XRD-independent stability control.

## Diagnostic Boundaries

Family labels are `automated_family_label` values with pending expert review.
No Family score enters the leaderboard or readiness decision.

Occupancy uses `identifiability_ratio = pair_l1 / parent_noise_l1_q95` at a
fixed 0.95 noise quantile. Ratios below 1 are ambiguous and cannot enter a
ranking metric. A result of 0 identifiable pairs out of 4,000 is a valid
negative diagnostic result.

## Readiness

The single acceptance command is `python -m
alloy_xrd_bench.hea_benchmark_readiness`. Release requires zero parent leakage,
passing controls, the complete prediction matrix, OOD and perturbation results,
the frozen bootstrap, and checksum/license evidence. Family pending review and
occupancy non-identifiability are disclosed but nonblocking.
