# SiteXRD-Bench HEA Scoring Protocol v1.1

## Scope

HEA v1.1 publishes one leaderboard task: masked first- and second-shell
Warren-Cowley SRO prediction. Family and occupancy identifiability are
diagnostics. They are not ground-truth leaderboards, qualification gates, or
components of an overall score.

## Submission Validity

Each requested split must contain exactly one prediction for every test
`structure_state_id`. Every row records model, seed, split column,
`xrd_sample_id`, `parent_group_id`, and the complete SRO prediction vector.
Missing, duplicate, unexpected, non-finite, or identity-mismatched rows
invalidate the run. Predictions are never imputed.

All children and observation views inherit their parent's split. A parent group
appearing in multiple split values invalidates the release.

## SRO Metrics

For each structure, MAE is calculated only over valid SRO mask entries. Errors
are averaged within `parent_group_id`, then macro-averaged across parents. The
primary metric is masked parent-macro MAE. Masked RMSE and first-/second-shell
Spearman correlations are secondary and remain separate.

Composition Ridge is the deterministic reference. Relative improvement is

`100 * (1 - candidate_mae / ridge_mae)`.

Whether a model exceeds Ridge is a scientific result, not a dataset-readiness
condition.

## Required Matrix

- Splits: `id_split`, `composition_ood_split`, `chemistry_ood_split`,
  `sro_ood_split`.
- Composition Ridge: one independent run per split, interface seed 2027.
- Peak XGBoost, ResNet1D, Spectrum Transformer, and the physics-constrained
  Transformer: seeds 2027-2031 per split.
- Checkpoints cannot be reused between split definitions.
- ExtraTrees is optional appendix evidence.

## Perturbation Evaluation

Peak XGBoost, ResNet1D, Spectrum Transformer, and the physics-constrained
Transformer reload the corresponding ID checkpoint and predict frozen v1.1
perturbation views. Errors aggregate first
within `structure_state_id`, then across `parent_group_id`. Reports include
clean MAE, perturbed MAE, retention, severity curves, and missing joins.
Composition Ridge is an XRD-independent negative control with retention 1.0.

## Statistical Reporting

Random-seed mean and standard deviation are reported for stochastic models.
Parent-paired bootstrap uncertainty uses 2,000 replicates, seed 2027, and
percentile 95% intervals. Reports include absolute MAE difference, relative
Ridge improvement, interval bounds, and probability of improvement. Seed
standard deviations and bootstrap confidence intervals must not be conflated.

## Diagnostic Tracks

Family values are automated, unadjudicated labels used for support and conflict
audits. They do not produce a v1.1 leaderboard score.

Occupancy uses frozen simulated perturbations with noise quantile 0.95 and
`identifiability_ratio = pair_l1 / parent_noise_l1_q95`. Ratio below 1 is
ambiguous and excluded from ranking metrics. The threshold cannot be lowered to
meet a quota. A 0/4,000 identifiable result is publishable as a negative
diagnostic. The external experimental sanity patterns carry no exact SRO or
component truth and never enter leaderboard scoring; no repeated-measurement
experimental XRD series is available in v1.1.

## Readiness

SRO-core readiness requires zero parent leakage, exact oracle and degraded
shuffle controls, the complete prediction matrix, all ID/OOD/perturbation
evaluations, the frozen bootstrap, and passing manifest/checksum/license audits.
Family pending review, occupancy non-identifiability, and failure to beat Ridge
are nonblocking disclosures.
