# HEA Benchmark Model Validation Design

## Objective

Validate the pre-release HEA portion of SiteXRD-Bench as a physically meaningful
inverse-problem benchmark. The experiment must test benchmark learnability,
shortcut resistance, group isolation, perturbation robustness, OOD behavior,
and scoring correctness. It is not intended to establish a final public
leaderboard while the dataset readiness gate remains below its 95% manual-review
pass threshold.

## Data Boundary

The source dataset is read-only:

```text
/home/GJM/data/hea/benchmark_v1
```

All generated environments, checkpoints, predictions, logs, and reports live
under:

```text
/home/GJM/runs/sitexrd_hea_validation_v1
```

The run uses the released parent-group assignments in `task_samples.parquet`.
No child occupancy state or observation view may cross its parent group's split.
Clean XRD is used for training. Frozen observation variants are used for
perturbed validation and test evaluation.

## Tasks

The first validation run covers:

1. parent-family qualification;
2. masked first- and second-shell Warren-Cowley SRO regression;
3. same-composition occupancy pair ranking and retrieval;
4. clean-to-perturbed robustness;
5. ID, composition-OOD, chemistry-OOD, and SRO-OOD generalization;
6. negative controls and evaluator sanity checks.

Gamma/gamma-prime tasks are excluded because no Gamma dataset was found on the
workstation. XDecomposer is therefore recorded as not applicable to this HEA-only
run rather than assigned a misleading score.

## Baselines

The model suite is deliberately diagnostic:

- Oracle predictions verify that every metric reaches its theoretical optimum.
- Shuffled-label controls verify that learned performance collapses toward the
  appropriate chance or no-skill level.
- Composition-only mean and ridge models quantify metadata shortcuts.
- Peak-feature XGBoost tests information available in detected peak summaries.
- A full-pattern 1D residual CNN is the primary learned spectral baseline.
- A compact 1D Transformer tests global sequence relationships.
- A Siamese residual encoder with a RankNet loss evaluates same-composition
  occupancy ranking and retrieval.
- XRD-AutoAnalyzer is evaluated only for the parent-family qualification gate
  if its phase-label adapter can be configured without exposing forbidden
  structure metadata. It is not treated as an SRO model.

Learned neural baselines run with seeds 2027, 2028, and 2029. Model capacity and
training budget are kept comparable enough that the validation tests data and
task behavior rather than an unconstrained hyperparameter search.

## Interfaces

One dataset loader resolves Parquet manifest rows to HDF5 grid rows and returns
the observed pattern, allowed metadata, target masks, parent group, and split
labels. Model adapters consume that common batch contract and emit a common
prediction schema. The evaluator reads only the prediction schema and frozen
manifests; it does not import model code.

The prediction schema includes sample and parent identifiers, family
probabilities, masked SRO entries, an occupancy score or embedding, and optional
uncertainty. Every adapter must produce complete predictions for the eligible
rows of the evaluated slice.

## Evaluation

Family qualification reports Macro-F1, balanced accuracy, confusion matrices,
and per-class recall. SRO recovery reports masked parent-macro MAE, RMSE, and
first-/second-shell Spearman correlation. Occupancy evaluation reports pairwise
accuracy, MRR, and Top-3 retrieval accuracy.

Each core metric is reported for ID, composition-OOD, chemistry-OOD, and SRO-OOD
test slices. Robustness reports clean-to-perturbed retention. Seed means and
standard deviations are mandatory. Group-aware bootstrap intervals may be
added after the first complete run but cannot replace seed reporting.

The report also applies the frozen diagnostic flags from `SCORING_PROTOCOL.md`,
including shortcut risk, occupancy blindness, perturbation brittleness, OOD
collapse, and overconfidence where uncertainty is available.

## Validation Criteria

The benchmark passes this internal validation only if all of the following hold:

1. Oracle predictions attain the exact optimum for every implemented metric.
2. Shuffled-label controls return to chance or no-skill behavior.
3. Automated audits find zero parent-group leakage.
4. At least one XRD model improves over the composition-only SRO baseline.
5. At least one XRD model exceeds random same-composition ranking.
6. Perturbation and OOD slices produce measurable, interpretable degradation
   rather than identical scores or total pipeline failure.
7. Repeated seeds produce finite metrics and no missing eligible predictions.

Failure of a criterion is retained as a benchmark finding. It must not be hidden
by averaging it into an overall score.

## Failure Handling

Missing grid shards, invalid HDF5 row indices, duplicate IDs, non-finite targets,
and partial predictions are hard failures. Unsupported external models are
reported as `not_applicable` or `blocked` with an explicit reason. Training
failures retain their logs and do not mutate dataset manifests.

## Deliverables

The isolated remote run directory contains environment metadata, resolved data
inventory, leakage audit, per-model configuration, checkpoints, predictions,
seed-level metrics, aggregate metrics, and a Markdown validation report. The
report begins with the `pre-release` qualification and the current readiness
status so that internal results cannot be mistaken for a frozen public
leaderboard.
