# HEA Benchmark v1.1 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a versioned HEA benchmark v1.1 whose family, SRO, occupancy-identifiability, OOD, and perturbation evaluations have reliable labels, adequate parent-group support, leakage-free splits, calibrated gates, and reproducible baseline evidence.

**Architecture:** Keep `/home/GJM/data/hea/benchmark_v1` immutable and build `/home/GJM/data/hea/benchmark_v1_1` from versioned manifests. Separate dataset-readiness gates from model-capability gates, split the overloaded family task into supported heads, calibrate occupancy identifiability against observation noise, and require every published slice to pass a single readiness command before model training.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, h5py, pyarrow, scikit-learn, XGBoost, PyTorch, unittest, micromamba, parent-group bootstrap, two RTX 4090 GPUs.

---

## Why Remediation Is Required

1. The SRO ID task is healthy: Transformer parent-macro MAE is `0.3397 +/- 0.0011`, 24.44% better than composition-only Ridge.
2. Family evaluation is statistically unstable: ID-test has 8,008 FCC, 5,149 BCC, 17 complex, and 1 HCP sample. Four-class Macro-F1 gives 50% of its weight to 18 rare test samples.
3. Family labels need adjudication: 79/1,000 reviewed rows failed, concentrated in 61 current FCC and 18 current BCC labels.
4. Occupancy ranking is underpowered: only 29/4,000 pairs are above the current clean-XRD L1 threshold; ID-test has only 2 eligible candidates.
5. The fixed `0.01` identifiability threshold is not tied to observation noise, although 131,340 perturbed observations are available for calibration.
6. OOD and perturbation slices exist but have not been evaluated.
7. `family_gate=0.90` is a model qualification gate. It must not be reused as a dataset publication gate.

## Target Release Gates

The v1.1 dataset is publishable only when all dataset gates pass:

- zero parent-group leakage for every split column;
- zero pending expert adjudications in the 1,000-row review sample;
- manual-review agreement at least 95% after adjudication, with disagreements retained in provenance;
- every primary family class has at least 500 train, 100 validation, and 100 test parent groups;
- occupancy ranking has at least 500 train, 100 validation, and 100 test parent groups with at least two noise-calibrated identifiable candidates per group;
- at least 1,000 ambiguous occupancy pairs are retained for abstention/identifiability evaluation;
- ID, composition-OOD, chemistry-OOD, SRO-OOD, and perturbation prediction files are complete;
- oracle controls are exact and seeded shuffled controls degrade;
- all metrics are parent-group aggregated and include confidence intervals.

Model gates remain separate:

- family Macro-F1 >= 0.90 is required only for the complete-capability model badge;
- SRO skill is reported relative to composition-only Ridge;
- no model score can mark the dataset ready or unready by itself.

### Task 1: Freeze v1 And Define The v1.1 Contract

**Files:**
- Create: `configs/hea_benchmark_v1_1.json`
- Create: `tests/test_hea_remediation_contract.py`
- Modify: `docs/BENCHMARK_CONTRACT.md`

- [ ] **Step 1: Write a failing contract test**

```python
def test_v1_1_separates_dataset_and_model_gates(self):
    config = json.loads(Path("configs/hea_benchmark_v1_1.json").read_text())
    self.assertEqual("/home/GJM/data/hea/benchmark_v1", config["source_dataset_root"])
    self.assertEqual("/home/GJM/data/hea/benchmark_v1_1", config["output_dataset_root"])
    self.assertEqual(100, config["dataset_gates"]["min_test_parent_groups_per_primary_family"])
    self.assertEqual(100, config["dataset_gates"]["min_identifiable_test_parent_groups"])
    self.assertEqual(0.90, config["model_gates"]["complete_capability_family_macro_f1"])
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_remediation_contract.py -v`

Expected: FAIL because `configs/hea_benchmark_v1_1.json` does not exist.

- [ ] **Step 3: Add the versioned contract**

Define immutable source/output roots, family task vocabularies, five split columns, noise-calibration quantile `0.95`, minimum parent-group counts, review requirements, seeds `[2027, 2028, 2029, 2030, 2031]`, and separate `dataset_gates` and `model_gates` objects.

- [ ] **Step 4: Document that v1 is immutable**

Add a contract rule that generated v1.1 paths must resolve under `/home/GJM/data/hea/benchmark_v1_1` and no command may write under the v1 source root.

- [ ] **Step 5: Verify GREEN and commit**

Run: `python -m unittest discover -s tests -p test_hea_remediation_contract.py -v`

Commit: `git commit -m "docs: define HEA benchmark v1.1 remediation contract"`

### Task 2: Complete Expert Family-Label Adjudication

**Files:**
- Create: `alloy_xrd_bench/hea_family_adjudication.py`
- Create: `schemas/hea_family_adjudication.schema.json`
- Create: `tests/test_hea_family_adjudication.py`
- Remote input: `/home/GJM/data/hea/benchmark_v1/manifests/pilot_review_queue.parquet`
- Remote output: `/home/GJM/data/hea/benchmark_v1_1/review/family_adjudication.parquet`

- [ ] **Step 1: Write failing export/import tests**

Test that the export contains `structure_state_id`, current label, proposed label, PTM/CNA evidence, reviewer ID, review date, decision, and notes. Test that import rejects empty reviewer IDs, labels outside the frozen vocabulary, duplicate structure IDs, and pending decisions.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_family_adjudication.py -v`

Expected: FAIL because the adjudication module does not exist.

- [ ] **Step 3: Implement conflict export**

Export all 79 failed rows plus the 8 PTM/CNA conflicts. Preserve the observed transition evidence: FCC->BCC 46, FCC->HCP 11, FCC->complex 4, BCC->complex 15, and BCC->HCP 3.

- [ ] **Step 4: Require independent human decisions**

The importer must require an expert-provided `expert_label` and must not convert an automated PTM/CNA suggestion into a human label. Rows may be `accept_current`, `accept_proposed`, or `exclude`; every decision remains in provenance.

- [ ] **Step 5: Rebuild versioned labels**

Apply adjudications to a copied v1.1 manifest, never to v1. Recompute review agreement and label counts from the applied manifest.

- [ ] **Step 6: Enforce the review gate**

Run the importer and require: 1,000 reviewed, 0 pending, agreement >= 0.95, and a non-empty reviewer ID for every conflict.

- [ ] **Step 7: Verify GREEN and commit**

Run: `python -m unittest discover -s tests -p test_hea_family_adjudication.py -v`

Commit: `git commit -m "feat: add expert family-label adjudication workflow"`

### Task 3: Split Family Evaluation Into Supported Heads

**Files:**
- Create: `alloy_xrd_bench/hea_family_tasks.py`
- Create: `tests/test_hea_family_tasks.py`
- Modify: `alloy_xrd_bench/hea_validation.py`
- Modify: `configs/hea_benchmark_v1_1.json`

- [ ] **Step 1: Write failing task-mapping tests**

```python
def test_family_heads_do_not_hide_ordering_labels(self):
    self.assertEqual("fcc", lattice_family("l12_like"))
    self.assertEqual("l12", ordering_prototype("l12_like"))
    self.assertEqual("b2", ordering_prototype("b2_like"))
    self.assertIsNone(ordering_prototype("hcp_like"))
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_family_tasks.py -v`

- [ ] **Step 3: Implement three isolated heads**

Implement:

- `lattice_family`: FCC vs BCC, with L12 mapped to FCC and B2 mapped to BCC;
- `ordering_prototype`: disordered vs L12 vs B2, only for eligible cubic structures;
- `rare_open_set`: HCP/complex/unknown, reported as open-set AUROC/F1 only after support gates pass.

- [ ] **Step 4: Build parent-group-stratified splits**

Assign parent groups, not observations, and keep all clean/perturbed/counterfactual views together. Refuse release when a primary class has fewer than 500/100/100 train/validation/test parent groups.

- [ ] **Step 5: Add class-balanced training and per-class reporting**

Use inverse-frequency sampling or class-weighted cross entropy for the family head. Always report the confusion matrix, per-class precision/recall, support, Macro-F1, and balanced accuracy.

- [ ] **Step 6: Verify and commit**

Run: `python -m unittest discover -s tests -p 'test_hea_family_*.py' -v`

Commit: `git commit -m "feat: split HEA family evaluation into supported heads"`

### Task 4: Calibrate Identifiability Against Observation Noise

**Files:**
- Create: `alloy_xrd_bench/hea_identifiability.py`
- Create: `tests/test_hea_identifiability.py`
- Remote input: `/home/GJM/data/hea/benchmark_v1/xrd/runs/hea_observation_pilot_v1/xrd_samples.parquet`
- Remote output: `/home/GJM/data/hea/benchmark_v1_1/reports/identifiability_calibration.json`

- [ ] **Step 1: Write failing noise-calibration tests**

Construct two parents with clean and perturbed spectra. Assert that the calibration computes a parent-level 95th-percentile noise distance and labels a counterfactual identifiable only when its clean-pair distance exceeds that calibrated noise floor.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_identifiability.py -v`

- [ ] **Step 3: Implement parent-level noise floors**

For every source spectrum, compute clean-to-observation L1 and cosine distances across its perturbations. Store parent-level quantiles plus global fallback quantiles for parents without sufficient variants.

- [ ] **Step 4: Replace the fixed threshold with an effect ratio**

Define `identifiability_ratio = clean_pair_l1 / max(parent_noise_l1_q95, 1e-12)`. Mark a pair identifiable when the ratio is at least 1 and the direction is consistent under bootstrap resampling of observation variants.

- [ ] **Step 5: Preserve ambiguous pairs**

Keep low-ratio pairs as the abstention/identifiability subset. Do not make them ranking negatives and do not lower the noise floor to satisfy quotas.

- [ ] **Step 6: Verify and commit**

Run: `python -m unittest discover -s tests -p test_hea_identifiability.py -v`

Commit: `git commit -m "feat: calibrate occupancy identifiability from observation noise"`

### Task 5: Regenerate A Statistically Supported Occupancy Subset

**Files:**
- Modify: `alloy_xrd_bench/occupancy_expansion.py`
- Create: `tests/test_hea_occupancy_remediation.py`
- Create: `scripts/workstation/build_hea_occupancy_v1_1.sh`

- [ ] **Step 1: Write failing candidate-selection tests**

Assert that candidate generation preserves composition and lattice, produces unique occupancy fingerprints, retains at least two candidates per selected parent, and ranks candidates by a deterministic joint objective without reading model predictions.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_occupancy_remediation.py -v`

- [ ] **Step 3: Expand the deterministic candidate pool**

Generate at least 128 occupancy candidates per parent across local swap, clustering, anti-clustering, B2-like ordering, L12-like ordering, and Monte Carlo SRO objectives. Preserve fixed geometry and exact composition.

- [ ] **Step 4: Add physically informed diversity**

Balance parent selection by FCC/BCC lattice, composition cardinality, SRO bin, and atomic-scattering contrast. Prefer parents containing element pairs with materially different X-ray form factors without changing their natural benchmark labels.

- [ ] **Step 5: Select after noise-calibrated simulation**

Simulate all candidates with the frozen XRD configuration, compute calibrated identifiability ratios, and select up to four diverse identifiable candidates per parent. Candidate selection may use physical XRD distance but must never use model performance or test predictions.

- [ ] **Step 6: Enforce ranking and ambiguity quotas**

Require at least 500/100/100 train/validation/test identifiable parent groups, each with at least two candidates. Separately retain at least 1,000 ambiguous pairs for abstention evaluation. Fail generation if either quota is unmet.

- [ ] **Step 7: Verify and commit**

Run: `python -m unittest discover -s tests -p test_hea_occupancy_remediation.py -v`

Commit: `git commit -m "feat: regenerate noise-supported HEA occupancy pairs"`

### Task 6: Add OOD And Perturbation Evaluation

**Files:**
- Modify: `alloy_xrd_bench/hea_validation.py`
- Create: `alloy_xrd_bench/hea_observation_eval.py`
- Create: `tests/test_hea_ood_perturbation.py`
- Modify: `scripts/workstation/run_hea_validation.sh`

- [ ] **Step 1: Write failing split-column tests**

Run a fixture through `id_split`, `composition_ood_split`, `chemistry_ood_split`, and `sro_ood_split`. Assert that each model trains only on that split column's train groups and predicts exactly its test groups.

- [ ] **Step 2: Write failing observation-join tests**

Assert that every perturbed observation joins to exactly one clean source structure and inherits labels without becoming an independent parent group.

- [ ] **Step 3: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_ood_perturbation.py -v`

- [ ] **Step 4: Implement split-column training**

Add `--split-column` to `train` and reject any parent appearing in both train and test for the selected column. Save the split column in checkpoints and prediction rows.

- [ ] **Step 5: Implement matched perturbation evaluation**

Evaluate clean and observation variants with the same checkpoint. Aggregate errors first within source structure, then across parent groups. Report clean score, perturbed score, retention, perturbation-severity curves, and missing joins.

- [ ] **Step 6: Run the frozen model matrix**

Run composition Ridge and Peak XGBoost once per split. Run ResNet and Transformer for seeds 2027-2031 on both GPUs. Do not reuse an ID-trained checkpoint for an OOD split whose train membership differs.

- [ ] **Step 7: Verify and commit**

Run: `python -m unittest discover -s tests -p test_hea_ood_perturbation.py -v`

Commit: `git commit -m "feat: evaluate HEA OOD and perturbation slices"`

### Task 7: Add Parent-Paired Statistical Evidence

**Files:**
- Create: `alloy_xrd_bench/hea_statistics.py`
- Create: `tests/test_hea_statistics.py`
- Modify: `alloy_xrd_bench/hea_validation_plot.py`

- [ ] **Step 1: Write failing paired-bootstrap tests**

Use a fixture where one model is better for every parent. Assert a positive improvement estimate, a positive 95% confidence lower bound, and deterministic output for seed 2027.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_statistics.py -v`

- [ ] **Step 3: Implement 10,000-draw parent bootstrap**

Resample parent groups with replacement, keeping all targets and observation variants within their parent. Report MAE difference, relative improvement, 95% confidence interval, and probability of improvement.

- [ ] **Step 4: Add family and ranking intervals**

Use stratified parent bootstrap for family metrics and group bootstrap for ranking metrics. Refuse intervals when a class or ranking group has insufficient support.

- [ ] **Step 5: Update figures**

Plot confidence intervals separately from random-seed standard deviations. Label each uncertainty source explicitly.

- [ ] **Step 6: Verify and commit**

Run: `python -m unittest discover -s tests -p test_hea_statistics.py -v`

Commit: `git commit -m "feat: add parent-paired HEA benchmark statistics"`

### Task 8: Build The v1.1 Readiness Gate And Release Report

**Files:**
- Create: `alloy_xrd_bench/hea_readiness.py`
- Create: `tests/test_hea_readiness.py`
- Modify: `alloy_xrd_bench/hea_validation.py`
- Remote output: `/home/GJM/runs/sitexrd_hea_validation_v1_1/validation_report.md`

- [ ] **Step 1: Write a failing complete-readiness fixture**

Create a fixture satisfying every dataset gate and assert `benchmark_ready=true`. Remove one family group, one occupancy group, one prediction ID, and one expert adjudication in separate tests and assert the corresponding fixed failure code.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -p test_hea_readiness.py -v`

- [ ] **Step 3: Implement machine-readable failure codes**

Emit fixed codes including `family_support_insufficient`, `occupancy_support_insufficient`, `review_incomplete`, `parent_leakage`, `prediction_incomplete`, `ood_missing`, and `perturbation_missing`.

- [ ] **Step 4: Generate the release report**

Report dataset gates and model results in separate sections. Begin with `pre-release` until every dataset gate passes. Include label transitions, support counts, noise calibration, ID/OOD/perturbation metrics, paired confidence intervals, and known limitations.

- [ ] **Step 5: Run complete verification**

Run locally: `python -m unittest discover -s tests`

Run remotely: `bash /home/GJM/xrd-benchmark/scripts/workstation/run_hea_validation.sh`

Verify that every expected prediction file contains exactly the eligible sample IDs and that no GPU job or failed log remains.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: complete HEA benchmark v1.1 readiness workflow"`

## Execution Order And Stop Conditions

1. Tasks 1-2 are blocking: do not regenerate splits before label adjudication is complete.
2. Task 3 may proceed only after adjudicated labels are applied.
3. Task 4 must finish before Task 5; do not tune identifiability thresholds to meet quotas.
4. If Task 5 cannot meet occupancy quotas under calibrated noise, publish occupancy as an identifiability/abstention task and remove ranking from the primary benchmark.
5. Task 6 can begin after Tasks 1 and 3 for family/SRO, but occupancy OOD results wait for Task 5.
6. Task 8 is the only component allowed to set `benchmark_ready=true`.

## Expected Final Scientific Position

- HEA-SRO remains the validated primary inverse task.
- Family is reported as supported, isolated heads rather than an unstable four-class mixture.
- Occupancy ranking is published only if calibrated support quotas pass; otherwise non-identifiability detection becomes the primary occupancy task.
- OOD and perturbation results are mandatory, not optional appendix diagnostics.
- Dataset readiness and model capability are reported separately.
