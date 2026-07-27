# HEA Benchmark Model Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible pre-release validation suite that determines whether SiteXRD-Bench HEA spectra support family recognition, SRO recovery, same-composition occupancy ranking, and controlled OOD/perturbation evaluation without parent-group leakage.

**Architecture:** Add a small validation package with separate data, metric, model, and orchestration modules. All models consume one manifest/HDF5 batch contract and write one prediction schema; the evaluator is model-independent. Local unit tests establish data and metric behavior before code is copied to the read-only-data workstation and executed under an isolated GPU environment.

**Tech Stack:** Python 3.11, pandas, pyarrow, h5py, NumPy, SciPy, scikit-learn, XGBoost, PyTorch, unittest, micromamba, two NVIDIA RTX 4090 GPUs.

---

## File Structure

- Create `alloy_xrd_bench/hea_validation_data.py`: manifest loading, SRO vectorization, HDF5 access, family mapping, and group-leakage audit.
- Create `alloy_xrd_bench/hea_validation_metrics.py`: masked SRO, family, ranking, robustness, and prediction-completeness metrics.
- Create `alloy_xrd_bench/hea_validation_models.py`: composition ridge, peak XGBoost, residual CNN, Transformer, and Siamese ranking models.
- Create `alloy_xrd_bench/hea_validation.py`: CLI for audit, controls, training, prediction, evaluation, and report aggregation.
- Create `configs/hea_validation_v1.json`: frozen model budgets, seeds, task vocabulary, paths, and slices.
- Create `environment-validation.yml`: isolated workstation dependencies.
- Create `tests/test_hea_validation_data.py`: data-contract and leakage tests.
- Create `tests/test_hea_validation_metrics.py`: exact metric and control tests.
- Create `tests/test_hea_validation_models.py`: shape, masking, and deterministic smoke tests.
- Create `scripts/workstation/run_hea_validation.sh`: remote environment and ordered execution.

### Task 1: Freeze The Validation Contract

**Files:**
- Create: `configs/hea_validation_v1.json`
- Test: `tests/test_hea_validation_data.py`

- [ ] **Step 1: Write the failing contract test**

```python
class ValidationContractTests(unittest.TestCase):
    def test_contract_has_fixed_vocab_and_seeds(self):
        config = json.loads(Path("configs/hea_validation_v1.json").read_text())
        self.assertEqual(["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Si"], config["elements"])
        self.assertEqual([2027, 2028, 2029], config["seeds"])
        self.assertEqual(128, len(config["sro_targets"]))
        self.assertEqual("pre-release", config["result_qualification"])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_hea_validation_data.ValidationContractTests -v`

Expected: FAIL because `configs/hea_validation_v1.json` does not exist.

- [ ] **Step 3: Add the frozen configuration**

Define eight elements, the Cartesian 64 directed pairs for shell 1 followed by shell 2, seeds 2027-2029, dataset root `/home/GJM/data/hea/benchmark_v1`, output root `/home/GJM/runs/sitexrd_hea_validation_v1`, 4,001 input points, family mapping, train/validation/test split names, and bounded epoch/tree budgets.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run: `python -m unittest tests.test_hea_validation_data.ValidationContractTests -v`

Expected: PASS.

### Task 2: Implement Manifest And HDF5 Loading

**Files:**
- Create: `alloy_xrd_bench/hea_validation_data.py`
- Modify: `tests/test_hea_validation_data.py`

- [ ] **Step 1: Write failing vectorization and loader tests**

```python
def test_vectorize_sro_preserves_mask(self):
    values, mask = vectorize_sro('{"Al->Co": -0.25}', "{}", self.targets)
    self.assertEqual(-0.25, values[self.targets.index("shell1:Al->Co")])
    self.assertTrue(mask[self.targets.index("shell1:Al->Co")])
    self.assertFalse(mask[self.targets.index("shell2:Al->Co")])

def test_grid_reader_rejects_wrong_sample_id(self):
    with self.assertRaisesRegex(ValueError, "sample ID mismatch"):
        read_grid_row(self.h5_path, 0, expected_sample_id="wrong")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_hea_validation_data -v`

Expected: FAIL because loader functions are missing.

- [ ] **Step 3: Implement the data contract**

```python
@dataclass(frozen=True)
class ValidationRow:
    structure_state_id: str
    parent_group_id: str
    xrd_sample_id: str
    grid_shard_path: str
    grid_row_index: int
    family: str
    composition: np.ndarray
    sro_values: np.ndarray
    sro_mask: np.ndarray
    splits: dict[str, str]

def vectorize_sro(shell1_json: str, shell2_json: str, targets: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...
def read_grid_row(path: Path, row_index: int, expected_sample_id: str) -> np.ndarray: ...
def load_validation_rows(dataset_root: Path, config: Mapping[str, Any]) -> list[ValidationRow]: ...
```

HDF5 reads must verify row bounds, `xrd_sample_id`, shape `(4001,)`, finite values, and non-negative intensity. Paths are resolved under the dataset root and cannot escape it.

- [ ] **Step 4: Add the parent-group leakage audit**

```python
def audit_parent_group_leakage(rows: Sequence[ValidationRow], split_name: str) -> list[str]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        assignments[row.parent_group_id].add(row.splits[split_name])
    return sorted(group for group, splits in assignments.items() if len(splits) > 1)
```

- [ ] **Step 5: Run data tests and the existing suite**

Run: `python -m unittest tests.test_hea_validation_data -v`

Run: `python -m unittest discover -s tests`

Expected: PASS.

### Task 3: Implement Exact Evaluator Metrics

**Files:**
- Create: `alloy_xrd_bench/hea_validation_metrics.py`
- Create: `tests/test_hea_validation_metrics.py`

- [ ] **Step 1: Write failing oracle, mask, ranking, and completeness tests**

```python
def test_oracle_sro_is_exact(self):
    result = masked_parent_macro_sro(self.truth, self.truth, self.mask, self.parents, self.shells)
    self.assertEqual(0.0, result["masked_parent_macro_mae"])
    self.assertEqual(0.0, result["rmse"])

def test_masked_entries_do_not_affect_error(self):
    prediction = self.truth.copy()
    prediction[~self.mask] = 999.0
    self.assertEqual(0.0, masked_parent_macro_sro(...)["masked_parent_macro_mae"])

def test_perfect_ranking(self):
    result = occupancy_ranking_metrics([0.1, 0.9], [0.1, 0.9], ["g", "g"], identifiable=[True, True])
    self.assertEqual(1.0, result["pair_accuracy"])
    self.assertEqual(1.0, result["mrr"])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_hea_validation_metrics -v`

Expected: FAIL because metric functions are missing.

- [ ] **Step 3: Implement model-independent metrics**

Implement family Macro-F1/balanced accuracy without silently dropping classes, masked parent-macro SRO MAE/RMSE, shell Spearman with undefined correlations reported as null, ranking PairAcc/MRR/Top-3, clean-to-perturbed retention, and complete ID validation. Reuse score composition functions from `alloy_xrd_bench.scoring` after raw metrics are computed.

- [ ] **Step 4: Verify oracle and shuffled-label behavior**

Add a deterministic shuffled-label fixture with seed 2027 and assert that it is worse than the oracle for SRO and ranking.

- [ ] **Step 5: Run metric and existing tests**

Run: `python -m unittest tests.test_hea_validation_metrics -v`

Run: `python -m unittest discover -s tests`

Expected: PASS.

### Task 4: Implement Diagnostic Classical Baselines

**Files:**
- Create: `alloy_xrd_bench/hea_validation_models.py`
- Create: `tests/test_hea_validation_models.py`

- [ ] **Step 1: Write failing masked ridge and feature tests**

```python
def test_masked_ridge_emits_all_targets(self):
    model = PerTargetRidge(alpha=1.0).fit(self.x, self.y, self.mask)
    prediction = model.predict(self.x[:2])
    self.assertEqual((2, 128), prediction.shape)
    self.assertTrue(np.isfinite(prediction).all())

def test_peak_features_have_fixed_shape(self):
    features = peak_features(self.pattern, self.two_theta, max_peaks=32)
    self.assertEqual((98,), features.shape)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_hea_validation_models -v`

Expected: FAIL because models and feature extraction are missing.

- [ ] **Step 3: Implement composition mean/ridge**

Fit one ridge model per SRO target using only rows where that target mask is true. Persist target means for targets with no eligible training rows. Never expose truth masks as numeric input features beyond the declared composition metadata.

- [ ] **Step 4: Implement peak-feature XGBoost**

Use `scipy.signal.find_peaks` to extract the 32 strongest peak positions, intensities, and widths plus total/centroid statistics. Fit one multi-output wrapper with bounded tree depth and deterministic seeds.

- [ ] **Step 5: Run model unit tests**

Run: `python -m unittest tests.test_hea_validation_models -v`

Expected: PASS.

### Task 5: Implement Neural Spectral And Ranking Models

**Files:**
- Modify: `alloy_xrd_bench/hea_validation_models.py`
- Modify: `tests/test_hea_validation_models.py`

- [ ] **Step 1: Write failing shape and masked-loss tests**

```python
def test_resnet_and_transformer_emit_128_targets(self):
    batch = torch.zeros(2, 1, 4001)
    self.assertEqual((2, 128), ResNet1D(128)(batch).shape)
    self.assertEqual((2, 128), SpectrumTransformer(128)(batch).shape)

def test_masked_mse_ignores_missing_targets(self):
    prediction = torch.tensor([[1.0, 999.0]])
    truth = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    self.assertEqual(1.0, masked_mse(prediction, truth, mask).item())
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_hea_validation_models -v`

Expected: FAIL because neural classes are missing.

- [ ] **Step 3: Implement compact models**

Implement a four-stage residual 1D CNN with adaptive pooling and a patchified Transformer with positional embeddings. Both use a 128-output masked SRO head and optional four-class family head. The Siamese ranker shares the residual encoder and predicts a scalar order-distance score optimized with RankNet binary cross-entropy on eligible within-parent pairs.

- [ ] **Step 4: Implement deterministic training utilities**

Seed Python, NumPy, and PyTorch; enable deterministic validation; use early stopping on validation parent-macro MAE; save the best state dict, optimizer-free inference config, and seed metadata.

- [ ] **Step 5: Run neural smoke tests**

Run: `python -m unittest tests.test_hea_validation_models -v`

Expected: PASS on CPU with a two-sample batch.

### Task 6: Add CLI, Prediction Schema, And Reports

**Files:**
- Create: `alloy_xrd_bench/hea_validation.py`
- Modify: `pyproject.toml`
- Create: `scripts/workstation/run_hea_validation.sh`

- [ ] **Step 1: Write a failing CLI smoke test**

Add a temporary two-shard fixture and run:

```python
result = subprocess.run(
    [sys.executable, "-m", "alloy_xrd_bench.hea_validation", "audit", "--dataset", str(root), "--config", str(config)],
    capture_output=True,
    text=True,
)
self.assertEqual(0, result.returncode, result.stderr)
self.assertEqual(0, json.loads(result.stdout)["parent_group_leakage_count"])
```

- [ ] **Step 2: Implement CLI subcommands**

Provide `audit`, `controls`, `train`, `evaluate`, and `report`. Prediction Parquet rows contain model, seed, slice, sample ID, parent group, family probabilities, 128 SRO predictions, occupancy score, and uncertainty when present.

- [ ] **Step 3: Implement ordered workstation runner**

The shell script creates `sitexrd-validate`, installs the editable project, runs unit tests, runs `audit`, runs oracle and shuffled controls, trains classical baselines, trains three neural seeds across the two GPUs, evaluates all slices, and writes `validation_report.md`. It exits immediately on any failed audit or incomplete prediction file.

- [ ] **Step 4: Run local CLI tests and full suite**

Run: `python -m unittest tests.test_hea_validation_cli -v`

Run: `python -m unittest discover -s tests`

Expected: PASS.

### Task 7: Deploy And Execute On The Workstation

**Files:**
- Create: `environment-validation.yml`
- Remote output: `/home/GJM/runs/sitexrd_hea_validation_v1`

- [ ] **Step 1: Create the isolated GPU environment**

Use Python 3.11, PyTorch with CUDA support compatible with the workstation driver, scikit-learn, XGBoost, pandas, pyarrow, h5py, scipy, and pytest. Verify `torch.cuda.is_available()` and two visible devices before training.

- [ ] **Step 2: Copy only versioned implementation files**

Transfer the validation modules, tests, config, environment, and runner into `/home/GJM/xrd-benchmark` without copying local data or overwriting the remote dataset.

- [ ] **Step 3: Run audit and control gates**

Run the remote `audit` and `controls` commands first. Required outputs are zero group leakage, oracle optimum, finite metrics, and shuffled-label degradation. Stop before training if any gate fails.

- [ ] **Step 4: Run classical baselines**

Train composition ridge and peak XGBoost, evaluate ID and all OOD slices, and persist predictions and metrics.

- [ ] **Step 5: Run neural baselines**

Train 1D ResNet, Transformer, and Siamese RankNet for seeds 2027-2029. Use separate GPU jobs with isolated log and checkpoint directories. Wait for every job to finish and fail the run if any expected seed output is missing.

- [ ] **Step 6: Evaluate frozen observation variants**

Run clean and perturbed evaluation on the matching validation/test parent groups and compute retention without treating repeated observation views as independent structures.

- [ ] **Step 7: Generate final internal report**

Produce seed-level and aggregate CSV/JSON plus `validation_report.md`, beginning with the `pre-release` qualification and the 92.1% manual-review pass rate. Report every failed validation criterion and every blocked/not-applicable external model.

### Task 8: Verify Completion

**Files:**
- Read: `/home/GJM/runs/sitexrd_hea_validation_v1/validation_report.md`

- [ ] **Step 1: Run local regression tests**

Run: `python -m unittest discover -s tests`

Expected: PASS.

- [ ] **Step 2: Verify remote artifacts**

Check that all three neural seeds, both classical baselines, oracle, shuffled controls, every required slice, and the report are present. Validate prediction completeness against eligible sample IDs.

- [ ] **Step 3: Summarize scientific interpretation**

Report whether XRD beats composition-only for SRO, whether same-composition ranking beats chance, how performance changes under perturbation/OOD, and whether the benchmark passes each internal validation criterion. Do not call the result a public leaderboard while `benchmark_ready=false`.
