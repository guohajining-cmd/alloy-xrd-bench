# SiteXRD-Bench KDD Paper Design

## 1. Objective

Design one self-contained KDD Datasets and Benchmarks paper that balances two
contributions:

1. a hierarchical XRD benchmark dataset with auditable physical ground truth;
2. an evaluation framework that distinguishes spectral fit from correct latent
   physical recovery.

The benchmark covers only two material tracks:

- chemical-order and short-range-order inference in BCC/FCC high-entropy
  alloys;
- gamma/gamma-prime overlapping-component decomposition and physical-parameter
  recovery in Ni-based superalloys.

Catalyst occupancy and downstream mechanical-property or creep-property
prediction are outside the benchmark scope. Their scientific relevance may be
used only as motivation.

## 2. Central Claim

The paper asks whether a model can recover physically meaningful chemical
order and overlapping diffraction components from powder XRD rather than only
reconstructing a similar observed pattern.

The evidence chain is:

```text
relaxed or parameterized structure truth
  -> SRO, sublattice, and phase-parameter truth
  -> indexed reflections
  -> phase/HKL component profiles
  -> perturbed observed XRD
```

Every benchmark claim must point to a released artifact, a reproducible
experiment, a table or figure generated from frozen outputs, or an audited
primary citation.

## 3. Contributions

The submission will make four contributions:

1. **Hierarchical data:** structures, order descriptors, indexed reflections,
   component profiles, and observed XRD are connected by stable provenance.
2. **Complementary inverse-problem tracks:** HEA chemical order tests subtle
   occupancy-dependent evidence, while gamma/gamma-prime decomposition tests
   overlapping phase evidence.
3. **Physics-aware evaluation:** spectral reconstruction, component recovery,
   physical parameters, and chemical order are scored separately.
4. **Controlled generalization and identifiability:** fixed OOD splits,
   metadata-only controls, same-composition occupancy variants, and
   spectrally-similar/latent-different challenge sets expose shortcuts and
   ambiguity.

Novelty language such as "first" or "to our knowledge" is prohibited until a
dated literature audit supports it.

## 4. Task Contract

### 4.1 HEA-SRO

- Input: observed XRD, acquisition metadata, and, in the primary setting,
  composition plus BCC/FCC parent topology.
- Target: masked first- and second-shell Warren-Cowley matrices.
- Primary metrics: masked macro-MAE and per-shell Spearman correlation.
- Eligibility: valid parent-lattice mapping, valid neighbor shells, and at
  least two chemical species.

### 4.2 Gamma-Decomp

- Input: an observed pattern containing gamma and/or gamma-prime components.
- Target: an unordered set of phase/HKL component profiles with center, area,
  width, profile parameters, and phase identity.
- Primary metrics: matched component precision/recall, center and width error,
  profile distance, and phase/HKL assignment accuracy.
- Matching: deterministic optimal bipartite matching. Invalid phase/HKL pairs
  receive prohibitive cost; valid-pair cost is the equal-weight mean of
  normalized center error, width error, and profile distance. Normalizers and
  center gates are fitted on training data and frozen before test evaluation;
  exact ties are resolved lexicographically by true and predicted component ID.

Phase weight and acquisition parameters remain generation metadata for
diagnostic slices. Phase-specific lattice parameters, lattice misfit, and
gamma-prime site occupancy are not targets in the Ni-Al release.

Exact HEA atom-index recovery is not a task because equivalent-site
permutations do not define a physically unique target.

## 5. Evaluation Design

Whole-pattern reconstruction is reported for every applicable task but cannot
replace latent-target metrics. A forward-consistency score reconstructs XRD
from predicted physical variables and is also reported separately.

Required evaluation slices are:

- in-distribution;
- composition OOD;
- occupancy/order OOD;
- instrument/perturbation OOD;
- simulation-to-experiment at only the label levels actually supported by
  public experimental data;
- identifiable, weakly identifiable, and non-identifiable challenge subsets.

All structure-derived augmentations and mixture-derived observations remain in
the split of their underlying source group. Canonical hashes, near-duplicate
groups, occupancy-variant groups, and mixture provenance are assigned before
splitting.

Probabilistic models are evaluated with a proper scoring rule, interval or
prediction-set coverage, and calibration error. Thresholds for identifiability
sets are selected using training and validation data, never test performance.

## 6. Baseline Design

The minimum credible baseline suite is:

1. metadata-only control;
2. detected-peak feature model;
3. full-pattern 1D residual CNN;
4. sequence model such as a 1D Transformer or state-space model;
5. physics-constrained latent-to-XRD model;
6. constrained pseudo-Voigt fitting for Gamma-Decomp.

All learned models use the same splits, information settings, metric
implementation, seed policy, and prediction schema. At least three seeds are
reported. The metadata-only baseline is mandatory because it measures
composition and generation shortcuts.

## 7. Required Experiments

The main results answer five questions:

1. Does XRD provide chemical-order information beyond metadata?
2. How much does controlled OOD evaluation reduce performance relative to a
   random split?
3. Does hierarchical supervision improve latent recovery?
4. Can models recover gamma/gamma-prime components and associated physical
   parameters under overlap?
5. Do uncertainty estimates respond to spectrally similar samples with
   different latent truth?

Required ablations remove, one at a time:

- weak and superlattice peaks;
- composition metadata;
- component-level supervision;
- perturbation randomization;
- controlled grouping, replacing it with an ordinary random split;
- latent losses, leaving only whole-pattern reconstruction.

Null and negative results remain in the benchmark report.

## 8. Main-Paper Structure

The eight-page submission body is budgeted as follows:

| Section | Approximate pages | Reviewer question |
|---|---:|---|
| Abstract and Introduction | 1.0 | Why is this benchmark needed? |
| Related Work | 0.6 | What is missing from existing resources? |
| Benchmark Formulation | 0.9 | What are the inputs, latent truth, and observations? |
| Dataset Construction | 1.3 | How are the two tracks generated and audited? |
| Tasks, Splits, and Metrics | 1.4 | What is predicted and how is it scored? |
| Baselines and Experiments | 1.5 | What works, and are shortcuts present? |
| Identifiability Analysis | 0.7 | When is latent recovery ambiguous? |
| Limitations and Conclusion | 0.6 | What can and cannot be concluded? |

The page budget will be rechecked against the official KDD 2027 call when it is
released.

## 9. Figures and Tables

The main paper contains at most five figures and three tables:

- Figure 1: hierarchical data and observation model;
- Figure 2: same-composition/different-SRO examples and gamma/gamma-prime
  overlap examples;
- Figure 3: task inputs, outputs, eligibility, and OOD split design;
- Figure 4: primary ID and OOD benchmark results;
- Figure 5: spectral distance versus latent distance with calibration results;
- Table 1: audited comparison with related XRD datasets and benchmarks;
- Table 2: track, label, eligibility, and split statistics;
- Table 3: primary baseline results.

Every quantitative figure and table is generated by a checked-in script from
frozen manifests or predictions. Extended per-element, per-HKL, per-instrument,
and per-seed results belong in the appendix.

## 10. Non-Data Artifacts

Before large baseline runs, the project must freeze:

```text
task specifications
prediction schemas
reference metrics and unit tests
split manifests and leakage audits
identifiability protocol
baseline training/prediction interface
literature comparison matrix
dataset and benchmark cards
```

The artifact dependency order is:

```text
task contract
  -> prediction schema and metrics
  -> split and leakage rules
  -> identifiability sets
  -> baselines and ablations
  -> frozen tables and figures
  -> final ACM manuscript
```

## 11. Acceptance Criteria

The paper design is complete only when:

- both primary tasks have machine-readable inputs, targets, masks, and metrics;
- a perfect synthetic prediction reaches the exact metric optimum;
- a spectrally accurate but physically wrong prediction fails latent metrics;
- all splits pass structure, near-duplicate, mixture, and augmentation leakage
  audits;
- metadata-only controls are included in every relevant result table;
- experimental claims do not exceed the available annotation level;
- all headline counts and results are generated from a frozen release;
- reviewers can access data and code without requesting permission;
- the eight-page body is self-contained and contains the primary results;
- KDD 2027 formatting and track requirements are revalidated before
  submission.

## 12. Explicit Limitations

The benchmark's computational structure is ground truth for generated samples,
but this does not prove that conventional powder XRD uniquely determines atomic
occupancy in experiments. Periodic supercells may encode order through weak
reflections differently from experimental diffuse scattering. Scattering-factor
similarity, finite resolution, texture, strain, nanoscale broadening, and
unmodeled phases can make latent recovery non-identifiable. The paper therefore
requires identifiability and uncertainty analysis and forbids equating low
spectral error with unique structural recovery.
