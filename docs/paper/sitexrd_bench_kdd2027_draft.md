# SiteXRD-Bench: A Hierarchical Benchmark for Chemical-Order Inference and Overlapping-Peak Decomposition in Structural Alloys

> KDD 2027 Datasets & Benchmarks Track working draft. Numerical results, final
> dataset counts, and literature citations must be inserted only after the
> corresponding artifacts and experiments are frozen.

## TL;DR

SiteXRD-Bench evaluates whether models can recover physically meaningful
chemical order and overlapping diffraction components from powder XRD, rather
than merely reconstructing a visually similar pattern. It provides hierarchical
ground truth and controlled OOD splits for high-entropy alloys and Ni-based
superalloys.

## Abstract

Atomic site occupancy, chemical ordering, and phase-resolved peak components
are central latent variables in multicomponent structural alloys, yet recovering
them from powder X-ray diffraction (XRD) is often non-identifiable. Distinct
atomic arrangements can produce nearly indistinguishable patterns, while
overlapping reflections, peak broadening, background signals, and instrumental
perturbations can hide weak occupancy-sensitive evidence. Existing XRD
benchmarks primarily evaluate phase or symmetry classification, structure
generation, peak detection, or whole-pattern reconstruction. They do not
separately test whether a model recovers the latent chemical order and the
physical components responsible for an observed pattern.

We introduce SiteXRD-Bench, a hierarchical benchmark for two complementary
inverse problems in structural alloys. The high-entropy-alloy track evaluates
the inference of permutation-invariant chemical-order descriptors, including
first- and second-shell Warren-Cowley short-range order, under controlled
composition and parent-lattice conditions. The Ni-based-superalloy track
evaluates permutation-invariant decomposition of overlapping gamma and
gamma-prime reflections into phase/HKL components with peak centers, widths,
areas, and profile parameters. Each generated sample
retains provenance from its relaxed structure and occupancy labels to indexed
reflections, component profiles, and the perturbed observed XRD pattern.

SiteXRD-Bench defines evaluation at three distinct levels: whole-pattern
reconstruction, component-peak recovery, and chemical-order recovery. It includes composition,
occupancy, instrument, and simulation-to-experiment shifts, along with explicit
identifiability sets containing structurally different but spectrally similar
samples. Standardized baselines and forward-consistency checks expose cases in
which a model fits the observed spectrum while producing a structurally
incorrect interpretation. SiteXRD-Bench therefore supports data-mining research
on constrained inverse problems, structured latent prediction, OOD
generalization, and uncertainty under physical non-identifiability.

## 1. Introduction

Powder XRD is widely used because it is non-destructive, scalable, and sensitive
to long-range crystal structure. Its inverse problem is nevertheless difficult:
the measured one-dimensional pattern is an aggregate of many reflections, and
multiple latent structures may be compatible with nearly the same observation.
This ambiguity is especially consequential when the scientific target is not a
coarse phase label but chemical order or phase-resolved parameters.

Two alloy settings expose complementary forms of this problem. In high-entropy
alloys (HEAs), structures with the same composition and nominal BCC or FCC
parent lattice can differ in atomic arrangement and short-range order (SRO).
Their dominant Bragg peaks may remain similar while occupancy-dependent weak
intensity changes are subtle. In Ni-based superalloys, the FCC gamma matrix and
L1_2 gamma-prime precipitate have closely related lattices. Their reflections
can overlap, and assigning the observed intensity to the correct phase/HKL
components is easily confounded by broadening and strain.

Most machine-learning evaluations collapse these distinct failure modes into a
single pattern-level metric. A low reconstruction error does not establish that
the model recovered the correct SRO or assigned the correct gamma/gamma-prime
components. Conversely, an inverse
problem may be physically non-identifiable at a given noise level, in which case
a responsible model should express uncertainty rather than return an
overconfident point estimate.

SiteXRD-Bench is designed around this separation between observational fit and
latent correctness. Its contributions are:

1. A shared hierarchical representation linking relaxed structures, chemical
   order, indexed reflections, component profiles, and observed XRD across HEA
   and Ni-based-superalloy tracks.
2. Controlled same-composition occupancy variants and gamma/gamma-prime mixture
   constructions that reduce composition shortcuts and expose physical
   non-identifiability.
3. Task-specific metrics for SRO, component sets, phase/HKL identity, peak
   parameters, and forward spectral consistency.
4. Fixed composition, occupancy, instrument, and simulation-to-experiment OOD
   splits, accompanied by baseline implementations and reproducible generation
   and audit procedures.

## 2. Related Work

This section will position the benchmark against five bodies of work without
claiming novelty until the literature audit is complete:

- crystallographic refinement of occupancy, ordering, and multiphase patterns;
- XRD phase identification and symmetry classification benchmarks;
- neural structure generation and spectrum-to-structure inverse models;
- Bragg-peak detection, profile fitting, and overlapping-component separation;
- scientific-ML benchmarks for structured prediction, OOD generalization, and
  calibrated uncertainty.

The comparison table should record, for every related dataset or benchmark,
whether it provides atomic occupancy targets, SRO, indexed component peaks,
phase-specific lattice parameters, controlled same-composition variants,
experimental patterns, OOD splits, uncertainty evaluation, and public code.

## 3. Benchmark Formulation

### 3.1 Hierarchical observation model

For a latent material state z and acquisition condition n, the benchmark uses a
forward hierarchy

```text
z = structure, composition, occupancy/order, phase parameters
r = indexed reflections generated from z
c = broadened phase/HKL component profiles generated from r
y = observed pattern produced from c under acquisition condition n
```

The observed pattern can be written schematically as

```text
y(theta) = sum_k c_k(theta; z, n) + background(theta; n) + noise(theta; n).
```

Models receive y and one of several declared information settings, such as known
composition or known parent phase. Targets at different hierarchy levels are
never silently exposed as inputs.

### 3.2 Track A: HEA chemical-order inference

The HEA target is permutation-invariant chemical order rather than an absolute
atom index. For chemical species a and b and neighbor shell s, the primary
target is the Warren-Cowley coefficient

```text
alpha_ab^(s) = 1 - P(b | a, s) / c_b,
```

where c_b is the global concentration of b. The benchmark stores the complete
species-pair matrix for the first and second shells, an element mask, occupancy
entropy, parent topology, and provenance to the relaxed structure.

The benchmark contains two information settings:

- HEA-known: composition and BCC/FCC parent topology are given;
- HEA-blind: only the pattern and acquisition metadata are given.

The primary setting is HEA-known because it isolates occupancy information from
composition and phase recognition. HEA-blind is a harder auxiliary setting.

### 3.3 Track B: gamma/gamma-prime decomposition

The superalloy target is an unordered set of component profiles and their
physical attributes. Each component includes phase identity, HKL indices,
integrated intensity, center, width, and profile parameters. Phase fraction and
acquisition parameters are retained as generation metadata, but phase-specific
lattice constants, lattice misfit, and sublattice occupancy are not prediction
targets in the Ni-Al release.

Because multiple same-phase components may exchange order without changing the
prediction, component evaluation uses optimal bipartite matching rather than a
fixed output ordering.

### 3.4 Identifiability sets

The benchmark explicitly marks pairs or groups with similar observed patterns
but different latent targets. For samples i and j, it records a spectral
distance d_y and a latent distance d_z. Cases with low d_y and high d_z form
non-identifiable or weakly identifiable challenge sets. Results on these sets
must include uncertainty or prediction-set coverage, not only point error.

## 4. Dataset Construction

### 4.1 Shared provenance and quality control

Every accepted structure receives a stable ID, canonical structure hash, source
record, composition, quality result, and task-eligibility flags. Raw relaxed
structures are preserved. Derived labels are deterministic functions of the
structure and are versioned with the label implementation and configuration.
Rejected structures remain in an accounted manifest with explicit reasons.

### 4.2 HEA structures and controlled occupancy variants

The public relaxed-structure source currently contains 84,024 records spanning
Al, Co, Cr, Cu, Fe, Mn, Ni, and Si. After quality control, accepted structures
are mapped to final BCC/FCC topology when the mapping is valid. Structures whose
final topology conflicts with the source parent retain both values so that
relaxation-induced transitions are not hidden.

The final release must augment naturally occurring structures with controlled
same-composition variants. Within each selected composition and parent lattice,
atomic arrangements are varied to span random, ordering, clustering, and
anti-clustering regimes while holding composition and the declared structural
controls fixed. This controlled subset is required for occupancy-specific
evaluation; the unpaired source corpus alone is not sufficient evidence that a
model learned SRO rather than composition or lattice scale.

### 4.3 Ni-based gamma/gamma-prime structures and mixtures

The Ni-based track pairs FCC gamma structures with L1_2 gamma-prime structures.
The data generator varies phase weight, broadening, strain, crystallite size,
background, zero shift, and noise according to declared ranges. Mixture truth
is stored at the component level before components are summed into the observed
pattern.

The release must distinguish physically relaxed pairs from parameterized
counterfactual pairs and must identify the potential or forward model used for
each structure. Synthetic mixtures are not presented as experimental two-phase
microstructures.

### 4.4 XRD generation and perturbations

Clean patterns use a fixed reference configuration and retain the unbroadened
reflection table. Experimental perturbations are generated separately with
recorded parameters. Training perturbations may be sampled online, whereas
validation and test perturbations are frozen. No augmented observation crosses
the split of its underlying structure or mixture.

### 4.5 Experimental evaluation subset

The simulation-to-experiment split will include only public patterns with
licenses and labels sufficient for the claimed evaluation. If phase-resolved or
occupancy labels are unavailable, the pattern may support reconstruction or
limited phase evaluation but cannot be counted as full latent ground truth. The
final abstract will state exactly which experimental target levels are present.

## 5. Tasks, Splits, and Metrics

### 5.1 Tasks

SiteXRD-Bench defines two primary tasks:

1. HEA-SRO: infer first- and second-shell SRO under the HEA-known setting.
2. Gamma-Decomp: recover the unordered gamma/gamma-prime component set.

HEA same-composition occupancy pairs are retained as an ambiguity and abstention
diagnostic. They are not an exact-ranking leaderboard because none of the 4,000
current pairs exceeds the frozen simulated-noise identifiability threshold.

### 5.2 Splits

The main paper reports:

- ID: random groups after duplicate and provenance isolation;
- composition OOD: held-out formulas or composition regions;
- occupancy OOD: held-out SRO/order regimes within known chemistry;
- instrument OOD: held-out perturbation families or parameter ranges;
- simulation-to-experiment: public experimental patterns unavailable in
  synthetic training.

Structure hashes, near duplicates, mixtures, and all augmented observations are
grouped before splitting. Additional potential and topology shifts are reported
in the appendix.

### 5.3 Metrics

We publish separate task scorecards rather than one cross-task number.
**HEA-SRO** reports order recovery and its ID/OOD/perturbation retention.
**Gamma-Decomp** reports component, phase/HKL-group, peak-parameter, and
calibration metrics. Family labels and HEA occupancy identifiability are
diagnostics and do not enter either leaderboard.

HEA-SRO computes masked MAE within each parent group and then macro-averages
parents. Relative to the mandatory composition-only baseline, its normalized
score is

\[
S_{SRO}=100\max(0,1-MAE_{model}/MAE_{base}).
\]

RMSE and per-shell Spearman are reported separately. Weakly and
non-identifiable occupancy pairs are not forced into exact classification.
They evaluate prediction-set coverage, abstention, risk-coverage, and whether
model uncertainty responds to empirical ambiguity. Identifiability thresholds
are selected on train/validation data and frozen before test evaluation.

Robustness is reported as perturbed-to-clean retention for HEA-SRO and as
component-metric retention for Gamma-Decomp. OOD results remain separated by
shift type. No overall composite score or complete-capability designation is
published.

Gamma-Decomp first matches predicted and true components using a declared cost
over phase, center, width, and normalized profile distance. It then reports
component precision/recall, matched center and width errors, profile error, and
phase/HKL assignment accuracy. The full-pattern task additionally evaluates
overlap-region localization, while the fixed-window task separately evaluates
single- versus multiple-component classification. Phase weight remains a separately reported
generation diagnostic and is not part of the primary leaderboard target.

All tasks report whole-pattern reconstruction separately. A forward-consistency
metric regenerates the pattern from predicted physical parameters, but this
metric never substitutes for latent-target accuracy. Probabilistic submissions
also report proper scoring rules, interval coverage, calibration error, and
risk-coverage curves. Confidence intervals use a two-level group-aware
bootstrap over model seeds and parent groups, keeping all counterfactuals and
observation views together.

## 6. Baselines

The benchmark will provide a compact but credible baseline suite:

1. Metadata-only controls using composition, parent phase, and acquisition
   parameters without XRD.
2. Peak-feature models using detected peak positions/intensities with linear,
   tree-based, or shallow neural predictors.
3. A standard 1D residual CNN operating on the full pattern.
4. A 1D Transformer or state-space sequence model with task-specific heads.
5. A physics-constrained encoder-decoder that predicts latent parameters and
   reconstructs XRD through a differentiable or surrogate forward layer.
6. For Gamma-Decomp, a conventional constrained pseudo-Voigt fitting baseline.

The metadata-only control is mandatory: if it matches the XRD model, the task is
dominated by composition or generation shortcuts. All baselines use identical
splits and are reported with multiple random seeds.

## 7. Experiments and Analysis

The final paper will answer six questions:

1. How much occupancy/order information is recoverable beyond metadata-only
   shortcuts?
2. Does hierarchical supervision improve latent recovery without sacrificing
   whole-pattern reconstruction?
3. Which OOD shift causes the largest degradation in each track?
4. Do models express uncertainty on spectrally similar but structurally
   different samples?
5. How much do weak peaks, superlattice reflections, and composition metadata
   contribute?
6. How large is the simulation-to-experiment gap under the labels actually
   available in the public experimental subset?

The required ablations are: remove weak peaks; remove composition metadata;
remove component supervision; train without perturbation randomization; and
replace controlled same-composition splits with ordinary random splits. The
last ablation quantifies shortcut inflation.

The main result table will contain ID and primary OOD performance for both
primary tasks. A second compact table will report calibration and identifiability-set
performance. Detailed per-element, per-HKL, per-instrument, and per-potential
results belong in the appendix.

## 8. Limitations and Responsible Use

SiteXRD-Bench provides computational structure and diffraction truth, not direct
proof that conventional experimental powder XRD uniquely determines atomic
occupancy. Periodic supercells can convert occupancy variation into weak
reflections that do not perfectly represent diffuse scattering in real samples.
Scattering-factor similarity, finite resolution, texture, microstrain,
nanostructure, and unmodeled phases can make the inverse problem fundamentally
ambiguous. The benchmark therefore evaluates identifiability and uncertainty
and forbids interpreting low reconstruction error as unique structural
recovery.

The dataset must document the licenses and biases of every source structure,
potential, and experimental pattern. Models trained on the benchmark should not
be used as a substitute for crystallographic refinement or complementary
measurements without domain validation.

## 9. Conclusion

SiteXRD-Bench reframes alloy XRD analysis as a hierarchical structured inverse
problem. By separating observed-pattern fit from component and chemical-order
correctness, it enables evaluation of whether data-mining models recover
physically meaningful latent variables under controlled and OOD conditions.
The benchmark's two tracks expose complementary challenges: subtle
occupancy-dependent evidence in HEAs and overlapping phase evidence in
gamma/gamma-prime superalloys. Public data, fixed splits, baseline code, and
auditable generation procedures are intended to support reproducible progress
on physically constrained inverse learning.

## Planned Main-Paper Figures and Tables

- Figure 1: hierarchical structure-to-reflection-to-observation data model.
- Figure 2: controlled HEA SRO pairs and gamma/gamma-prime overlap examples.
- Figure 3: tasks, eligibility masks, and OOD split design.
- Figure 4: main ID/OOD leaderboard.
- Figure 5: identifiability and calibration analysis.
- Table 1: comparison with existing XRD datasets and benchmarks.
- Table 2: dataset and split statistics.
- Table 3: main baseline results.

## Citation Audit Queue

No citation should be invented. Before converting this draft to LaTeX, collect
primary references for:

- occupancy-sensitive and anomalous/resonant XRD;
- HEA SRO measurement and simulation;
- gamma/gamma-prime overlapping-peak decomposition;
- XRD machine-learning datasets and inverse models;
- permutation-invariant set prediction and component matching;
- uncertainty under inverse-problem non-identifiability;
- scientific benchmark and data-card best practices.
