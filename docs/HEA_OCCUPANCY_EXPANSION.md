# HEA Occupancy and SRO Expansion

The canonical layer remains immutable: 84,019 source-relaxed DFT structures and
one clean 4,001-point XRD pattern per structure. Derived occupancy structures
and experimental observation views are separate artifacts and must be reported
separately.

## Fidelity labels

- `source_relaxed_dft`: source final-relaxation structure.
- `fixed_lattice_counterfactual`: composition-preserving site permutation with
  the source lattice and fractional coordinates held fixed.
- `mlip_relaxed_counterfactual`: reserved for a later admitted-MLIP relaxation
  layer; the current generator never assigns this label.

The fixed-lattice tier isolates occupancy-dependent intensity changes. It is not
a claim of thermodynamic stability or new DFT relaxation.

## Counterfactual production

```powershell
python -m alloy_xrd_bench.occupancy_xrd counterfactual-plan `
  --dataset <dataset> --run-id hea_sro_cf_v1 --parents 1000 `
  --variants-per-parent 4 --config configs/sro_counterfactual_v1.json

python -m alloy_xrd_bench.occupancy_xrd counterfactual-generate `
  --dataset <dataset> --run-id hea_sro_cf_v1 --shard-index 0 --resume

python -m alloy_xrd_bench.occupancy_xrd counterfactual-finalize `
  --dataset <dataset> --run-id hea_sro_cf_v1
```

The four deterministic strategies are local swap, randomized occupancy,
ordering-enhanced occupancy, and clustering-enhanced occupancy. Ordering and
clustering are audited against the randomized member of the same parent group.
Parents must contain at least 27 atoms; smaller periodic cells often have an
effectively complete first-neighbor graph, making SRO invariant to site
permutations and invalidating enhanced-ordering/clustering labels.
Runs above 1,000 parents require the dataset scientific gate to report
`benchmark_ready=true`.

## Benchmark manifests

```powershell
python -m alloy_xrd_bench.occupancy_xrd benchmark-build `
  --dataset <dataset> --counterfactual-run hea_sro_cf_v1 `
  --canonical-run hea_relaxed_clean_xrd_v1
```

This writes task samples, occupancy pairs, parent-group split assignments, and
composition-only control features. Parent structures, counterfactuals, and all
observation views share one ID split.

## Frozen observations

```powershell
python -m alloy_xrd_bench.occupancy_xrd observations-freeze `
  --dataset <dataset> --run-id hea_observation_v1 `
  --config configs/xrd_perturbation_v1.json --variants 5 `
  --splits validation,test
```

Observation patterns are reconstructed from indexed peak components and include
instrument broadening, microstrain, crystallite size, polynomial background,
mixed Poisson/Gaussian noise, zero shift, and March-Dollase texture. Parameters
and seeds are stored per sample. Training uses the same deterministic parameter
sampler online rather than storing repeated grids.

Interrupted observation runs validate and reuse completed shards. After a run
finishes, audit all grids, manifests, parameters, checksums, and split grouping:

```powershell
python -m alloy_xrd_bench.occupancy_xrd observations-audit `
  --dataset <dataset> --run-id hea_observation_v1
```

## Release boundary

`classification_confidence` is quality metadata, not a scientific target. The
existing rule-assisted pilot adjudication is not an independent expert review.
The 10,000-parent production run remains blocked until the label audit and
expert review satisfy the 95% release threshold.
