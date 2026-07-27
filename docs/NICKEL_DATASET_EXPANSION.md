# Nickel Dataset Expansion

## Recommended Priority

The benchmark should not expand by collecting every structure that contains
nickel. The useful expansion order is:

1. ingest target-specific Ni3Al DFT configurations;
2. add Ni-Al defect, strain, temperature, and competing-phase structures;
3. query broad materials databases for chemistry and prototype coverage;
4. use experimental CIF and diffraction refinements as validation anchors.

## Tier 1: Immediately Actionable

### DaRUS-4480 Ni3Al DFT training set

- Dataset: https://doi.org/10.18419/DARUS-4480
- Paper: https://doi.org/10.1016/j.actamat.2024.120423
- License: CC BY 4.0
- Local file: `local_potential_cache/pot_ni3al_mtp_darus_yield_anomaly_2024/downloads/train.cfg`
- Verified size: 1,162 configurations
- Labels: energy, forces, and stress for every configuration
- Atom counts: 96, 107, 108, and 144
- Chemistry: stoichiometric Ni3Al plus one-vacancy variants

This is the highest-priority source because it is already downloaded,
hash-verified, target-specific, and much larger than the current 16 generated
nickel initial structures. It should be converted into canonical structures,
deduplicated, assigned parent/trajectory groups, and split by structural family
rather than by individual frame.

### DaRUS-3579 Ni3Al spin/APB training set

- Dataset: https://doi.org/10.18419/DARUS-3579
- Paper: https://doi.org/10.1016/j.actamat.2023.118986
- License: CC BY 4.0
- Archive: `MTP-training-set.tar.gz`, 580,894,549 bytes
- Content: VASP training structures for bulk Ni3Al, 100 APB, and 111 APB

This source is valuable for defect and ordering OOD tests. The local archive
probe is incomplete, so the full archive must be downloaded and inventoried
before its configuration count or phase balance is reported.

### Materials Cloud Al-Ni dataset

- Dataset: https://archive.materialscloud.org/records/wejc0-9tz40
- DOI: https://doi.org/10.24435/materialscloud:dq-ax
- License: CC BY 4.0
- Archive: `dataset-AlNi.tgz`, 369,256,120 bytes
- Scope: pure Ni, equiatomic AlNi, liquid configurations, and nucleation data

This source broadens temperature and phase coverage, but it is primarily an
AlNi/B2/liquid dataset. It should be used for phase-OOD and negative controls,
not presented as a gamma-prime occupancy dataset.

## Tier 2: Broad Computed Structure Sources

| Source | What it provides | Best benchmark use | Main limitation |
| --- | --- | --- | --- |
| OQMD | Large computed crystal database; API reported 91,382 Ni-containing calculations on 2026-07-23 | Ni-Al-Ti-Ta-Co-Cr prototype and chemistry expansion | duplicates, hypothetical phases, no occupancy truth |
| Materials Project | Relaxed inorganic structures and computed properties | stable/near-stable Ni compounds and prototype anchors | API key/terms review; mostly small ordered cells |
| JARVIS-DFT | DFT structures and properties | independent calculation-source OOD | limited superalloy-specific coverage |
| AFLOWlib | Enumerated prototypes and computed compounds | crystallographic diversity and prototype OOD | many ideal ordered compounds |
| NOMAD | Published DFT calculations and trajectories | paper-specific structures with calculation provenance | heterogeneous metadata and licenses |
| Alexandria | Millions of enumerated DFT structures | large chemistry/prototype pool | must aggressively filter and deduplicate |
| MPtrj/MatPES/OMat24 | Large atomistic training trajectories | relaxation and model-source OOD | highly correlated frames; not occupancy labels |

For broad databases, retain only structures meeting explicit rules, for
example:

```text
elements subset of {Ni, Al, Ti, Ta, Co, Cr, Mo, W, Re}
Ni atomic fraction >= 0.40
phase/prototype in {FCC, L12, B2, D019, D024, relevant TCP controls}
energy above hull <= chosen threshold
canonical structure hash is unique
```

## Tier 3: Experimental Anchors

Use COD and, where institutional access permits, ICSD to collect experimental
lattice parameters and refined occupancies. Search the primary literature for
Rietveld, neutron diffraction, anomalous/resonant XRD, and single-crystal
refinement of Ni3Al and Ni3(Al,Ti,Ta).

Experimental records should be a smaller validation set. Most commercial
nickel superalloys do not have a unique atom-by-atom structure because they are
multiphase, compositionally disordered, and process dependent.

## Ingestion Rules

Every imported structure must record:

```text
source_dataset_id
source_record_or_frame_id
source_url_and_doi
license
calculation_or_measurement_method
composition
structure_hash
parent_trajectory_or_family_id
energy_force_stress_availability
occupancy_label_origin
```

Frames from the same relaxation, molecular-dynamics trajectory, defect family,
or parent crystal must stay in the same data split. Otherwise a large imported
trajectory can inflate sample count while leaking nearly identical structures
between training and test sets.

## Practical Target

A defensible first nickel release can be built from:

- 1,162 DaRUS-4480 configurations as the core Ni3Al corpus;
- a curated subset of DaRUS-3579 APB/defect structures;
- a filtered subset of the Materials Cloud Al-Ni dataset for phase and
  temperature OOD;
- 200-500 deliberately generated occupancy/antisite variants;
- a small experimental validation set with literature-backed lattice and
  occupancy information.

This produces meaningful structural diversity without claiming that every DFT
snapshot is an independent alloy or that generic Ni-containing compounds carry
site-occupancy ground truth.
