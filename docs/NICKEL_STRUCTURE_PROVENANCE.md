# Nickel Structure Provenance

## Current Provenance Statement

The current nickel-alloy structures are benchmark-generated structures. They
were not downloaded from AFLOW, OQMD, COD, Materials Project, or ICSD, and they
must not be described as collected experimental structures.

The direct generation chain is:

```text
configs/alloy_recipes_v0_1.yml
  -> alloy_xrd_bench/structures.py
  -> data/shared/structures_initial/*.json
  -> MLIP relaxation
  -> data/potentials/{potential_id}/generated/structures_relaxed/*
```

## Initial Structures

The recipe catalog defines 16 nickel-alloy initial structures arranged into
eight gamma/gamma-prime pairs:

| Alloy system | Gamma structures | Gamma-prime structures | Pair count |
| --- | ---: | ---: | ---: |
| Ni-Al | 2 | 2 | 2 |
| Ni-Al-Ti | 2 | 2 | 2 |
| Ni-Al-Ta | 2 | 2 | 2 |
| Ni-Al-Ti-Ta | 2 | 2 | 2 |

All structures use 2 x 2 x 2 conventional supercells. Gamma structures use a
canonical FCC basis and deterministic `pair_balance_occupancy`: 32 candidate
decorations are generated from each recipe seed, and the candidate with the
smallest two-shell pair-balance score is retained. This is an internal
SQS-like heuristic, not an ATAT/mcsqs calculation and not an experimental atom
configuration.

Gamma-prime structures use the canonical L12 basis. Al, Ti, and Ta are assigned
to the corner sublattice according to the recipe, and Ni is assigned to the
face-center sublattice. The generator stores discrete atom decorations; it does
not currently store experimentally refined partial occupancies.

## Crystallographic Reference Versus File Origin

The FCC and L12 fractional coordinates are implemented directly in
`alloy_xrd_bench/structures.py`. The L12 topology has been independently
cross-checked against the AFLOW `AB3_cP4_221_a_c-001` prototype and OQMD AlNi3
entry 17890:

- AFLOW: https://aflow.org/p/AB3_cP4_221_a_c-001
- OQMD API: https://oqmd.org/oqmdapi/formationenergy?composition=AlNi3

These records are crystallographic references, not the file source of the
current local JSON structures. The local structures pre-exist the source
registry and were generated from the project recipe file.

## Parameter Status

The initial lattice constants are benchmark configuration values:

| Alloy system | Initial lattice parameter (angstrom) |
| --- | ---: |
| Ni-Al | 3.57 |
| Ni-Al-Ti | 3.58 |
| Ni-Al-Ta | 3.59 |
| Ni-Al-Ti-Ta | 3.59 |

At present, these values are not tied to specific experimental records. The
nominal gamma/gamma-prime compositions and partitioning rules are also benchmark
design choices rather than measured phase compositions. They must be described
as controlled simulation parameters until literature or experimental anchors
are attached.

## Relaxed Structures

The 16 initial structures have 50 nickel-alloy relaxed derivatives:

| Potential | Nickel structures | Potential source |
| --- | ---: | --- |
| CHGNet 0.3.0 | 16 | doi:10.1038/s42256-023-00716-3 |
| MACE-MP-0b3-medium | 16 | arXiv:2401.00096 |
| MatGL/M3GNet MatPES-PBE-2025.2 | 16 | doi/arXiv:2212.07687 |
| Ni3Al MTP | 2 | doi:10.18419/DARUS-4480 |

These are MLIP-relaxed derivatives of the generated initial structures. They
are not DFT structures or experimentally refined structures unless a future
manifest row explicitly records such a source.

## Paper-Ready Wording

Use:

> We constructed parameterized gamma-FCC and gamma-prime-L12 supercells from
> canonical crystallographic prototypes. Element decorations, compositions,
> and random seeds were controlled by versioned benchmark recipes, after which
> the structures were relaxed using registered machine-learning interatomic
> potentials. The resulting structures are simulated benchmark assets rather
> than collected experimental crystal structures.

Do not use:

> We collected nickel-superalloy structures from AFLOW/OQMD.

Do not call the configured compositions, lattice parameters, or partitioning
rules experimental ground truth until each value is linked to an admissible
experimental source with measurement conditions and uncertainty.
