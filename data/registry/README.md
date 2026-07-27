# Registry

Global benchmark registries live here:

- `potentials_catalog.yml`
- `mlip_candidates.yml`
- `nickel_structure_sources.yml`
- `relax_configs.json`
- `xrd_configs.json`

Each potential catalog row points to its own `data/potentials/{potential_id}/`
workspace.

`mlip_candidates.yml` is a pre-admission registry. Candidate rows do not create
`data/potentials/{potential_id}/` directories and do not enter benchmark
manifests until they are accepted into `potentials_catalog.yml`.

`nickel_structure_sources.yml` separates external crystallographic references
from benchmark-generated gamma/gamma-prime occupancy variants. A source entry
does not enter the released structure corpus until its identity, citation, and
redistribution policy have passed review.
