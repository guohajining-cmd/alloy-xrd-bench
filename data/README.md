# Data Directory

This repository currently stores the benchmark contract, schemas, and templates.
Large benchmark payloads should be placed in versioned release artifacts rather
than committed directly.

Expected release layout:

```text
registry/
  potentials_catalog.yml
  relax_configs.json
  xrd_configs.json
shared/
  recipes/
  structures_initial/
  manifests/
    initial_structures.parquet
    gamma_prime_pairs.parquet
potentials/
  {potential_id}/
    potential/
      metadata.yml
      checksums.sha256
      files/
    generated/
      manifests/
      relaxation_logs/
      structures_relaxed/
      xrd/
        grid/
        peaks/
    audits/
local_potential_cache/
  {candidate_id}/
    downloads/
    download_audit.json
    checksums.sha256
releases/
  v0.1/
    manifests/
    potentials/
    shared/
    audits/
```

Each `potential_id` owns one folder under `data/potentials/`. Potential files or
metadata go in `potential/`; everything generated with that potential goes in
`generated/`.

Initial structures are shared benchmark inputs and live in `data/shared/`. Their
manifests are referenced by each potential-specific relaxation run rather than
copied into every potential folder.

Candidate MLIP downloads stay outside the release tree under
`local_potential_cache/` until the model is accepted into
`potentials_catalog.yml`. This keeps unclear-license or missing-engine model
files out of `data/potentials/` while preserving hash-audited provenance.

For gamma/gamma-prime systems, the shared pair manifest records initial
structure pairs. Each accepted potential also writes a
`generated/manifests/gamma_prime_pairs.parquet` file that resolves those pairs to
its own relaxed structure IDs.
