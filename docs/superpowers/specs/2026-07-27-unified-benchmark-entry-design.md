# SiteXRD-Bench Unified Entry Design

## Goal

Expose the local HEA-SRO and Gamma-Decomp releases as one versioned benchmark,
`SiteXRD-Bench v1.1`, without moving or rewriting the verified underlying data.

## Chosen Structure

The benchmark has one root and two independent primary tracks:

1. `hea_sro`: masked first- and second-shell Warren-Cowley SRO prediction.
2. `gamma_decomp`: gamma/gamma-prime phase and HKL-group component recovery.

The tracks share release metadata, versioning, provenance expectations, and a
single registry. They retain separate schemas, splits, metrics, and result
tables. No cross-track composite score is defined.

## Unified Root

The local root is:

```text
data/benchmarks/siteXRD-bench-v1_1/
```

It gains two entry files:

```text
README.md
benchmark_registry.json
```

Existing verified data remains in place:

```text
hea/benchmark_v1/
hea/benchmark_v1_1/
gamma-decomp-v1_1/gamma-decomp-core-v1.1/
gamma-decomp-v1_1/gamma-decomp-provenance-v1.1/
```

## Registry Contract

`benchmark_registry.json` records:

- benchmark ID, display name, schema version, and release version;
- the two primary track IDs and human-readable names;
- relative paths to each track's inputs, labels, splits, and task registry;
- sample counts and point counts;
- primary and secondary metrics;
- diagnostic-only tasks and explicit exclusions;
- checksum manifest paths;
- the rule that track scores remain independent.

All paths are relative to the unified root so the release remains portable.

## Human Entry Point

The root `README.md` explains:

- that Gamma-Decomp means a gamma/gamma-prime two-phase decomposition task;
- which directories are canonical data and which are derived labels;
- how to choose a track;
- that 88,019 HEA task rows include 84,019 canonical structures plus 4,000
  diagnostic counterfactual rows;
- that Gamma-Decomp contains 10,000 full patterns and 21,116 windows;
- that the tracks must not be concatenated into one training table without an
  explicit multi-task loader.

## Validation

A unified validation command or module must fail when:

- a registered path is missing or escapes the benchmark root;
- declared sample counts differ from Parquet manifests;
- a checksum manifest is missing;
- a track ID or primary metric is absent;
- a composite cross-track score is enabled.

The validation is read-only. It must not update dataset manifests, audit files,
or checksums.

## Compatibility

No existing IDs, paths, hashes, splits, or XRD arrays change. Existing HEA and
Gamma loaders continue to work. The unified registry is an orchestration layer,
not a new merged scientific dataset.

## Acceptance Criteria

- One root registry exposes both primary tracks.
- Registry paths resolve on the transferred local release.
- Counts match 84,019 canonical HEA XRD patterns, 88,019 HEA task rows, 10,000
  Gamma full patterns, and 21,116 Gamma windows.
- Both checksum manifests and task-specific schemas remain reachable.
- Unified validation is read-only and passes on the local release.
- Existing tests remain green.
