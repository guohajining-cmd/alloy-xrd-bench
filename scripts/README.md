# Scripts

Utility scripts for local benchmark development.

Generate a minimal valid workspace:

```powershell
python scripts/make_minimal_workspace.py --root data/examples/minimal_workspace
python -m alloy_xrd_bench.validate data/examples/minimal_workspace --json
```

Register a potential in a workspace:

```powershell
python -m alloy_xrd_bench.potentials register --help
```

Simulate XRD from relaxed structures:

```powershell
python -m alloy_xrd_bench.xrd simulate --help
```

Generate averaged XRD:

```powershell
python -m alloy_xrd_bench.xrd average --help
```

Build an occupancy-aware HEA XRD benchmark:

```powershell
python -m alloy_xrd_bench.occupancy_xrd inventory --csv <source.csv> --output <dataset>
python -m alloy_xrd_bench.occupancy_xrd label --dataset <dataset> --pilot-size 1000
python -m alloy_xrd_bench.occupancy_xrd split --dataset <dataset>
python -m alloy_xrd_bench.occupancy_xrd simulate --dataset <dataset> --subset pilot --augmentations 1
python -m alloy_xrd_bench.occupancy_xrd review-export --dataset <dataset> --output <review-dir> --batch-size 50
python -m alloy_xrd_bench.occupancy_xrd review-apply --dataset <dataset> --input <review.csv>
python -m alloy_xrd_bench.occupancy_xrd audit --dataset <dataset>
```

The staged workflow stores XRD grids in HDF5 shards and peak tables in Parquet.
`audit` reports engineering validity separately from the 95% manual pilot-review
gate required for `benchmark_ready`. Workstation XRD production supports a separate
`xrd-artifact` purpose that defers semantic labels and does not bypass the release
gate.

Run the smoke relaxation workflow:

```powershell
python -m alloy_xrd_bench.relax smoke --help
```

Run the CHGNet relaxation workflow:

```powershell
python -m alloy_xrd_bench.relax chgnet --help
```

Generate release audits:

```powershell
python -m alloy_xrd_bench.audit --help
```

Run physics sanity audits:

```powershell
python -m alloy_xrd_bench.quality --help
```

Generate release metadata:

```powershell
python -m alloy_xrd_bench.metadata --help
```

Package release files:

```powershell
python -m alloy_xrd_bench.package --help
```

Check release handoff readiness:

```powershell
python -m alloy_xrd_bench.release_readiness --help
```

Publication metadata template:

```powershell
Get-Content templates/publication_record.yml
```
