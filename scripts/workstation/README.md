# HEA XRD Workstation Production

The production flow is CPU-only and never stores SSH credentials in scripts.

```bash
export PROJECT_ROOT="$HOME/xrd-benchmark"
export DATASET_DIR="$HOME/data/hea/benchmark_v1"

bash scripts/workstation/setup_environment.sh
bash scripts/workstation/smoke_hea_xrd.sh
bash scripts/workstation/benchmark_hea_shard.sh
bash scripts/workstation/prepare_full_hea_run.sh --mode xrd-only
bash scripts/workstation/submit_hea_xrd.sh
```

If the workstation cannot reach the micromamba download endpoint, transfer the
Linux archive first and set `MICROMAMBA_ARCHIVE=/path/to/micromamba.tar.bz2`.

After the Slurm array has no failed or running tasks:

```bash
bash scripts/workstation/finalize_hea_xrd.sh
```

`xrd-only` produces a label-neutral structure-to-XRD artifact run and only requires
the structure engineering audit. It records `purpose=xrd-artifact` and defers phase,
subset, split, and superlattice annotations until labels are frozen.

Use `--mode benchmark-release` only after all 1,000 pilot reviews are complete,
their pass rate is at least 95%, and the dataset audit reports
`benchmark_ready=true`. Failed array indices can be resubmitted directly; valid
completed shards are skipped by `--resume`.

## Occupancy/SRO expansion

Run the 1,000-parent pilot first:

```bash
bash scripts/workstation/prepare_hea_counterfactual.sh pilot
bash scripts/workstation/submit_hea_counterfactual.sh
bash scripts/workstation/finalize_hea_counterfactual.sh
```

After physical review, use `full` to request 10,000 parents. The CLI rejects a
full run while `benchmark_ready=false`.

Freeze five deterministic observation variants for validation and test only
after `benchmark-build` has published the task and split manifests:

```bash
bash scripts/workstation/freeze_hea_observations.sh
```

Training perturbations should call the same parameter sampler online and must
not be stored as additional independent material structures.

## Dataset-paper visuals

After both formal v1.1 releases are frozen, generate the traceable crystal/XRD
figures and synchronize them into the paper tree with:

```bash
bash scripts/workstation/generate_dataset_paper_visuals.sh
```

The runner uses seed 2027 and refuses an HEA root without the formal task,
composition, and structure manifests.
