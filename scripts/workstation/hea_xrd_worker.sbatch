#!/usr/bin/env bash
#SBATCH --job-name=hea-xrd
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=logs/hea_xrd_%A_%a.out
#SBATCH --error=logs/hea_xrd_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR was not exported}"
RUN_ID="${RUN_ID:-hea_relaxed_clean_xrd_v1}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"

cd "$PROJECT_ROOT"
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd simulate \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --resume
