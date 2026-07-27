#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
RUN_ID="${RUN_ID:-hea_sro_cf_v1}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
PLAN_PATH="$DATASET_DIR/derived/occupancy/$RUN_ID/run_plan.json"

cd "$PROJECT_ROOT"
mkdir -p logs
shard_count=$("$MAMBA_EXE" run -n sitexrd python -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["shard_count"])' \
  "$PLAN_PATH")
sbatch --array="0-$((shard_count - 1))%24" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,DATASET_DIR=$DATASET_DIR,RUN_ID=$RUN_ID,MAMBA_EXE=$MAMBA_EXE" \
  scripts/workstation/hea_counterfactual_worker.sbatch
