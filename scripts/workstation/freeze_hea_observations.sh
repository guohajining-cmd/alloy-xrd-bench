#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
RUN_ID="${RUN_ID:-hea_observation_v1}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"

cd "$PROJECT_ROOT"
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd observations-freeze \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" \
  --config configs/xrd_perturbation_v1.json --variants 5 --splits validation,test \
  --shard-size 256
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd observations-audit \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
