#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
RUN_ID="${RUN_ID:-hea_smoke_16}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
PYTHON=("$MAMBA_EXE" run -n sitexrd python)

cd "$PROJECT_ROOT"
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd simulate-plan \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" --subset all \
  --purpose xrd-artifact --config configs/xrd_config_default.json \
  --shard-size 16 --max-structures 16
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd simulate \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" --shard-index 0 --resume
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd simulate-finalize \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd audit \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
