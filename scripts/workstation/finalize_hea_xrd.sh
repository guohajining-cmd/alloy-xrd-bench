#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
RUN_ID="${RUN_ID:-hea_relaxed_clean_xrd_v1}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
PYTHON=("$MAMBA_EXE" run -n sitexrd python)

cd "$PROJECT_ROOT"
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd simulate-finalize \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd audit \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
cd "$DATASET_DIR/xrd/runs/$RUN_ID"
sha256sum --check checksums.sha256
