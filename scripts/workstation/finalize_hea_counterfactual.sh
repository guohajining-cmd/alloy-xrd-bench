#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
RUN_ID="${RUN_ID:-hea_sro_cf_v1}"
CANONICAL_RUN="${CANONICAL_RUN:-hea_relaxed_clean_xrd_v1}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"

cd "$PROJECT_ROOT"
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd counterfactual-finalize \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID"
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd benchmark-build \
  --dataset "$DATASET_DIR" --counterfactual-run "$RUN_ID" --canonical-run "$CANONICAL_RUN"
