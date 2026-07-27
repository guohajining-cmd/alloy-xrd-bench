#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
RUN_ID="${RUN_ID:-hea_sro_cf_v1}"
MODE="${1:-pilot}"

case "$MODE" in
  pilot) PARENTS=1000 ;;
  full) PARENTS=10000 ;;
  *) echo "Usage: $0 [pilot|full]" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
"$MAMBA_EXE" run -n sitexrd python -m alloy_xrd_bench.occupancy_xrd counterfactual-plan \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" --parents "$PARENTS" \
  --variants-per-parent 4 --config configs/sro_counterfactual_v1.json --shard-size 64
