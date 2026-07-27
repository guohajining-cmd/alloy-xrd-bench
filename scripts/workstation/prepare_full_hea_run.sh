#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to the benchmark_v1 directory}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
SOURCE_CSV="$DATASET_DIR/source/hea.2023-04-06.csv"
EXPECTED_SHA256="42722654902f03efde4e1d53284741033f4c43b77d7ba5ced29e4cf873c455fe"
PYTHON=("$MAMBA_EXE" run -n sitexrd python)
MODE="xrd-only"

if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--mode" ]]; then
    echo "Usage: $0 [--mode xrd-only|benchmark-release]" >&2
    exit 2
  fi
  MODE="$2"
fi

case "$MODE" in
  xrd-only)
    PURPOSE="xrd-artifact"
    RUN_ID="${RUN_ID:-hea_relaxed_clean_xrd_v1}"
    ;;
  benchmark-release)
    PURPOSE="benchmark-release"
    RUN_ID="${RUN_ID:-hea_benchmark_release_v1}"
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
echo "$EXPECTED_SHA256  $SOURCE_CSV" | sha256sum --check --status

"${PYTHON[@]}" -m alloy_xrd_bench.occupancy_xrd simulate-plan \
  --dataset "$DATASET_DIR" --run-id "$RUN_ID" --subset all \
  --purpose "$PURPOSE" --config configs/xrd_config_default.json --shard-size 256
