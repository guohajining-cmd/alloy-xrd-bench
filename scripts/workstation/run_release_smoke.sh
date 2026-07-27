#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"
cd "$project_root"
"$micromamba" run -n sitexrd-validate python -m alloy_xrd_bench.release_smoke \
  --hea-dataset "${SITEXRD_HEA_DATASET:-/home/GJM/data/hea/benchmark_v1}" \
  --hea-config "$project_root/configs/hea_validation_v1.json" \
  --gamma-release "${SITEXRD_GAMMA_RELEASE:-/home/GJM/data/gamma-decomp-v1_1/gamma-decomp-core-v1.1}" \
  --output "${SITEXRD_SMOKE_OUTPUT:-/home/GJM/runs/sitexrd_release_smoke_v1_1}"
