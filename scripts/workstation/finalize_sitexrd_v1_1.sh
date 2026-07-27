#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
hea_run="${SITEXRD_HEA_RUN_ROOT:-/home/GJM/runs/sitexrd_hea_validation_v1_1}"
gamma_run="${SITEXRD_GAMMA_RUN_ROOT:-/home/GJM/runs/gamma_decomp_validation_v1_1}"
output="${SITEXRD_BENCHMARK_READINESS:-/home/GJM/runs/sitexrd_benchmark_v1_1_readiness.json}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"

cd "$project_root"
"$micromamba" run -n sitexrd-validate python \
  -m alloy_xrd_bench.benchmark_readiness \
  --hea-readiness "$hea_run/hea_benchmark_readiness.json" \
  --gamma-readiness "$gamma_run/gamma_benchmark_readiness.json" \
  --output "$output"
