#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
hea_run="${SITEXRD_HEA_RUN_ROOT:-/home/GJM/runs/sitexrd_hea_validation_v1_1}"
gamma_run="${SITEXRD_GAMMA_RUN_ROOT:-/home/GJM/runs/gamma_decomp_validation_v1_1}"
smoke_run="${SITEXRD_SMOKE_OUTPUT:-/home/GJM/runs/sitexrd_release_smoke_v1_1}"
publication_record="${SITEXRD_PUBLICATION_RECORD:-$project_root/publication_record.json}"
output="${SITEXRD_KDD_READINESS:-/home/GJM/runs/sitexrd_kdd_submission_readiness.json}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"

cd "$project_root"
"$micromamba" run -n sitexrd-validate python \
  -m alloy_xrd_bench.benchmark_readiness \
  --hea-readiness "$hea_run/hea_benchmark_readiness.json" \
  --gamma-readiness "$gamma_run/gamma_benchmark_readiness.json" \
  --paper "$project_root/docs/paper/sitexrd_bench_kdd2027.tex" \
  --hea-paper-manifest "$hea_run/paper_artifacts/paper_artifacts.json" \
  --gamma-paper-manifest "$gamma_run/paper_artifacts/gamma_paper_artifacts.json" \
  --release-smoke "$smoke_run/release_smoke.json" \
  --environment-lock "$project_root/environment-validation-linux-64.lock" \
  --external-sanity "$project_root/data/external_sanity/status.json" \
  --publication-record "$publication_record" \
  --output "$output" --require-submission
