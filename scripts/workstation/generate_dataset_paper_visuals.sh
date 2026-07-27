#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/project/xrd-benchmark}"
hea_release_root="${SITEXRD_HEA_RELEASE_ROOT:-/home/GJM/data/hea/benchmark_v1_1}"
gamma_release_root="${SITEXRD_GAMMA_RELEASE_ROOT:-/home/GJM/data/releases/gamma-decomp-v1_1}"
output="${SITEXRD_PAPER_VISUAL_ROOT:-/home/GJM/runs/sitexrd_benchmark_v1_1/paper_visuals}"
paper_generated="$project_root/docs/paper/generated"
python_cmd="${SITEXRD_PYTHON:-python}"

cd "$project_root"

"$python_cmd" -m alloy_xrd_bench.dataset_paper_visuals \
  --hea-release-root "$hea_release_root" \
  --gamma-release-root "$gamma_release_root" \
  --output "$output" \
  --seed 2027

mkdir -p "$paper_generated"
for artifact in \
  dataset_overview.png \
  hea_sro_example.png \
  gamma_decomp_example.png \
  dataset_coverage.png \
  dataset_visual_examples.json \
  dataset_visual_manifest.json \
  compact_primary_results.tex
do
  install -m 0644 "$output/$artifact" "$paper_generated/$artifact"
done

printf 'Dataset paper visuals generated at %s and synchronized to %s\n' \
  "$output" "$paper_generated"
