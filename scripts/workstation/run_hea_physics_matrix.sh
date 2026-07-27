#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
dataset="${SITEXRD_HEA_DATASET:-/home/GJM/data/hea/benchmark_v1}"
observation_dataset="${SITEXRD_HEA_V1_1_DATASET:-/home/GJM/data/hea/benchmark_v1_1}"
run_root="${SITEXRD_HEA_PHYSICS_RUN_ROOT:-/home/GJM/runs/sitexrd_hea_validation_v1_1_physics_appendix}"
observation_run="${SITEXRD_HEA_OBSERVATION_RUN:-hea_observation_v1_1}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"
python_cmd=("$micromamba" run -n sitexrd-validate python)
config="$project_root/configs/hea_validation_v1.json"
perturbation_config="$project_root/configs/xrd_perturbation_v1.json"
splits=(id_split composition_ood_split chemistry_ood_split sro_ood_split)
seeds=(2027 2028 2029 2030 2031)

run_is_complete() {
  local seed="$1" split_column="$2"
  local stem="physics-transformer_seed${seed}_${split_column}_test"
  local submission="$run_root/predictions/submissions/physics-transformer/$seed/${split_column}_test"
  [[ -s "$run_root/predictions/$stem.parquet" \
     && -s "$run_root/predictions/$stem.metrics.json" \
     && -s "$run_root/predictions/$stem.pt" \
     && -s "$submission/metadata.json" \
     && -s "$submission/sro.parquet" ]]
}

prediction_is_complete() {
  local seed="$1" split_column="$2"
  local stem="physics-transformer_seed${seed}_${split_column}_test"
  [[ -s "$run_root/predictions/$stem.parquet" \
     && -s "$run_root/predictions/$stem.metrics.json" \
     && -s "$run_root/predictions/$stem.pt" ]]
}

score_is_complete() {
  local seed="$1" split_column="$2"
  local score="$run_root/scores/physics-transformer/$seed/${split_column}_test"
  [[ -s "$score/metrics.json" && -s "$score/metrics_by_slice.parquet" ]]
}

run_worker() {
  local gpu="$1" job_index=0
  for split_column in "${splits[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( job_index % 2 == gpu )); then
        if ! run_is_complete "$seed" "$split_column"; then
          if prediction_is_complete "$seed" "$split_column"; then
            "${python_cmd[@]}" -m alloy_xrd_bench.hea_validation materialize-submission \
              --dataset "$dataset" \
              --predictions "$run_root/predictions/physics-transformer_seed${seed}_${split_column}_test.parquet" \
              --output "$run_root/predictions" --model physics-transformer \
              --seed "$seed" --split-column "$split_column"
          else
            CUDA_VISIBLE_DEVICES="$gpu" "${python_cmd[@]}" \
              -m alloy_xrd_bench.hea_validation train \
              --model physics-transformer --dataset "$dataset" --config "$config" \
              --output "$run_root/predictions" --split-column "$split_column" --seed "$seed" \
              > "$run_root/physics-transformer_${split_column}_seed${seed}.run.json" \
              2> "$run_root/physics-transformer_${split_column}_seed${seed}.log"
          fi
        fi
        if ! score_is_complete "$seed" "$split_column"; then
          "${python_cmd[@]}" -m alloy_xrd_bench.evaluate score \
            --dataset "$dataset" \
            --predictions "$run_root/predictions/submissions/physics-transformer/$seed/${split_column}_test" \
            --output "$run_root/scores/physics-transformer/$seed/${split_column}_test" \
            --config "$config"
        fi
      fi
      job_index=$((job_index + 1))
    done
  done
}

mkdir -p "$run_root/predictions" "$run_root/scores" "$run_root/perturbation/physics-transformer" "$run_root/evidence"
cd "$project_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_worker 0 & worker_0=$!
run_worker 1 & worker_1=$!
wait "$worker_0"
wait "$worker_1"

perturbation_pids=()
for seed in "${seeds[@]}"; do
  output="$run_root/perturbation/physics-transformer/$seed"
  if [[ ! -s "$output/perturbation_summary.json" ]]; then
    "${python_cmd[@]}" -m alloy_xrd_bench.hea_perturbation evaluate \
      --dataset "$dataset" --observation-dataset "$observation_dataset" \
      --observation-run "$observation_run" \
      --checkpoint "$run_root/predictions/physics-transformer_seed${seed}_id_split_test.pt" \
      --clean-metrics "$run_root/predictions/physics-transformer_seed${seed}_id_split_test.metrics.json" \
      --config "$config" --perturbation-config "$perturbation_config" --output "$output" &
    perturbation_pids+=("$!")
    if (( ${#perturbation_pids[@]} == 2 )); then
      wait "${perturbation_pids[0]}"
      wait "${perturbation_pids[1]}"
      perturbation_pids=()
    fi
  fi
done
for pid in "${perturbation_pids[@]}"; do wait "$pid"; done

"${python_cmd[@]}" -m alloy_xrd_bench.hea_perturbation summarize \
  --input "$run_root/perturbation" --output "$run_root/evidence/perturbation_summary.json" \
  --minimum-seeds 5 --required-models physics-transformer
