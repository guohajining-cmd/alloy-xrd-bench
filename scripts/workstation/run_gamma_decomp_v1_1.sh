#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
release="${SITEXRD_GAMMA_RELEASE:-/home/GJM/data/gamma-decomp-v1_1/gamma-decomp-core-v1.1}"
run_root="${SITEXRD_GAMMA_RUN_ROOT:-/home/GJM/runs/gamma_decomp_validation_v1_1}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"
python_cmd=("$micromamba" run -n sitexrd-validate python)
tasks=(full window)
seeds=(2027 2028 2029)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

score_is_complete() {
  local score="$1" task="$2"
  [[ -s "$score/scorecard.json" \
     && -s "$score/per_parent_pair.parquet" \
     && -s "$score/per_instrument_slice.parquet" ]] || return 1
  if [[ "$task" == "window" ]]; then
    [[ -s "$score/per_difficulty.parquet" ]] || return 1
  fi
}

run_is_complete() {
  local output="$1" task="$2"
  [[ -s "$output/training_summary.json" && -s "$output/best_model.pt" \
     ]] || return 1
  score_is_complete "$output/validation_score" "$task" \
    && score_is_complete "$output/test_score" "$task"
}

baseline_is_complete() {
  local output="$1" task="$2"
  score_is_complete "$output/score" "$task"
}

rescore_existing_submission() {
  local source_release="$1" task="$2" split="$3" output="$4"
  if [[ -s "$output/submission/submission_components.parquet" \
     && -s "$output/submission/submission_samples.parquet" ]] \
     && ! baseline_is_complete "$output" "$task"; then
    "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap evaluate \
      --release "$source_release" --task "$task" --split "$split" \
      --submission "$output/submission" --output "$output/score"
  fi
}

rescore_neural_submissions() {
  local source_release="$1" task="$2" output="$3"
  local split
  for split in validation test; do
    if [[ -s "$output/${split}_submission/submission_components.parquet" \
       && -s "$output/${split}_submission/submission_samples.parquet" ]] \
       && ! score_is_complete "$output/${split}_score" "$task"; then
      "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap evaluate \
        --release "$source_release" --task "$task" --split "$split" \
        --submission "$output/${split}_submission" --output "$output/${split}_score"
    fi
  done
}

comparison_is_complete() {
  local comparison="$1"
  [[ -s "$comparison/comparison.json" \
     && -s "$comparison/bootstrap_samples.parquet" \
     && -s "$comparison/paired_parent_metrics.parquet" ]]
}

training_pids=()

run_neural() {
  local gpu="$1" source_release="$2" task="$3" model="$4" output="$5"
  shift 5
  rescore_neural_submissions "$source_release" "$task" "$output"
  if ! run_is_complete "$output" "$task"; then
    CUDA_VISIBLE_DEVICES="$gpu" "${python_cmd[@]}" \
      -m alloy_xrd_bench.gamma_overlap train-baseline \
      --release "$source_release" --task "$task" --split test --model "$model" \
      "$@" --output "$output" &
    training_pids+=("$!")
  fi
}

wait_training() {
  local training_pid
  for training_pid in "${training_pids[@]}"; do
    wait "$training_pid"
  done
  training_pids=()
}

mkdir -p "$(dirname "$run_root")"
exec 9>"${run_root}.lock"
if ! flock -n 9; then
  echo "Gamma-Decomp v1.1 runner is already active for $run_root" >&2
  exit 75
fi
mkdir -p "$run_root"
cd "$project_root"
snapshot="$run_root/run_snapshot.sha256"
if [[ -s "$snapshot" ]]; then
  if ! sha256sum --check --status "$snapshot"; then
    echo "Gamma-Decomp code snapshot changed; use a new run root" >&2
    exit 78
  fi
else
  sha256sum \
    alloy_xrd_bench/gamma_overlap_public.py \
    alloy_xrd_bench/gamma_overlap_training.py \
    alloy_xrd_bench/gamma_overlap_models.py \
    alloy_xrd_bench/gamma_overlap_compare.py \
    alloy_xrd_bench/gamma_overlap_readiness.py \
    alloy_xrd_bench/gamma_overlap_ablation.py \
    configs/nickel_gamma_overlap_v1.json \
    scripts/workstation/run_gamma_decomp_v1_1.sh \
    > "$snapshot"
fi
"${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap validate-release --release "$release" \
  > "$run_root/release_validation.json"
random_split_release="$run_root/ablations/random_split/release"
if [[ ! -s "$random_split_release/ablation_metadata.json" ]]; then
  "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap_ablation \
    --release "$release" --output "$random_split_release" --seed 2027
fi

for task in "${tasks[@]}"; do
  oracle_submission="$run_root/oracle/$task/validation_submission"
  oracle_score="$run_root/oracle/$task/validation_score"
  if ! score_is_complete "$oracle_score" "$task"; then
    if [[ -s "$oracle_submission/submission_components.parquet" \
       && -s "$oracle_submission/submission_samples.parquet" ]]; then
      "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap evaluate \
        --release "$release" --task "$task" --split validation \
        --submission "$oracle_submission" --output "$oracle_score"
      continue
    fi
    "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap make-oracle \
      --release "$release" --task "$task" --split validation --output "$oracle_submission"
    "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap evaluate \
      --release "$release" --task "$task" --split validation \
      --submission "$oracle_submission" --output "$oracle_score"
  fi
done

for task in "${tasks[@]}"; do
  for split in validation test; do
    output="$run_root/no-spectrum/$task/$split"
    rescore_existing_submission "$release" "$task" "$split" "$output"
    if ! baseline_is_complete "$output" "$task"; then
      "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap train-baseline \
        --release "$release" --task "$task" --split "$split" --model no-spectrum \
        --output "$output"
    fi
  done

  for seed in "${seeds[@]}"; do
    for split in validation test; do
      output="$run_root/peak-feature/$task/$seed/$split"
      rescore_existing_submission "$release" "$task" "$split" "$output"
      if ! baseline_is_complete "$output" "$task"; then
        "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap train-baseline \
          --release "$release" --task "$task" --split "$split" --model peak-feature \
          --seed "$seed" --n-estimators 300 --output "$output"
      fi
    done

    run_neural 0 "$release" "$task" resnet-set \
      "$run_root/resnet-set/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0
    run_neural 1 "$release" "$task" transformer-set \
      "$run_root/transformer-set/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0
    wait_training

    run_neural 0 "$release" "$task" transformer-set \
      "$run_root/physics-constrained/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0.1
    run_neural 1 "$release" "$task" transformer-set \
      "$run_root/reconstruction-only/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 1 \
      --class-weight 0 --parameter-weight 0 --existence-weight 0 --mask-weight 0
    wait_training

    run_neural 0 "$release" "$task" transformer-set \
      "$run_root/ablations/weak_peaks/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0 \
      --minimum-relative-intensity 0.05
    run_neural 1 "$release" "$task" transformer-set \
      "$run_root/ablations/no_perturbation_training/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0 \
      --training-spectrum-view clean
    wait_training

    run_neural 0 "$random_split_release" "$task" transformer-set \
      "$run_root/ablations/random_split/$task/$seed" \
      --seed "$seed" --epochs 50 --patience 10 --reconstruction-weight 0
    wait_training
  done
done

for split in validation test; do
  output="$run_root/bic/window/$split"
  rescore_existing_submission "$release" window "$split" "$output"
  if ! baseline_is_complete "$output" window; then
    "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap train-baseline \
      --release "$release" --task window --split "$split" --model bic --output "$output"
  fi
done

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    comparison="$run_root/comparisons/peak-feature/$task/$seed"
    if ! comparison_is_complete "$comparison"; then
      "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap compare-baselines \
        --reference "$run_root/no-spectrum/$task/validation/score" \
        --candidate "$run_root/peak-feature/$task/$seed/validation/score" \
        --output "$comparison" --replicates 2000 --seed 2027
    fi
    for model in resnet-set transformer-set physics-constrained reconstruction-only; do
      comparison="$run_root/comparisons/$model/$task/$seed"
      if ! comparison_is_complete "$comparison"; then
        "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap compare-baselines \
          --reference "$run_root/no-spectrum/$task/validation/score" \
          --candidate "$run_root/$model/$task/$seed/validation_score" \
          --output "$comparison" --replicates 2000 --seed 2027
      fi
    done
  done
done

set +e
"${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap_readiness \
  --release "$release" --run-root "$run_root" --output "$run_root/gamma_benchmark_readiness.json"
readiness_status=$?
set -e
if [[ "$readiness_status" -ne 0 && "$readiness_status" -ne 2 ]]; then
  exit "$readiness_status"
fi
if [[ "$readiness_status" -eq 0 ]]; then
  "${python_cmd[@]}" -m alloy_xrd_bench.gamma_overlap_paper_artifacts \
    --run-root "$run_root" --output "$run_root/paper_artifacts" --require-complete
fi
