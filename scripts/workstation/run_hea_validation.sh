#!/usr/bin/env bash
set -euo pipefail

project_root="${SITEXRD_PROJECT_ROOT:-/home/GJM/xrd-benchmark}"
source_dataset="${SITEXRD_HEA_DATASET:-/home/GJM/data/hea/benchmark_v1}"
output_dataset="${SITEXRD_HEA_V1_1_DATASET:-/home/GJM/data/hea/benchmark_v1_1}"
run_root="${SITEXRD_HEA_RUN_ROOT:-/home/GJM/runs/sitexrd_hea_validation_v1_1}"
micromamba="${MICROMAMBA_EXE:-/home/GJM/.local/bin/micromamba}"
config="$project_root/configs/hea_validation_v1.json"
release_config="$project_root/configs/hea_benchmark_v1_1.json"
perturbation_config="$project_root/configs/xrd_perturbation_v1.json"
observation_run="${SITEXRD_HEA_OBSERVATION_RUN:-hea_observation_v1_1}"
source_license="CC-BY-4.0"
python_cmd=("$micromamba" run -n sitexrd-validate python)
splits=(id_split composition_ood_split chemistry_ood_split sro_ood_split)
seeds=(2027 2028 2029 2030 2031)
xrd_models=(peak-xgboost resnet1d transformer)

mkdir -p "$(dirname "$run_root")"
exec 9>"${run_root}.lock"
if ! flock -n 9; then
  echo "HEA v1.1 runner is already active for $run_root" >&2
  exit 75
fi

run_is_complete() {
  local model="$1" seed="$2" split_column="$3"
  local stem="${model//-/_}_seed${seed}_${split_column}_test"
  local submission="$run_root/predictions/submissions/$model/$seed/${split_column}_test"
  [[ -s "$run_root/predictions/$stem.parquet" ]] || return 1
  [[ -s "$run_root/predictions/$stem.metrics.json" ]] || return 1
  [[ -s "$submission/metadata.json" && -s "$submission/sro.parquet" ]] || return 1
  if [[ "$model" == "peak-xgboost" ]]; then
    [[ -s "$run_root/predictions/$stem.joblib" ]] || return 1
  elif [[ "$model" == "resnet1d" || "$model" == "transformer" ]]; then
    [[ -s "$run_root/predictions/$stem.pt" ]] || return 1
  fi
}

score_is_complete() {
  local model="$1" seed="$2" split_column="$3"
  local score="$run_root/scores/$model/$seed/${split_column}_test"
  [[ -s "$score/metrics.json" && -s "$score/metrics_by_slice.parquet" ]]
}

mkdir -p "$run_root/predictions" "$run_root/scores" "$run_root/comparisons" "$run_root/evidence"
cd "$project_root"
snapshot="$run_root/run_snapshot.sha256"
if [[ -s "$snapshot" ]]; then
  if ! sha256sum --check --status "$snapshot"; then
    echo "HEA v1.1 code snapshot changed; use a new run root" >&2
    exit 78
  fi
else
  sha256sum \
    alloy_xrd_bench/hea_prediction_audit.py \
    alloy_xrd_bench/hea_benchmark_readiness.py \
    configs/hea_benchmark_v1_1.json \
    configs/hea_validation_v1.json \
    scripts/workstation/run_hea_validation.sh \
    > "$snapshot"
fi

if ! "$micromamba" env list | awk '{print $1}' | grep -qx sitexrd-validate; then
  "$micromamba" create -y -f "$project_root/environment-validation.yml"
fi

"${python_cmd[@]}" -m unittest discover -s tests -p 'test_hea_*.py' -v

# Create only the v1.1 view. The release module rejects v1 or descendants as output.
"${python_cmd[@]}" -m alloy_xrd_bench.hea_release_package \
  --source "$source_dataset" --output "$output_dataset" --config "$release_config" \
  --source-license "$source_license" --report "$run_root/release_package.bootstrap.json"

"${python_cmd[@]}" -m alloy_xrd_bench.occupancy_xrd observations-freeze \
  --dataset "$source_dataset" --output-dataset "$output_dataset" \
  --parent-groups "$output_dataset/manifests/occupancy_pairs.parquet" \
  --run-id "$observation_run" --config "$perturbation_config" \
  --variants 5 --splits train,validation,test \
  > "$run_root/observations_freeze.json"
"${python_cmd[@]}" -m alloy_xrd_bench.occupancy_xrd observations-audit \
  --dataset "$output_dataset" --run-id "$observation_run" \
  > "$run_root/evidence/observation_audit.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_identifiability \
  --dataset "$source_dataset" --observation-dataset "$output_dataset" \
  --observation-run "$observation_run" --output-root "$output_dataset" --quantile 0.95 \
  > "$run_root/identifiability.run.json"
cp "$output_dataset/reports/identifiability_calibration.json" "$run_root/evidence/identifiability.json"

"${python_cmd[@]}" -m alloy_xrd_bench.hea_validation audit \
  --dataset "$source_dataset" --config "$config" \
  > "$run_root/evidence/canonical_audit.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_validation controls \
  --dataset "$source_dataset" --config "$config" --split all --seed 2027 \
  > "$run_root/evidence/controls.json"
set +e
"${python_cmd[@]}" -m alloy_xrd_bench.hea_family_tasks \
  --labels "$source_dataset/manifests/task_samples.parquet" \
  --output "$run_root/evidence/family_support.json"
family_status=$?
set -e
if [[ "$family_status" -ne 0 && "$family_status" -ne 2 ]]; then
  exit "$family_status"
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
for split_column in "${splits[@]}"; do
  if ! run_is_complete composition-ridge 2027 "$split_column"; then
    "${python_cmd[@]}" -m alloy_xrd_bench.hea_validation train \
      --model composition-ridge --dataset "$source_dataset" --config "$config" \
      --output "$run_root/predictions" --split-column "$split_column" --seed 2027 \
      > "$run_root/composition-ridge_${split_column}_seed2027.run.json"
  fi
  ridge_submission="$run_root/predictions/submissions/composition-ridge/2027/${split_column}_test"
  if ! score_is_complete composition-ridge 2027 "$split_column"; then
    "${python_cmd[@]}" -m alloy_xrd_bench.evaluate score \
      --dataset "$source_dataset" --predictions "$ridge_submission" \
      --output "$run_root/scores/composition-ridge/2027/${split_column}_test" --config "$config"
  fi

  for seed in "${seeds[@]}"; do
    if ! run_is_complete peak-xgboost "$seed" "$split_column"; then
      "${python_cmd[@]}" -m alloy_xrd_bench.hea_validation train \
        --model peak-xgboost --dataset "$source_dataset" --config "$config" \
        --output "$run_root/predictions" --split-column "$split_column" --seed "$seed" \
        > "$run_root/peak-xgboost_${split_column}_seed${seed}.run.json"
    fi

    training_pids=()
    if ! run_is_complete resnet1d "$seed" "$split_column"; then
      CUDA_VISIBLE_DEVICES=0 "${python_cmd[@]}" -m alloy_xrd_bench.hea_validation train \
        --model resnet1d --dataset "$source_dataset" --config "$config" \
        --output "$run_root/predictions" --split-column "$split_column" --seed "$seed" \
        > "$run_root/resnet1d_${split_column}_seed${seed}.run.json" \
        2> "$run_root/resnet1d_${split_column}_seed${seed}.log" &
      training_pids+=("$!")
    fi
    if ! run_is_complete transformer "$seed" "$split_column"; then
      CUDA_VISIBLE_DEVICES=1 "${python_cmd[@]}" -m alloy_xrd_bench.hea_validation train \
        --model transformer --dataset "$source_dataset" --config "$config" \
        --output "$run_root/predictions" --split-column "$split_column" --seed "$seed" \
        > "$run_root/transformer_${split_column}_seed${seed}.run.json" \
        2> "$run_root/transformer_${split_column}_seed${seed}.log" &
      training_pids+=("$!")
    fi
    for training_pid in "${training_pids[@]}"; do
      wait "$training_pid"
    done
    for model in "${xrd_models[@]}"; do
      submission="$run_root/predictions/submissions/$model/$seed/${split_column}_test"
      if ! score_is_complete "$model" "$seed" "$split_column"; then
        "${python_cmd[@]}" -m alloy_xrd_bench.evaluate score \
          --dataset "$source_dataset" --predictions "$submission" \
          --output "$run_root/scores/$model/$seed/${split_column}_test" --config "$config"
      fi
    done
  done
done

"${python_cmd[@]}" -m alloy_xrd_bench.hea_prediction_audit \
  --dataset "$source_dataset" --predictions-root "$run_root/predictions" \
  --output "$run_root/evidence/prediction_completeness.json"
"${python_cmd[@]}" -m alloy_xrd_bench.evaluate validate \
  --dataset "$source_dataset" \
  --predictions "$run_root/predictions/submissions/transformer/2027/id_split_test" \
  --config "$config" > "$run_root/evidence/evaluator_smoke.json"

"${python_cmd[@]}" -m alloy_xrd_bench.hea_ood \
  --scores-root "$run_root/scores" --model transformer \
  --seeds "${seeds[@]}" \
  --split-columns composition_ood_split chemistry_ood_split sro_ood_split \
  --output "$run_root/evidence/ood_summary.json"

for model in "${xrd_models[@]}"; do
  for seed in "${seeds[@]}"; do
    checkpoint_stem="${model//-/_}"
    checkpoint="$run_root/predictions/${checkpoint_stem}_seed${seed}_id_split_test.pt"
    if [[ "$model" == "peak-xgboost" ]]; then
      checkpoint="$run_root/predictions/peak_xgboost_seed${seed}_id_split_test.joblib"
    fi
    "${python_cmd[@]}" -m alloy_xrd_bench.hea_perturbation evaluate \
      --dataset "$source_dataset" --observation-dataset "$output_dataset" \
      --observation-run "$observation_run" --checkpoint "$checkpoint" \
      --clean-metrics "$run_root/predictions/${checkpoint_stem}_seed${seed}_id_split_test.metrics.json" \
      --config "$config" --perturbation-config "$perturbation_config" \
      --output "$run_root/perturbation/$model/$seed"
  done
done
"${python_cmd[@]}" -m alloy_xrd_bench.hea_perturbation summarize \
  --input "$run_root/perturbation" --output "$run_root/evidence/perturbation_summary.json" \
  --minimum-seeds 5 --required-models peak-xgboost resnet1d transformer physics-transformer \
  --composition-ridge-metrics "$run_root/predictions/composition_ridge_seed2027_id_split_test.metrics.json"

comparison_reference="$run_root/scores/composition-ridge/2027/id_split_test"
comparison_candidate="$run_root/comparison_inputs/transformer"
mkdir -p "$comparison_candidate"
for seed in "${seeds[@]}"; do
  target="$comparison_candidate/$seed/id_split_test"
  mkdir -p "$target"
  cp "$run_root/scores/transformer/$seed/id_split_test/metrics_by_slice.parquet" "$target/metrics_by_slice.parquet"
done
"${python_cmd[@]}" -m alloy_xrd_bench.evaluate compare \
  --reference "$comparison_reference" --candidate "$comparison_candidate" \
  --bootstrap-replicates 2000 --bootstrap-seed 2027 \
  --output "$run_root/evidence/model_comparison.json"
cp "$run_root/evidence/model_comparison.json" "$run_root/evidence/bootstrap_summary.json"

"${python_cmd[@]}" -m alloy_xrd_bench.hea_release_package \
  --source "$source_dataset" --output "$output_dataset" --config "$release_config" \
  --source-license "$source_license" --report "$run_root/evidence/release_package.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_validation report \
  --run-root "$run_root" --config "$config" > "$run_root/report.run.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_validation_plot \
  --summary "$run_root/validation_summary.json" --output-dir "$run_root/figures" \
  > "$run_root/figures.run.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_paper_artifacts \
  --summary "$run_root/validation_summary.json" \
  --calibrated-pairs "$output_dataset/manifests/occupancy_pairs_calibrated.parquet" \
  --perturbation-root "$run_root/perturbation" --output "$run_root/paper_artifacts" \
  > "$run_root/paper_artifacts.run.json"
"${python_cmd[@]}" -m alloy_xrd_bench.hea_benchmark_readiness \
  --evidence-root "$run_root/evidence" --output "$run_root/hea_benchmark_readiness.json"
