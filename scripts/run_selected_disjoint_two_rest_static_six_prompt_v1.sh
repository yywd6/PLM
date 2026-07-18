#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-}"
STATIC_PROMPT_CONFIG="${EXPERIMENT_CONFIG:-${STATIC_PROMPT_CONFIG:-configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml}}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-outputs/selected_two_rest_visual_baseline_v7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selected_two_rest_ablation_static_six_prompt_v7_uniform_scoring}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"
RUN_ONLY="${RUN_ONLY:-}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SEED_ARGS=()
RESUME_ARGS=()
if [[ -n "${EXPERIMENT_SEED}" ]]; then
  if ! [[ "${EXPERIMENT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: EXPERIMENT_SEED must be a non-negative integer" >&2
    exit 2
  fi
  SEED_ARGS+=(--seed "${EXPERIMENT_SEED}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

training_artifacts_valid() {
  "${PYTHON_BIN}" -m utils.artifact_integrity training \
    --checkpoint "$1" --completion "$2" --quarantine
}

evaluation_artifacts_valid() {
  "${PYTHON_BIN}" -m utils.artifact_integrity evaluation \
    --mean-metrics "$1" \
    --per-category "$2" \
    --metrics "$3" \
    --sample-scores "$4" \
    --completion "$5" \
    --quarantine
}

REAL3D_RUNS=(
  "car chicken"
  "candybar fish"
  "duck starfish"
)
ANOMALY_SHAPENET_RUNS=(
  "bowl5 cup0"
  "bottle0 tap0"
  "helmet0 jar0"
)

validate_disjoint_runs() {
  local dataset_name="$1"
  shift
  local pair
  local first_category
  local second_category
  local category
  declare -A seen=()
  for pair in "$@"; do
    read -r first_category second_category <<< "${pair}"
    for category in "${first_category}" "${second_category}"; do
      if [[ -n "${seen[${category}]:-}" ]]; then
        echo "Duplicate selected category for ${dataset_name}: ${category}" >&2
        return 1
      fi
      seen["${category}"]=1
    done
  done
  if [[ "${#seen[@]}" -ne 6 ]]; then
    echo "${dataset_name}: expected 6 distinct categories, found ${#seen[@]}" >&2
    return 1
  fi
}

run_pair() {
  local dataset_name="$1"
  local data_root="$2"
  local first_category="$3"
  local second_category="$4"
  local pair_name="${first_category}+${second_category}"
  if [[ -n "${RUN_ONLY}" && "${RUN_ONLY}" != "${dataset_name}:${pair_name}" ]]; then
    return 0
  fi
  local baseline_dataset_root="${BASELINE_OUTPUT_ROOT}/${dataset_name}"
  local output_dataset_root="${OUTPUT_ROOT}/${dataset_name}"
  local baseline_run_dir="${baseline_dataset_root}/${pair_name}"
  local run_dir="${output_dataset_root}/${pair_name}"
  local baseline_checkpoint="${STAGE1_CHECKPOINT:-${baseline_run_dir}/best.pth}"
  local baseline_completion="${baseline_checkpoint%/*}/training_complete.yaml"
  local checkpoint="${run_dir}/best.pth"

  mkdir -p "${baseline_dataset_root}" "${output_dataset_root}"

  echo "============================================================"
  echo "Dataset=${dataset_name} static-six-prompt=${pair_name}"
  echo "Stage-2 experiment config=${STATIC_PROMPT_CONFIG}"
  echo "Prompt/residual count is read from and validated against the config"
  echo "Experiment seed=${EXPERIMENT_SEED:-config-default}"
  echo "============================================================"

  if training_artifacts_valid \
      "${baseline_checkpoint}" "${baseline_completion}"; then
    echo "Stage 1 already complete; skip: ${baseline_checkpoint}"
  elif [[ "${SKIP_STAGE1}" == "1" ]]; then
    echo "ERROR: SKIP_STAGE1=1 but Stage 1 checkpoint is invalid: ${baseline_checkpoint}" >&2
    return 1
  else
    "${PYTHON_BIN}" train.py --config "${BASELINE_CONFIG}" --protocol one_rest --dataset_name "${dataset_name}" --data_root "${data_root}" --train_categories "${first_category}" "${second_category}" --output_root "${baseline_dataset_root}" --batch_size "${BATCH_SIZE}" --use_static_prompt false --freeze_visual_adapter false --baseline_checkpoint "" --global_alpha 0.3
    baseline_checkpoint="${baseline_run_dir}/best.pth"
  fi

  if [[ "${RESUME}" == "1" ]] && training_artifacts_valid \
      "${checkpoint}" "${run_dir}/training_complete.yaml"; then
    echo "Static Prompt stage already complete; skip: ${checkpoint}"
  else
    "${PYTHON_BIN}" train.py --config "${STATIC_PROMPT_CONFIG}" --protocol one_rest --dataset_name "${dataset_name}" --data_root "${data_root}" --train_categories "${first_category}" "${second_category}" --baseline_checkpoint "${baseline_checkpoint}" --output_root "${output_dataset_root}" --batch_size "${BATCH_SIZE}" "${SEED_ARGS[@]}" "${RESUME_ARGS[@]}"
  fi

  if [[ "${RESUME}" == "1" ]] \
      && evaluation_artifacts_valid \
        "${run_dir}/mean_metrics.json" \
        "${run_dir}/per_category_metrics.json" \
        "${run_dir}/metrics.json" \
        "${run_dir}/sample_scores.npz" \
        "${run_dir}/training_complete.yaml" \
      && [[ "${run_dir}/mean_metrics.json" -nt "${checkpoint}" ]]; then
    echo "Test already complete; skip: ${run_dir}/mean_metrics.json"
  else
    "${PYTHON_BIN}" test.py --config "${STATIC_PROMPT_CONFIG}" --protocol one_rest --dataset_name "${dataset_name}" --data_root "${data_root}" --train_categories "${first_category}" "${second_category}" --output_root "${output_dataset_root}" --batch_size "${BATCH_SIZE}" "${SEED_ARGS[@]}"
  fi
}

run_dataset() {
  local dataset_name="$1"
  local data_root="$2"
  shift 2
  local pair
  local first_category
  local second_category
  for pair in "$@"; do
    read -r first_category second_category <<< "${pair}"
    run_pair "${dataset_name}" "${data_root}" "${first_category}" "${second_category}"
  done
}

validate_disjoint_runs Real3D "${REAL3D_RUNS[@]}"
validate_disjoint_runs AnomalyShapeNet "${ANOMALY_SHAPENET_RUNS[@]}"

run_dataset Real3D /home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048 "${REAL3D_RUNS[@]}"
run_dataset AnomalyShapeNet /home/objectdec/data/Point3D/Anomaly_shapeNet_btp_fps2048 "${ANOMALY_SHAPENET_RUNS[@]}"

if [[ -z "${RUN_ONLY}" ]]; then
  "${PYTHON_BIN}" scripts/summarize_selected_two_rest.py \
    --output-root "${OUTPUT_ROOT}" --expected-runs 3
else
  "${PYTHON_BIN}" scripts/summarize_selected_two_rest.py \
    --output-root "${OUTPUT_ROOT}" --expected-runs 0
fi

echo "All six static-six-Prompt runs completed."
echo "Checkpoints and metrics: ${OUTPUT_ROOT}"
echo "Dataset-level summary: ${OUTPUT_ROOT}/selected_two_rest_summary.json"
