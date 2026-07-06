#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/one_rest_geometric_mode_prompt_v7.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/one_rest_visual_baseline_v7.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/one_rest_geometric_mode_prompt_v7_all}"
BASELINE_OUTPUT_BASE="${BASELINE_OUTPUT_BASE:-outputs/one_rest_visual_baseline_v7}"
GEOMETRIC_MODE_VERSION="${GEOMETRIC_MODE_VERSION:-se3_mode_prompt_v7_gate_sinkhorn}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

run_dataset() {
  local dataset_name="$1"
  local data_root="$2"
  local output_root="${OUTPUT_BASE}/${dataset_name}"
  local baseline_output_root="${BASELINE_OUTPUT_BASE}/${dataset_name}"
  mkdir -p "${output_root}" "${baseline_output_root}"

  mapfile -t categories < <(
    "${PYTHON_BIN}" -c \
      "from data.anomaly_datasets import PointCloudDataset; print('\\n'.join(PointCloudDataset.PRESETS['${dataset_name}']))"
  )

  echo "Dataset ${dataset_name}: ${#categories[@]} source categories"
  for train_category in "${categories[@]}"; do
    local run_dir="${output_root}/${train_category}"
    local checkpoint="${run_dir}/best.pth"
    local completion="${run_dir}/training_complete.yaml"
    local metrics="${run_dir}/per_category_metrics.json"
    local mode_statistics="${run_dir}/mode_statistics.json"
    local baseline_run_dir="${baseline_output_root}/${train_category}"
    local baseline_checkpoint="${baseline_run_dir}/best.pth"
    local baseline_completion="${baseline_run_dir}/training_complete.yaml"

    echo "------------------------------------------------------------"
    echo "Stage 1 visual baseline: dataset=${dataset_name} source=${train_category}"
    echo "------------------------------------------------------------"
    if [[ "${RESUME}" == "1" && -f "${baseline_checkpoint}" && -f "${baseline_completion}" ]]; then
      echo "Completed visual baseline found; skip: ${baseline_checkpoint}"
    else
      "${PYTHON_BIN}" train.py \
        --config "${BASELINE_CONFIG}" \
        --protocol one_rest \
        --dataset_name "${dataset_name}" \
        --data_root "${data_root}" \
        --train_category "${train_category}" \
        --output_root "${baseline_output_root}" \
        --batch_size "${BATCH_SIZE}"
    fi

    echo "============================================================"
    echo "Geometric mode prompting: dataset=${dataset_name} source=${train_category}"
    echo "============================================================"

    if [[ "${RESUME}" == "1" && -f "${checkpoint}" && -f "${completion}" ]] && \
       rg -q "geometric_mode_version: ${GEOMETRIC_MODE_VERSION}" "${completion}"; then
      echo "Completed training found; skip: ${checkpoint}"
    else
      "${PYTHON_BIN}" train.py \
        --config "${CONFIG}" \
        --protocol one_rest \
        --dataset_name "${dataset_name}" \
        --data_root "${data_root}" \
        --train_category "${train_category}" \
        --output_root "${output_root}" \
        --baseline_checkpoint "${baseline_checkpoint}" \
        --batch_size "${BATCH_SIZE}"
    fi

    if [[ "${RESUME}" == "1" && -f "${metrics}" && -f "${mode_statistics}" && \
          "${metrics}" -nt "${checkpoint}" ]]; then
      echo "Current test metrics found; skip: ${metrics}"
    else
      "${PYTHON_BIN}" test.py \
        --config "${CONFIG}" \
        --protocol one_rest \
        --dataset_name "${dataset_name}" \
        --data_root "${data_root}" \
        --train_category "${train_category}" \
        --output_root "${output_root}" \
        --baseline_checkpoint "${baseline_checkpoint}" \
        --batch_size "${BATCH_SIZE}"
    fi
  done

  "${PYTHON_BIN}" summarize_one_rest.py \
    --dataset_name "${dataset_name}" \
    --output_root "${output_root}"
}

run_dataset \
  Real3D \
  /home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048

run_dataset \
  AnomalyShapeNet \
  /home/objectdec/data/Point3D/Anomaly_shapeNet_btp_fps2048

echo "All v7 two-stage runs completed: ${OUTPUT_BASE}"
echo "Visual baselines: ${BASELINE_OUTPUT_BASE}"
