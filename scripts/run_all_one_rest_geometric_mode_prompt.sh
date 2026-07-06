#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/one_rest_geometric_mode_prompt.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/one_rest_geometric_mode_prompt_all}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

run_dataset() {
  local dataset_name="$1"
  local data_root="$2"
  local output_root="${OUTPUT_BASE}/${dataset_name}"
  mkdir -p "${output_root}"

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

    echo "============================================================"
    echo "Geometric mode prompting: dataset=${dataset_name} source=${train_category}"
    echo "============================================================"

    if [[ "${RESUME}" == "1" && -f "${checkpoint}" && -f "${completion}" ]] && \
       rg -q "geometric_mode_version: se3_mode_prompt_v6_patch_routing" "${completion}"; then
      echo "Completed training found; skip: ${checkpoint}"
    else
      "${PYTHON_BIN}" train.py \
        --config "${CONFIG}" \
        --protocol one_rest \
        --dataset_name "${dataset_name}" \
        --data_root "${data_root}" \
        --train_category "${train_category}" \
        --output_root "${output_root}" \
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

echo "All geometric mode prompting runs completed: ${OUTPUT_BASE}"
