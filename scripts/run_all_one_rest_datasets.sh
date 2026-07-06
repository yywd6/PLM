#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/one_rest_geometric_cap_dap.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/one_rest_geometric_cap_dap_all}"
RESUME="${RESUME:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

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
    local completion="${run_dir}/training_complete.yaml"
    local checkpoint="${run_dir}/best.pth"
    local metrics="${run_dir}/per_category_metrics.json"

    echo "============================================================"
    echo "Dataset=${dataset_name} source=${train_category}"
    echo "============================================================"

    local resume_ready=0
    if [[ "${RESUME}" == "1" && -f "${completion}" && -f "${checkpoint}" ]] && \
       rg -q "prompt_suffix_mode: prompt_template_v1" "${completion}"; then
      resume_ready=1
    fi

    if [[ "${resume_ready}" == "1" ]]; then
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

    if [[ "${RESUME}" == "1" && -f "${metrics}" && "${metrics}" -nt "${checkpoint}" ]]; then
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
  "Real3D" \
  "/home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048"

run_dataset \
  "AnomalyShapeNet" \
  "/home/objectdec/data/Point3D/Anomaly_shapeNet_btp_fps2048"

echo "All one-rest runs completed: ${OUTPUT_BASE}"
