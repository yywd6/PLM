#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/objectdec/anaconda3/envs/B/bin/python}"
CONFIG="${CONFIG:-configs/one_rest_geometric_cap_dap.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/top3_accuracy_category_semantic}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-}"
if [[ -z "${PROMPT_TEMPLATE}" ]]; then
  PROMPT_TEMPLATE="a point cloud patch of {article} {category}"
fi

run_dataset() {
  local dataset_name="$1"
  local data_root="$2"
  shift 2
  local output_root="${OUTPUT_BASE}/${dataset_name}"

  mkdir -p "${output_root}"
  for train_category in "$@"; do
    local run_dir="${output_root}/${train_category}"
    local checkpoint="${run_dir}/best.pth"
    local completion="${run_dir}/training_complete.yaml"
    local metrics="${run_dir}/per_category_metrics.json"

    echo "============================================================"
    echo "Dataset=${dataset_name} source=${train_category}"
    echo "Prompt=${PROMPT_TEMPLATE}"
    echo "============================================================"

    if [[ "${RESUME}" == "1" && -f "${checkpoint}" && -f "${completion}" ]] && \
       rg -Fq "geometric_prompt_suffix: ${PROMPT_TEMPLATE}" "${completion}" && \
       rg -Fq "use_route_aux: true" "${completion}"; then
      echo "Completed training found; skip: ${checkpoint}"
    else
      "${PYTHON_BIN}" train.py \
        --config "${CONFIG}" \
        --protocol one_rest \
        --dataset_name "${dataset_name}" \
        --data_root "${data_root}" \
        --train_category "${train_category}" \
        --output_root "${output_root}" \
        --batch_size "${BATCH_SIZE}" \
        --geometric_prompt_suffix "${PROMPT_TEMPLATE}"
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
        --batch_size "${BATCH_SIZE}" \
        --geometric_prompt_suffix "${PROMPT_TEMPLATE}"
    fi
  done

  "${PYTHON_BIN}" summarize_one_rest.py \
    --dataset_name "${dataset_name}" \
    --output_root "${output_root}"
}

# Top-3 source categories by mean Point AUROC in completed fixed-prompt runs.
run_dataset \
  Real3D \
  /home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048 \
  car shell fish

run_dataset \
  AnomalyShapeNet \
  /home/objectdec/data/Point3D/Anomaly_shapeNet_btp_fps2048 \
  tap0 vase3 headset0

echo "Top-3 category-semantic runs completed: ${OUTPUT_BASE}"
