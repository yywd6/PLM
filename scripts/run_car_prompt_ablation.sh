#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/one_rest_geometric_cap_dap.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/car_prompt_ablation}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

variants=(fixed_generic category_only category_semantic)
templates=(
  "point cloud patch"
  "{category}"
  "a point cloud patch of {article} {category}"
)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  template="${templates[$index]}"
  output_root="${OUTPUT_BASE}/${variant}"
  run_dir="${output_root}/car"
  checkpoint="${run_dir}/best.pth"
  completion="${run_dir}/training_complete.yaml"
  metrics="${run_dir}/mean_metrics.json"

  echo "============================================================"
  echo "Prompt ablation: ${variant}"
  echo "Template: ${template}"
  echo "============================================================"

  if [[ "${RESUME}" == "1" && -f "${checkpoint}" && -f "${completion}" ]] && \
     rg -q "prompt_suffix_mode: prompt_template_v1" "${completion}"; then
    echo "Completed training found; skip: ${checkpoint}"
  else
    "${PYTHON_BIN}" train.py \
      --config "${CONFIG}" \
      --protocol one_rest \
      --dataset_name Real3D \
      --data_root "${DATA_ROOT}" \
      --train_category car \
      --output_root "${output_root}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --geometric_prompt_suffix "${template}" \
      --lambda_geometry_invariance 0
  fi

  if [[ "${RESUME}" == "1" && -f "${metrics}" && "${metrics}" -nt "${checkpoint}" ]]; then
    echo "Current test metrics found; skip: ${metrics}"
  else
    "${PYTHON_BIN}" test.py \
      --config "${CONFIG}" \
      --protocol one_rest \
      --dataset_name Real3D \
      --data_root "${DATA_ROOT}" \
      --train_category car \
      --output_root "${output_root}" \
      --batch_size "${BATCH_SIZE}" \
      --geometric_prompt_suffix "${template}"
  fi
done

"${PYTHON_BIN}" scripts/summarize_car_prompt_ablation.py \
  --output_base "${OUTPUT_BASE}"
