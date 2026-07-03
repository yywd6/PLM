#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/one_rest.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/one_rest}"

mkdir -p "${OUTPUT_ROOT}"
mapfile -t CATEGORIES < <(
  "${PYTHON_BIN}" -c 'from data.anomaly_datasets import PointCloudDataset; print("\n".join(PointCloudDataset.PRESETS["Real3D"]))'
)

for TRAIN_CATEGORY in "${CATEGORIES[@]}"; do
  echo "Train source category: ${TRAIN_CATEGORY}"
  "${PYTHON_BIN}" train.py \
    --config "${CONFIG}" \
    --protocol one_rest \
    --train_category "${TRAIN_CATEGORY}" \
    --output_root "${OUTPUT_ROOT}"

  echo "Test target categories: all Real3D categories except ${TRAIN_CATEGORY}"
  "${PYTHON_BIN}" test_standard_aupro.py \
    --config "${CONFIG}" \
    --protocol one_rest \
    --train_category "${TRAIN_CATEGORY}" \
    --output_root "${OUTPUT_ROOT}"
done

"${PYTHON_BIN}" summarize_one_rest.py \
  --dataset_name Real3D \
  --output_root "${OUTPUT_ROOT}"
