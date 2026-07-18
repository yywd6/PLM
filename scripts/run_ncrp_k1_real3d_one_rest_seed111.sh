#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/objectdec/PycharmProjects/Zero-shot}"
PYTHON_BIN="${PYTHON_BIN:-/home/objectdec/anaconda3/envs/B/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"
SEED="${SEED:-111}"
TRAIN_STAGE1_IF_NEEDED="${TRAIN_STAGE1_IF_NEEDED:-1}"
RUN_SEAHORSE="${RUN_SEAHORSE:-1}"
RUN_SHELL="${RUN_SHELL:-1}"
RUN_STARFISH="${RUN_STARFISH:-1}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/one_rest_visual_baseline_v7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ncrp_k1_real3d_one_rest_seed111}"

show_help() {
  cat <<'EOF'
Serially train and test NCRP-K1 for three Real3D one-rest runs:
  source=seahorse -> evaluate the other 11 categories
  source=shell    -> evaluate the other 11 categories
  source=starfish -> evaluate the other 11 categories

The source category is explicitly excluded from test loading, per-category
metrics, and mean metrics. Existing valid Stage-1 adapters are reused. The
current default Stage-1 files may be empty; with TRAIN_STAGE1_IF_NEEDED=1 the
script quarantines invalid artifacts and trains only the missing adapter.

Full run:
  cd /home/objectdec/PycharmProjects/Zero-shot
  PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
  BATCH_SIZE=4 RESUME=1 SEED=111 \
  bash scripts/run_ncrp_k1_real3d_one_rest_seed111.sh

Only shell:
  RUN_SEAHORSE=0 RUN_SHELL=1 RUN_STARFISH=0 \
  bash scripts/run_ncrp_k1_real3d_one_rest_seed111.sh

Require pre-existing valid Stage-1 adapters:
  TRAIN_STAGE1_IF_NEEDED=0 \
  STAGE1_ROOT=outputs/my_valid_one_rest_stage1 \
  bash scripts/run_ncrp_k1_real3d_one_rest_seed111.sh

Environment variables:
  PROJECT_ROOT PYTHON_BIN DATA_ROOT BATCH_SIZE RESUME SEED
  TRAIN_STAGE1_IF_NEEDED RUN_SEAHORSE RUN_SHELL RUN_STARFISH
  STAGE1_ROOT OUTPUT_ROOT
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unsupported argument: $1" >&2
  show_help >&2
  exit 2
fi

validate_flag() {
  if [[ "$2" != "0" && "$2" != "1" ]]; then
    echo "ERROR: $1 must be 0 or 1, got $2" >&2
    exit 2
  fi
}
for item in \
    "RESUME:${RESUME}" \
    "TRAIN_STAGE1_IF_NEEDED:${TRAIN_STAGE1_IF_NEEDED}" \
    "RUN_SEAHORSE:${RUN_SEAHORSE}" \
    "RUN_SHELL:${RUN_SHELL}" \
    "RUN_STARFISH:${RUN_STARFISH}"; do
  validate_flag "${item%%:*}" "${item#*:}"
done
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SEED must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

resolve_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${PROJECT_ROOT}" "$1"
  fi
}

if [[ "${PROJECT_ROOT}" != /* ]]; then
  PROJECT_ROOT="$(pwd)/${PROJECT_ROOT}"
fi
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "ERROR: project root missing: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_BIN="$(resolve_path "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

DATA_ROOT="$(resolve_path "${DATA_ROOT}")"
STAGE1_ROOT="$(resolve_path "${STAGE1_ROOT}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
BASELINE_CONFIG="${PROJECT_ROOT}/configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml"
K1_CONFIG="${PROJECT_ROOT}/configs/ncrp_k1.yaml"
for path in "${DATA_ROOT}" "${BASELINE_CONFIG}" "${K1_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
TOTAL_LOG="${OUTPUT_ROOT}/sequence.log"
RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
if [[ "${RESUME}" == "0" && -e "${TOTAL_LOG}" ]]; then
  mv "${TOTAL_LOG}" "${TOTAL_LOG}.previous.${RUN_TIMESTAMP}"
fi
exec > >(tee -a "${TOTAL_LOG}") 2>&1

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
trap 'status=$?; echo "[$(timestamp)] NCRP-K1 one-rest sequence finished status=${status}"' EXIT

archive_dir() {
  local directory="$1"
  if [[ -e "${directory}" ]]; then
    mv "${directory}" "${directory}.previous.${RUN_TIMESTAMP}"
    echo "[$(timestamp)] archived ${directory}"
  fi
}

training_valid() {
  "${PYTHON_BIN}" -m utils.artifact_integrity training \
    --checkpoint "$1" --completion "$2" --quarantine
}

evaluation_valid() {
  local directory="$1"
  "${PYTHON_BIN}" -m utils.artifact_integrity evaluation \
    --mean-metrics "${directory}/mean_metrics.json" \
    --per-category "${directory}/per_category_metrics.json" \
    --metrics "${directory}/metrics.json" \
    --sample-scores "${directory}/sample_scores.npz" \
    --diagnostics "${directory}/ncrp_diagnostics.json" \
    --completion "${directory}/training_complete.yaml" --quarantine
}

validate_excluded_source_metrics() {
  local directory="$1"
  local source_category="$2"
  "${PYTHON_BIN}" -B -c \
    'import json,sys
from pathlib import Path
from data.anomaly_datasets import PointCloudDataset
d=Path(sys.argv[1]); source=sys.argv[2]
mean=json.load(open(d/"mean_metrics.json",encoding="utf-8"))
per=json.load(open(d/"per_category_metrics.json",encoding="utf-8"))
expected=[c for c in PointCloudDataset.PRESETS["Real3D"] if c != source]
assert mean.get("train_category")==source
assert mean.get("test_categories")==expected
assert per.get("test_categories")==expected
assert list(per.get("metrics",{}))==expected
assert source not in mean["test_categories"] and source not in per["metrics"]
assert len(expected)==11
print(f"one-rest exclusion OK: source={source}, evaluated_categories={expected}")' \
    "${directory}" "${source_category}"
}

validate_k1_checkpoint() {
  local directory="$1"
  local source_category="$2"
  "${PYTHON_BIN}" -B -c \
    'import json,sys,torch
from pathlib import Path
d=Path(sys.argv[1]); source=sys.argv[2]; seed=int(sys.argv[3])
c=torch.load(d/"best.pth",map_location="cpu",weights_only=False)
g=json.load(open(d/"ncrp_diagnostics.json",encoding="utf-8"))
assert c.get("train_categories")==[source]
assert int(c.get("seed",-1))==seed
assert c.get("residual_prompt_enabled") is True
assert int(c.get("residual_num_bases",-1))==1
assert c.get("static_prompt_version") in {"ncrp_k1", "ncrp_a1_k1_single_residual"}
assert int(g.get("trainable_parameter_count",-1))==6400
assert g.get("basis_usage")==[1.0]
assert g.get("assignment_diagnostics_status")=="not_applicable_single_basis"
print(f"NCRP-K1 checkpoint OK: source={source}, seed={seed}")' \
    "${directory}" "${source_category}" "${SEED}"
}

ensure_stage1() {
  local source_category="$1"
  local stage1_dataset_root="${STAGE1_ROOT}/Real3D"
  local stage1_dir="${stage1_dataset_root}/${source_category}"
  local checkpoint="${stage1_dir}/best.pth"
  local completion="${stage1_dir}/training_complete.yaml"

  if training_valid "${checkpoint}" "${completion}"; then
    echo "[$(timestamp)] reuse valid Stage 1: ${checkpoint}"
    return 0
  fi
  if [[ "${TRAIN_STAGE1_IF_NEEDED}" != "1" ]]; then
    echo "ERROR: invalid Stage 1 for ${source_category}: ${checkpoint}" >&2
    echo "Set TRAIN_STAGE1_IF_NEEDED=1 or provide a valid STAGE1_ROOT." >&2
    return 1
  fi

  echo "[$(timestamp)] train missing Stage 1 source=${source_category}"
  mkdir -p "${stage1_dataset_root}"
  "${PYTHON_BIN}" train.py \
    --config "${BASELINE_CONFIG}" \
    --protocol one_rest \
    --dataset_name Real3D \
    --data_root "${DATA_ROOT}" \
    --train_categories "${source_category}" \
    --exclude_train_category_from_test true \
    --zero_shot_target true \
    --use_target_anomaly_for_training false \
    --output_root "${stage1_dataset_root}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --use_static_prompt false \
    --freeze_visual_adapter false \
    --baseline_checkpoint "" \
    --global_alpha 0.3
  training_valid "${checkpoint}" "${completion}"
}

run_source() {
  local source_category="$1"
  local output_dataset_root="${OUTPUT_ROOT}/Real3D"
  local run_dir="${output_dataset_root}/${source_category}"
  local stage1_checkpoint
  local resume_checkpoint=""
  local start end status

  echo "[$(timestamp)] START source=${source_category} seed=${SEED} method=NCRP-K1 one-rest"
  echo "config=${K1_CONFIG} output_dir=${run_dir}"
  echo "evaluation=all Real3D categories except ${source_category}"
  stage1_checkpoint="${STAGE1_ROOT}/Real3D/${source_category}/best.pth"
  ensure_stage1 "${source_category}"
  echo "stage1_checkpoint=${stage1_checkpoint}"

  if [[ "${RESUME}" == "1" ]] \
      && training_valid "${run_dir}/best.pth" "${run_dir}/training_complete.yaml"; then
    echo "[$(timestamp)] Stage 2 already complete; skip training: ${run_dir}/best.pth"
  else
    if [[ "${RESUME}" == "0" ]]; then
      archive_dir "${run_dir}"
    elif [[ -s "${run_dir}/last.pth" ]]; then
      if "${PYTHON_BIN}" -B -c \
        'import sys,torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
assert p.get("train_categories")==[sys.argv[2]]
assert int(p.get("seed",-1))==int(sys.argv[3])
assert p.get("residual_prompt_enabled") is True
assert int(p.get("residual_num_bases",-1))==1
assert "optimizer" in p and "scheduler" in p' \
        "${run_dir}/last.pth" "${source_category}" "${SEED}"; then
        resume_checkpoint="${run_dir}/last.pth"
        echo "[$(timestamp)] resume Stage 2 from ${resume_checkpoint}"
      else
        "${PYTHON_BIN}" -B -c \
          'from utils.artifact_integrity import quarantine_corrupt_file; import sys; print(quarantine_corrupt_file(sys.argv[1]))' \
          "${run_dir}/last.pth"
      fi
    fi

    local resume_args=()
    if [[ -n "${resume_checkpoint}" ]]; then
      resume_args+=(--resume_checkpoint "${resume_checkpoint}")
    fi
    "${PYTHON_BIN}" train.py \
      --config "${K1_CONFIG}" \
      --protocol one_rest \
      --dataset_name Real3D \
      --data_root "${DATA_ROOT}" \
      --train_categories "${source_category}" \
      --exclude_train_category_from_test true \
      --zero_shot_target true \
      --use_target_anomaly_for_training false \
      --baseline_checkpoint "${stage1_checkpoint}" \
      --output_root "${output_dataset_root}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      "${resume_args[@]}"
    training_valid "${run_dir}/best.pth" "${run_dir}/training_complete.yaml"
  fi

  start="$(date '+%s')"
  status=0
  if [[ "${RESUME}" == "1" ]] \
      && evaluation_valid "${run_dir}" \
      && [[ "${run_dir}/mean_metrics.json" -nt "${run_dir}/best.pth" ]]; then
    echo "[$(timestamp)] evaluation already complete; validating excluded source"
  elif "${PYTHON_BIN}" test.py \
      --config "${K1_CONFIG}" \
      --protocol one_rest \
      --dataset_name Real3D \
      --data_root "${DATA_ROOT}" \
      --train_categories "${source_category}" \
      --exclude_train_category_from_test true \
      --zero_shot_target true \
      --use_target_anomaly_for_training false \
      --output_root "${output_dataset_root}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}"; then
    status=0
  else
    status=$?
  fi
  end="$(date '+%s')"
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(timestamp)] END source=${source_category} status=${status} duration_seconds=$((end-start))"
    return "${status}"
  fi

  training_valid "${run_dir}/best.pth" "${run_dir}/training_complete.yaml"
  evaluation_valid "${run_dir}"
  validate_k1_checkpoint "${run_dir}" "${source_category}"
  validate_excluded_source_metrics "${run_dir}" "${source_category}"
  echo "[$(timestamp)] END source=${source_category} status=0 evaluation_duration_seconds=$((end-start))"
}

echo "[$(timestamp)] NCRP-K1 Real3D one-rest sequence start"
echo "sources=seahorse,shell,starfish seed=${SEED} batch_size=${BATCH_SIZE}"
echo "test rule=exclude the current source category from all reported metrics"
echo "stage1_root=${STAGE1_ROOT} output_root=${OUTPUT_ROOT}"

[[ "${RUN_SEAHORSE}" == "1" ]] && run_source seahorse
[[ "${RUN_SHELL}" == "1" ]] && run_source shell
[[ "${RUN_STARFISH}" == "1" ]] && run_source starfish

echo "[$(timestamp)] all requested NCRP-K1 one-rest runs completed"
echo "results=${OUTPUT_ROOT}/Real3D/{seahorse,shell,starfish}"
