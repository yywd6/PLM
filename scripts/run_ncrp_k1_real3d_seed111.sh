#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/objectdec/PycharmProjects/Zero-shot}"
PYTHON_BIN="${PYTHON_BIN:-/home/objectdec/anaconda3/envs/B/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"
SEED="${SEED:-111}"
RUN_CAR_CHICKEN="${RUN_CAR_CHICKEN:-1}"
RUN_CANDYBAR_FISH="${RUN_CANDYBAR_FISH:-1}"
RUN_DUCK_STARFISH="${RUN_DUCK_STARFISH:-1}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/selected_two_rest_visual_baseline_v7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ncrp_k1_real3d_seed111}"

show_help() {
  cat <<'EOF'
Run the NCRP-K1 main method on three Real3D two-rest source pairs,
serially with seed 111. Stage 1 adapters are read-only.

Full run:
  cd /home/objectdec/PycharmProjects/Zero-shot
  PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
  BATCH_SIZE=4 RESUME=1 SEED=111 \
  bash scripts/run_ncrp_k1_real3d_seed111.sh

Only candybar+fish:
  RUN_CAR_CHICKEN=0 RUN_CANDYBAR_FISH=1 RUN_DUCK_STARFISH=0 \
  RESUME=1 bash scripts/run_ncrp_k1_real3d_seed111.sh

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
    "RUN_CAR_CHICKEN:${RUN_CAR_CHICKEN}" \
    "RUN_CANDYBAR_FISH:${RUN_CANDYBAR_FISH}" \
    "RUN_DUCK_STARFISH:${RUN_DUCK_STARFISH}"; do
  validate_flag "${item%%:*}" "${item#*:}"
done
if [[ "${SEED}" != "111" ]]; then
  echo "ERROR: this controlled experiment fixes SEED=111, got ${SEED}" >&2
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

STAGE1_ROOT="$(resolve_path "${STAGE1_ROOT}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
CONFIG="${PROJECT_ROOT}/configs/ncrp_k1.yaml"
RUNNER="${PROJECT_ROOT}/scripts/run_selected_disjoint_two_rest_static_six_prompt_v1.sh"
for path in "${CONFIG}" "${RUNNER}"; do
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing or empty required file: ${path}" >&2
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
trap 'status=$?; echo "[$(timestamp)] NCRP-K1 sequence finished status=${status}"' EXIT

archive_dir() {
  local directory="$1"
  if [[ -e "${directory}" ]]; then
    mv "${directory}" "${directory}.previous.${RUN_TIMESTAMP}"
    echo "[$(timestamp)] archived ${directory}"
  fi
}

require_nonempty() {
  if [[ ! -s "$1" ]]; then
    echo "ERROR: missing or empty artifact: $1" >&2
    return 1
  fi
}

validate_stage1() {
  local checkpoint="$1"
  local completion="${checkpoint%/*}/training_complete.yaml"
  "${PYTHON_BIN}" -m utils.artifact_integrity training \
    --checkpoint "${checkpoint}" --completion "${completion}" --quarantine
}

validate_k1() {
  local directory="$1"
  local pair="$2"
  local required
  for required in \
      best.pth last.pth training_complete.yaml config.yaml train.log test.log \
      mean_metrics.json per_category_metrics.json metrics.json \
      sample_scores.npz ncrp_diagnostics.json; do
    require_nonempty "${directory}/${required}" || return 1
  done
  "${PYTHON_BIN}" -m utils.artifact_integrity training \
    --checkpoint "${directory}/best.pth" \
    --completion "${directory}/training_complete.yaml" --quarantine
  "${PYTHON_BIN}" -m utils.artifact_integrity training \
    --checkpoint "${directory}/last.pth" \
    --completion "${directory}/training_complete.yaml" --quarantine
  "${PYTHON_BIN}" -m utils.artifact_integrity evaluation \
    --mean-metrics "${directory}/mean_metrics.json" \
    --per-category "${directory}/per_category_metrics.json" \
    --metrics "${directory}/metrics.json" \
    --sample-scores "${directory}/sample_scores.npz" \
    --diagnostics "${directory}/ncrp_diagnostics.json" \
    --completion "${directory}/training_complete.yaml" --quarantine
  "${PYTHON_BIN}" -B -c \
    'import json,sys,torch,yaml,numpy as np
from pathlib import Path
d=Path(sys.argv[1]); pair=sys.argv[2]; seed=int(sys.argv[3])
c=torch.load(d/"best.pth",map_location="cpu",weights_only=False)
m=json.load(open(d/"mean_metrics.json",encoding="utf-8"))
g=json.load(open(d/"ncrp_diagnostics.json",encoding="utf-8"))
t=yaml.safe_load(open(d/"training_complete.yaml",encoding="utf-8"))
assert int(c.get("seed",-1))==seed==int(m.get("seed",-1))==int(g.get("seed",-1))
assert c.get("residual_prompt_enabled") is True and int(c.get("residual_num_bases",-1))==1
assert c.get("static_prompt_version") in {"ncrp_k1", "ncrp_a1_k1_single_residual"}
assert t.get("complete") is True and int(t.get("residual_num_bases",-1))==1
assert g.get("basis_usage")==[1.0]
assert g.get("assignment_diagnostics_status")=="not_applicable_single_basis"
assert g.get("normalized_assignment_entropy") is None
assert g.get("basis_gram_off_diagonal_mean")==0.0
assert int(g.get("trainable_parameter_count",-1))==6400
with np.load(d/"sample_scores.npz",allow_pickle=False) as z:
 assert z["basis_assignments"].shape[-1]==1 and z["basis_similarities"].shape[-1]==1
assert m.get("train_category")==pair
print(f"NCRP-K1 integrity OK pair={pair} path={d}")' \
    "${directory}" "${pair}" "${SEED}"
}

run_pair() {
  local first="$1"
  local second="$2"
  local pair="${first}+${second}"
  local stage1="${STAGE1_ROOT}/Real3D/${pair}/best.pth"
  local run_dir="${OUTPUT_ROOT}/Real3D/${pair}"
  local start end status resume_checkpoint=""

  echo "[$(timestamp)] START pair=${pair} seed=${SEED} method=NCRP-K1"
  echo "config=${CONFIG} batch_size=${BATCH_SIZE} output_dir=${run_dir}"
  echo "stage1_checkpoint=${stage1} SKIP_STAGE1=1 orthogonalization=soft num_bases=1"
  validate_stage1 "${stage1}"
  if [[ "${RESUME}" == "1" ]] && validate_k1 "${run_dir}" "${pair}"; then
    echo "[$(timestamp)] SKIP pair=${pair}: valid completed artifacts"
    return 0
  fi
  if [[ "${RESUME}" == "0" ]]; then
    archive_dir "${run_dir}"
  elif [[ -s "${run_dir}/last.pth" ]]; then
    if "${PYTHON_BIN}" -B -c \
      'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert p.get("residual_prompt_enabled") is True and int(p.get("residual_num_bases",-1))==1 and int(p.get("seed",-1))==111 and "optimizer" in p and "scheduler" in p' \
      "${run_dir}/last.pth"; then
      resume_checkpoint="${run_dir}/last.pth"
      echo "[$(timestamp)] resume from valid checkpoint=${resume_checkpoint}"
    else
      "${PYTHON_BIN}" -B -c \
        'from utils.artifact_integrity import quarantine_corrupt_file; import sys; print(quarantine_corrupt_file(sys.argv[1]))' \
        "${run_dir}/last.pth"
    fi
  fi

  start="$(date '+%s')"
  status=0
  if env \
      PYTHON_BIN="${PYTHON_BIN}" BATCH_SIZE="${BATCH_SIZE}" RESUME="${RESUME}" \
      RUN_ONLY="Real3D:${pair}" SKIP_STAGE1=1 \
      STAGE1_CHECKPOINT="${stage1}" EXPERIMENT_CONFIG="${CONFIG}" \
      EXPERIMENT_SEED="${SEED}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      RESUME_CHECKPOINT="${resume_checkpoint}" \
      bash "${RUNNER}"; then
    status=0
  else
    status=$?
  fi
  end="$(date '+%s')"
  echo "[$(timestamp)] END pair=${pair} status=${status} duration_seconds=$((end-start))"
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
  validate_k1 "${run_dir}" "${pair}"
}

echo "[$(timestamp)] NCRP-K1 Real3D seed-111 sequence start"
echo "pairs=car+chicken,candybar+fish,duck+starfish seed=${SEED} batch_size=${BATCH_SIZE}"
echo "stage1_root=${STAGE1_ROOT} output_root=${OUTPUT_ROOT}"

[[ "${RUN_CAR_CHICKEN}" == "1" ]] && run_pair car chicken
[[ "${RUN_CANDYBAR_FISH}" == "1" ]] && run_pair candybar fish
[[ "${RUN_DUCK_STARFISH}" == "1" ]] && run_pair duck starfish

echo "[$(timestamp)] NCRP-K1 runs complete: ${OUTPUT_ROOT}/Real3D"
