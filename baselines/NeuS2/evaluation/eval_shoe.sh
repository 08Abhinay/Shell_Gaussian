#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 <shoe_name> [gpu_id] [n_steps] [output_name]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${NEUS2_DATA_ROOT:-/data/abelde/datasets/processed/neus2_shoes}"
PYTHON_BIN="${NEUS2_PYTHON:-${NEUS2_ROOT}/neus2_env/bin/python}"
CONFIG="${NEUS2_CONFIG:-dtu.json}"
EVAL_TRANSFORM_NAME="${NEUS2_EVAL_TRANSFORM:-transform_test.json}"

SHOE_NAME="$1"
GPU_ID="${2:-${NEUS2_GPU:-}}"
N_STEPS="${3:-${NEUS2_N_STEPS:-10000}}"
OUTPUT_NAME="${4:-${NEUS2_OUTPUT_NAME:-${SHOE_NAME}_neus2_${N_STEPS}}}"
SCENE_PATH="${DATA_ROOT}/${SHOE_NAME}/${EVAL_TRANSFORM_NAME}"
OUTPUT_PATH="${NEUS2_ROOT}/output/${OUTPUT_NAME}"
SNAPSHOT_PATH="${OUTPUT_PATH}/checkpoints/${N_STEPS}.msgpack"
COMBINED_LOG="${OUTPUT_PATH}/eval_test_log.txt"

if [[ ! -f "${SCENE_PATH}" ]]; then
  echo "Missing eval transform: ${SCENE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SNAPSHOT_PATH}" ]]; then
  echo "Missing snapshot: ${SNAPSHOT_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${NEUS2_ROOT}/build${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

EXTRA_SOURCE="${NEUS2_EVAL_EXTRA_ARGS:-${NEUS2_EXTRA_ARGS:-}}"
EXTRA_ARGS=()
if [[ -n "${EXTRA_SOURCE}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_SOURCE}"
fi

EVAL_VIEW_COUNT="$("${PYTHON_BIN}" - "${SCENE_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(len(data["frames"]))
PY
)"

if [[ "${NEUS2_EVAL_VIEWS:-all}" == "all" ]]; then
  EVAL_VIEWS=()
  for ((view=0; view<EVAL_VIEW_COUNT; view++)); do
    EVAL_VIEWS+=("${view}")
  done
else
  read -r -a EVAL_VIEWS <<< "$(printf '%s' "${NEUS2_EVAL_VIEWS}" | tr ',' ' ')"
fi

cd "${NEUS2_ROOT}"

: > "${COMBINED_LOG}"

echo "Eval scene: ${SCENE_PATH}"
echo "Snapshot: ${SNAPSHOT_PATH}"
echo "Views: ${EVAL_VIEWS[*]}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

for VIEW_ID in "${EVAL_VIEWS[@]}"; do
  "${PYTHON_BIN}" scripts/run.py \
    --scene "${SCENE_PATH}" \
    --name "${OUTPUT_NAME}" \
    --network "${CONFIG}" \
    --load_snapshot "${SNAPSHOT_PATH}" \
    --test \
    --test_camera_view "${VIEW_ID}" \
    "${EXTRA_ARGS[@]}"

  cat "${OUTPUT_PATH}/eval_log.txt" >> "${COMBINED_LOG}"
  printf '\n' >> "${COMBINED_LOG}"
done

awk '
  /camera_view:/ {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^PSNR=/) {
        sub("PSNR=", "", $i)
        psnr += $i
      }
      if ($i ~ /^SSIM=/) {
        sub("SSIM=", "", $i)
        ssim += $i
      }
      if ($i ~ /^LPIPS=/) {
        sub("LPIPS=", "", $i)
        lpips += $i
      }
    }
    n += 1
  }
  END {
    if (n > 0) {
      printf("AVERAGE_TEST PSNR=%.12f SSIM=%.12f LPIPS=%.12f views=%d\n", psnr / n, ssim / n, lpips / n, n)
    }
  }
' "${COMBINED_LOG}" | tee -a "${COMBINED_LOG}"

cp "${COMBINED_LOG}" "${OUTPUT_PATH}/eval_log.txt"
