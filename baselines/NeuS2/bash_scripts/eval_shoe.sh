#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <shoe_name> [gpu_id] [n_steps]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${NEUS2_ENV:-/home/ab5298/anaconda3/envs/neus2}"
PYTHON_BIN="${NEUS2_PYTHON:-${ENV_DIR}/bin/python}"
DATA_ROOT="${NEUS2_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation}"
OUTPUT_ROOT="${NEUS2_OUTPUT_ROOT:-${NEUS2_ROOT}/output/golden_set_evaluation_blender_final}"
EXPERIMENT_TAG="${NEUS2_EXPERIMENT_TAG:-}"
CONFIG="${NEUS2_CONFIG:-dtu.json}"
EVAL_TRANSFORM_NAME="${NEUS2_EVAL_TRANSFORM:-transform_test.json}"

SHOE_NAME="$1"
GPU_ID="${2:-${NEUS2_GPU:-2}}"
N_STEPS="${3:-${NEUS2_N_STEPS:-15000}}"
SCENE_PATH="${DATA_ROOT}/${SHOE_NAME}/${EVAL_TRANSFORM_NAME}"
if [[ -n "${EXPERIMENT_TAG}" ]]; then
    OUTPUT_PATH="${OUTPUT_ROOT}/${EXPERIMENT_TAG}/${SHOE_NAME}"
else
    OUTPUT_PATH="${OUTPUT_ROOT}/${SHOE_NAME}"
fi
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
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing NeuS2 Python: ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONPATH="${NEUS2_ROOT}/build${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

EVAL_VIEW_COUNT="$("${PYTHON_BIN}" - "${SCENE_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(len(json.load(handle)["frames"]))
PY
)"
if [[ "${EVAL_VIEW_COUNT}" -le 0 ]]; then
    echo "Expected at least one held-out view, found ${EVAL_VIEW_COUNT}" >&2
    exit 1
fi

cd "${NEUS2_ROOT}"
: > "${COMBINED_LOG}"

echo "Eval scene: ${SCENE_PATH}"
echo "Snapshot: ${SNAPSHOT_PATH}"
echo "Views: 0-$((EVAL_VIEW_COUNT - 1))"
echo "Background: white"
echo "Physical GPU: ${GPU_ID}"

"${PYTHON_BIN}" scripts/run.py \
    --scene "${SCENE_PATH}" \
    --name "${SHOE_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --network "${CONFIG}" \
    --load_snapshot "${SNAPSHOT_PATH}" \
    --test \
    --test_all_views \
    --white_bkgd

cp "${OUTPUT_PATH}/eval_log.txt" "${COMBINED_LOG}"

awk -v expected="${EVAL_VIEW_COUNT}" '
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
        if (n != expected) {
            exit 2
        }
        printf "AVERAGE_TEST PSNR=%.12f SSIM=%.12f LPIPS=%.12f views=%d\n", psnr / n, ssim / n, lpips / n, n
    }
' "${COMBINED_LOG}" | tee -a "${COMBINED_LOG}"

cp "${COMBINED_LOG}" "${OUTPUT_PATH}/eval_log.txt"
