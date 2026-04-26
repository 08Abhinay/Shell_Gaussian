#!/usr/bin/env bash
set -euo pipefail

SHOE_NAME="${1:?Usage: $0 <shoe_name> [gpu_id] [n_steps]}"
GPU_ID="${2:-${NEUS2_GPU:-0}}"
N_STEPS="${3:-${NEUS2_N_STEPS:-10000}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${NEUS2_ENV:-${NEUS2_ROOT}/neus2_env}"
DATA_ROOT="${NEUS2_DATA_ROOT:-/data/abelde/datasets/processed/neus2_shoes}"
CONFIG="${NEUS2_CONFIG:-dtu.json}"
TRANSFORM_NAME="${NEUS2_TRAIN_TRANSFORM:-transform_train.json}"

SCENE_PATH="${DATA_ROOT}/${SHOE_NAME}/${TRANSFORM_NAME}"
OUTPUT_NAME="${SHOE_NAME}_neus2_${N_STEPS}"
OUTPUT_DIR="${NEUS2_ROOT}/output/${OUTPUT_NAME}"
LOG_DIR="${OUTPUT_DIR}/logs"
CACHE_ROOT="${NEUS2_CACHE_ROOT:-/data/abelde/.cache}"

if [[ ! -f "${SCENE_PATH}" ]]; then
    echo "Missing scene transform: ${SCENE_PATH}" >&2
    exit 1
fi

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Missing conda env: ${ENV_DIR}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data/abelde/.conda/pkgs}"
export PYTHONPATH="${NEUS2_ROOT}/build${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v conda >/dev/null 2>&1; then
    for CONDA_HOOK in "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "${CONDA_HOOK}" ]]; then
            source "${CONDA_HOOK}"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH" >&2
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"
set -u

cd "${NEUS2_ROOT}"

echo "Training shoe: ${SHOE_NAME}"
echo "Scene: ${SCENE_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG}"
echo "Steps: ${N_STEPS}"
echo "GPU: ${GPU_ID}"

python -u scripts/run.py \
    --scene "${SCENE_PATH}" \
    --name "${OUTPUT_NAME}" \
    --network "${CONFIG}" \
    --n_steps "${N_STEPS}" \
    2>&1 | tee "${LOG_DIR}/train.log"
