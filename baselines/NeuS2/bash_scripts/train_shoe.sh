#!/usr/bin/env bash
set -euo pipefail

SHOE_NAME="${1:?Usage: $0 <shoe_name> [gpu_id] [n_steps]}"
GPU_ID="${2:-${NEUS2_GPU:-2}}"
N_STEPS="${3:-${NEUS2_N_STEPS:-15000}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${NEUS2_ENV:-/home/ab5298/anaconda3/envs/neus2}"
PYTHON_BIN="${NEUS2_PYTHON:-${ENV_DIR}/bin/python}"
DATA_ROOT="${NEUS2_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_neus2}"
OUTPUT_ROOT="${NEUS2_OUTPUT_ROOT:-${NEUS2_ROOT}/output/golden_set_evaluation_blender_final}"
EXPERIMENT_TAG="${NEUS2_EXPERIMENT_TAG:-}"
CONFIG="${NEUS2_CONFIG:-dtu.json}"
TRANSFORM_NAME="${NEUS2_TRAIN_TRANSFORM:-transform_train.json}"
MARCHING_CUBES_RES="${NEUS2_MARCHING_CUBES_RES:-512}"
CACHE_ROOT="${NEUS2_CACHE_ROOT:-/home/ab5298/.cache}"

SCENE_PATH="${DATA_ROOT}/${SHOE_NAME}/${TRANSFORM_NAME}"
if [[ -n "${EXPERIMENT_TAG}" ]]; then
    OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_TAG}/${SHOE_NAME}"
else
    OUTPUT_DIR="${OUTPUT_ROOT}/${SHOE_NAME}"
fi
LOG_DIR="${OUTPUT_DIR}/logs"

if [[ ! -f "${SCENE_PATH}" ]]; then
    echo "Missing scene transform: ${SCENE_PATH}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing NeuS2 Python: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export PYTHONPATH="${NEUS2_ROOT}/build${PYTHONPATH:+:${PYTHONPATH}}"

cd "${NEUS2_ROOT}"

echo "Training shoe: ${SHOE_NAME}"
echo "Scene: ${SCENE_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG}"
echo "Steps: ${N_STEPS}"
echo "Marching cubes: ${MARCHING_CUBES_RES}"
echo "Physical GPU: ${GPU_ID}"

"${PYTHON_BIN}" -u scripts/run.py \
    --scene "${SCENE_PATH}" \
    --name "${SHOE_NAME}" \
    --output_path "${OUTPUT_DIR}" \
    --network "${CONFIG}" \
    --n_steps "${N_STEPS}" \
    --marching_cubes_res "${MARCHING_CUBES_RES}" \
    --skip_post_train_eval \
    2>&1 | tee "${LOG_DIR}/train.log"
