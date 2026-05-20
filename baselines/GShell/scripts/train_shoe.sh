#!/bin/bash
set -eo pipefail

SHOE_NAME="${1:?Usage: $0 <SHOE_NAME> [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${PROJECT_DIR}/GShell_env"
DATASET_ROOT="${GSHELL_DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes}"
CONFIG_PATH="${GSHELL_CONFIG:-${PROJECT_DIR}/configs/shoes_mc_normfix.json}"
OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-_normfix_new}"
OUTPUT_ROOT="${GSHELL_OUTPUT_ROOT:-${PROJECT_DIR}/output}"
DATASET_DIR="${DATASET_ROOT}/${SHOE_NAME}"
OUT_DIR="${OUTPUT_ROOT}/${SHOE_NAME}${OUT_SUFFIX}"
LOG_DIR="${OUT_DIR}/logs"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: dataset not found at ${DATASET_DIR}"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config not found at ${CONFIG_PATH}"
    exit 1
fi

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="/data/abelde/.cache/huggingface"
export TORCH_HOME="/data/abelde/.cache/torch"
export XDG_CACHE_HOME="/data/abelde/.cache"
export CONDA_PKGS_DIRS="/data/abelde/.conda/pkgs"
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
export PATH="${ENV_DIR}/bin:${PATH}"

if ! command -v ninja >/dev/null 2>&1; then
    echo "Error: ninja is required to build PyTorch C++ extensions, but it is not on PATH."
    echo "Install ninja into ${ENV_DIR} or make it available before launching training."
    exit 1
fi

CONDA_GCC="${ENV_DIR}/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="${ENV_DIR}/bin/x86_64-conda-linux-gnu-g++"
if [[ ! -x "${CONDA_GCC}" || ! -x "${CONDA_GXX}" ]]; then
    echo "Error: Conda GCC/G++ wrappers were not found in ${ENV_DIR}/bin."
    echo "CUDA 11.7 needs a GCC/G++ 11-compatible host compiler for local extension builds."
    exit 1
fi

export CC="${CONDA_GCC}"
export CXX="${CONDA_GXX}"
export CUDAHOSTCXX="${CONDA_GXX}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"

echo "Training shoe: ${SHOE_NAME}"
echo "Dataset: ${DATASET_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Output: ${OUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "CXX: ${CXX}"

python -u train_gshelltet_polycam.py \
    --config "$CONFIG_PATH" \
    --trainset_path "$DATASET_DIR" \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "${LOG_DIR}/train.log"
