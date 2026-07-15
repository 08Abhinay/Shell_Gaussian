#!/bin/bash
set -eo pipefail

SHOE_NAME="${1:?Usage: $0 <SHOE_NAME> [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ENV_DIR="/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv"
LOCAL_DATASET_ROOT="/storage/Abhinay/home_ab5298/dataset/datasets/processed/fab_evaluation_final"

if [ -d "${LOCAL_DATASET_ROOT}" ]; then
    DEFAULT_ENV_DIR="${LOCAL_ENV_DIR}"
    DEFAULT_DATASET_ROOT="${LOCAL_DATASET_ROOT}"
    DEFAULT_CONFIG_PATH="${PROJECT_DIR}/configs/shoes_mc_normfix_512_768_depth.json"
    DEFAULT_OUT_SUFFIX="_depth"
    DEFAULT_OUTPUT_ROOT="${PROJECT_DIR}/output/fab_evaluation_final_depth"
else
    DEFAULT_ENV_DIR="${PROJECT_DIR}/GShell_env"
    DEFAULT_DATASET_ROOT="/data/abelde/datasets/processed/gshell_shoes"
    DEFAULT_CONFIG_PATH="${PROJECT_DIR}/configs/shoes_mc_normfix_512_768.json"
    DEFAULT_OUT_SUFFIX="_normfix"
    DEFAULT_OUTPUT_ROOT="${PROJECT_DIR}/output"
fi

ENV_DIR="${GSHELL_CONDA_ENV:-${DEFAULT_ENV_DIR}}"
DATASET_ROOT="${GSHELL_DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
CONFIG_PATH="${GSHELL_CONFIG:-${DEFAULT_CONFIG_PATH}}"
OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-${DEFAULT_OUT_SUFFIX}}"
OUTPUT_ROOT="${GSHELL_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
TRAINER_MODE="${GSHELL_TRAINER:-auto}"
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

if [ ! -d "$ENV_DIR" ]; then
    echo "Error: conda env not found at ${ENV_DIR}"
    exit 1
fi

resolve_json_path() {
    local requested="$1"
    local fallback="$2"
    local resolved="${requested:-${fallback}}"
    if [[ -z "${resolved}" ]]; then
        return 1
    fi
    if [[ "${resolved}" = /* ]]; then
        printf '%s\n' "${resolved}"
    else
        printf '%s\n' "${DATASET_DIR}/${resolved}"
    fi
}

HAS_SPLIT_JSON=0
HAS_SINGLE_JSON=0
if [ -f "${DATASET_DIR}/transforms_train.json" ] && [ -f "${DATASET_DIR}/transforms_test.json" ]; then
    HAS_SPLIT_JSON=1
fi
if [ -f "${DATASET_DIR}/transforms.json" ]; then
    HAS_SINGLE_JSON=1
fi

if [[ "${TRAINER_MODE}" == "auto" ]]; then
    if [[ "${HAS_SPLIT_JSON}" == "1" ]]; then
        TRAINER_MODE="synthetic"
    elif [[ "${HAS_SINGLE_JSON}" == "1" ]]; then
        TRAINER_MODE="polycam"
    else
        echo "Error: unsupported dataset layout in ${DATASET_DIR}"
        exit 1
    fi
fi

if [[ "${TRAINER_MODE}" == "synthetic" ]]; then
    if [[ "${HAS_SPLIT_JSON}" != "1" ]]; then
        echo "Error: synthetic trainer needs transforms_train.json and transforms_test.json in ${DATASET_DIR}"
        exit 1
    fi
    TRAIN_ENTRY="train_gshelltet_synthetic.py"
    TRAIN_ARGS=(
        --config "$CONFIG_PATH"
        --ref_mesh "$DATASET_DIR"
        --out-dir "$OUT_DIR"
    )
elif [[ "${TRAINER_MODE}" == "polycam" ]]; then
    if [[ "${HAS_SPLIT_JSON}" == "1" ]]; then
        DEFAULT_TRAIN_JSON="transforms_train.json"
        DEFAULT_VALIDATE_JSON="transforms_test.json"
    elif [[ "${HAS_SINGLE_JSON}" == "1" ]]; then
        DEFAULT_TRAIN_JSON="transforms.json"
        DEFAULT_VALIDATE_JSON="transforms.json"
    else
        echo "Error: polycam trainer could not find usable transforms JSONs in ${DATASET_DIR}"
        exit 1
    fi

    TRAIN_JSON_PATH="$(resolve_json_path "${GSHELL_TRAIN_TRANSFORMS_JSON:-}" "${DEFAULT_TRAIN_JSON}")"
    VALIDATE_JSON_PATH="$(resolve_json_path "${GSHELL_VALIDATE_TRANSFORMS_JSON:-}" "${DEFAULT_VALIDATE_JSON}")"

    if [[ ! -f "${TRAIN_JSON_PATH}" ]]; then
        echo "Error: train transforms json not found at ${TRAIN_JSON_PATH}"
        exit 1
    fi
    if [[ ! -f "${VALIDATE_JSON_PATH}" ]]; then
        echo "Error: validate transforms json not found at ${VALIDATE_JSON_PATH}"
        exit 1
    fi

    TRAIN_ENTRY="train_gshelltet_polycam.py"
    TRAIN_ARGS=(
        --config "$CONFIG_PATH"
        --trainset_path "$DATASET_DIR"
        --testset_path "$DATASET_DIR"
        --train-transforms-json "$TRAIN_JSON_PATH"
        --validate-transforms-json "$VALIDATE_JSON_PATH"
        --out-dir "$OUT_DIR"
    )
else
    echo "Error: unsupported GSHELL_TRAINER='${TRAINER_MODE}'. Expected auto, synthetic, or polycam."
    exit 1
fi

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="/data/abelde/.cache/huggingface"
export TORCH_HOME="/data/abelde/.cache/torch"
export XDG_CACHE_HOME="/data/abelde/.cache"
export CONDA_PKGS_DIRS="/data/abelde/.conda/pkgs"
if [ -f "/storage/Abhinay/home_ab5298/anaconda3/etc/profile.d/conda.sh" ]; then
    # Prefer the shared conda install available on this machine.
    source /storage/Abhinay/home_ab5298/anaconda3/etc/profile.d/conda.sh
else
    eval "$(conda shell.bash hook)"
fi
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
echo "Trainer: ${TRAIN_ENTRY}"
echo "Trainer mode: ${TRAINER_MODE}"
if [[ "${TRAINER_MODE}" == "polycam" ]]; then
    echo "Train JSON: ${TRAIN_JSON_PATH}"
    echo "Validate JSON: ${VALIDATE_JSON_PATH}"
fi
echo "Env: ${ENV_DIR}"
echo "CXX: ${CXX}"

python -u "${TRAIN_ENTRY}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
