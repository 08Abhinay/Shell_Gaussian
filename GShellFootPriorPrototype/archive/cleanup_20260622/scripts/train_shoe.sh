#!/bin/bash
set -eo pipefail

SHOE_NAME="${1:?Usage: $0 <SHOE_NAME> [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_ENV_DIR="${PROJECT_DIR}/../baselines/GShell/GShell_env"
ENV_DIR="${FOOTSHELL_ENV_DIR:-${DEFAULT_ENV_DIR}}"
DATASET_ROOT="${GSHELL_DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes}"
CONFIG_PATH="${GSHELL_CONFIG:-${PROJECT_DIR}/configs/shoes_mc_normfix.json}"
OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-_normfix_new}"
DATASET_DIR="${DATASET_ROOT}/${SHOE_NAME}"
OUT_DIR="${PROJECT_DIR}/output/${SHOE_NAME}${OUT_SUFFIX}"
LOG_DIR="${OUT_DIR}/logs"
FOOT_PRIOR_ROOT="${FOOTSHELL_FOOT_PRIOR_ROOT:-${PROJECT_DIR}/output/foot_prior_debug}"
FOOT_PRIOR_ALIGNMENT="${FOOTSHELL_FOOT_PRIOR_ALIGNMENT:-${FOOT_PRIOR_ROOT}/${SHOE_NAME}/alignment.json}"
PSEUDO_LAST_ROOT="${FOOTSHELL_PSEUDO_LAST_ROOT:-${PROJECT_DIR}/output/pseudo_last_section_loft}"
PSEUDO_LAST_SDF="${FOOTSHELL_PSEUDO_LAST_SDF:-${PSEUDO_LAST_ROOT}/${SHOE_NAME}/pseudo_last_sdf.npz}"
PSEUDO_LAST_SECTIONS="${FOOTSHELL_PSEUDO_LAST_SECTIONS:-${PSEUDO_LAST_ROOT}/${SHOE_NAME}/pseudo_last_sections.npz}"

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

CONFIG_USES_FOOT_PRIOR="$(
python - "$CONFIG_PATH" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as f:
    config = json.load(f)
print("1" if config.get("use_foot_prior", False) else "0")
PY
)"
FOOT_PRIOR_ARGS=()
if [[ "${CONFIG_USES_FOOT_PRIOR}" == "1" ]]; then
    if [ ! -f "$FOOT_PRIOR_ALIGNMENT" ]; then
        echo "Error: foot-prior alignment not found at ${FOOT_PRIOR_ALIGNMENT}"
        echo "Run scripts/prepare_foot_prior_for_shoe.py for this shoe first, or set FOOTSHELL_FOOT_PRIOR_ALIGNMENT."
        exit 1
    fi
    FOOT_PRIOR_ARGS+=(--foot_prior_alignment_path "$FOOT_PRIOR_ALIGNMENT")
fi

CONFIG_USES_PSEUDO_LAST_PRIOR="$(
python - "$CONFIG_PATH" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as f:
    config = json.load(f)
print("1" if config.get("use_pseudo_last_prior", False) else "0")
PY
)"
PSEUDO_LAST_ARGS=()
if [[ "${CONFIG_USES_PSEUDO_LAST_PRIOR}" == "1" ]]; then
    if [ ! -f "$PSEUDO_LAST_SDF" ]; then
        echo "Error: pseudo-last SDF not found at ${PSEUDO_LAST_SDF}"
        echo "Run scripts/run_pseudo_last_builder.py for this shoe first, or set FOOTSHELL_PSEUDO_LAST_SDF."
        exit 1
    fi
    if [ ! -f "$PSEUDO_LAST_SECTIONS" ]; then
        echo "Error: pseudo-last sections not found at ${PSEUDO_LAST_SECTIONS}"
        echo "Run scripts/run_pseudo_last_builder.py for this shoe first, or set FOOTSHELL_PSEUDO_LAST_SECTIONS."
        exit 1
    fi
    PSEUDO_LAST_ARGS+=(
        --pseudo_last_sdf_path "$PSEUDO_LAST_SDF"
        --pseudo_last_sections_path "$PSEUDO_LAST_SECTIONS"
    )
fi

cd "$PROJECT_DIR"

echo "Training shoe: ${SHOE_NAME}"
echo "Dataset: ${DATASET_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "Env: ${ENV_DIR}"
if [[ "${CONFIG_USES_FOOT_PRIOR}" == "1" ]]; then
    echo "Foot alignment: ${FOOT_PRIOR_ALIGNMENT}"
fi
if [[ "${CONFIG_USES_PSEUDO_LAST_PRIOR}" == "1" ]]; then
    echo "Pseudo-last SDF: ${PSEUDO_LAST_SDF}"
    echo "Pseudo-last sections: ${PSEUDO_LAST_SECTIONS}"
fi
echo "CXX: ${CXX}"

python -u train_gshelltet_polycam.py \
    --config "$CONFIG_PATH" \
    --trainset_path "$DATASET_DIR" \
    --out-dir "$OUT_DIR" \
    "${FOOT_PRIOR_ARGS[@]}" \
    "${PSEUDO_LAST_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/train.log"
