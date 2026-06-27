#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/abelde/projects/active/Shell_Gaussian"
GSHELL_DIR="${PROJECT_ROOT}/baselines/GShell"
ENV_DIR="${GSHELL_ENV_DIR:-${GSHELL_DIR}/GShell_env}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
BLENDER="${BLENDER:-${ENV_DIR}/opt/blender-4.2.21-linux-x64/blender}"
DATASET_ROOT="${DATASET_ROOT:-/data/abelde/datasets/processed/external_shoes_canonical/nike-air-jordan/multi_elevation_360}"
CONFIG_PATH="${CONFIG_PATH:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768_depth2.json}"
OUT_DIR="${OUT_DIR:-${GSHELL_DIR}/output/external_shoes_canonical_gshell/nike-air-jordan_multi_elevation_360_depth2_train_val_512_768}"
LOG_DIR="${OUT_DIR}/logs"
MESH_NPZ="${MESH_NPZ:-${OUT_DIR}/canonical_mesh/nike-air-jordan_canonical.npz}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-768}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v min_free="${MIN_FREE_MB}" '
      {
        idx=$1; free=$2
        gsub(/ /, "", idx); gsub(/ /, "", free)
        if (free >= min_free) print idx, free
      }
    ' \
  | sort -k2,2nr \
  | head -n 1 \
  | awk '{print $1}'
}

mkdir -p "${LOG_DIR}" "$(dirname "${MESH_NPZ}")"
BATCH_LOG="${BATCH_LOG:-${LOG_DIR}/batch_$(date -u +%Y%m%d_%H%M%S).log}"
TRAIN_LOG="${LOG_DIR}/train.log"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[$(timestamp)] Nike depth-second experiment starting"
echo "[$(timestamp)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(timestamp)] ENV_DIR=${ENV_DIR}"
echo "[$(timestamp)] PYTHON=${PYTHON}"
echo "[$(timestamp)] BLENDER=${BLENDER}"
echo "[$(timestamp)] DATASET_ROOT=${DATASET_ROOT}"
echo "[$(timestamp)] CONFIG_PATH=${CONFIG_PATH}"
echo "[$(timestamp)] OUT_DIR=${OUT_DIR}"
echo "[$(timestamp)] BATCH_LOG=${BATCH_LOG}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[$(timestamp)] ERROR: python not executable at ${PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${BLENDER}" ]]; then
  echo "[$(timestamp)] ERROR: blender not executable at ${BLENDER}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[$(timestamp)] ERROR: config missing at ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_ROOT}/train/transforms.json" || ! -f "${DATASET_ROOT}/val/transforms.json" ]]; then
  echo "[$(timestamp)] ERROR: train/val transforms missing under ${DATASET_ROOT}" >&2
  exit 1
fi

export HF_HOME="/data/abelde/.cache/huggingface"
export TORCH_HOME="/data/abelde/.cache/torch"
export XDG_CACHE_HOME="/data/abelde/.cache"
export CONDA_PKGS_DIRS="/data/abelde/.conda/pkgs"
export PATH="${ENV_DIR}/bin:${PATH}"

CONDA_GCC="${ENV_DIR}/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="${ENV_DIR}/bin/x86_64-conda-linux-gnu-g++"
if [[ -x "${CONDA_GCC}" && -x "${CONDA_GXX}" ]]; then
  export CC="${CONDA_GCC}"
  export CXX="${CONDA_GXX}"
  export CUDAHOSTCXX="${CONDA_GXX}"
fi
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LD_LIBRARY_PATH:-}"

echo "[$(timestamp)] Exporting canonical Nike mesh"
"${BLENDER}" --background --python "${PROJECT_ROOT}/FootShellGaussian/scripts/export_external_canonical_mesh_npz.py" -- \
  --shoe nike-air-jordan \
  --output "${MESH_NPZ}"

echo "[$(timestamp)] Generating second-layer invdepth targets"
"${PYTHON}" "${GSHELL_DIR}/scripts/generate_invdepth_second.py" \
  --dataset-root "${DATASET_ROOT}" \
  --mesh-npz "${MESH_NPZ}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --overwrite

GPU_ID="${GPU_ID:-$(pick_gpu)}"
if [[ -z "${GPU_ID}" ]]; then
  echo "[$(timestamp)] ERROR: no GPU has at least ${MIN_FREE_MB} MiB free" >&2
  nvidia-smi || true
  exit 1
fi
echo "[$(timestamp)] Selected physical GPU ${GPU_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cd "${GSHELL_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[$(timestamp)] Starting training with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON}" -u train_gshelltet_polycam.py \
  --config "${CONFIG_PATH}" \
  --trainset_path "${DATASET_ROOT}/train" \
  --testset_path "${DATASET_ROOT}/val" \
  --out-dir "${OUT_DIR}" \
  2>&1 | tee "${TRAIN_LOG}"

echo "[$(timestamp)] Done"
