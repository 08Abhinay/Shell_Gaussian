#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/abelde/projects/active/Shell_Gaussian"
GSHELL_DIR="${PROJECT_ROOT}/baselines/GShell"
ENV_DIR="${GSHELL_ENV_DIR:-${GSHELL_DIR}/GShell_env}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768_msdf_mlp.json}"
OUT_ROOT="${OUT_ROOT:-${GSHELL_DIR}/output/msdf_mlp_true_comparison_20260616}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"

NIKE_FULL_ROOT="/data/abelde/datasets/processed/external_shoes_canonical/nike-air-jordan/multi_elevation_360"
TURN_ROOT="/data/abelde/datasets/processed/gshell_shoes_turntable_canonical"

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

run_train() {
  local run_name="$1"
  local trainset_path="$2"
  local testset_path="$3"
  local out_dir="${OUT_ROOT}/${run_name}"
  local log_dir="${out_dir}/logs"
  local train_log="${log_dir}/train.log"

  mkdir -p "${log_dir}"

  echo "[$(timestamp)] ====================================================================="
  echo "[$(timestamp)] Starting ${run_name}"
  echo "[$(timestamp)] trainset_path=${trainset_path}"
  echo "[$(timestamp)] testset_path=${testset_path:-<same as train>}"
  echo "[$(timestamp)] out_dir=${out_dir}"

  if [[ ! -f "${trainset_path}/transforms.json" ]]; then
    echo "[$(timestamp)] ERROR: missing ${trainset_path}/transforms.json" >&2
    exit 1
  fi
  if [[ -n "${testset_path}" && ! -f "${testset_path}/transforms.json" ]]; then
    echo "[$(timestamp)] ERROR: missing ${testset_path}/transforms.json" >&2
    exit 1
  fi

  local cmd=(
    "${PYTHON}" -u train_gshelltet_polycam.py
    --config "${CONFIG_PATH}"
    --trainset_path "${trainset_path}"
    --out-dir "${out_dir}"
  )
  if [[ -n "${testset_path}" ]]; then
    cmd+=(--testset_path "${testset_path}")
  fi

  (
    cd "${GSHELL_DIR}"
    "${cmd[@]}"
  ) 2>&1 | tee "${train_log}"

  echo "[$(timestamp)] Finished ${run_name}"
}

mkdir -p "${OUT_ROOT}/logs"
BATCH_LOG="${BATCH_LOG:-${OUT_ROOT}/logs/batch_$(date -u +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[$(timestamp)] mSDF MLP comparison batch starting"
echo "[$(timestamp)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(timestamp)] GSHELL_DIR=${GSHELL_DIR}"
echo "[$(timestamp)] ENV_DIR=${ENV_DIR}"
echo "[$(timestamp)] PYTHON=${PYTHON}"
echo "[$(timestamp)] CONFIG_PATH=${CONFIG_PATH}"
echo "[$(timestamp)] OUT_ROOT=${OUT_ROOT}"
echo "[$(timestamp)] BATCH_LOG=${BATCH_LOG}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[$(timestamp)] ERROR: python not executable at ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[$(timestamp)] ERROR: config missing at ${CONFIG_PATH}" >&2
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

GPU_ID="${GPU_ID:-$(pick_gpu)}"
if [[ -z "${GPU_ID}" ]]; then
  echo "[$(timestamp)] ERROR: no GPU has at least ${MIN_FREE_MB} MiB free" >&2
  nvidia-smi || true
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[$(timestamp)] Selected physical GPU ${GPU_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cat > "${OUT_ROOT}/run_manifest.txt" <<EOF
mSDF MLP true comparison batch
started_utc=$(timestamp)
config=${CONFIG_PATH}
physical_gpu=${GPU_ID}

runs:
  nike-air-jordan_full_views_msdfmlp
    train=${NIKE_FULL_ROOT}/train
    val=${NIKE_FULL_ROOT}/val
  sandal_nike-calm-slide_msdfmlp
    train=${TURN_ROOT}/Nike-Calm-Slide-Cinnamon-Monarch
  boot_ugg-classic-short_msdfmlp
    train=${TURN_ROOT}/Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler
  shoe_nike-cortez_msdfmlp
    train=${TURN_ROOT}/Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail
EOF

run_train \
  "nike-air-jordan_full_views_msdfmlp" \
  "${NIKE_FULL_ROOT}/train" \
  "${NIKE_FULL_ROOT}/val"

run_train \
  "sandal_nike-calm-slide_msdfmlp" \
  "${TURN_ROOT}/Nike-Calm-Slide-Cinnamon-Monarch" \
  ""

run_train \
  "boot_ugg-classic-short_msdfmlp" \
  "${TURN_ROOT}/Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler" \
  ""

run_train \
  "shoe_nike-cortez_msdfmlp" \
  "${TURN_ROOT}/Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail" \
  ""

echo "[$(timestamp)] mSDF MLP comparison batch finished"
