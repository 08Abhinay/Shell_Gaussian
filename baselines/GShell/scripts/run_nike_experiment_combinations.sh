#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/abelde/projects/active/Shell_Gaussian"
GSHELL_DIR="${PROJECT_ROOT}/baselines/GShell"
ENV_DIR="${GSHELL_ENV_DIR:-${GSHELL_DIR}/GShell_env}"
PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/abelde/datasets/processed/external_shoes_canonical/nike-air-jordan/multi_elevation_360}"
OUT_ROOT="${OUT_ROOT:-${GSHELL_DIR}/output/experiment_combinations/nike-air-jordan_full_views_20260617}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"

CONFIG_A="${CONFIG_A:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768.json}"
CONFIG_B="${CONFIG_B:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768_depth.json}"
CONFIG_C="${CONFIG_C:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768_msdf_mlp.json}"
CONFIG_D="${CONFIG_D:-${GSHELL_DIR}/configs/shoes_mc_normfix_512_768_depth_msdf_mlp.json}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v min_free="${MIN_FREE_MB}" '
      {
        idx=$1; free=$2
        gsub(/ /, "", idx); gsub(/ /, "", free)
        if (free >= min_free) print idx, free
      }
    ' \
  | sort -k2,2nr \
  | head -n 2 \
  | awk '{print $1}'
}

run_train() {
  local run_name="$1"
  local config_path="$2"
  local gpu_id="$3"
  local out_dir="${OUT_ROOT}/${run_name}"
  local log_dir="${out_dir}/logs"
  local train_log="${log_dir}/train.log"

  mkdir -p "${log_dir}"

  {
    echo "[$(timestamp)] Starting ${run_name}"
    echo "[$(timestamp)] config=${config_path}"
    echo "[$(timestamp)] CUDA_VISIBLE_DEVICES=${gpu_id}"
    echo "[$(timestamp)] out_dir=${out_dir}"
    (
      cd "${GSHELL_DIR}"
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      "${PYTHON}" -u train_gshelltet_polycam.py \
        --config "${config_path}" \
        --trainset_path "${DATASET_ROOT}/train" \
        --testset_path "${DATASET_ROOT}/val" \
        --out-dir "${out_dir}"
    )
    echo "[$(timestamp)] Finished ${run_name}"
  } 2>&1 | tee "${train_log}"
}

run_pair() {
  local name1="$1"
  local config1="$2"
  local name2="${3:-}"
  local config2="${4:-}"

  if [[ "${#GPUS[@]}" -ge 2 && -n "${name2}" ]]; then
    run_train "${name1}" "${config1}" "${GPUS[0]}" &
    local pid1=$!
    run_train "${name2}" "${config2}" "${GPUS[1]}" &
    local pid2=$!
    local status1=0
    local status2=0
    wait "${pid1}" || status1=$?
    wait "${pid2}" || status2=$?
    if [[ "${status1}" -ne 0 || "${status2}" -ne 0 ]]; then
      echo "[$(timestamp)] ERROR: parallel pair failed: ${name1} status=${status1}, ${name2} status=${status2}" >&2
      exit 1
    fi
  else
    run_train "${name1}" "${config1}" "${GPUS[0]}"
    if [[ -n "${name2}" ]]; then
      run_train "${name2}" "${config2}" "${GPUS[0]}"
    fi
  fi
}

mkdir -p "${OUT_ROOT}/logs"
BATCH_LOG="${BATCH_LOG:-${OUT_ROOT}/logs/batch_$(date -u +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[$(timestamp)] Nike experiment-combinations batch starting"
echo "[$(timestamp)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(timestamp)] GSHELL_DIR=${GSHELL_DIR}"
echo "[$(timestamp)] ENV_DIR=${ENV_DIR}"
echo "[$(timestamp)] PYTHON=${PYTHON}"
echo "[$(timestamp)] DATASET_ROOT=${DATASET_ROOT}"
echo "[$(timestamp)] OUT_ROOT=${OUT_ROOT}"
echo "[$(timestamp)] BATCH_LOG=${BATCH_LOG}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[$(timestamp)] ERROR: python not executable at ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_ROOT}/train/transforms.json" || ! -f "${DATASET_ROOT}/val/transforms.json" ]]; then
  echo "[$(timestamp)] ERROR: train/val transforms missing under ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/train/invdepth" || ! -d "${DATASET_ROOT}/val/invdepth" ]]; then
  echo "[$(timestamp)] ERROR: invdepth directories missing under train/val in ${DATASET_ROOT}" >&2
  exit 1
fi
for config_path in "${CONFIG_A}" "${CONFIG_B}" "${CONFIG_C}" "${CONFIG_D}"; do
  if [[ ! -f "${config_path}" ]]; then
    echo "[$(timestamp)] ERROR: config missing at ${config_path}" >&2
    exit 1
  fi
done

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

mapfile -t GPUS < <(pick_gpus)
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[$(timestamp)] ERROR: no GPU has at least ${MIN_FREE_MB} MiB free" >&2
  nvidia-smi || true
  exit 1
fi

echo "[$(timestamp)] Selected physical GPUs: ${GPUS[*]}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cat > "${OUT_ROOT}/run_manifest.txt" <<EOF
Nike Air Jordan full-view experiment combinations
started_utc=$(timestamp)
dataset_root=${DATASET_ROOT}
selected_physical_gpus=${GPUS[*]}

A_baseline_no_depth_no_msdfmlp
  use_depth=false
  use_msdf_mlp=false
  config=${CONFIG_A}

B_invdepth_no_msdfmlp
  use_depth=true
  use_msdf_mlp=false
  config=${CONFIG_B}

C_msdfmlp_no_depth
  use_depth=false
  use_msdf_mlp=true
  config=${CONFIG_C}

D_invdepth_msdfmlp
  use_depth=true
  use_msdf_mlp=true
  config=${CONFIG_D}
EOF

run_pair \
  "A_baseline_no_depth_no_msdfmlp" "${CONFIG_A}" \
  "B_invdepth_no_msdfmlp" "${CONFIG_B}"

run_pair \
  "C_msdfmlp_no_depth" "${CONFIG_C}" \
  "D_invdepth_msdfmlp" "${CONFIG_D}"

echo "[$(timestamp)] Nike experiment-combinations batch finished"
