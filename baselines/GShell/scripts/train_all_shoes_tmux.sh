#!/bin/bash
set -euo pipefail

GPU_ID="${1:-0}"
SESSION_NAME="${2:-gshell_all_shoes}"
shift $(( $# >= 2 ? 2 : $# ))
REQUESTED_SHOES=("$@")

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_shoe.sh"
DATASET_ROOT="${GSHELL_DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes}"
CONFIG_PATH="${GSHELL_CONFIG:-${PROJECT_DIR}/configs/shoes_mc_normfix.json}"
OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-_normfix}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

if [[ ! -x "$(command -v tmux)" ]]; then
    echo "Error: tmux is not available on PATH."
    exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Error: dataset root not found at ${DATASET_ROOT}"
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config not found at ${CONFIG_PATH}"
    exit 1
fi

if [[ "${INSIDE_BATCH_TMUX:-0}" != "1" ]]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Error: tmux session '${SESSION_NAME}' already exists."
        exit 1
    fi

    RUN_ROOT="${PROJECT_DIR}/output/batch_runs/${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_ROOT}"
    BATCH_LOG="${RUN_ROOT}/batch.log"

    cmd=(
        env
        "INSIDE_BATCH_TMUX=1"
        "BATCH_LOG=${BATCH_LOG}"
        "GSHELL_DATASET_ROOT=${DATASET_ROOT}"
        "GSHELL_CONFIG=${CONFIG_PATH}"
        "GSHELL_OUT_SUFFIX=${OUT_SUFFIX}"
        "SKIP_EXISTING=${SKIP_EXISTING}"
        "$0"
        "${GPU_ID}"
        "${SESSION_NAME}"
    )
    if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
        cmd+=("${REQUESTED_SHOES[@]}")
    fi

    printf -v quoted_cmd '%q ' "${cmd[@]}"
    tmux new-session -d -s "${SESSION_NAME}" "cd '${PROJECT_DIR}' && ${quoted_cmd}"

    echo "Started tmux session: ${SESSION_NAME}"
    echo "Batch log: ${BATCH_LOG}"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

mkdir -p "$(dirname "${BATCH_LOG}")"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[$(timestamp)] Batch training started"
echo "[$(timestamp)] GPU=${GPU_ID}"
echo "[$(timestamp)] DATASET_ROOT=${DATASET_ROOT}"
echo "[$(timestamp)] CONFIG_PATH=${CONFIG_PATH}"
echo "[$(timestamp)] OUT_SUFFIX=${OUT_SUFFIX}"
echo "[$(timestamp)] SKIP_EXISTING=${SKIP_EXISTING}"

if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
    SHOES=("${REQUESTED_SHOES[@]}")
else
    mapfile -t SHOES < <(
        find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
    )
fi

if [[ ${#SHOES[@]} -eq 0 ]]; then
    echo "[$(timestamp)] No shoes found to train."
    exit 1
fi

success_count=0
skip_count=0
fail_count=0

for shoe in "${SHOES[@]}"; do
    dataset_dir="${DATASET_ROOT}/${shoe}"
    out_dir="${PROJECT_DIR}/output/${shoe}${OUT_SUFFIX}"

    if [[ ! -d "${dataset_dir}" ]]; then
        echo "[$(timestamp)] MISSING ${shoe}: dataset not found at ${dataset_dir}"
        fail_count=$((fail_count + 1))
        continue
    fi

    if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/mesh/mesh.obj" && -f "${out_dir}/validate/metrics.txt" ]]; then
        echo "[$(timestamp)] SKIP ${shoe}: existing completed output at ${out_dir}"
        skip_count=$((skip_count + 1))
        continue
    fi

    echo "[$(timestamp)] START ${shoe}"
    if GSHELL_DATASET_ROOT="${DATASET_ROOT}" \
       GSHELL_CONFIG="${CONFIG_PATH}" \
       GSHELL_OUT_SUFFIX="${OUT_SUFFIX}" \
       bash "${TRAIN_SCRIPT}" "${shoe}" "${GPU_ID}"; then
        echo "[$(timestamp)] DONE ${shoe}"
        success_count=$((success_count + 1))
    else
        echo "[$(timestamp)] FAIL ${shoe}"
        fail_count=$((fail_count + 1))
    fi
done

echo "[$(timestamp)] Batch training finished"
echo "[$(timestamp)] success=${success_count} skip=${skip_count} fail=${fail_count}"
