#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_shoe.sh"

SESSION_NAME="${NEUS2_TMUX_SESSION:-neus2_all_shoes}"
SHOE_LIST="${NEUS2_SHOE_LIST:-${SCRIPT_DIR}/shoes.txt}"
N_STEPS="${NEUS2_N_STEPS:-10000}"
RUN_ID="${NEUS2_RUN_ID:-${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${NEUS2_LOG_DIR:-${NEUS2_ROOT}/output/batch_runs/${RUN_ID}}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

load_gpus() {
    if [[ -n "${NEUS2_GPUS:-}" ]]; then
        read -r -a GPUS <<< "$(printf '%s' "${NEUS2_GPUS}" | tr ',' ' ')"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
    else
        GPUS=(0)
    fi
}

load_gpus

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "No GPUs found. Set NEUS2_GPUS, for example: NEUS2_GPUS=\"0 1 2 3\"" >&2
    exit 1
fi

if [[ ! -f "${SHOE_LIST}" ]]; then
    echo "Missing shoe list: ${SHOE_LIST}" >&2
    exit 1
fi

if [[ "${NEUS2_INSIDE_TMUX:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not available on PATH" >&2
        exit 1
    fi

    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "tmux session already exists: ${SESSION_NAME}"
        echo "Attach with: tmux attach -t ${SESSION_NAME}"
        exit 1
    fi

    mkdir -p "${LOG_DIR}"

    cmd=(
        env
        "NEUS2_INSIDE_TMUX=1"
        "NEUS2_ROOT=${NEUS2_ROOT}"
        "NEUS2_SHOE_LIST=${SHOE_LIST}"
        "NEUS2_N_STEPS=${N_STEPS}"
        "NEUS2_GPUS=${GPUS[*]}"
        "NEUS2_RUN_ID=${RUN_ID}"
        "NEUS2_LOG_DIR=${LOG_DIR}"
    )

    for name in NEUS2_DATA_ROOT NEUS2_CONFIG NEUS2_TRAIN_TRANSFORM NEUS2_ENV NEUS2_CACHE_ROOT HF_HOME TORCH_HOME XDG_CACHE_HOME CONDA_PKGS_DIRS; do
        if [[ -n "${!name:-}" ]]; then
            cmd+=("${name}=${!name}")
        fi
    done
    cmd+=("${SCRIPT_PATH}")

    printf -v quoted_cmd '%q ' "${cmd[@]}"
    printf -v quoted_root '%q' "${NEUS2_ROOT}"
    tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_root} && ${quoted_cmd}"

    echo "Started tmux session: ${SESSION_NAME}"
    echo "Shoes: ${SHOE_LIST}"
    echo "GPUs: ${GPUS[*]}"
    echo "Steps: ${N_STEPS}"
    echo "Logs: ${LOG_DIR}"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/batch.log") 2>&1

mapfile -t SHOES < <(awk 'NF && $1 !~ /^#/ { print $1 }' "${SHOE_LIST}")

if [[ "${#SHOES[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] No shoes found in ${SHOE_LIST}" >&2
    exit 1
fi

echo "[$(timestamp)] NeuS2 batch started"
echo "[$(timestamp)] Shoes: ${#SHOES[@]}"
echo "[$(timestamp)] GPUs: ${GPUS[*]}"
echo "[$(timestamp)] Steps: ${N_STEPS}"
echo "[$(timestamp)] Logs: ${LOG_DIR}"

PIDS=()

for WORKER_IDX in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[${WORKER_IDX}]}"
    (
        set -euo pipefail
        for ((i=WORKER_IDX; i<${#SHOES[@]}; i+=${#GPUS[@]})); do
            SHOE_NAME="${SHOES[$i]}"
            LOG_FILE="${LOG_DIR}/${SHOE_NAME}.log"

            echo "[$(timestamp)] START ${SHOE_NAME} on GPU ${GPU_ID}"
            if bash "${TRAIN_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" "${N_STEPS}" > "${LOG_FILE}" 2>&1; then
                echo "[$(timestamp)] DONE ${SHOE_NAME} on GPU ${GPU_ID}"
            else
                echo "[$(timestamp)] FAIL ${SHOE_NAME} on GPU ${GPU_ID}; see ${LOG_FILE}"
                exit 1
            fi
        done
    ) &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED=1
    fi
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo "[$(timestamp)] One or more NeuS2 jobs failed. Logs are in ${LOG_DIR}" >&2
    exit 1
fi

echo "[$(timestamp)] All NeuS2 jobs finished. Logs are in ${LOG_DIR}"
