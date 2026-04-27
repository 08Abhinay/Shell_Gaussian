#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SUGAR_ROOT="${SUGAR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_shoe.sh"

SESSION_NAME="${SUGAR_TMUX_SESSION:-sugar_all_shoes}"
SHOE_LIST="${SUGAR_SHOE_LIST:-${SCRIPT_DIR}/shoes.txt}"
RUN_ID="${SUGAR_BATCH_RUN_ID:-${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${SUGAR_BATCH_LOG_DIR:-${SUGAR_ROOT}/output/sugar_batch_runs/${RUN_ID}}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

load_gpus() {
    if [[ -n "${SUGAR_GPUS:-}" ]]; then
        read -r -a GPUS <<< "$(printf '%s' "${SUGAR_GPUS}" | tr ',' ' ')"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
    else
        GPUS=(0)
    fi
}

load_gpus

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "No GPUs found. Set SUGAR_GPUS, for example: SUGAR_GPUS=\"0 1 2 3\"" >&2
    exit 1
fi

if [[ ! -f "${SHOE_LIST}" ]]; then
    echo "Missing shoe list: ${SHOE_LIST}" >&2
    exit 1
fi

if [[ "${SUGAR_INSIDE_TMUX:-0}" != "1" ]]; then
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
        "SUGAR_INSIDE_TMUX=1"
        "SUGAR_ROOT=${SUGAR_ROOT}"
        "SUGAR_SHOE_LIST=${SHOE_LIST}"
        "SUGAR_GPUS=${GPUS[*]}"
        "SUGAR_BATCH_RUN_ID=${RUN_ID}"
        "SUGAR_BATCH_LOG_DIR=${LOG_DIR}"
    )

    for name in \
        SUGAR_ENV SUGAR_DATA_ROOT SUGAR_OUTPUT_ROOT SUGAR_CACHE_ROOT \
        SUGAR_GS_ITERATIONS SUGAR_GS_RESOLUTION SUGAR_REGULARIZATION SUGAR_SURFACE_LEVEL \
        SUGAR_MESH_VERTICES SUGAR_GAUSSIANS_PER_TRIANGLE SUGAR_REFINEMENT_END_ITER \
        SUGAR_SKIP_3DGS SUGAR_SKIP_EXISTING SUGAR_RENDER_PREVIEWS \
        SUGAR_EVAL SUGAR_WHITE_BACKGROUND SUGAR_SQUARE_SIZE SUGAR_PORT_BASE; do
        if [[ -n "${!name:-}" ]]; then
            cmd+=("${name}=${!name}")
        fi
    done
    cmd+=("bash" "${SCRIPT_PATH}")

    printf -v quoted_cmd '%q ' "${cmd[@]}"
    printf -v quoted_root '%q' "${SUGAR_ROOT}"
    tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_root} && ${quoted_cmd}"

    echo "Started tmux session: ${SESSION_NAME}"
    echo "Shoes: ${SHOE_LIST}"
    echo "GPUs: ${GPUS[*]}"
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

echo "[$(timestamp)] SuGaR batch started"
echo "[$(timestamp)] Shoes: ${#SHOES[@]}"
echo "[$(timestamp)] GPUs: ${GPUS[*]}"
echo "[$(timestamp)] Logs: ${LOG_DIR}"

PIDS=()

for WORKER_IDX in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[${WORKER_IDX}]}"
    (
        set -euo pipefail
        for ((i=WORKER_IDX; i<${#SHOES[@]}; i+=${#GPUS[@]})); do
            SHOE_NAME="${SHOES[$i]}"
            SAFE_NAME="$(printf '%s' "${SHOE_NAME}" | tr -c '[:alnum:]_.-' '_')"
            LOG_FILE="${LOG_DIR}/${SAFE_NAME}.log"

            echo "[$(timestamp)] START ${SHOE_NAME} on GPU ${GPU_ID}"
            if bash "${TRAIN_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" > "${LOG_FILE}" 2>&1; then
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
    echo "[$(timestamp)] One or more SuGaR jobs failed. Logs are in ${LOG_DIR}" >&2
    exit 1
fi

echo "[$(timestamp)] All SuGaR jobs finished. Logs are in ${LOG_DIR}"
