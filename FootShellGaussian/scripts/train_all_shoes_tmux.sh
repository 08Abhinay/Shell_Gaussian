#!/bin/bash
set -euo pipefail

MIN_FREE_MB="${MIN_FREE_MB:-51200}"  # 50 GB minimum free VRAM
SESSION_NAME="${1:-gshell_all_shoes}"
shift $(( $# >= 1 ? 1 : $# ))
REQUESTED_SHOES=("${@+"$@"}")

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_shoe.sh"
DATASET_ROOT="${GSHELL_DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes}"
CONFIG_PATH="${GSHELL_CONFIG:-${PROJECT_DIR}/configs/shoes_mc_normfix.json}"
OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-_normfix}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | while IFS=', ' read -r idx free; do
        if [[ "${free}" -ge "${MIN_FREE_MB}" ]]; then
            echo "${idx}"
        fi
    done
}

# Claim a GPU that is not already used by a running job in this batch.
# BUSY_GPUS is a space-separated string of GPU indices currently in use.
claim_gpu() {
    local busy="$1"
    local gpu
    while IFS= read -r gpu; do
        if [[ -n "${gpu}" && ! " ${busy} " == *" ${gpu} "* ]]; then
            echo "${gpu}"
            return 0
        fi
    done < <(get_free_gpus)
    return 1
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

# ── tmux launcher (outer invocation) ──────────────────────────────────────────
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
        "MIN_FREE_MB=${MIN_FREE_MB}"
        "${SCRIPT_PATH}"
        "${SESSION_NAME}"
    )
    if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
        cmd+=("${REQUESTED_SHOES[@]}")
    fi

    printf -v quoted_cmd '%q ' "${cmd[@]}"
    tmux new-session -d -s "${SESSION_NAME}" "cd '${PROJECT_DIR}' && ${quoted_cmd}"

    echo "Started tmux session: ${SESSION_NAME}"
    echo "Batch log: ${BATCH_LOG}"
    echo "Min free VRAM: ${MIN_FREE_MB} MiB"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

# ── main loop (inside tmux) ──────────────────────────────────────────────────
mkdir -p "$(dirname "${BATCH_LOG}")"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "[$(timestamp)] Batch training started"
echo "[$(timestamp)] MIN_FREE_MB=${MIN_FREE_MB}"
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

echo "[$(timestamp)] Total shoes to process: ${#SHOES[@]}"

free_gpus_now="$(get_free_gpus | tr '\n' ' ')"
echo "[$(timestamp)] GPUs with >=${MIN_FREE_MB} MiB free right now: ${free_gpus_now:-none}"

# Track background jobs: PID -> shoe name, PID -> GPU index
# Use plain strings instead of associative arrays to avoid set -u issues.
PIDS=""          # space-separated list of tracked PIDs
PID_SHOES=""     # |-separated "pid:shoe" entries
PID_GPUS=""      # |-separated "pid:gpu" entries
success_count=0
skip_count=0
fail_count=0

_lookup() { # _lookup "pid:val|pid:val|..." pid
    local entries="$1" key="$2"
    echo "${entries}" | tr '|' '\n' | grep "^${key}:" | head -1 | cut -d: -f2-
}

_remove() { # _remove "pid:val|pid:val|..." pid
    local entries="$1" key="$2"
    echo "${entries}" | tr '|' '\n' | { grep -v "^${key}:" || true; } | paste -sd'|' -
}

track_job() { # track_job pid shoe gpu
    PIDS="${PIDS:+${PIDS} }$1"
    PID_SHOES="${PID_SHOES:+${PID_SHOES}|}$1:$2"
    PID_GPUS="${PID_GPUS:+${PID_GPUS}|}$1:$3"
}

reap_finished() {
    local new_pids=""
    for pid in ${PIDS}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            local shoe gpu
            shoe="$(_lookup "${PID_SHOES}" "${pid}")"
            gpu="$(_lookup "${PID_GPUS}" "${pid}")"
            wait "${pid}" && {
                echo "[$(timestamp)] DONE ${shoe} (GPU ${gpu})"
                success_count=$((success_count + 1))
            } || {
                echo "[$(timestamp)] FAIL ${shoe} (GPU ${gpu})"
                fail_count=$((fail_count + 1))
            }
            PID_SHOES="$(_remove "${PID_SHOES}" "${pid}")"
            PID_GPUS="$(_remove "${PID_GPUS}" "${pid}")"
        else
            new_pids="${new_pids:+${new_pids} }${pid}"
        fi
    done
    PIDS="${new_pids}"
}

busy_gpu_list() {
    echo "${PID_GPUS}" | tr '|' '\n' | cut -d: -f2 | tr '\n' ' '
}

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

    # Wait until a free GPU (not already used by a running job) is available
    while true; do
        reap_finished
        GPU_ID="$(claim_gpu "$(busy_gpu_list)")" && break
        sleep 30
    done

    echo "[$(timestamp)] START ${shoe} on GPU ${GPU_ID}"
    (
        GSHELL_DATASET_ROOT="${DATASET_ROOT}" \
        GSHELL_CONFIG="${CONFIG_PATH}" \
        GSHELL_OUT_SUFFIX="${OUT_SUFFIX}" \
        bash "${TRAIN_SCRIPT}" "${shoe}" "${GPU_ID}"
    ) &
    track_job "$!" "${shoe}" "${GPU_ID}"
done

# Wait for all remaining jobs
while [[ -n "${PIDS}" ]]; do
    reap_finished
    [[ -n "${PIDS}" ]] && sleep 10
done

echo "[$(timestamp)] Batch training finished"
echo "[$(timestamp)] success=${success_count} skip=${skip_count} fail=${fail_count}"
