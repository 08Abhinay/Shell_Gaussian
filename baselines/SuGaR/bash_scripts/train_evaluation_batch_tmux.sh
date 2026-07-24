#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SHOE_LAUNCHER="${SCRIPT_DIR}/train_evaluation_shoe_tmux.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_LIST="${SCRIPT_DIR}/evaluation_shoes.txt"
DEFAULT_OUTPUT="${PROJECT_ROOT}/baselines/SuGaR/output/golden_set_evaluation_blender_final"

usage() {
    cat <<'EOF'
Usage:
  train_evaluation_batch_tmux.sh --gpus ID,ID [options]

Options:
  --gpus ID,ID       Physical GPUs used by independent sequential workers.
  --shoe-list FILE   Shoe list; defaults to evaluation_shoes.txt.
  --session NAME     tmux session; defaults to sugar_final_five.
  --output-root DIR  Output root for per-shoe runs.
  --retries N        Retry each failed shoe N times; defaults to 1.
  --overwrite-data   Rebuild each derived SuGaR scene before training.
EOF
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_batch() {
    local list="${SUGAR_BATCH_SHOE_LIST:?}"
    local output="${SUGAR_BATCH_OUTPUT_ROOT:?}"
    local log_dir="${SUGAR_BATCH_LOG_DIR:?}"
    local overwrite="${SUGAR_BATCH_OVERWRITE_DATA:-0}"
    local gpu_text="${SUGAR_BATCH_GPUS:?}"
    local retries="${SUGAR_BATCH_RETRIES:-1}"
    local status_file="${log_dir}/status.tsv"
    local gpus=()
    read -r -a gpus <<< "$(tr ',' ' ' <<<"${gpu_text}")"
    mapfile -t shoes < <(awk 'NF && $1 !~ /^#/ {print $1}' "${list}")
    mkdir -p "${log_dir}"
    printf 'timestamp\tshoe\tgpu\tstatus\tattempt\n' >"${status_file}"
    exec > >(tee -a "${log_dir}/batch.log") 2>&1

    echo "[$(timestamp)] SuGaR evaluation batch started"
    echo "Shoes: ${shoes[*]}"
    echo "Physical GPUs: ${gpus[*]}"
    echo "Retries per failed shoe: ${retries}"
    echo "Output: ${output}"

    local pids=()
    local worker gpu
    for worker in "${!gpus[@]}"; do
        gpu="${gpus[$worker]}"
        (
            set -uo pipefail
            local index shoe final_failed=0
            for ((index=worker; index<${#shoes[@]}; index+=${#gpus[@]})); do
                shoe="${shoes[$index]}"
                args=(
                    --shoe "${shoe}"
                    --gpu "${gpu}"
                    --output-root "${output}"
                    --foreground
                )
                if [[ "${overwrite}" == "1" ]]; then
                    args+=(--overwrite-data)
                fi
                local attempt success=0 max_attempts
                max_attempts="$((retries + 1))"
                for ((attempt=1; attempt<=max_attempts; attempt++)); do
                    echo "[$(timestamp)] START ${shoe} on GPU ${gpu} (attempt ${attempt}/${max_attempts})"
                    if bash "${SHOE_LAUNCHER}" "${args[@]}"; then
                        success=1
                        printf '%s\t%s\t%s\tcomplete\t%s\n' \
                            "$(timestamp)" "${shoe}" "${gpu}" "${attempt}" >>"${status_file}"
                        echo "[$(timestamp)] COMPLETE ${shoe} on GPU ${gpu}"
                        break
                    fi
                    if [[ "${attempt}" -lt "${max_attempts}" ]]; then
                        printf '%s\t%s\t%s\tretrying\t%s\n' \
                            "$(timestamp)" "${shoe}" "${gpu}" "${attempt}" >>"${status_file}"
                        echo "[$(timestamp)] RETRY ${shoe} on GPU ${gpu}"
                    fi
                done
                if [[ "${success}" != "1" ]]; then
                    final_failed=1
                    printf '%s\t%s\t%s\tfailed\t%s\n' \
                        "$(timestamp)" "${shoe}" "${gpu}" "${max_attempts}" >>"${status_file}"
                    echo "[$(timestamp)] FAILED ${shoe} on GPU ${gpu}; continuing queue." >&2
                fi
            done
            exit "${final_failed}"
        ) >"${log_dir}/worker_${worker}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done

    local failed=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" != "0" ]]; then
        echo "[$(timestamp)] All shoes were attempted, but one or more failed." >&2
        echo "Status report: ${status_file}" >&2
        exit 1
    fi
    echo "[$(timestamp)] SuGaR evaluation batch complete"
    echo "Status report: ${status_file}"
}

if [[ "${SUGAR_BATCH_INSIDE_TMUX:-0}" == "1" ]]; then
    run_batch
    exit 0
fi

GPUS=""
SHOE_LIST="${DEFAULT_LIST}"
SESSION="sugar_final_five"
OUTPUT_ROOT="${DEFAULT_OUTPUT}"
OVERWRITE_DATA=0
RETRIES=1
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="${2:?--gpus requires indices}"; shift 2 ;;
        --shoe-list) SHOE_LIST="${2:?--shoe-list requires a path}"; shift 2 ;;
        --session) SESSION="${2:?--session requires a name}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?--output-root requires a path}"; shift 2 ;;
        --retries) RETRIES="${2:?--retries requires a count}"; shift 2 ;;
        --overwrite-data) OVERWRITE_DATA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${GPUS}" ]] || {
    echo "--gpus is required." >&2
    exit 2
}
[[ "${RETRIES}" =~ ^[0-9]+$ ]] || {
    echo "--retries must be a nonnegative integer." >&2
    exit 2
}
[[ -f "${SHOE_LIST}" && -x "${SHOE_LAUNCHER}" ]] || {
    echo "Missing shoe list or single-shoe launcher." >&2
    exit 2
}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

LOG_DIR="${OUTPUT_ROOT}/batch_runs/${SESSION}"
mkdir -p "${LOG_DIR}"
cp "${SHOE_LIST}" "${LOG_DIR}/shoes.txt"
cmd=(
    env
    "SUGAR_BATCH_INSIDE_TMUX=1"
    "SUGAR_BATCH_SHOE_LIST=${LOG_DIR}/shoes.txt"
    "SUGAR_BATCH_OUTPUT_ROOT=${OUTPUT_ROOT}"
    "SUGAR_BATCH_LOG_DIR=${LOG_DIR}"
    "SUGAR_BATCH_OVERWRITE_DATA=${OVERWRITE_DATA}"
    "SUGAR_BATCH_GPUS=${GPUS}"
    "SUGAR_BATCH_RETRIES=${RETRIES}"
    bash
    "${SCRIPT_PATH}"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_root '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION}" "cd ${quoted_root} && ${quoted_cmd}"

echo "Started tmux session: ${SESSION}"
echo "GPUs: ${GPUS}"
echo "Batch log: ${LOG_DIR}/batch.log"
echo "Worker logs: ${LOG_DIR}/worker_*"
echo "Output: ${OUTPUT_ROOT}"
