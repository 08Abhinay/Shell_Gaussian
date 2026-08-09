#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${TURNTABLE_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE="${TURNTABLE_PIPELINE:-${PROJECT_ROOT}/dataset_tools_blender/pipeline.py}"
PYTHON="${TURNTABLE_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}"
LOG_ROOT="${TURNTABLE_LOG_ROOT:-${PROJECT_ROOT}/dataset_tools_blender/logs/turntable_other_baselines}"
MIN_FREE_MB="${TURNTABLE_MIN_FREE_MB:-20000}"

SHOES=(
    air_jordan_1
    female_gymnasts_shoes
    red_high_heel_shoes
    sandals_0001
    birkenstock_arizona_sandal
)

usage() {
    cat <<'EOF'
Usage:
  prepare_other_baselines_five.sh --gpus ID,ID [options]

Options:
  --gpus ID,ID      Two physical GPUs for SuGaR feature extraction.
  --session NAME    tmux session name (default: turntable_other_baselines_five).
  --log-root PATH   Runtime log root.
  --overwrite       Transactionally replace existing scenes.
  -h, --help        Show this help.
EOF
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

parse_gpus() {
    local text="$1"
    read -r -a GPUS <<< "$(printf '%s' "${text}" | tr ',' ' ')"
    if [[ "${#GPUS[@]}" -ne 2 ]]; then
        echo "Exactly two GPUs are required." >&2
        exit 2
    fi
    if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
        echo "The two GPU indices must be different." >&2
        exit 2
    fi
    local gpu
    for gpu in "${GPUS[@]}"; do
        if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU index: ${gpu}" >&2
            exit 2
        fi
    done
}

check_gpu_memory() {
    local gpu="$1" free_mb
    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" \
            | tr -d ' '
    )"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || (( free_mb < MIN_FREE_MB )); then
        echo "GPU ${gpu} has ${free_mb:-unknown} MiB free; require ${MIN_FREE_MB}." >&2
        exit 2
    fi
}

run_inside_tmux() {
    local gpu_text="${TURNTABLE_GPUS:?Missing TURNTABLE_GPUS}"
    local log_dir="${TURNTABLE_LOG_DIR:?Missing TURNTABLE_LOG_DIR}"
    local overwrite="${TURNTABLE_OVERWRITE:-0}"
    parse_gpus "${gpu_text}"
    mkdir -p "${log_dir}"
    exec > >(tee -a "${log_dir}/batch.log") 2>&1

    local extra=()
    if [[ "${overwrite}" == "1" ]]; then
        extra+=(--overwrite)
    fi

    echo "[$(timestamp)] Turntable preparation started"
    echo "[$(timestamp)] Shoes: ${SHOES[*]}"
    echo "[$(timestamp)] SuGaR GPUs: ${GPUS[*]}"

    (
        set -euo pipefail
        local shoe
        for shoe in "${SHOES[@]}"; do
            echo "[$(timestamp)] G-Shell START ${shoe}"
            "${PYTHON}" "${PIPELINE}" prepare-gshell-turntable \
                --shoe "${shoe}" "${extra[@]}"
            echo "[$(timestamp)] NeuralUDF START ${shoe}"
            "${PYTHON}" "${PIPELINE}" prepare-neuraludf-turntable \
                --shoe "${shoe}" "${extra[@]}"
        done
    ) >"${log_dir}/gshell_neuraludf_worker.log" 2>&1 &
    local adapter_pid=$!

    local sugar_pids=() worker gpu index shoe
    for worker in 0 1; do
        gpu="${GPUS[${worker}]}"
        (
            set -euo pipefail
            for ((index=worker; index<${#SHOES[@]}; index+=2)); do
                shoe="${SHOES[${index}]}"
                echo "[$(timestamp)] SuGaR START ${shoe} on GPU ${gpu}"
                "${PYTHON}" "${PIPELINE}" prepare-sugar-turntable \
                    --shoe "${shoe}" --gpu "${gpu}" "${extra[@]}"
            done
        ) >"${log_dir}/sugar_worker_${worker}_gpu${gpu}.log" 2>&1 &
        sugar_pids+=("$!")
    done

    local failed=0 pid
    if ! wait "${adapter_pid}"; then
        failed=1
    fi
    for pid in "${sugar_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "[$(timestamp)] A preparation worker failed; inspect ${log_dir}." >&2
        exit 1
    fi

    for shoe in "${SHOES[@]}"; do
        "${PYTHON}" "${PIPELINE}" validate-gshell-turntable --shoe "${shoe}"
        "${PYTHON}" "${PIPELINE}" validate-neuraludf-turntable --shoe "${shoe}"
        "${PYTHON}" "${PIPELINE}" validate-sugar-turntable --shoe "${shoe}"
    done
    echo "[$(timestamp)] All 15 turntable scenes validated"
}

if [[ "${TURNTABLE_INSIDE_TMUX:-0}" == "1" ]]; then
    run_inside_tmux
    exit 0
fi

GPU_TEXT=""
SESSION_NAME="turntable_other_baselines_five"
OVERWRITE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_TEXT="${2:?--gpus requires ID,ID}"
            shift 2
            ;;
        --session)
            SESSION_NAME="${2:?--session requires a name}"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="${2:?--log-root requires a path}"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${PYTHON}" || ! -f "${PIPELINE}" ]]; then
    echo "Missing Python environment or centralized pipeline." >&2
    exit 2
fi
if [[ -z "${GPU_TEXT}" ]]; then
    echo "Pass --gpus ID,ID." >&2
    exit 2
fi
parse_gpus "${GPU_TEXT}"
check_gpu_memory "${GPUS[0]}"
check_gpu_memory "${GPUS[1]}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

cmd=(
    env
    "TURNTABLE_INSIDE_TMUX=1"
    "TURNTABLE_GPUS=${GPUS[*]}"
    "TURNTABLE_LOG_DIR=${LOG_DIR}"
    "TURNTABLE_OVERWRITE=${OVERWRITE}"
    "TURNTABLE_PROJECT_ROOT=${PROJECT_ROOT}"
    "TURNTABLE_PIPELINE=${PIPELINE}"
    "TURNTABLE_PYTHON=${PYTHON}"
    bash
    "${SCRIPT_PATH}"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_root '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${quoted_root} && ${quoted_cmd}"

echo "Started tmux session: ${SESSION_NAME}"
echo "SuGaR GPUs: ${GPUS[*]}"
echo "Batch log: ${LOG_DIR}/batch.log"
echo "Worker logs: ${LOG_DIR}"
