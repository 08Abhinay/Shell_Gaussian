#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${SUGAR_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PIPELINE="${SUGAR_PREP_PIPELINE:-${PROJECT_ROOT}/dataset_tools_blender/pipeline.py}"
MANIFEST="${SUGAR_PREP_MANIFEST:-${PROJECT_ROOT}/dataset_tools_blender/manifests/baseline_evaluation_manifest.json}"
PYTHON="${SUGAR_PREP_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}"
DEFAULT_LOG_ROOT="${PROJECT_ROOT}/baselines/SuGaR/output/dataset_preparation_runs"

usage() {
    cat <<'EOF'
Usage:
  prepare_evaluation_dataset_tmux.sh --all --gpus 2,3 [options]
  prepare_evaluation_dataset_tmux.sh --shoe NAME [--shoe NAME ...] --gpus 2,3 [options]
  prepare_evaluation_dataset_tmux.sh --shoe-list FILE --gpus 2,3 [options]

Options:
  --all                 Prepare every shoe in baseline_evaluation_manifest.json.
  --shoe NAME           Prepare one shoe; repeat for multiple shoes.
  --shoe-list FILE      Prepare newline-separated shoe names from FILE.
  --gpu ID              Use one physical GPU.
  --gpus ID,ID          Use multiple physical GPUs.
  --session NAME        Override the generated tmux session name.
  --log-root PATH       Override the runtime log root.
  --overwrite           Rebuild existing scenes instead of validating/skipping.
  -h, --help            Show this help.

Each shoe is prepared through dataset_tools_blender/pipeline.py. Work is split
across the requested GPUs, existing valid scenes are skipped, and every selected
scene is validated before the tmux session exits successfully.
EOF
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

parse_gpus() {
    local text="$1"
    read -r -a GPUS <<< "$(printf '%s' "${text}" | tr ',' ' ')"
    if [[ "${#GPUS[@]}" -eq 0 ]]; then
        echo "At least one GPU is required." >&2
        exit 2
    fi
    local gpu
    for gpu in "${GPUS[@]}"; do
        if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "Invalid physical GPU index: ${gpu}" >&2
            exit 2
        fi
    done
}

run_inside_tmux() {
    local shoe_list="${SUGAR_PREP_SHOE_LIST:?Missing SUGAR_PREP_SHOE_LIST}"
    local log_dir="${SUGAR_PREP_LOG_DIR:?Missing SUGAR_PREP_LOG_DIR}"
    local gpu_text="${SUGAR_PREP_GPUS:?Missing SUGAR_PREP_GPUS}"
    local overwrite="${SUGAR_PREP_OVERWRITE:-0}"

    parse_gpus "${gpu_text}"
    mapfile -t SHOES < <(awk 'NF && $1 !~ /^#/ { print $1 }' "${shoe_list}")
    if [[ "${#SHOES[@]}" -eq 0 ]]; then
        echo "No shoes found in ${shoe_list}" >&2
        exit 2
    fi

    mkdir -p "${log_dir}"
    exec > >(tee -a "${log_dir}/batch.log") 2>&1

    echo "[$(timestamp)] SuGaR dataset preparation started"
    echo "[$(timestamp)] Shoes: ${#SHOES[@]}"
    echo "[$(timestamp)] GPUs: ${GPUS[*]}"
    echo "[$(timestamp)] Pipeline: ${PIPELINE}"
    echo "[$(timestamp)] Output: golden_set_evaluation_blender_sugar"
    echo "[$(timestamp)] Overwrite: ${overwrite}"

    local prepare_extra=()
    if [[ "${overwrite}" == "1" ]]; then
        prepare_extra+=(--overwrite)
    fi

    local pids=()
    local worker_index gpu
    for worker_index in "${!GPUS[@]}"; do
        gpu="${GPUS[${worker_index}]}"
        (
            set -euo pipefail
            local i shoe
            for ((i=worker_index; i<${#SHOES[@]}; i+=${#GPUS[@]})); do
                shoe="${SHOES[$i]}"
                echo "[$(timestamp)] START ${shoe} on GPU ${gpu}"
                "${PYTHON}" "${PIPELINE}" prepare-sugar \
                    --shoe "${shoe}" \
                    --gpu "${gpu}" \
                    "${prepare_extra[@]}"
                echo "[$(timestamp)] PREPARED ${shoe} on GPU ${gpu}"
            done
        ) >"${log_dir}/worker_${worker_index}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done

    local failed=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "[$(timestamp)] Preparation failed; inspect worker logs in ${log_dir}" >&2
        exit 1
    fi

    local shoe
    for shoe in "${SHOES[@]}"; do
        echo "[$(timestamp)] VALIDATE ${shoe}"
        "${PYTHON}" "${PIPELINE}" validate-sugar --shoe "${shoe}"
    done
    echo "[$(timestamp)] All selected SuGaR datasets validated"
}

if [[ "${SUGAR_PREP_INSIDE_TMUX:-0}" == "1" ]]; then
    run_inside_tmux
    exit 0
fi

ALL=0
OVERWRITE=0
GPU_TEXT="${SUGAR_PREP_GPUS:-}"
SESSION_NAME=""
LOG_ROOT="${SUGAR_PREP_LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
SHOE_LIST_INPUT=""
SHOES=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --all)
            ALL=1
            shift
            ;;
        --shoe)
            SHOES+=("${2:?--shoe requires a name}")
            shift 2
            ;;
        --shoe-list)
            SHOE_LIST_INPUT="${2:?--shoe-list requires a path}"
            shift 2
            ;;
        --gpu)
            GPU_TEXT="${2:?--gpu requires an index}"
            shift 2
            ;;
        --gpus)
            GPU_TEXT="${2:?--gpus requires indices}"
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

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python executable: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${PIPELINE}" || ! -f "${MANIFEST}" ]]; then
    echo "Missing pipeline or evaluation manifest." >&2
    exit 2
fi
if [[ -z "${GPU_TEXT}" ]]; then
    echo "Pass --gpu ID or --gpus ID,ID." >&2
    exit 2
fi
parse_gpus "${GPU_TEXT}"

selection_modes=0
(( ALL == 1 )) && selection_modes=$((selection_modes + 1))
(( ${#SHOES[@]} > 0 )) && selection_modes=$((selection_modes + 1))
[[ -n "${SHOE_LIST_INPUT}" ]] && selection_modes=$((selection_modes + 1))
if [[ "${selection_modes}" -ne 1 ]]; then
    echo "Choose exactly one of --all, --shoe, or --shoe-list." >&2
    exit 2
fi

if [[ "${ALL}" == "1" ]]; then
    mapfile -t SHOES < <(jq -r '.shoes[].name' "${MANIFEST}")
elif [[ -n "${SHOE_LIST_INPUT}" ]]; then
    if [[ ! -f "${SHOE_LIST_INPUT}" ]]; then
        echo "Missing shoe list: ${SHOE_LIST_INPUT}" >&2
        exit 2
    fi
    mapfile -t SHOES < <(awk 'NF && $1 !~ /^#/ { print $1 }' "${SHOE_LIST_INPUT}")
fi

mapfile -t SHOES < <(printf '%s\n' "${SHOES[@]}" | awk 'NF && !seen[$0]++')
for shoe in "${SHOES[@]}"; do
    if ! jq -e --arg name "${shoe}" '.shoes[] | select(.name == $name)' \
        "${MANIFEST}" >/dev/null; then
        echo "Shoe is not reviewed in the evaluation manifest: ${shoe}" >&2
        exit 2
    fi
done

RUN_ID="${SUGAR_PREP_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
RESOLVED_SHOE_LIST="${LOG_DIR}/shoes.txt"
if [[ -z "${SESSION_NAME}" ]]; then
    SESSION_NAME="sugar_prepare_${RUN_ID}"
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
printf '%s\n' "${SHOES[@]}" >"${RESOLVED_SHOE_LIST}"

cmd=(
    env
    "SUGAR_PREP_INSIDE_TMUX=1"
    "SUGAR_PREP_SHOE_LIST=${RESOLVED_SHOE_LIST}"
    "SUGAR_PREP_LOG_DIR=${LOG_DIR}"
    "SUGAR_PREP_GPUS=${GPUS[*]}"
    "SUGAR_PREP_OVERWRITE=${OVERWRITE}"
    "SUGAR_PROJECT_ROOT=${PROJECT_ROOT}"
    "SUGAR_PREP_PIPELINE=${PIPELINE}"
    "SUGAR_PREP_MANIFEST=${MANIFEST}"
    "SUGAR_PREP_PYTHON=${PYTHON}"
    bash
    "${SCRIPT_PATH}"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_root '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_root} && ${quoted_cmd}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Shoes: ${#SHOES[@]}"
echo "GPUs: ${GPUS[*]}"
echo "Resolved list: ${RESOLVED_SHOE_LIST}"
echo "Batch log: ${LOG_DIR}/batch.log"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
