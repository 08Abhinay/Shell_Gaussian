#!/usr/bin/env bash
set -euo pipefail

# Permanent launcher for the canonical bottom-slab mSDF conditioning experiment.
# Usage:
#   bash configs/run_bottom_slab_training.sh single <SHOE_NAME> [GPU_ID]
#   bash configs/run_bottom_slab_training.sh all [SESSION_NAME] [SHOE_NAME ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
    echo "Usage:"
    echo "  bash configs/run_bottom_slab_training.sh single <SHOE_NAME> [GPU_ID]"
    echo "  bash configs/run_bottom_slab_training.sh all [SESSION_NAME] [SHOE_NAME ...]"
    exit 1
fi
shift

DATASET_ROOT="${GSHELL_DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes_size_metadata}"
CONFIG_PATH="${GSHELL_CONFIG:-${SCRIPT_DIR}/shoes_mc_bottom_slab_512.json}"
OUT_ROOT="${GSHELL_OUT_ROOT:-${PROJECT_DIR}/output/bottom_slab}"
ENV_DIR="${FOOTSHELL_ENV_DIR:-/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env}"
MIN_FREE_MB="${MIN_FREE_MB:-51200}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

usage() {
    echo "Usage:"
    echo "  bash configs/run_bottom_slab_training.sh single <SHOE_NAME> [GPU_ID]"
    echo "  bash configs/run_bottom_slab_training.sh all [SESSION_NAME] [SHOE_NAME ...]"
    echo
    echo "Environment overrides:"
    echo "  GSHELL_DATASET_ROOT=${DATASET_ROOT}"
    echo "  GSHELL_CONFIG=${CONFIG_PATH}"
    echo "  GSHELL_OUT_ROOT=${OUT_ROOT}"
    echo "  FOOTSHELL_ENV_DIR=${ENV_DIR}"
    echo "  MIN_FREE_MB=${MIN_FREE_MB}"
    echo "  SKIP_EXISTING=${SKIP_EXISTING}"
}

activate_env() {
    if [[ ! -d "${ENV_DIR}" ]]; then
        echo "Error: environment not found at ${ENV_DIR}" >&2
        exit 1
    fi

    # Conda activation scripts may read unset variables; keep the launcher strict
    # everywhere else, but relax nounset only around activation.
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_DIR}"
    set -u

    export PATH="${ENV_DIR}/bin:${PATH}"
    export HF_HOME="/data/abelde/.cache/huggingface"
    export TORCH_HOME="/data/abelde/.cache/torch"
    export XDG_CACHE_HOME="/data/abelde/.cache"
    export CONDA_PKGS_DIRS="/data/abelde/.conda/pkgs"

    local conda_gcc="${ENV_DIR}/bin/x86_64-conda-linux-gnu-gcc"
    local conda_gxx="${ENV_DIR}/bin/x86_64-conda-linux-gnu-g++"
    if [[ ! -x "${conda_gcc}" || ! -x "${conda_gxx}" ]]; then
        echo "Error: Conda GCC/G++ wrappers not found in ${ENV_DIR}/bin." >&2
        exit 1
    fi

    export CC="${conda_gcc}"
    export CXX="${conda_gxx}"
    export CUDAHOSTCXX="${conda_gxx}"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ENV_DIR}/lib:${ENV_DIR}/lib64:${LD_LIBRARY_PATH:-}"
}

validate_config_is_clean_bottom_slab() {
    python - "${CONFIG_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r") as f:
    cfg = json.load(f)

errors = []
if cfg.get("use_foot_prior", False):
    errors.append("use_foot_prior must be false for this launcher")
if cfg.get("use_pseudo_last_prior", False):
    errors.append("use_pseudo_last_prior must be false for this launcher")
if not cfg.get("use_bottom_msdf_conditioning", False):
    errors.append("use_bottom_msdf_conditioning must be true for this launcher")

if errors:
    print("Config is not a clean bottom-slab config:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)
PY
}

run_single() {
    local shoe="${1:?Missing shoe name}"
    local gpu_id="${2:-0}"
    local dataset_dir="${DATASET_ROOT}/${shoe}"
    local out_dir="${OUT_ROOT}/${shoe}"
    local log_dir="${out_dir}/logs"

    if [[ ! -d "${dataset_dir}" ]]; then
        echo "Error: dataset not found at ${dataset_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Error: config not found at ${CONFIG_PATH}" >&2
        exit 1
    fi

    validate_config_is_clean_bottom_slab
    mkdir -p "${log_dir}"

    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    activate_env

    cd "${PROJECT_DIR}"

    echo "[$(timestamp)] Training shoe: ${shoe}"
    echo "[$(timestamp)] Dataset: ${dataset_dir}"
    echo "[$(timestamp)] Config: ${CONFIG_PATH}"
    echo "[$(timestamp)] Output: ${out_dir}"
    echo "[$(timestamp)] GPU: ${gpu_id}"
    echo "[$(timestamp)] Env: ${ENV_DIR}"
    echo "[$(timestamp)] Pseudo-last prior: disabled"
    echo "[$(timestamp)] Foot/support prior: disabled"
    echo "[$(timestamp)] Bottom mSDF conditioning: enabled"

    python -u train_gshelltet_polycam.py \
        --config "${CONFIG_PATH}" \
        --trainset_path "${dataset_dir}" \
        --out-dir "${out_dir}" \
        2>&1 | tee "${log_dir}/train.log"
}

get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | while IFS=', ' read -r idx free; do
        if [[ "${free}" -ge "${MIN_FREE_MB}" ]]; then
            echo "${idx}"
        fi
    done
}

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

run_all() {
    local session_name="${1:-bottom_slab_all_shoes}"
    if [[ $# -gt 0 ]]; then
        shift
    fi
    local requested_shoes=("${@}")

    if [[ ! -x "$(command -v tmux)" ]]; then
        echo "Error: tmux is not available on PATH." >&2
        exit 1
    fi
    if [[ ! -d "${DATASET_ROOT}" ]]; then
        echo "Error: dataset root not found at ${DATASET_ROOT}" >&2
        exit 1
    fi
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Error: config not found at ${CONFIG_PATH}" >&2
        exit 1
    fi
    validate_config_is_clean_bottom_slab

    if [[ "${INSIDE_BOTTOM_SLAB_TMUX:-0}" != "1" ]]; then
        if tmux has-session -t "${session_name}" 2>/dev/null; then
            echo "Error: tmux session '${session_name}' already exists." >&2
            exit 1
        fi

        local run_root="${OUT_ROOT}/batch_runs/${session_name}_$(date -u +%Y%m%d_%H%M%S)"
        local batch_log="${run_root}/batch.log"
        mkdir -p "${run_root}"

        local cmd=(
            env
            "INSIDE_BOTTOM_SLAB_TMUX=1"
            "BATCH_LOG=${batch_log}"
            "GSHELL_DATASET_ROOT=${DATASET_ROOT}"
            "GSHELL_CONFIG=${CONFIG_PATH}"
            "GSHELL_OUT_ROOT=${OUT_ROOT}"
            "FOOTSHELL_ENV_DIR=${ENV_DIR}"
            "MIN_FREE_MB=${MIN_FREE_MB}"
            "SKIP_EXISTING=${SKIP_EXISTING}"
            "${SCRIPT_PATH}"
            all
            "${session_name}"
        )
        if [[ ${#requested_shoes[@]} -gt 0 ]]; then
            cmd+=("${requested_shoes[@]}")
        fi

        printf -v quoted_cmd '%q ' "${cmd[@]}"
        tmux new-session -d -s "${session_name}" "cd '${PROJECT_DIR}' && ${quoted_cmd}"

        echo "Started tmux session: ${session_name}"
        echo "Batch log: ${batch_log}"
        echo "Attach with: tmux attach -t ${session_name}"
        return 0
    fi

    mkdir -p "$(dirname "${BATCH_LOG}")"
    exec > >(tee -a "${BATCH_LOG}") 2>&1

    echo "[$(timestamp)] Bottom-slab batch started"
    echo "[$(timestamp)] DATASET_ROOT=${DATASET_ROOT}"
    echo "[$(timestamp)] CONFIG_PATH=${CONFIG_PATH}"
    echo "[$(timestamp)] OUT_ROOT=${OUT_ROOT}"
    echo "[$(timestamp)] ENV_DIR=${ENV_DIR}"
    echo "[$(timestamp)] MIN_FREE_MB=${MIN_FREE_MB}"
    echo "[$(timestamp)] SKIP_EXISTING=${SKIP_EXISTING}"

    local shoes=()
    if [[ ${#requested_shoes[@]} -gt 0 ]]; then
        shoes=("${requested_shoes[@]}")
    else
        mapfile -t shoes < <(find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
    fi

    if [[ ${#shoes[@]} -eq 0 ]]; then
        echo "[$(timestamp)] No shoes found to train."
        exit 1
    fi

    echo "[$(timestamp)] Total shoes to process: ${#shoes[@]}"
    echo "[$(timestamp)] GPUs with >=${MIN_FREE_MB} MiB free: $(get_free_gpus | tr '\n' ' ')"

    local pids=""
    local pid_shoes=""
    local pid_gpus=""
    local success_count=0
    local skip_count=0
    local fail_count=0

    _lookup() {
        local entries="$1" key="$2"
        echo "${entries}" | tr '|' '\n' | grep "^${key}:" | head -1 | cut -d: -f2-
    }

    _remove() {
        local entries="$1" key="$2"
        echo "${entries}" | tr '|' '\n' | { grep -v "^${key}:" || true; } | paste -sd'|' -
    }

    track_job() {
        pids="${pids:+${pids} }$1"
        pid_shoes="${pid_shoes:+${pid_shoes}|}$1:$2"
        pid_gpus="${pid_gpus:+${pid_gpus}|}$1:$3"
    }

    busy_gpu_list() {
        echo "${pid_gpus}" | tr '|' '\n' | cut -d: -f2 | tr '\n' ' '
    }

    reap_finished() {
        local new_pids=""
        local pid shoe gpu
        for pid in ${pids}; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                shoe="$(_lookup "${pid_shoes}" "${pid}")"
                gpu="$(_lookup "${pid_gpus}" "${pid}")"
                if wait "${pid}"; then
                    echo "[$(timestamp)] DONE ${shoe} (GPU ${gpu})"
                    success_count=$((success_count + 1))
                else
                    echo "[$(timestamp)] FAIL ${shoe} (GPU ${gpu})"
                    fail_count=$((fail_count + 1))
                fi
                pid_shoes="$(_remove "${pid_shoes}" "${pid}")"
                pid_gpus="$(_remove "${pid_gpus}" "${pid}")"
            else
                new_pids="${new_pids:+${new_pids} }${pid}"
            fi
        done
        pids="${new_pids}"
    }

    local shoe dataset_dir out_dir gpu_id
    for shoe in "${shoes[@]}"; do
        dataset_dir="${DATASET_ROOT}/${shoe}"
        out_dir="${OUT_ROOT}/${shoe}"

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

        while true; do
            reap_finished
            gpu_id="$(claim_gpu "$(busy_gpu_list)")" && break
            sleep 30
        done

        echo "[$(timestamp)] START ${shoe} on GPU ${gpu_id}"
        (
            GSHELL_DATASET_ROOT="${DATASET_ROOT}" \
            GSHELL_CONFIG="${CONFIG_PATH}" \
            GSHELL_OUT_ROOT="${OUT_ROOT}" \
            FOOTSHELL_ENV_DIR="${ENV_DIR}" \
            bash "${SCRIPT_PATH}" single "${shoe}" "${gpu_id}"
        ) &
        track_job "$!" "${shoe}" "${gpu_id}"
    done

    while [[ -n "${pids}" ]]; do
        reap_finished
        [[ -n "${pids}" ]] && sleep 10
    done

    echo "[$(timestamp)] Bottom-slab batch finished"
    echo "[$(timestamp)] success=${success_count} skip=${skip_count} fail=${fail_count}"
}

case "${MODE}" in
    single)
        if [[ $# -lt 1 ]]; then
            usage
            exit 1
        fi
        run_single "$@"
        ;;
    all)
        run_all "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac
