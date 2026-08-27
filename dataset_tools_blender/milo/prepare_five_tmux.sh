#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE="${PROJECT_ROOT}/dataset_tools_blender/pipeline.py"
TOOLS_PYTHON="/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python"
MILO_PYTHON="/home/ab5298/anaconda3/envs/milo/bin/python"
MILO_SOURCE="/home/ab5298/milo_runtime/source/milo"
COLMAP_BIN="/storage/Abhinay/conda_envs/colmap/bin/colmap"
MANIFEST="${PROJECT_ROOT}/dataset_tools_blender/evaluation_manifest.json"
RAW_ROOT="/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb"
GSHELL_FULL="/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation"
GSHELL_TURNTABLE="/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation_turntable"
SUGAR_FULL="/home/ab5298/dataset/datasets/processed/sugar/golden_set_evaluation"
SUGAR_TURNTABLE="/home/ab5298/dataset/datasets/processed/sugar/golden_set_evaluation_turntable"
MILO_FULL="/home/ab5298/dataset/datasets/processed/milo/golden_set_evaluation"
MILO_TURNTABLE="/home/ab5298/dataset/datasets/processed/milo/golden_set_evaluation_turntable"
DEFAULT_LOG_ROOT="/home/ab5298/Outputs/FootShellGaussian/MILo/dataset_preparation_runs"

SHOES=(
    air_jordan_1
    birkenstock_arizona_sandal
    female_gymnasts_shoes
    red_high_heel_shoes
    sandals_0001
)

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

usage() {
    cat <<'EOF'
Usage: prepare_five_tmux.sh [--gpus 0,1] [--session NAME] [--log-root DIR]

Launch the pilot-first MILo full and turntable dataset workflow in tmux.
The command returns immediately after creating the detached session.
EOF
}

if [[ "${MILO_DATASET_INSIDE_TMUX:-0}" != "1" ]]; then
    GPU_TEXT="0,1"
    SESSION_NAME=""
    LOG_ROOT="${DEFAULT_LOG_ROOT}"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --gpus)
                GPU_TEXT="${2:?--gpus requires two comma-separated IDs}"
                shift 2
                ;;
            --session)
                SESSION_NAME="${2:?--session requires a name}"
                shift 2
                ;;
            --log-root)
                LOG_ROOT="${2:?--log-root requires a directory}"
                shift 2
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

    IFS=',' read -r -a REQUESTED_GPUS <<< "${GPU_TEXT}"
    if [[ "${#REQUESTED_GPUS[@]}" -ne 2 ||
          ! "${REQUESTED_GPUS[0]}" =~ ^[0-9]+$ ||
          ! "${REQUESTED_GPUS[1]}" =~ ^[0-9]+$ ||
          "${REQUESTED_GPUS[0]}" == "${REQUESTED_GPUS[1]}" ]]; then
        echo "--gpus must contain two distinct non-negative GPU IDs." >&2
        exit 2
    fi
    RUN_STAMP="$(date -u +"%Y%m%d_%H%M%S")"
    SESSION_NAME="${SESSION_NAME:-milo_dataset_five_${RUN_STAMP}}"
    LOG_DIR="${LOG_ROOT}/${RUN_STAMP}"
    mkdir -p "${LOG_DIR}"
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "tmux session already exists: ${SESSION_NAME}" >&2
        exit 1
    fi

    printf -v TMUX_COMMAND \
        'env MILO_DATASET_INSIDE_TMUX=1 MILO_DATASET_GPUS=%q MILO_DATASET_LOG_DIR=%q MILO_DATASET_SESSION=%q bash %q' \
        "${GPU_TEXT}" "${LOG_DIR}" "${SESSION_NAME}" "${SCRIPT_PATH}"
    tmux new-session -d -s "${SESSION_NAME}" "${TMUX_COMMAND}"
    echo "SESSION=${SESSION_NAME}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "ATTACH=tmux attach -t ${SESSION_NAME}"
    exit 0
fi

GPU_TEXT="${MILO_DATASET_GPUS:?Missing MILO_DATASET_GPUS}"
LOG_DIR="${MILO_DATASET_LOG_DIR:?Missing MILO_DATASET_LOG_DIR}"
SESSION_NAME="${MILO_DATASET_SESSION:?Missing MILO_DATASET_SESSION}"
IFS=',' read -r -a GPUS <<< "${GPU_TEXT}"
if [[ "${#GPUS[@]}" -ne 2 ]]; then
    echo "The inner workflow requires exactly two GPUs." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/batch.log") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/home/ab5298/milo_runtime/cache/tmp"
export XDG_CACHE_HOME="/home/ab5298/milo_runtime/cache/xdg"
export CUDA_CACHE_PATH="/home/ab5298/milo_runtime/cache/cuda"
export TORCH_EXTENSIONS_DIR="/home/ab5298/milo_runtime/cache/torch_extensions"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}"

checksum_sources() {
    local shoe root
    local roots=("${RAW_ROOT}" "${MILO_SOURCE}")
    for shoe in "${SHOES[@]}"; do
        roots+=(
            "${GSHELL_FULL}/${shoe}"
            "${GSHELL_TURNTABLE}/${shoe}"
            "${SUGAR_FULL}/${shoe}"
            "${SUGAR_TURNTABLE}/${shoe}"
        )
    done
    for root in "${roots[@]}"; do
        [[ -e "${root}" ]] || {
            echo "Missing checksum source: ${root}" >&2
            return 1
        }
    done
    {
        for root in "${roots[@]}"; do
            find "${root}" -type f -print0
        done
    } | LC_ALL=C sort -z | xargs -0 sha256sum
}

write_status() {
    local state="$1"
    local exit_code="$2"
    local sources_equal="$3"
    printf '{\n  "state": "%s",\n  "exit_code": %s,\n  "sources_equal": %s,\n  "session": "%s",\n  "updated_utc": "%s"\n}\n' \
        "${state}" "${exit_code}" "${sources_equal}" \
        "${SESSION_NAME}" "$(timestamp)" >"${LOG_DIR}/status.json"
}

finish() {
    local exit_code="$?"
    trap - EXIT
    set +e
    checksum_sources >"${LOG_DIR}/source_checksums_after.txt"
    local checksum_exit="$?"
    local sources_equal=false
    if [[ "${checksum_exit}" -eq 0 ]] && cmp -s \
        "${LOG_DIR}/source_checksums_before.txt" \
        "${LOG_DIR}/source_checksums_after.txt"; then
        sources_equal=true
    else
        echo "Source checksum verification failed or sources changed." >&2
        exit_code=1
    fi
    if [[ "${exit_code}" -eq 0 ]]; then
        : >"${LOG_DIR}/SUCCESS"
        write_status success 0 "${sources_equal}"
        echo "[$(timestamp)] MILo dataset workflow completed successfully."
    else
        : >"${LOG_DIR}/FAILED"
        write_status failed "${exit_code}" "${sources_equal}"
        echo "[$(timestamp)] MILo dataset workflow failed with exit code ${exit_code}." >&2
    fi
    exit "${exit_code}"
}
trap finish EXIT

require_free_space() {
    local available_kib
    available_kib="$(df -Pk /home/ab5298 | awk 'NR == 2 {print $4}')"
    if [[ -z "${available_kib}" || "${available_kib}" -lt 5242880 ]]; then
        echo "Less than 5 GiB remains under /home/ab5298; stopping." >&2
        return 1
    fi
}

check_gpu() {
    local gpu="$1"
    local free_mib
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if [[ ! "${free_mib}" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ||
          "${free_mib//[[:space:]]/}" -lt 20000 ]]; then
        echo "GPU ${gpu} does not have the required 20 GiB free." >&2
        return 1
    fi
}

run_tools() {
    "${TOOLS_PYTHON}" "${PIPELINE}" "$@"
}

prepare_validate_full() {
    local shoe="$1"
    local gpu="$2"
    require_free_space
    run_tools prepare-milo \
        --shoe "${shoe}" --gpu "${gpu}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_FULL}" --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_FULL}" --colmap-bin "${COLMAP_BIN}"
    run_tools validate-milo \
        --shoe "${shoe}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_FULL}" --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_FULL}"
}

prepare_validate_turntable() {
    local shoe="$1"
    local gpu="$2"
    require_free_space
    run_tools prepare-milo-turntable \
        --shoe "${shoe}" --gpu "${gpu}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_TURNTABLE}" \
        --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_TURNTABLE}" --colmap-bin "${COLMAP_BIN}"
    run_tools validate-milo-turntable \
        --shoe "${shoe}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_TURNTABLE}" \
        --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_TURNTABLE}"
}

loader_smoke() {
    local scene="$1"
    local expected_train="$2"
    local expected_test="$3"
    local ply_before ply_after output
    ply_before="$(sha256sum "${scene}/points3d.ply")"
    output="$({
        cd "${MILO_SOURCE}"
        PYTHONPATH="${MILO_SOURCE}" "${MILO_PYTHON}" - \
            "${scene}" "${expected_train}" "${expected_test}" <<'PY'
import json
import sys
from pathlib import Path

from scene.dataset_readers import readNerfSyntheticInfo

scene = Path(sys.argv[1])
expected_train = int(sys.argv[2])
expected_test = int(sys.argv[3])
if not (scene / "points3d.ply").is_file():
    raise SystemExit("points3d.ply is missing before MILo loader smoke test")
info = readNerfSyntheticInfo(str(scene), white_background=True, eval=True)
if len(info.train_cameras) != expected_train:
    raise SystemExit(
        f"MILo train count {len(info.train_cameras)} != {expected_train}"
    )
if len(info.test_cameras) != expected_test:
    raise SystemExit(
        f"MILo test count {len(info.test_cameras)} != {expected_test}"
    )
if info.point_cloud is None or not len(info.point_cloud.points):
    raise SystemExit("MILo did not load the prepared sparse point cloud")
print(json.dumps({
    "loader": "MILo Blender",
    "scene": scene.name,
    "train_count": len(info.train_cameras),
    "test_count": len(info.test_cameras),
    "sparse_point_count": len(info.point_cloud.points),
}, sort_keys=True))
PY
    } 2>&1)"
    printf '%s\n' "${output}"
    if grep -q "Generating random point cloud" <<<"${output}"; then
        echo "MILo generated random points instead of using points3d.ply." >&2
        return 1
    fi
    ply_after="$(sha256sum "${scene}/points3d.ply")"
    if [[ "${ply_before}" != "${ply_after}" ]]; then
        echo "MILo loader changed points3d.ply." >&2
        return 1
    fi
}

worker() {
    local gpu="$1"
    shift
    local shoe
    for shoe in "$@"; do
        echo "[$(timestamp)] START full ${shoe} on GPU ${gpu}"
        prepare_validate_full "${shoe}" "${gpu}"
        echo "[$(timestamp)] START turntable ${shoe} on GPU ${gpu}"
        prepare_validate_turntable "${shoe}" "${gpu}"
        echo "[$(timestamp)] COMPLETE both variants ${shoe}"
    done
}

echo "[$(timestamp)] MILo five-shoe dataset workflow started."
echo "Session: ${SESSION_NAME}"
echo "GPUs: ${GPUS[*]}"
echo "Full output: ${MILO_FULL}"
echo "Turntable output: ${MILO_TURNTABLE}"
check_gpu "${GPUS[0]}"
check_gpu "${GPUS[1]}"
require_free_space
write_status running 0 false
checksum_sources >"${LOG_DIR}/source_checksums_before.txt"

{
    echo "[$(timestamp)] PILOT full air_jordan_1"
    prepare_validate_full air_jordan_1 "${GPUS[0]}"
    loader_smoke "${MILO_FULL}/air_jordan_1" 150 30
} 2>&1 | tee -a "${LOG_DIR}/pilot_full.log"

{
    echo "[$(timestamp)] PILOT turntable air_jordan_1"
    prepare_validate_turntable air_jordan_1 "${GPUS[0]}"
    loader_smoke "${MILO_TURNTABLE}/air_jordan_1" 30 6
} 2>&1 | tee -a "${LOG_DIR}/pilot_turntable.log"

echo "[$(timestamp)] Both Air Jordan pilots passed; starting remaining shoes."
worker "${GPUS[0]}" \
    birkenstock_arizona_sandal red_high_heel_shoes \
    > >(tee -a "${LOG_DIR}/worker_0_gpu${GPUS[0]}.log") 2>&1 &
worker_0_pid="$!"
worker "${GPUS[1]}" \
    female_gymnasts_shoes sandals_0001 \
    > >(tee -a "${LOG_DIR}/worker_1_gpu${GPUS[1]}.log") 2>&1 &
worker_1_pid="$!"

worker_failed=0
if ! wait "${worker_0_pid}"; then
    worker_failed=1
fi
if ! wait "${worker_1_pid}"; then
    worker_failed=1
fi
if [[ "${worker_failed}" -ne 0 ]]; then
    echo "At least one MILo dataset worker failed." >&2
    exit 1
fi

echo "[$(timestamp)] Running final validation for all ten scenes."
for shoe in "${SHOES[@]}"; do
    run_tools validate-milo \
        --shoe "${shoe}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_FULL}" --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_FULL}"
    run_tools validate-milo-turntable \
        --shoe "${shoe}" \
        --manifest "${MANIFEST}" --source-root "${RAW_ROOT}" \
        --input-root "${GSHELL_TURNTABLE}" \
        --full-input-root "${GSHELL_FULL}" \
        --output-root "${MILO_TURNTABLE}"
done
