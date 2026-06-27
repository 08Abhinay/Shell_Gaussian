#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FOOTSHELL_ROOT="${PROJECT_ROOT}/FootShellGaussian"
GSHELL_ROOT="${PROJECT_ROOT}/baselines/GShell"
GSHELL_ENV="${GSHELL_ROOT}/GShell_env"
REQUESTED_SHOES=("${@}")

DATASET_ROOT="${DATASET_ROOT:-/data/abelde/datasets/processed/gshell_shoes}"
FINAL_ROOT="${FINAL_ROOT:-${FOOTSHELL_ROOT}/output/final}"
GSHELL_OUTPUT_ROOT="${GSHELL_OUTPUT_ROOT:-${FINAL_ROOT}/gshell}"
ALIGNMENT_ROOT="${ALIGNMENT_ROOT:-${FINAL_ROOT}/foot_aware_alignment}"
PSEUDO_LAST_ROOT="${PSEUDO_LAST_ROOT:-${FINAL_ROOT}/pseudo_last}"

GSHELL_CONFIG="${GSHELL_CONFIG:-${GSHELL_ROOT}/configs/shoes_mc_normfix_512_768.json}"
GSHELL_OUT_SUFFIX="${GSHELL_OUT_SUFFIX:-_turntable}"
MIN_FREE_MB="${MIN_FREE_MB:-51200}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-0}"
ALLOWED_GPUS="${ALLOWED_GPUS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
POST_GPU="${POST_GPU:-0}"
PSEUDO_LAST_EXTRA_ARGS="${PSEUDO_LAST_EXTRA_ARGS:-}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
SHOE_LIST="${SHOE_LIST:-}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

normalize_shoe_name() {
    local raw="$1"
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    printf '%s' "${raw}"
}

build_requested_shoes() {
    local raw item
    if [[ ${#REQUESTED_SHOES[@]} -gt 0 && -n "${SHOE_LIST}" ]]; then
        echo "Set requested shoes either as positional args or SHOE_LIST, not both." >&2
        exit 1
    fi

    if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
        return 0
    fi

    if [[ -z "${SHOE_LIST}" ]]; then
        return 0
    fi

    IFS=',' read -r -a raw_shoes <<< "${SHOE_LIST}"
    REQUESTED_SHOES=()
    for item in "${raw_shoes[@]}"; do
        raw="$(normalize_shoe_name "${item}")"
        if [[ -n "${raw}" ]]; then
            REQUESTED_SHOES+=("${raw}")
        fi
    done
}

build_scene_args() {
    local shoe
    local -n out_ref="$1"
    out_ref=()
    for shoe in "${REQUESTED_SHOES[@]}"; do
        out_ref+=(--scene "${shoe}${GSHELL_OUT_SUFFIX}")
    done
}

build_shoe_args() {
    local shoe
    local -n out_ref="$1"
    out_ref=()
    for shoe in "${REQUESTED_SHOES[@]}"; do
        out_ref+=(--shoe-name "${shoe}")
    done
}

build_requested_shoes

mkdir -p "${FINAL_ROOT}/logs"
PIPELINE_LOG="${PIPELINE_LOG:-${FINAL_ROOT}/logs/final_pipeline_$(date -u +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[$(timestamp)] Final pipeline started"
echo "[$(timestamp)] DATASET_ROOT=${DATASET_ROOT}"
echo "[$(timestamp)] FINAL_ROOT=${FINAL_ROOT}"
echo "[$(timestamp)] GSHELL_OUTPUT_ROOT=${GSHELL_OUTPUT_ROOT}"
echo "[$(timestamp)] ALIGNMENT_ROOT=${ALIGNMENT_ROOT}"
echo "[$(timestamp)] PSEUDO_LAST_ROOT=${PSEUDO_LAST_ROOT}"
echo "[$(timestamp)] GSHELL_CONFIG=${GSHELL_CONFIG}"
echo "[$(timestamp)] GSHELL_OUT_SUFFIX=${GSHELL_OUT_SUFFIX}"
echo "[$(timestamp)] MIN_FREE_MB=${MIN_FREE_MB}"
echo "[$(timestamp)] MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
echo "[$(timestamp)] ALLOWED_GPUS=${ALLOWED_GPUS:-all}"
echo "[$(timestamp)] SKIP_EXISTING=${SKIP_EXISTING}"
echo "[$(timestamp)] POST_GPU=${POST_GPU}"
echo "[$(timestamp)] START_STAGE=${START_STAGE}"
echo "[$(timestamp)] END_STAGE=${END_STAGE}"
if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
    echo "[$(timestamp)] REQUESTED_SHOES=${REQUESTED_SHOES[*]}"
else
    echo "[$(timestamp)] REQUESTED_SHOES=all"
fi

case "${START_STAGE}:${END_STAGE}" in
    1:1|1:2|1:3|1:4|2:2|2:3|2:4|3:3|3:4|4:4) ;;
    *)
        echo "START_STAGE and END_STAGE must define an increasing range within 1..4; got ${START_STAGE}..${END_STAGE}" >&2
        exit 1
        ;;
esac

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Missing DATASET_ROOT: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${GSHELL_CONFIG}" ]]; then
    echo "Missing GSHELL_CONFIG: ${GSHELL_CONFIG}" >&2
    exit 1
fi
if [[ ! -d "${GSHELL_ENV}" ]]; then
    echo "Missing GShell env: ${GSHELL_ENV}" >&2
    exit 1
fi

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
    echo "[$(timestamp)] Stage 1/4: GShell training"
    stage1_cmd=(
        env
        "INSIDE_BATCH_TMUX=1"
        "BATCH_LOG=${FINAL_ROOT}/logs/gshell_batch.log"
        "GSHELL_DATASET_ROOT=${DATASET_ROOT}"
        "GSHELL_CONFIG=${GSHELL_CONFIG}"
        "GSHELL_OUT_SUFFIX=${GSHELL_OUT_SUFFIX}"
        "GSHELL_OUTPUT_ROOT=${GSHELL_OUTPUT_ROOT}"
        "SKIP_EXISTING=${SKIP_EXISTING}"
        "MIN_FREE_MB=${MIN_FREE_MB}"
        "MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
        "ALLOWED_GPUS=${ALLOWED_GPUS}"
        bash "${GSHELL_ROOT}/scripts/train_all_shoes_tmux.sh" final_gshell_batch
    )
    if [[ ${#REQUESTED_SHOES[@]} -gt 0 ]]; then
        stage1_cmd+=("${REQUESTED_SHOES[@]}")
    fi
    "${stage1_cmd[@]}"
else
    echo "[$(timestamp)] Skipping Stage 1/4: GShell training"
fi

echo "[$(timestamp)] Activating GShell env for post-processing"
set +u
eval "$(conda shell.bash hook)"
conda activate "${GSHELL_ENV}"
set -u
export PATH="${GSHELL_ENV}/bin:${PATH}"
export LIBRARY_PATH="${GSHELL_ENV}/lib/stubs:${GSHELL_ENV}/targets/x86_64-linux/lib/stubs:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${GSHELL_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${POST_GPU}"

build_scene_args scene_args
build_shoe_args shoe_args

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
    echo "[$(timestamp)] Stage 2/4: watertight mesh export"
    pushd "${GSHELL_ROOT}" >/dev/null
    stage2_cmd=(
        python "${GSHELL_ROOT}/scripts/export_watertight_meshes.py"
        --output-root "${GSHELL_OUTPUT_ROOT}"
        --config "${GSHELL_CONFIG}"
        --overwrite
    )
    if [[ ${#scene_args[@]} -gt 0 ]]; then
        stage2_cmd+=("${scene_args[@]}")
    fi
    "${stage2_cmd[@]}"
    popd >/dev/null
else
    echo "[$(timestamp)] Skipping Stage 2/4: watertight mesh export"
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
    echo "[$(timestamp)] Stage 3/4: foot-aware alignment"
    stage3_cmd=(
        python "${FOOTSHELL_ROOT}/scripts/run_foot_aware_alignment_pipeline.py"
        --mesh-root "${GSHELL_OUTPUT_ROOT}"
        --output-root "${ALIGNMENT_ROOT}"
        --device cuda
        --overwrite
    )
    if [[ ${#shoe_args[@]} -gt 0 ]]; then
        stage3_cmd+=("${shoe_args[@]}")
    fi
    "${stage3_cmd[@]}"
else
    echo "[$(timestamp)] Skipping Stage 3/4: foot-aware alignment"
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
    echo "[$(timestamp)] Stage 4/4: pseudo-last build"
    # shellcheck disable=SC2086
    stage4_cmd=(
        python "${FOOTSHELL_ROOT}/scripts/run_pseudo_last_builder.py"
        --alignment-root "${ALIGNMENT_ROOT}"
        --output-root "${PSEUDO_LAST_ROOT}"
        --device cuda
        --allow-non-normal
        --overwrite
    )
    if [[ ${#shoe_args[@]} -gt 0 ]]; then
        stage4_cmd+=("${shoe_args[@]}")
    fi
    if [[ -n "${PSEUDO_LAST_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        extra_args=( ${PSEUDO_LAST_EXTRA_ARGS} )
        stage4_cmd+=("${extra_args[@]}")
    fi
    "${stage4_cmd[@]}"
else
    echo "[$(timestamp)] Skipping Stage 4/4: pseudo-last build"
fi

echo "[$(timestamp)] Final pipeline completed"
echo "[$(timestamp)] Log: ${PIPELINE_LOG}"
