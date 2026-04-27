#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash bash_scripts/train_shoe.sh <shoe_name_or_scene_path> [gpu_id]

Runs the full SuGaR pipeline for one shoe:
  1. vanilla 3DGS training, unless a checkpoint already exists or SUGAR_SKIP_3DGS=1
  2. coarse SuGaR density/sdf regularization
  3. coarse mesh extraction
  4. mesh-bound SuGaR refinement
  5. UV textured OBJ extraction
  6. quick headless preview PNG/GIF generation

Useful overrides:
  SUGAR_DATA_ROOT=/data/abelde/datasets/raw/golden_set
  SUGAR_OUTPUT_ROOT=<SuGaR>/output/sugar_runs
  SUGAR_GS_ITERATIONS=15000
  SUGAR_GS_RESOLUTION=2
  SUGAR_REGULARIZATION=density
  SUGAR_MESH_VERTICES=200000
  SUGAR_GAUSSIANS_PER_TRIANGLE=6
  SUGAR_REFINEMENT_END_ITER=9000
  SUGAR_GS_OUTPUT_DIR=/path/to/existing/3dgs/output
  SUGAR_SKIP_3DGS=1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
    usage
    exit 0
fi

SHOE_ARG="$1"
GPU_ID="${2:-${SUGAR_GPU:-0}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUGAR_ROOT="${SUGAR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_DIR="${SUGAR_ENV:-${SUGAR_ROOT}/SuGaR_env}"
DATA_ROOT="${SUGAR_DATA_ROOT:-/data/abelde/datasets/raw/golden_set}"
OUTPUT_ROOT="${SUGAR_OUTPUT_ROOT:-${SUGAR_ROOT}/output/sugar_runs}"
CACHE_ROOT="${SUGAR_CACHE_ROOT:-/data/abelde/.cache}"

GS_ITERATIONS="${SUGAR_GS_ITERATIONS:-15000}"
GS_RESOLUTION="${SUGAR_GS_RESOLUTION:-2}"
REGULARIZATION="${SUGAR_REGULARIZATION:-density}"
ESTIMATION_FACTOR="${SUGAR_ESTIMATION_FACTOR:-0.2}"
NORMAL_FACTOR="${SUGAR_NORMAL_FACTOR:-0.2}"
SURFACE_LEVEL="${SUGAR_SURFACE_LEVEL:-0.3}"
MESH_VERTICES="${SUGAR_MESH_VERTICES:-200000}"
GAUSSIANS_PER_TRIANGLE="${SUGAR_GAUSSIANS_PER_TRIANGLE:-6}"
REFINEMENT_END_ITER="${SUGAR_REFINEMENT_END_ITER:-9000}"
SQUARE_SIZE="${SUGAR_SQUARE_SIZE:-8}"
EVAL_SPLIT="${SUGAR_EVAL:-True}"
WHITE_BACKGROUND="${SUGAR_WHITE_BACKGROUND:-False}"
SKIP_EXISTING="${SUGAR_SKIP_EXISTING:-1}"
SKIP_3DGS="${SUGAR_SKIP_3DGS:-0}"
RENDER_PREVIEWS="${SUGAR_RENDER_PREVIEWS:-1}"
PORT_BASE="${SUGAR_PORT_BASE:-6010}"
VIEWER_PORT="$((PORT_BASE + GPU_ID))"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

strip_decimal() {
    local value="$1"
    value="${value//./}"
    value="${value//-/m}"
    printf '%s' "${value}"
}

sanitize_name() {
    printf '%s' "$1" | tr -c '[:alnum:]_.-' '_'
}

resolve_scene() {
    local arg="$1"
    if [[ -d "${arg}" && -d "${arg}/sparse" ]]; then
        SCENE_PATH="$(cd "${arg}" && pwd)"
        SHOE_NAME="$(basename "$(dirname "${SCENE_PATH}")")"
    elif [[ -d "${arg}/undistorted" ]]; then
        SCENE_PATH="$(cd "${arg}/undistorted" && pwd)"
        SHOE_NAME="$(basename "${arg}")"
    else
        SHOE_NAME="${arg}"
        SCENE_PATH="${DATA_ROOT}/${SHOE_NAME}/undistorted"
    fi
}

activate_env() {
    if [[ ! -d "${ENV_DIR}" ]]; then
        echo "Missing SuGaR env: ${ENV_DIR}" >&2
        exit 1
    fi

    export HOME=/data/abelde
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
    export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_ROOT}/nv/ComputeCache}"
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data/abelde/.conda/pkgs}"
    export TMPDIR="${TMPDIR:-/data/abelde/tmp}"
    export CUDA_HOME="${ENV_DIR}"
    export PATH="${ENV_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
    mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${CONDA_PKGS_DIRS}" "${TMPDIR}"
}

run_cmd() {
    echo "[$(timestamp)] $*"
    "$@"
}

resolve_scene "${SHOE_ARG}"
if [[ ! -d "${SCENE_PATH}" ]]; then
    echo "Missing scene directory: ${SCENE_PATH}" >&2
    exit 1
fi

if [[ ! -d "${SCENE_PATH}/sparse" ]]; then
    echo "Scene does not look like a COLMAP dataset: ${SCENE_PATH}" >&2
    exit 1
fi

SHOE_SAFE="$(sanitize_name "${SHOE_NAME}")"
RUN_ID="${SUGAR_RUN_ID:-${SHOE_SAFE}_gs${GS_ITERATIONS}_${REGULARIZATION}_v${MESH_VERTICES}_g${GAUSSIANS_PER_TRIANGLE}_r${REFINEMENT_END_ITER}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
GS_OUTPUT_DIR="${SUGAR_GS_OUTPUT_DIR:-${RUN_DIR}/vanilla_3dgs}"
COARSE_DIR="${RUN_DIR}/coarse"
COARSE_MESH_DIR="${RUN_DIR}/coarse_mesh"
REFINED_DIR="${RUN_DIR}/refined"
REFINED_MESH_DIR="${RUN_DIR}/refined_mesh"
PREVIEW_DIR="${RUN_DIR}/previews"
mkdir -p "${LOG_DIR}" "${COARSE_DIR}" "${COARSE_MESH_DIR}" "${REFINED_DIR}" "${REFINED_MESH_DIR}" "${PREVIEW_DIR}"

exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1
activate_env
cd "${SUGAR_ROOT}"

echo "[$(timestamp)] SuGaR shoe pipeline"
echo "Shoe: ${SHOE_NAME}"
echo "Scene: ${SCENE_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "GPU: ${GPU_ID}"
echo "3DGS output: ${GS_OUTPUT_DIR}"
echo "Regularization: ${REGULARIZATION}"
echo "Mesh vertices: ${MESH_VERTICES}"
echo "Refinement end iteration: ${REFINEMENT_END_ITER}"

GS_PLY="${GS_OUTPUT_DIR}/point_cloud/iteration_${GS_ITERATIONS}/point_cloud.ply"
if [[ "${SKIP_3DGS}" == "1" ]]; then
    echo "[$(timestamp)] SUGAR_SKIP_3DGS=1, using existing 3DGS output."
elif [[ "${SKIP_EXISTING}" == "1" && -f "${GS_PLY}" ]]; then
    echo "[$(timestamp)] Found existing 3DGS checkpoint: ${GS_PLY}"
else
    mkdir -p "${GS_OUTPUT_DIR}"
    GS_ARGS=(
        "${ENV_DIR}/bin/python" "./gaussian_splatting/train.py"
        -s "${SCENE_PATH}"
        -m "${GS_OUTPUT_DIR}"
        --resolution "${GS_RESOLUTION}"
        --iterations "${GS_ITERATIONS}"
        --save_iterations "${GS_ITERATIONS}"
        --test_iterations "${GS_ITERATIONS}"
        --port "${VIEWER_PORT}"
    )
    if [[ "${EVAL_SPLIT}" == "True" || "${EVAL_SPLIT}" == "true" || "${EVAL_SPLIT}" == "1" ]]; then
        GS_ARGS+=(--eval)
    fi
    if [[ "${WHITE_BACKGROUND}" == "True" || "${WHITE_BACKGROUND}" == "true" || "${WHITE_BACKGROUND}" == "1" ]]; then
        GS_ARGS+=(-w)
    fi
    run_cmd "${GS_ARGS[@]}"
fi

if [[ ! -f "${GS_PLY}" ]]; then
    echo "Missing expected 3DGS checkpoint: ${GS_PLY}" >&2
    exit 1
fi

FACTOR_ESTIM="$(strip_decimal "${ESTIMATION_FACTOR}")"
FACTOR_NORMAL="$(strip_decimal "${NORMAL_FACTOR}")"
if [[ "${REGULARIZATION}" == "density" ]]; then
    COARSE_SCRIPT="train_coarse_density.py"
    COARSE_TAG="densityestim${FACTOR_ESTIM}_sdfnorm${FACTOR_NORMAL}"
elif [[ "${REGULARIZATION}" == "sdf" ]]; then
    COARSE_SCRIPT="train_coarse_sdf.py"
    COARSE_TAG="sdfestim${FACTOR_ESTIM}_sdfnorm${FACTOR_NORMAL}"
else
    echo "Unsupported SUGAR_REGULARIZATION=${REGULARIZATION}; use density or sdf for this modular script." >&2
    exit 1
fi

COARSE_MODEL="${COARSE_DIR}/sugarcoarse_3Dgs${GS_ITERATIONS}_${COARSE_TAG}/15000.pt"
if [[ "${SKIP_EXISTING}" == "1" && -f "${COARSE_MODEL}" ]]; then
    echo "[$(timestamp)] Found existing coarse model: ${COARSE_MODEL}"
else
    run_cmd "${ENV_DIR}/bin/python" "${COARSE_SCRIPT}" \
        -s "${SCENE_PATH}" \
        -c "${GS_OUTPUT_DIR}/" \
        -i "${GS_ITERATIONS}" \
        -o "${COARSE_DIR}" \
        -e "${ESTIMATION_FACTOR}" \
        -n "${NORMAL_FACTOR}" \
        --eval "${EVAL_SPLIT}" \
        --white_background "${WHITE_BACKGROUND}" \
        --gpu "${GPU_ID}"
fi

if [[ ! -f "${COARSE_MODEL}" ]]; then
    echo "Missing expected coarse model: ${COARSE_MODEL}" >&2
    exit 1
fi

LEVEL_TAG="$(strip_decimal "${SURFACE_LEVEL}")"
COARSE_MESH="${COARSE_MESH_DIR}/sugarmesh_3Dgs${GS_ITERATIONS}_${COARSE_TAG}_level${LEVEL_TAG}_decim${MESH_VERTICES}.ply"
if [[ "${SKIP_EXISTING}" == "1" && -f "${COARSE_MESH}" ]]; then
    echo "[$(timestamp)] Found existing coarse mesh: ${COARSE_MESH}"
else
    run_cmd "${ENV_DIR}/bin/python" extract_mesh.py \
        -s "${SCENE_PATH}" \
        -c "${GS_OUTPUT_DIR}/" \
        -i "${GS_ITERATIONS}" \
        -m "${COARSE_MODEL}" \
        -l "${SURFACE_LEVEL}" \
        -d "${MESH_VERTICES}" \
        -o "${COARSE_MESH_DIR}" \
        --project_mesh_on_surface_points True \
        --eval "${EVAL_SPLIT}" \
        --gpu "${GPU_ID}"
fi

if [[ ! -f "${COARSE_MESH}" ]]; then
    echo "Missing expected coarse mesh: ${COARSE_MESH}" >&2
    exit 1
fi

NORMAL_CONSISTENCY="${SUGAR_NORMAL_CONSISTENCY:-0.1}"
NORMAL_TAG="$(strip_decimal "${NORMAL_CONSISTENCY}")"
COARSE_MESH_STEM="$(basename "${COARSE_MESH}" .ply)"
REFINED_STEM="sugarfine_${COARSE_MESH_STEM#sugarmesh_}_normalconsistency${NORMAL_TAG}_gaussperface${GAUSSIANS_PER_TRIANGLE}"
REFINED_MODEL="${REFINED_DIR}/${REFINED_STEM}/${REFINEMENT_END_ITER}.pt"
if [[ "${SKIP_EXISTING}" == "1" && -f "${REFINED_MODEL}" ]]; then
    echo "[$(timestamp)] Found existing refined model: ${REFINED_MODEL}"
else
    run_cmd "${ENV_DIR}/bin/python" train_refined.py \
        -s "${SCENE_PATH}" \
        -c "${GS_OUTPUT_DIR}/" \
        -m "${COARSE_MESH}" \
        -o "${REFINED_DIR}" \
        -i "${GS_ITERATIONS}" \
        -n "${NORMAL_CONSISTENCY}" \
        -g "${GAUSSIANS_PER_TRIANGLE}" \
        -v "${MESH_VERTICES}" \
        -f "${REFINEMENT_END_ITER}" \
        --eval "${EVAL_SPLIT}" \
        --white_background "${WHITE_BACKGROUND}" \
        --gpu "${GPU_ID}" \
        --export_ply True
fi

if [[ ! -f "${REFINED_MODEL}" ]]; then
    echo "Missing expected refined model: ${REFINED_MODEL}" >&2
    exit 1
fi

REFINED_OBJ="${REFINED_MESH_DIR}/${REFINED_STEM}.obj"
if [[ "${SKIP_EXISTING}" == "1" && -f "${REFINED_OBJ}" ]]; then
    echo "[$(timestamp)] Found existing textured mesh: ${REFINED_OBJ}"
else
    run_cmd "${ENV_DIR}/bin/python" extract_refined_mesh_with_texture.py \
        -s "${SCENE_PATH}" \
        -c "${GS_OUTPUT_DIR}/" \
        -i "${GS_ITERATIONS}" \
        -m "${REFINED_MODEL}" \
        -o "${REFINED_MESH_DIR}" \
        -n "${GAUSSIANS_PER_TRIANGLE}" \
        --square_size "${SQUARE_SIZE}" \
        --eval "${EVAL_SPLIT}" \
        -g "${GPU_ID}"
fi

if [[ ! -f "${REFINED_OBJ}" ]]; then
    echo "Missing expected textured mesh: ${REFINED_OBJ}" >&2
    exit 1
fi

if [[ "${RENDER_PREVIEWS}" == "1" ]]; then
    run_cmd "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/render_mesh_previews.py" \
        --mesh "${COARSE_MESH}" \
        --out-dir "${PREVIEW_DIR}"
fi

echo "[$(timestamp)] SuGaR pipeline finished"
echo "3DGS PLY: ${GS_PLY}"
echo "Coarse mesh PLY: ${COARSE_MESH}"
echo "Refined textured OBJ: ${REFINED_OBJ}"
echo "Previews: ${PREVIEW_DIR}"
