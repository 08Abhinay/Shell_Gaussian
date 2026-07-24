#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${SUGAR_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SUGAR_ROOT="${PROJECT_ROOT}/baselines/SuGaR"
PIPELINE="${PROJECT_ROOT}/dataset_tools_blender/pipeline.py"
SUGAR_PYTHON="${SUGAR_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/SuGaR_env/SuGaR_env/bin/python}"
TOOLS_PYTHON="${SUGAR_TOOLS_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}"
DATA_ROOT="${SUGAR_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender_sugar}"
SOURCE_ROOT="${SUGAR_SOURCE_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender}"
OUTPUT_ROOT="${SUGAR_OUTPUT_ROOT:-${SUGAR_ROOT}/output/golden_set_evaluation_blender_pilot}"
MESH_METRICS_ROOT="${SUGAR_MESH_METRICS_ROOT:-${PROJECT_ROOT}/mesh_metrics/output/evaluations/sugar}"

usage() {
    cat <<'EOF'
Usage:
  train_evaluation_shoe_tmux.sh --shoe NAME --gpu ID [options]

Options:
  --shoe NAME       Reviewed evaluation shoe to prepare and train.
  --gpu ID          Physical GPU index.
  --session NAME    tmux session name.
  --output-root DIR Experiment output root.
  --overwrite-data  Rebuild the derived SuGaR dataset.
  --foreground      Run directly; intended for the permanent batch launcher.
  -h, --help        Show this help.

The resumable pipeline prepares/validates the dataset, trains vanilla 3DGS for
7,000 iterations, trains bounded dn-consistency SuGaR for 15,000 iterations,
extracts a one-million-vertex coarse mesh, refines for 15,000 iterations with
one Gaussian per triangle, exports textured and geometry meshes, evaluates the
30 explicit held-out views, and computes shared similarity-aligned mesh metrics.
EOF
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_stage() {
    echo "[$(timestamp)] RUN: $*"
    "$@"
}

run_pipeline() {
    local shoe="${SUGAR_EVAL_SHOE:?Missing SUGAR_EVAL_SHOE}"
    local physical_gpu="${SUGAR_EVAL_GPU:?Missing SUGAR_EVAL_GPU}"
    local overwrite_data="${SUGAR_EVAL_OVERWRITE_DATA:-0}"
    local scene_parent="${DATA_ROOT}/${shoe}"
    local scene="${scene_parent}/undistorted"
    local run_dir="${OUTPUT_ROOT}/${shoe}"
    local log_dir="${run_dir}/logs"
    local resource_dir="${run_dir}/resources"
    local batch_log="${SUGAR_EVAL_BATCH_LOG:?Missing SUGAR_EVAL_BATCH_LOG}"
    local started
    started="$(date +%s)"

    mkdir -p "${log_dir}" "${resource_dir}" "$(dirname "${batch_log}")"
    exec > >(tee -a "${batch_log}" "${log_dir}/pipeline.log") 2>&1

    export CUDA_VISIBLE_DEVICES="${physical_gpu}"
    export PYTHONUNBUFFERED=1
    export HOME=/storage/Abhinay/home_ab5298
    export XDG_CACHE_HOME="${HOME}/.cache"
    export TORCH_HOME="${XDG_CACHE_HOME}/torch"
    export TMPDIR="${HOME}/tmp"
    export PATH="$(dirname "${SUGAR_PYTHON}"):${PATH}"
    export PYTHONPATH="${SUGAR_ROOT}/gaussian_splatting:${SUGAR_ROOT}:${PYTHONPATH:-}"
    mkdir -p "${TORCH_HOME}" "${TMPDIR}"

    nvidia-smi -i "${physical_gpu}" \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -l 30 \
        >"${resource_dir}/gpu.csv" 2>&1 &
    local monitor_pid="$!"
    trap "kill ${monitor_pid} 2>/dev/null || true" EXIT

    echo "[$(timestamp)] SuGaR evaluation pipeline started"
    echo "Shoe: ${shoe}"
    echo "Physical GPU: ${physical_gpu}"
    echo "Dataset: ${scene_parent}"
    echo "Output: ${run_dir}"
    echo "Contract: 150 train / 30 test, RGBA masks, white background"

    local stale_scene=0
    local prepared_scene=0
    local -a prepare_args
    if [[ ! -f "${scene}/transforms_train.json" ||
          ! -f "${scene}/transforms_test.json" ]]; then
        stale_scene=1
    fi
    if [[ "${overwrite_data}" == "1" || ! -d "${scene_parent}" ||
          "${stale_scene}" == "1" ]]; then
        prepare_args=(prepare-sugar --shoe "${shoe}" --gpu "${physical_gpu}")
        if [[ "${overwrite_data}" == "1" || -d "${scene_parent}" ]]; then
            prepare_args+=(--overwrite)
        fi
        run_stage "${TOOLS_PYTHON}" "${PIPELINE}" "${prepare_args[@]}"
        prepared_scene=1
    fi
    if ! run_stage "${TOOLS_PYTHON}" "${PIPELINE}" validate-sugar --shoe "${shoe}"; then
        if [[ "${prepared_scene}" == "1" ]]; then
            echo "Prepared SuGaR scene still failed validation: ${scene_parent}" >&2
            exit 1
        fi
        echo "[$(timestamp)] Existing SuGaR scene is stale; rebuilding once."
        run_stage "${TOOLS_PYTHON}" "${PIPELINE}" \
            prepare-sugar --shoe "${shoe}" --gpu "${physical_gpu}" --overwrite
        run_stage "${TOOLS_PYTHON}" "${PIPELINE}" validate-sugar --shoe "${shoe}"
    fi

    local gs_dir="${run_dir}/vanilla_3dgs"
    local gs_ply="${gs_dir}/point_cloud/iteration_7000/point_cloud.ply"
    if [[ ! -f "${gs_ply}" ]]; then
        mkdir -p "${gs_dir}"
        (
            cd "${SUGAR_ROOT}/gaussian_splatting"
            run_stage "${SUGAR_PYTHON}" train.py \
                -s "${scene}" \
                -m "${gs_dir}" \
                --iterations 7000 \
                --save_iterations 7000 \
                --test_iterations 7000 \
                --eval \
                -w \
                --port "$((6100 + physical_gpu))"
        )
    else
        echo "[$(timestamp)] SKIP vanilla 3DGS: ${gs_ply}"
    fi
    [[ -f "${gs_ply}" ]] || {
        echo "Missing vanilla 3DGS checkpoint: ${gs_ply}" >&2
        exit 1
    }

    mapfile -t bbox_min < <(
        jq -r '.foreground_bbox.min[]' \
            "${scene_parent}/masked_colmap_manifest.json"
    )
    mapfile -t bbox_max < <(
        jq -r '.foreground_bbox.max[]' \
            "${scene_parent}/masked_colmap_manifest.json"
    )
    [[ "${#bbox_min[@]}" -eq 3 && "${#bbox_max[@]}" -eq 3 ]] || {
        echo "Invalid foreground bounds in dataset manifest." >&2
        exit 1
    }

    local coarse_result="${run_dir}/coarse_pilot_result.json"
    if [[ ! -f "${coarse_result}" ]]; then
        (
            cd "${SUGAR_ROOT}"
            run_stage "${SUGAR_PYTHON}" \
                training_scripts/train_coarse_masked_bounded.py \
                --scene-path "${scene}" \
                --checkpoint-path "${gs_dir}" \
                --output-root "${run_dir}" \
                --bbox-min "${bbox_min[@]}" \
                --bbox-max "${bbox_max[@]}" \
                --gpu 0 \
                --iteration 7000 \
                --max-gaussian-scale-ratio 0.05 \
                --surface-level 0.3 \
                --vertices 1000000
        )
    else
        echo "[$(timestamp)] SKIP coarse SuGaR: ${coarse_result}"
    fi

    local refinement_result="${run_dir}/refinement_run_result.json"
    if [[ ! -f "${refinement_result}" ]]; then
        (
            cd "${SUGAR_ROOT}"
            run_stage "${SUGAR_PYTHON}" \
                training_scripts/refine_masked_bounded_coarse.py \
                --coarse-result "${coarse_result}" \
                --gpu 0 \
                --refinement-iterations 15000 \
                --gaussians-per-triangle 1 \
                --vertices 1000000
        )
    else
        echo "[$(timestamp)] SKIP refinement: ${refinement_result}"
    fi

    local textured_mesh
    textured_mesh="$(jq -r '.refined_mesh_path' "${refinement_result}")"
    [[ -f "${textured_mesh}" ]] || {
        echo "Missing final textured mesh: ${textured_mesh}" >&2
        exit 1
    }
    local geometry_mesh="${run_dir}/final_geometry_mesh.ply"
    if [[ ! -f "${geometry_mesh}" ]]; then
        run_stage "${SUGAR_PYTHON}" - "${textured_mesh}" "${geometry_mesh}" <<'PY'
import sys
import open3d as o3d

source, destination = sys.argv[1:]
mesh = o3d.io.read_triangle_mesh(source)
if not mesh.has_triangles():
    raise RuntimeError(f"Final mesh has no triangles: {source}")
if not o3d.io.write_triangle_mesh(destination, mesh, write_ascii=False):
    raise RuntimeError(f"Could not write geometry mesh: {destination}")
PY
    fi

    (
        cd "${SUGAR_ROOT}"
        run_stage "${SUGAR_PYTHON}" \
            evaluation/eval_masked_blackbg_runs.py \
            --run-dir "${run_dir}" \
            --shoe "${shoe}" \
            --gpu 0 \
            --gs-iteration 7000 \
            --refined-iteration 15000 \
            --gaussians-per-triangle 1 \
            --white-background \
            --summary "${run_dir}/native_image_metrics.tsv"
    )

    local metric_output="${MESH_METRICS_ROOT}/${shoe}"
    if [[ ! -f "${metric_output}/geometry_metrics.json" ]]; then
        (
            cd "${PROJECT_ROOT}"
            run_stage "${TOOLS_PYTHON}" -m mesh_metrics.evaluate_mesh \
                --prediction "${geometry_mesh}" \
                --scene "${SOURCE_ROOT}/${shoe}" \
                --output "${metric_output}" \
                --training-view-set train \
                --save-aligned
        )
    else
        echo "[$(timestamp)] SKIP mesh metrics: ${metric_output}"
    fi

    local completed elapsed
    completed="$(date +%s)"
    elapsed="$((completed - started))"
    "${TOOLS_PYTHON}" - "${run_dir}" "${shoe}" "${physical_gpu}" \
        "${scene}" "${textured_mesh}" "${geometry_mesh}" "${elapsed}" <<'PY'
import json
import sys
from pathlib import Path

run, shoe, gpu, scene, textured, geometry, elapsed = sys.argv[1:]
payload = {
    "protocol": "golden_set_evaluation_blender_sugar_v1",
    "shoe": shoe,
    "physical_gpu": int(gpu),
    "scene": scene,
    "training": {
        "train_views": 150,
        "test_views": 30,
        "vanilla_3dgs_iterations": 7000,
        "coarse_regularization": "dn_consistency",
        "coarse_iterations": 15000,
        "max_gaussian_scale_ratio": 0.05,
        "coarse_mesh_vertices": 1000000,
        "surface_level": 0.3,
        "refinement_iterations": 15000,
        "gaussians_per_triangle": 1,
        "white_background": True,
        "uses_masks": True,
    },
    "textured_mesh": textured,
    "geometry_mesh": geometry,
    "elapsed_seconds": int(elapsed),
    "status": "complete",
}
Path(run, "run_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY
    echo "[$(timestamp)] SuGaR evaluation pipeline complete"
}

if [[ "${SUGAR_EVAL_INSIDE_TMUX:-0}" == "1" ]]; then
    run_pipeline
    exit 0
fi

SHOE=""
GPU=""
SESSION=""
OVERWRITE_DATA=0
FOREGROUND=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --shoe) SHOE="${2:?--shoe requires a name}"; shift 2 ;;
        --gpu) GPU="${2:?--gpu requires an index}"; shift 2 ;;
        --session) SESSION="${2:?--session requires a name}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?--output-root requires a path}"; shift 2 ;;
        --overwrite-data) OVERWRITE_DATA=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${SHOE}" && -n "${GPU}" ]] || {
    usage >&2
    exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU index: ${GPU}" >&2
    exit 2
}
for required in "${SUGAR_PYTHON}" "${TOOLS_PYTHON}"; do
    [[ -x "${required}" ]] || {
        echo "Missing Python executable: ${required}" >&2
        exit 2
    }
done

SESSION="${SESSION:-sugar_${SHOE}_gpu${GPU}}"
RUN_DIR="${OUTPUT_ROOT}/${SHOE}"
BATCH_LOG="${RUN_DIR}/logs/batch.log"
mkdir -p "$(dirname "${BATCH_LOG}")"

if [[ "${FOREGROUND}" == "1" ]]; then
    export SUGAR_EVAL_SHOE="${SHOE}"
    export SUGAR_EVAL_GPU="${GPU}"
    export SUGAR_EVAL_OVERWRITE_DATA="${OVERWRITE_DATA}"
    export SUGAR_EVAL_BATCH_LOG="${BATCH_LOG}"
    run_pipeline
    exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi
cmd=(
    env
    "SUGAR_EVAL_INSIDE_TMUX=1"
    "SUGAR_EVAL_SHOE=${SHOE}"
    "SUGAR_EVAL_GPU=${GPU}"
    "SUGAR_EVAL_OVERWRITE_DATA=${OVERWRITE_DATA}"
    "SUGAR_EVAL_BATCH_LOG=${BATCH_LOG}"
    "SUGAR_PROJECT_ROOT=${PROJECT_ROOT}"
    "SUGAR_DATA_ROOT=${DATA_ROOT}"
    "SUGAR_SOURCE_ROOT=${SOURCE_ROOT}"
    "SUGAR_OUTPUT_ROOT=${OUTPUT_ROOT}"
    "SUGAR_MESH_METRICS_ROOT=${MESH_METRICS_ROOT}"
    "SUGAR_PYTHON=${SUGAR_PYTHON}"
    "SUGAR_TOOLS_PYTHON=${TOOLS_PYTHON}"
    bash
    "${SCRIPT_PATH}"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_root '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION}" "cd ${quoted_root} && ${quoted_cmd}"

echo "Started tmux session: ${SESSION}"
echo "Shoe: ${SHOE}"
echo "Physical GPU: ${GPU}"
echo "Batch log: ${BATCH_LOG}"
echo "Output: ${RUN_DIR}"
