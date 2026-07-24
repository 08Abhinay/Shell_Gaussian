#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
NEUS2_ROOT="${NEUS2_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_shoe.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_shoe.sh"

SESSION_NAME="${NEUS2_TMUX_SESSION:-neus2_all_shoes}"
SHOE_LIST="${NEUS2_SHOE_LIST:-${SCRIPT_DIR}/shoes.txt}"
N_STEPS="${NEUS2_N_STEPS:-15000}"
MARCHING_CUBES_RES="${NEUS2_MARCHING_CUBES_RES:-512}"
RUN_EVAL="${NEUS2_RUN_EVAL:-0}"
RUN_MESH_METRICS="${NEUS2_RUN_MESH_METRICS:-0}"
RESUME_EXISTING="${NEUS2_RESUME_EXISTING:-0}"
RUN_ID="${NEUS2_RUN_ID:-${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${NEUS2_LOG_DIR:-${NEUS2_ROOT}/output/batch_runs/${RUN_ID}}"
OUTPUT_ROOT="${NEUS2_OUTPUT_ROOT:-${NEUS2_ROOT}/output/golden_set_evaluation_blender_final}"
EXPERIMENT_TAG="${NEUS2_EXPERIMENT_TAG:-}"
DATA_ROOT="${NEUS2_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_neus2}"
SOURCE_DATA_ROOT="${NEUS2_SOURCE_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender}"
MESH_METRICS_ROOT="${NEUS2_MESH_METRICS_ROOT:-/storage/Abhinay/Shell_Gaussian/mesh_metrics/output/evaluations/neus2}"
MESH_METRICS_PYTHON="${NEUS2_MESH_METRICS_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}"
MIN_REFERENCE_MASK_IOU="${NEUS2_MIN_REFERENCE_MASK_IOU:-0.98}"
NEUS2_PYTHON="${NEUS2_PYTHON:-${NEUS2_ENV:-/home/ab5298/anaconda3/envs/neus2}/bin/python}"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

output_dir_for() {
    local shoe_name="$1"
    if [[ -n "${EXPERIMENT_TAG}" ]]; then
        printf '%s/%s/%s\n' "${OUTPUT_ROOT}" "${EXPERIMENT_TAG}" "${shoe_name}"
    else
        printf '%s/%s\n' "${OUTPUT_ROOT}" "${shoe_name}"
    fi
}

load_gpus() {
    if [[ -n "${NEUS2_GPUS:-}" ]]; then
        read -r -a GPUS <<< "$(printf '%s' "${NEUS2_GPUS}" | tr ',' ' ')"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
    else
        GPUS=(0)
    fi
}

load_gpus
if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "No GPUs found. Set NEUS2_GPUS, for example: NEUS2_GPUS=2" >&2
    exit 1
fi
if [[ ! -f "${SHOE_LIST}" ]]; then
    echo "Missing shoe list: ${SHOE_LIST}" >&2
    exit 1
fi

if [[ "${NEUS2_INSIDE_TMUX:-0}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not available on PATH" >&2
        exit 1
    fi
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "tmux session already exists: ${SESSION_NAME}" >&2
        exit 1
    fi

    mkdir -p "${LOG_DIR}"
    cmd=(
        env
        "NEUS2_INSIDE_TMUX=1"
        "NEUS2_ROOT=${NEUS2_ROOT}"
        "NEUS2_SHOE_LIST=${SHOE_LIST}"
        "NEUS2_N_STEPS=${N_STEPS}"
        "NEUS2_MARCHING_CUBES_RES=${MARCHING_CUBES_RES}"
        "NEUS2_GPUS=${GPUS[*]}"
        "NEUS2_RUN_EVAL=${RUN_EVAL}"
        "NEUS2_RUN_MESH_METRICS=${RUN_MESH_METRICS}"
        "NEUS2_RESUME_EXISTING=${RESUME_EXISTING}"
        "NEUS2_RUN_ID=${RUN_ID}"
        "NEUS2_LOG_DIR=${LOG_DIR}"
        "NEUS2_OUTPUT_ROOT=${OUTPUT_ROOT}"
        "NEUS2_EXPERIMENT_TAG=${EXPERIMENT_TAG}"
        "NEUS2_DATA_ROOT=${DATA_ROOT}"
        "NEUS2_SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}"
        "NEUS2_MESH_METRICS_ROOT=${MESH_METRICS_ROOT}"
        "NEUS2_MESH_METRICS_PYTHON=${MESH_METRICS_PYTHON}"
        "NEUS2_MIN_REFERENCE_MASK_IOU=${MIN_REFERENCE_MASK_IOU}"
        "NEUS2_PYTHON=${NEUS2_PYTHON}"
    )
    for name in NEUS2_CONFIG NEUS2_TRAIN_TRANSFORM NEUS2_EVAL_TRANSFORM NEUS2_ENV NEUS2_CACHE_ROOT HF_HOME TORCH_HOME XDG_CACHE_HOME; do
        if [[ -n "${!name:-}" ]]; then
            cmd+=("${name}=${!name}")
        fi
    done
    cmd+=("${SCRIPT_PATH}")

    printf -v quoted_cmd '%q ' "${cmd[@]}"
    printf -v quoted_root '%q' "${NEUS2_ROOT}"
    tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_root} && ${quoted_cmd}"

    echo "Started tmux session: ${SESSION_NAME}"
    echo "Shoes: ${SHOE_LIST}"
    echo "GPUs: ${GPUS[*]}"
    echo "Steps: ${N_STEPS}"
    echo "Marching cubes: ${MARCHING_CUBES_RES}"
    echo "Output root: ${OUTPUT_ROOT}"
    echo "Logs: ${LOG_DIR}"
    exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/batch.log") 2>&1
mapfile -t SHOES < <(awk 'NF && $1 !~ /^#/ { print $1 }' "${SHOE_LIST}")
if [[ "${#SHOES[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] No shoes found in ${SHOE_LIST}" >&2
    exit 1
fi

echo "[$(timestamp)] NeuS2 batch started"
echo "[$(timestamp)] Shoes: ${#SHOES[@]}"
echo "[$(timestamp)] GPUs: ${GPUS[*]}"
echo "[$(timestamp)] Steps: ${N_STEPS}"
echo "[$(timestamp)] Marching cubes: ${MARCHING_CUBES_RES}"
echo "[$(timestamp)] Config: ${NEUS2_CONFIG:-dtu.json}"
echo "[$(timestamp)] Run held-out eval: ${RUN_EVAL}"
echo "[$(timestamp)] Run mesh metrics: ${RUN_MESH_METRICS}"
echo "[$(timestamp)] Resume existing training: ${RESUME_EXISTING}"
echo "[$(timestamp)] Minimum reference mask IoU: ${MIN_REFERENCE_MASK_IOU}"

FAILURE_MARKER="${LOG_DIR}/.failed"
rm -f "${FAILURE_MARKER}"
PIDS=()

for WORKER_IDX in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[${WORKER_IDX}]}"
    (
        set -euo pipefail
        for ((i=WORKER_IDX; i<${#SHOES[@]}; i+=${#GPUS[@]})); do
            if [[ -f "${FAILURE_MARKER}" ]]; then
                echo "[$(timestamp)] STOP worker ${WORKER_IDX}: another job failed"
                exit 1
            fi

            SHOE_NAME="${SHOES[$i]}"
            SHOE_LOG="${LOG_DIR}/${SHOE_NAME}.log"
            GPU_LOG="${LOG_DIR}/${SHOE_NAME}_gpu_memory.csv"
            RESOURCE_LOG="${LOG_DIR}/${SHOE_NAME}_resources.txt"
            OUTPUT_DIR="$(output_dir_for "${SHOE_NAME}")"
            MESH_PATH="${OUTPUT_DIR}/mesh/${N_STEPS}.obj"
            SNAPSHOT_PATH="${OUTPUT_DIR}/checkpoints/${N_STEPS}.msgpack"
            START_EPOCH="$(date +%s)"

            echo "[$(timestamp)] START ${SHOE_NAME} on GPU ${GPU_ID}"
            (
                while true; do
                    printf '%s,' "$(timestamp)"
                    nvidia-smi -i "${GPU_ID}" \
                        --query-gpu=memory.used \
                        --format=csv,noheader,nounits | tr -d ' '
                    sleep 2
                done
            ) > "${GPU_LOG}" &
            MONITOR_PID="$!"

            STATUS=0
            if [[ "${RESUME_EXISTING}" == "1" && -f "${MESH_PATH}" && -f "${SNAPSHOT_PATH}" ]]; then
                echo "[$(timestamp)] RESUME ${SHOE_NAME}: reusing checkpoint and mesh" | tee "${SHOE_LOG}"
                if [[ "${RUN_EVAL}" == "1" ]] && ! bash "${EVAL_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" "${N_STEPS}" >> "${SHOE_LOG}" 2>&1; then
                    STATUS=1
                fi
            else
                if ! bash "${TRAIN_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" "${N_STEPS}" > "${SHOE_LOG}" 2>&1; then
                    STATUS=1
                elif [[ "${RUN_EVAL}" == "1" ]] && ! bash "${EVAL_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}" "${N_STEPS}" >> "${SHOE_LOG}" 2>&1; then
                    STATUS=1
                fi
            fi

            kill "${MONITOR_PID}" 2>/dev/null || true
            wait "${MONITOR_PID}" 2>/dev/null || true
            END_EPOCH="$(date +%s)"
            PEAK_GPU_MB="$(awk -F, 'NF >= 2 && $2 + 0 > max { max = $2 + 0 } END { print max + 0 }' "${GPU_LOG}")"
            {
                echo "shoe=${SHOE_NAME}"
                echo "physical_gpu=${GPU_ID}"
                echo "wall_seconds=$((END_EPOCH - START_EPOCH))"
                echo "peak_gpu_memory_mb=${PEAK_GPU_MB}"
                echo "steps=${N_STEPS}"
                echo "marching_cubes_resolution=${MARCHING_CUBES_RES}"
                echo "config=${NEUS2_CONFIG:-dtu.json}"
            } > "${RESOURCE_LOG}"

            if [[ "${STATUS}" -eq 0 ]]; then
                if ! "${NEUS2_PYTHON}" - "${MESH_PATH}" "${DATA_ROOT}/${SHOE_NAME}/transform_train.json" "${MARCHING_CUBES_RES}" <<'PY' >> "${SHOE_LOG}" 2>&1
import json
import sys

import numpy as np
import trimesh

mesh_path, transform_path, resolution_text = sys.argv[1:]
mesh = trimesh.load(mesh_path, force="mesh", process=False)
if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
    raise SystemExit("NeuS2 mesh is empty")
vertices = np.asarray(mesh.vertices, dtype=np.float64)
if not np.isfinite(vertices).all():
    raise SystemExit("NeuS2 mesh contains non-finite vertices")
with open(transform_path, "r", encoding="utf-8") as handle:
    transforms = json.load(handle)
normalized = vertices * float(transforms["scale"]) + np.asarray(transforms["offset"])
boundary_tolerance = 1.0 / int(resolution_text)
if normalized.min() <= boundary_tolerance or normalized.max() >= 1.0 - boundary_tolerance:
    raise SystemExit(
        "NeuS2 mesh touches the extraction boundary: "
        f"normalized_bounds={np.stack((normalized.min(0), normalized.max(0))).tolist()}"
    )
print(
    "MESH_VALID "
    f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
    f"normalized_bounds={np.stack((normalized.min(0), normalized.max(0))).tolist()}"
)
PY
                then
                    STATUS=1
                fi
            fi

            if [[ "${STATUS}" -eq 0 && "${RUN_MESH_METRICS}" == "1" ]]; then
                METRIC_DIR="${MESH_METRICS_ROOT}/${SHOE_NAME}"
                if ! (
                    cd /storage/Abhinay/Shell_Gaussian
                    "${MESH_METRICS_PYTHON}" -m mesh_metrics.evaluate_mesh \
                        --prediction "${MESH_PATH}" \
                        --scene "${SOURCE_DATA_ROOT}/${SHOE_NAME}" \
                        --output "${METRIC_DIR}" \
                        --training-view-set train \
                        --minimum-reference-mask-iou "${MIN_REFERENCE_MASK_IOU}" \
                        --save-aligned
                ) >> "${SHOE_LOG}" 2>&1; then
                    STATUS=1
                fi
            fi

            if [[ "${STATUS}" -ne 0 ]]; then
                touch "${FAILURE_MARKER}"
                echo "[$(timestamp)] FAIL ${SHOE_NAME} on GPU ${GPU_ID}; see ${SHOE_LOG}"
                exit 1
            fi

            "${NEUS2_PYTHON}" - "${OUTPUT_DIR}" "${SHOE_NAME}" "${NEUS2_CONFIG:-dtu.json}" "${N_STEPS}" "${MARCHING_CUBES_RES}" "${RESOURCE_LOG}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

output, shoe, config, steps, resolution, resource_path = sys.argv[1:]
root = Path("/storage/Abhinay/Shell_Gaussian")
config_path = Path(config)
if not config_path.is_absolute():
    config_path = root / "baselines/NeuS2/configs/nerf" / config_path
resources = {}
for line in Path(resource_path).read_text(encoding="utf-8").splitlines():
    key, value = line.split("=", 1)
    resources[key] = value
payload = {
    "schema_version": 1,
    "shoe": shoe,
    "dataset": os.environ["NEUS2_DATA_ROOT"],
    "config": str(config_path.resolve()),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "source_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "steps": int(steps),
    "marching_cubes_resolution": int(resolution),
    "training_views": 150,
    "heldout_views": 30,
    "heldout_background": "white",
    "resources": resources,
}
path = Path(output) / "experiment_manifest.json"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
            echo "[$(timestamp)] DONE ${SHOE_NAME} on GPU ${GPU_ID}"
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
    echo "[$(timestamp)] NeuS2 queue stopped after a failure. Logs: ${LOG_DIR}" >&2
    exit 1
fi

echo "[$(timestamp)] All NeuS2 jobs finished. Logs: ${LOG_DIR}"
