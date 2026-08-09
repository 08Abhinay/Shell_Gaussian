#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/../.." && pwd)
TRAIN_SCRIPT=$ROOT/scripts/train_shoe_open.sh
CONFIG=${NEURALUDF_FULL_CONFIG:-$ROOT/confs/udf_shoes_open.conf}
RESOLUTION=${NEURALUDF_FULL_RESOLUTION:-512}
SEED=${NEURALUDF_SEED:-0}
THRESHOLD=${NEURALUDF_THRESHOLD:-0.005}
MIN_STORAGE_MB=${NEURALUDF_MIN_STORAGE_MB:-10240}
GPU_POLL_SECONDS=${NEURALUDF_GPU_POLL_SECONDS:-5}
VALIDATION_PYTHON=${NEURALUDF_VALIDATION_PYTHON:-/home/ab5298/anaconda3/envs/neuraludf/bin/python}
METRICS_PYTHON=${NEURALUDF_METRICS_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}
PIPELINE=$PROJECT_ROOT/dataset_tools_blender/pipeline.py
DATA_ROOT=${NEURALUDF_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/neuraludf/golden_set_evaluation}
SOURCE_SCENE_ROOT=${NEURALUDF_SOURCE_SCENE_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
GROUND_TRUTH_ROOT=${NEURALUDF_GROUND_TRUTH_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
OUTPUT_ROOT=${NEURALUDF_OUTPUT_ROOT:-$ROOT/output/golden_set_evaluation_blender_final}
METRICS_ROOT=${NEURALUDF_METRICS_ROOT:-$PROJECT_ROOT/mesh_metrics/output/evaluations/neuraludf_final}
VALIDATION_COMMAND=${NEURALUDF_VALIDATION_COMMAND:-validate-neuraludf}
NOVEL_VIEW_EVAL_SCRIPT=$ROOT/scripts/evaluate_novel_views.sh
BATCH_DIR=${NEURALUDF_BATCH_DIR:-$OUTPUT_ROOT/batch_runs/manual}
WORKER_ID=${NEURALUDF_WORKER_ID:-0}

if (( $# < 2 )); then
    echo "Usage: $0 GPU SHOE [SHOE ...]" >&2
    exit 2
fi

GPU=$1
shift
[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }
[[ $RESOLUTION == 512 ]] || { echo "Final NeuralUDF extraction resolution must be 512" >&2; exit 2; }
[[ $SEED == 0 ]] || { echo "Final NeuralUDF seed must be 0" >&2; exit 2; }
[[ $THRESHOLD == 0.005 ]] || { echo "Final NeuralUDF threshold must be 0.005" >&2; exit 2; }
[[ -x $VALIDATION_PYTHON ]] || { echo "Validation Python is missing: $VALIDATION_PYTHON" >&2; exit 1; }
[[ -x $METRICS_PYTHON ]] || { echo "Metrics Python is missing: $METRICS_PYTHON" >&2; exit 1; }
[[ -x $TRAIN_SCRIPT ]] || { echo "Training script is not executable: $TRAIN_SCRIPT" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "Configuration is missing: $CONFIG" >&2; exit 1; }
[[ -f $PIPELINE ]] || { echo "Dataset pipeline is missing: $PIPELINE" >&2; exit 1; }

mkdir -p "$BATCH_DIR"

sample_gpu_memory() {
    nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits |
        head -n 1 | tr -dc '0-9'
}

monitor_gpu_memory() {
    local stop_file=$1
    local peak_file=$2
    local peak=0
    local current

    while [[ ! -e $stop_file ]]; do
        current=$(sample_gpu_memory)
        if [[ $current =~ ^[0-9]+$ ]] && (( current > peak )); then
            peak=$current
        fi
        printf '%s\n' "$peak" > "$peak_file"
        sleep "$GPU_POLL_SECONDS"
    done

    current=$(sample_gpu_memory)
    if [[ $current =~ ^[0-9]+$ ]] && (( current > peak )); then
        peak=$current
    fi
    printf '%s\n' "$peak" > "$peak_file"
}

for SHOE in "$@"; do
    PREPARED_SCENE=$DATA_ROOT/$SHOE
    SOURCE_SCENE=$SOURCE_SCENE_ROOT/$SHOE
    GROUND_TRUTH_MESH=$GROUND_TRUTH_ROOT/$SHOE/reference_mesh.ply
    SHOE_ROOT=$OUTPUT_ROOT/$SHOE
    EXP_DIR=$SHOE_ROOT/udf_open
    LOG_DIR=$SHOE_ROOT/logs
    LOG=$LOG_DIR/train.log
    CHECKPOINT=$EXP_DIR/checkpoints/ckpt_300000.pth
    FINAL_MESH=$EXP_DIR/udf_meshes/udf_res512_step300000.ply
    METRIC_DIR=$METRICS_ROOT/$SHOE
    METRIC_JSON=$METRIC_DIR/geometry_metrics.json
    METADATA=$SHOE_ROOT/run_metadata.json

    [[ -d $PREPARED_SCENE ]] || { echo "Prepared scene is missing: $PREPARED_SCENE" >&2; exit 1; }
    [[ -d $SOURCE_SCENE ]] || { echo "Source evaluation scene is missing: $SOURCE_SCENE" >&2; exit 1; }
    [[ -f $GROUND_TRUTH_MESH ]] || { echo "Ground-truth mesh is missing: $GROUND_TRUTH_MESH" >&2; exit 1; }
    if [[ -d $SHOE_ROOT ]] && find "$SHOE_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "Fresh-run output already exists: $SHOE_ROOT" >&2
        exit 1
    fi
    if [[ -d $METRIC_DIR ]] && find "$METRIC_DIR" -mindepth 1 -print -quit | grep -q .; then
        echo "Final metric output already exists: $METRIC_DIR" >&2
        exit 1
    fi

    AVAILABLE_MB=$(df --output=avail -BM "$OUTPUT_ROOT" | tail -n 1 | tr -dc '0-9')
    if (( AVAILABLE_MB < MIN_STORAGE_MB )); then
        echo "Only ${AVAILABLE_MB} MiB remains; ${MIN_STORAGE_MB} MiB is required" >&2
        exit 1
    fi

    echo "[$(date -u +%FT%TZ)] Validating $SHOE"
    "$VALIDATION_PYTHON" "$PIPELINE" "$VALIDATION_COMMAND" --shoe "$SHOE"

    TRAIN_VIEW_COUNT=$(find "$PREPARED_SCENE/image" -maxdepth 1 -type f -name '*.png' | wc -l)

    mkdir -p "$LOG_DIR"
    START_EPOCH=$(date +%s)
    START_UTC=$(date -u +%FT%TZ)
    STOP_FILE=$BATCH_DIR/.gpu_monitor_worker${WORKER_ID}_${SHOE}.stop
    PEAK_FILE=$BATCH_DIR/.gpu_monitor_worker${WORKER_ID}_${SHOE}.peak
    rm -f "$STOP_FILE" "$PEAK_FILE"

    {
        echo "[$START_UTC] Starting $SHOE on physical GPU $GPU"
        echo "Configuration: $CONFIG"
        echo "Contract: ${TRAIN_VIEW_COUNT} RGB views + masks + exact cameras; seed 0; 300000 iterations"
        echo "Final mesh: $FINAL_MESH"
    } | tee "$LOG"

    monitor_gpu_memory "$STOP_FILE" "$PEAK_FILE" &
    MONITOR_PID=$!

    set +e
    NEURALUDF_SEED=$SEED NEURALUDF_THRESHOLD=$THRESHOLD \
        "$TRAIN_SCRIPT" "$SHOE" "$GPU" "$CONFIG" "$RESOLUTION" 2>&1 | tee -a "$LOG"
    TRAIN_STATUS=${PIPESTATUS[0]}
    set -e

    touch "$STOP_FILE"
    wait "$MONITOR_PID"
    PEAK_GPU_MB=$(cat "$PEAK_FILE")
    rm -f "$STOP_FILE" "$PEAK_FILE"

    END_EPOCH=$(date +%s)
    END_UTC=$(date -u +%FT%TZ)
    ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))

    if (( TRAIN_STATUS != 0 )); then
        echo "[$END_UTC] FAIL $SHOE (status $TRAIN_STATUS)" | tee -a "$LOG"
        exit "$TRAIN_STATUS"
    fi

    [[ -s $CHECKPOINT ]] || { echo "Final checkpoint is missing: $CHECKPOINT" >&2; exit 1; }
    [[ -s $FINAL_MESH ]] || { echo "Final MeshUDF surface is missing: $FINAL_MESH" >&2; exit 1; }

    echo "[$(date -u +%FT%TZ)] Rendering held-out views for $SHOE" | tee -a "$LOG"
    NEURALUDF_CONFIG=$CONFIG \
    NEURALUDF_SOURCE_ROOT=$SOURCE_SCENE_ROOT \
    NEURALUDF_PREPARED_ROOT=$DATA_ROOT \
        "$NOVEL_VIEW_EVAL_SCRIPT" "$GPU" "$SHOE" "$CHECKPOINT" \
        "$SHOE_ROOT/heldout_evaluation" | tee -a "$LOG"

    echo "[$(date -u +%FT%TZ)] Computing mesh metrics for $SHOE" | tee -a "$LOG"
    "$METRICS_PYTHON" -m mesh_metrics.evaluate_mesh \
        --prediction "$FINAL_MESH" \
        --scene "$SOURCE_SCENE" \
        --ground-truth "$GROUND_TRUTH_MESH" \
        --output "$METRIC_DIR" \
        --training-view-set train \
        --save-aligned | tee -a "$LOG"

    "$METRICS_PYTHON" - "$METRIC_JSON" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

required = {
    "accuracy_percent",
    "completeness_percent",
    "chamfer_l1_percent",
    "f_score_1_percent",
    "normal_consistency",
    "p95_distance_percent",
}
headline = payload.get("headline", {})
missing = sorted(required.difference(headline))
if missing:
    raise RuntimeError(f"Geometry metrics are incomplete: {missing}")
if not all(math.isfinite(float(headline[key])) for key in required):
    raise RuntimeError("Geometry metrics contain non-finite values")
PY

    CONFIG_SHA256=$(sha256sum "$CONFIG" | awk '{print $1}')
    CAMERA_SHA256=$(sha256sum "$PREPARED_SCENE/cameras_sphere.npz" | awk '{print $1}')
    GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
    if [[ -z $(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal) ]]; then
        GIT_DIRTY=false
    else
        GIT_DIRTY=true
    fi

    SHOE=$SHOE GPU=$GPU START_UTC=$START_UTC END_UTC=$END_UTC \
    ELAPSED_SECONDS=$ELAPSED_SECONDS PEAK_GPU_MB=$PEAK_GPU_MB \
    CONFIG=$CONFIG CONFIG_SHA256=$CONFIG_SHA256 CAMERA_SHA256=$CAMERA_SHA256 \
    GIT_COMMIT=$GIT_COMMIT GIT_DIRTY=$GIT_DIRTY CHECKPOINT=$CHECKPOINT \
    FINAL_MESH=$FINAL_MESH METRIC_JSON=$METRIC_JSON METADATA=$METADATA \
        "$VALIDATION_PYTHON" - <<'PY'
import json
import os

payload = {
    "schema_version": 1,
    "status": "success",
    "method": "NeuralUDF (masked open-surface configuration)",
    "shoe": os.environ["SHOE"],
    "physical_gpu": int(os.environ["GPU"]),
    "seed": 0,
    "iterations": 300000,
    "extraction_resolution": 512,
    "extraction_threshold": 0.005,
    "start_utc": os.environ["START_UTC"],
    "end_utc": os.environ["END_UTC"],
    "elapsed_seconds": int(os.environ["ELAPSED_SECONDS"]),
    "peak_gpu_memory_mib": int(os.environ["PEAK_GPU_MB"]),
    "config": os.environ["CONFIG"],
    "config_sha256": os.environ["CONFIG_SHA256"],
    "camera_archive_sha256": os.environ["CAMERA_SHA256"],
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty": os.environ["GIT_DIRTY"].lower() == "true",
    "checkpoint": os.environ["CHECKPOINT"],
    "final_mesh": os.environ["FINAL_MESH"],
    "geometry_metrics": os.environ["METRIC_JSON"],
}
with open(os.environ["METADATA"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

    echo "[$END_UTC] DONE $SHOE (GPU $GPU, ${ELAPSED_SECONDS}s, peak ${PEAK_GPU_MB} MiB)" |
        tee -a "$LOG"
done
