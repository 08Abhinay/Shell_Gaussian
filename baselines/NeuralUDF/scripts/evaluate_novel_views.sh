#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/../.." && pwd)
NEURALUDF_PYTHON=${NEURALUDF_PYTHON:-/home/ab5298/anaconda3/envs/neuraludf/bin/python}
METRICS_PYTHON=${NEURALUDF_METRICS_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}
CONFIG=${NEURALUDF_CONFIG:-$ROOT/confs/udf_shoes_open.conf}
SOURCE_ROOT=${NEURALUDF_SOURCE_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
PREPARED_ROOT=${NEURALUDF_PREPARED_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/neuraludf/golden_set_evaluation}
CHUNK_SIZE=${NEURALUDF_EVAL_CHUNK_SIZE:-1024}
SOURCE_VIEWS=${NEURALUDF_EVAL_SOURCE_VIEWS:-8}

if (( $# < 4 )); then
    echo "Usage: $0 GPU SHOE CHECKPOINT OUTPUT_DIR [renderer extra arguments...]" >&2
    exit 2
fi

GPU=$1
SHOE=$2
CHECKPOINT=$3
OUTPUT_DIR=$4
shift 4

[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }
[[ -x $NEURALUDF_PYTHON ]] || { echo "NeuralUDF Python is missing: $NEURALUDF_PYTHON" >&2; exit 1; }
[[ -x $METRICS_PYTHON ]] || { echo "Metrics Python is missing: $METRICS_PYTHON" >&2; exit 1; }
[[ -f $CHECKPOINT ]] || { echo "Checkpoint is missing: $CHECKPOINT" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "Configuration is missing: $CONFIG" >&2; exit 1; }

cd "$ROOT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$NEURALUDF_PYTHON" \
    evaluation/render_heldout_views.py \
    --shoe "$SHOE" \
    --conf "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --source-scene "$SOURCE_ROOT/$SHOE" \
    --prepared-scene "$PREPARED_ROOT/$SHOE" \
    --output "$OUTPUT_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --source-views "$SOURCE_VIEWS" \
    "$@"

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$METRICS_PYTHON" \
    evaluation/score_heldout_views.py \
    --evaluation "$OUTPUT_DIR" \
    --device cuda \
    --lpips-network vgg
