#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${NEURALUDF_PYTHON:-/home/ab5298/anaconda3/envs/neuraludf/bin/python}
DATA_ROOT=${NEURALUDF_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_neuraludf}

if (( $# < 2 || $# > 4 )); then
    echo "Usage: $0 SHOE GPU [CONFIG] [RESOLUTION]" >&2
    exit 2
fi

SHOE=$1
GPU=$2
CONFIG=${3:-confs/udf_shoes_open.conf}
RESOLUTION=${4:-512}
SEED=${NEURALUDF_SEED:-0}
THRESHOLD=${NEURALUDF_THRESHOLD:-0.005}

[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }
[[ $RESOLUTION =~ ^[1-9][0-9]*$ ]] || { echo "RESOLUTION must be a positive integer" >&2; exit 2; }
[[ -x $PYTHON ]] || { echo "NeuralUDF Python is not executable: $PYTHON" >&2; exit 1; }
[[ -d $DATA_ROOT/$SHOE ]] || { echo "Prepared NeuralUDF scene is missing: $DATA_ROOT/$SHOE" >&2; exit 1; }

if [[ $CONFIG != /* ]]; then
    CONFIG=$ROOT/$CONFIG
fi
[[ -f $CONFIG ]] || { echo "Config is missing: $CONFIG" >&2; exit 1; }

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
exec "$PYTHON" exp_runner_blending.py \
    --conf "$CONFIG" \
    --case "$SHOE" \
    --seed "$SEED" \
    --threshold "$THRESHOLD" \
    --resolution "$RESOLUTION" \
    --reg_weights_schedule
