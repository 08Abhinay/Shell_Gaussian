#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${NEURALUDF_PYTHON:-/home/ab5298/anaconda3/envs/neuraludf/bin/python}

if (( $# != 2 )); then
    echo "Usage: $0 SHOE GPU" >&2
    exit 2
fi

SHOE=$1
GPU=$2
[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }

FULL_CHECKPOINT=$ROOT/output/open_surface_full/$SHOE/udf_open/checkpoints/ckpt_300000.pth
FT_CHECKPOINT=$ROOT/output/open_surface_finetune/$SHOE/udf_open_ft/checkpoints/ckpt_050000.pth
[[ -s $FULL_CHECKPOINT ]] || { echo "Missing checkpoint: $FULL_CHECKPOINT" >&2; exit 1; }
[[ -s $FT_CHECKPOINT ]] || { echo "Missing checkpoint: $FT_CHECKPOINT" >&2; exit 1; }

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

for RESOLUTION in 128 256; do
    echo "[$(date -u +%FT%TZ)] Extracting $SHOE stage=300k resolution=$RESOLUTION"
    "$PYTHON" exp_runner_blending.py \
        --mode extract_udf_mesh \
        --conf confs/udf_shoes_open.conf \
        --case "$SHOE" \
        --seed 0 \
        --resolution "$RESOLUTION" \
        --init_checkpoint "$FULL_CHECKPOINT"
done

for RESOLUTION in 128 256; do
    echo "[$(date -u +%FT%TZ)] Extracting $SHOE stage=ft50k resolution=$RESOLUTION"
    "$PYTHON" exp_runner_blending.py \
        --mode extract_udf_mesh \
        --conf confs/udf_shoes_open_ft.conf \
        --case "$SHOE" \
        --seed 0 \
        --resolution "$RESOLUTION" \
        --init_checkpoint "$FT_CHECKPOINT"
done

echo "[$(date -u +%FT%TZ)] Extraction sweep completed for $SHOE"
