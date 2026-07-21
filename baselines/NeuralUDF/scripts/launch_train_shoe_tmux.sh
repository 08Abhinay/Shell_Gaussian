#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_SCRIPT=$ROOT/scripts/train_shoe.sh

if (( $# < 2 || $# > 5 )); then
    echo "Usage: $0 SHOE GPU [RUN_LABEL] [CONFIG] [RESOLUTION]" >&2
    exit 2
fi

SHOE=$1
GPU=$2
RUN_LABEL=${3:-masked_white_300k}
CONFIG=${4:-confs/udf_shoes.conf}
RESOLUTION=${5:-512}
MIN_FREE_MB=${NEURALUDF_MIN_FREE_MB:-20000}

[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }
SAFE_SHOE=$(printf '%s' "$SHOE" | tr -cs '[:alnum:]_-' '_')
SAFE_LABEL=$(printf '%s' "$RUN_LABEL" | tr -cs '[:alnum:]_-' '_')
SESSION=${NEURALUDF_SESSION:-neuraludf_${SAFE_SHOE}_${SAFE_LABEL}_gpu${GPU}}
LOG=${NEURALUDF_LOG:-$ROOT/output/logs/${SHOE}_${RUN_LABEL}.log}

tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "tmux session already exists: $SESSION" >&2
    exit 1
}
[[ ! -e $LOG || ${NEURALUDF_OVERWRITE_LOG:-0} == 1 ]] || {
    echo "Log already exists: $LOG" >&2
    exit 1
}

FREE_MB=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
(( FREE_MB >= MIN_FREE_MB )) || {
    echo "GPU $GPU has ${FREE_MB} MiB free; ${MIN_FREE_MB} MiB is required" >&2
    exit 1
}

mkdir -p "$(dirname "$LOG")"
printf -v COMMAND '%q %q %q %q %q 2>&1 | tee %q' \
    "$TRAIN_SCRIPT" "$SHOE" "$GPU" "$CONFIG" "$RESOLUTION" "$LOG"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$COMMAND"

echo "Started: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"
