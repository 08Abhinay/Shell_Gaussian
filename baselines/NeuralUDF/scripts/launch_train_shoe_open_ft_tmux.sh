#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_SCRIPT=$ROOT/scripts/train_shoe_open_ft.sh

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 SHOE GPU [RUN_LABEL]" >&2
    exit 2
fi

SHOE=$1
GPU=$2
RUN_LABEL=${3:-open_ft_50k}
SAFE_SHOE=$(printf '%s' "$SHOE" | tr -cs '[:alnum:]_-' '_')
SESSION=neuraludf_${SAFE_SHOE}_${RUN_LABEL}_gpu${GPU}
LOG=$ROOT/output/logs/${SHOE}_${RUN_LABEL}.log

tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "tmux session already exists: $SESSION" >&2
    exit 1
}
[[ ! -e $LOG ]] || { echo "Log already exists: $LOG" >&2; exit 1; }
FREE_MB=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
(( FREE_MB >= 20000 )) || { echo "GPU $GPU has only ${FREE_MB} MiB free" >&2; exit 1; }

mkdir -p "$(dirname "$LOG")"
printf -v COMMAND '%q %q %q 2>&1 | tee %q' "$TRAIN_SCRIPT" "$SHOE" "$GPU" "$LOG"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$COMMAND"

echo "Started: $SESSION"
echo "Log: $LOG"
