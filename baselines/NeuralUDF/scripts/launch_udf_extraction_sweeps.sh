#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNNER=$ROOT/scripts/run_udf_extraction_sweep.sh

SHOES=(air_jordan_1 birkenstock_arizona_sandal)
GPUS=(0 1)

for INDEX in 0 1; do
    SHOE=${SHOES[$INDEX]}
    GPU=${GPUS[$INDEX]}
    SESSION=neuraludf_extract_sweep_${SHOE}_gpu${GPU}
    LOG=$ROOT/output/logs/${SHOE}_extraction_sweep.log

    tmux has-session -t "$SESSION" 2>/dev/null && {
        echo "tmux session already exists: $SESSION" >&2
        exit 1
    }
    [[ ! -e $LOG ]] || { echo "Log already exists: $LOG" >&2; exit 1; }

    printf -v COMMAND '%q %q %q 2>&1 | tee %q' "$RUNNER" "$SHOE" "$GPU" "$LOG"
    tmux new-session -d -s "$SESSION" -c "$ROOT" "$COMMAND"
    echo "Started: $SESSION"
    echo "Log: $LOG"
done
