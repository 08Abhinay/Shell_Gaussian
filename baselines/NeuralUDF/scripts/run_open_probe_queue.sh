#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_SCRIPT=$ROOT/scripts/train_shoe_open.sh
CONFIG=${NEURALUDF_PROBE_CONFIG:-confs/udf_shoes_open_probe.conf}
RESOLUTION=${NEURALUDF_PROBE_RESOLUTION:-256}

if (( $# < 2 )); then
    echo "Usage: $0 GPU SHOE [SHOE ...]" >&2
    exit 2
fi

GPU=$1
shift
[[ $GPU =~ ^[0-9]+$ ]] || { echo "GPU must be a numeric physical GPU index" >&2; exit 2; }

mkdir -p "$ROOT/output/logs"
for SHOE in "$@"; do
    LOG=$ROOT/output/logs/${SHOE}_open_probe_25k.log
    if [[ -e $LOG && ${NEURALUDF_OVERWRITE_LOG:-0} != 1 ]]; then
        echo "Log already exists: $LOG" >&2
        exit 1
    fi

    printf '[%s] Starting %s on physical GPU %s\n' \
        "$(date -u +%FT%TZ)" "$SHOE" "$GPU" | tee "$LOG"

    set +e
    "$TRAIN_SCRIPT" "$SHOE" "$GPU" "$CONFIG" "$RESOLUTION" 2>&1 | tee -a "$LOG"
    STATUS=${PIPESTATUS[0]}
    set -e

    printf '[%s] Finished %s with status %s\n' \
        "$(date -u +%FT%TZ)" "$SHOE" "$STATUS" | tee -a "$LOG"
    if (( STATUS != 0 )); then
        exit "$STATUS"
    fi
done
