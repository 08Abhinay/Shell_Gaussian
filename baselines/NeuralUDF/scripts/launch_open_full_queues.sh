#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/../.." && pwd)
SCRIPT=$ROOT/scripts/launch_open_full_queues.sh
QUEUE_SCRIPT=$ROOT/scripts/run_open_full_queue.sh
DEFAULT_SHOE_LIST=$ROOT/scripts/evaluation_shoes.txt
DEFAULT_OUTPUT_ROOT=$ROOT/output/golden_set_evaluation_blender_final
MIN_FREE_MB=${NEURALUDF_MIN_FREE_MB:-20000}
MIN_STORAGE_MB=${NEURALUDF_MIN_STORAGE_MB:-10240}

usage() {
    cat <<EOF
Usage:
  $0 --gpus ID,ID --shoe-list PATH --session NAME

Options:
  --gpus       Comma-separated physical GPU indices.
  --shoe-list  Text file containing one shoe name per line.
  --session    tmux session name.
EOF
}

INTERNAL_RUN=0
GPUS_CSV=
SHOE_LIST=$DEFAULT_SHOE_LIST
SESSION=

while (( $# > 0 )); do
    case "$1" in
        --gpus)
            GPUS_CSV=${2:?Missing value for --gpus}
            shift 2
            ;;
        --shoe-list)
            SHOE_LIST=${2:?Missing value for --shoe-list}
            shift 2
            ;;
        --session)
            SESSION=${2:?Missing value for --session}
            shift 2
            ;;
        --internal-run)
            INTERNAL_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n $GPUS_CSV ]] || { echo "--gpus is required" >&2; exit 2; }
[[ -n $SESSION ]] || { echo "--session is required" >&2; exit 2; }
[[ -f $SHOE_LIST ]] || { echo "Shoe list is missing: $SHOE_LIST" >&2; exit 1; }
[[ -x $QUEUE_SCRIPT ]] || { echo "Queue script is not executable: $QUEUE_SCRIPT" >&2; exit 1; }

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
(( ${#GPUS[@]} > 0 )) || { echo "At least one GPU is required" >&2; exit 2; }
declare -A SEEN_GPUS=()
for GPU in "${GPUS[@]}"; do
    [[ $GPU =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $GPU" >&2; exit 2; }
    [[ -z ${SEEN_GPUS[$GPU]:-} ]] || { echo "Duplicate GPU index: $GPU" >&2; exit 2; }
    SEEN_GPUS[$GPU]=1
done

mapfile -t SHOES < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$SHOE_LIST")
(( ${#SHOES[@]} > 0 )) || { echo "Shoe list is empty: $SHOE_LIST" >&2; exit 2; }

OUTPUT_ROOT=${NEURALUDF_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}
DATA_ROOT=${NEURALUDF_DATA_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/neuraludf/golden_set_evaluation}
SOURCE_SCENE_ROOT=${NEURALUDF_SOURCE_SCENE_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
GROUND_TRUTH_ROOT=${NEURALUDF_GROUND_TRUTH_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
METRICS_ROOT=${NEURALUDF_METRICS_ROOT:-$PROJECT_ROOT/mesh_metrics/output/evaluations/neuraludf_final}
VALIDATION_COMMAND=${NEURALUDF_VALIDATION_COMMAND:-validate-neuraludf}
BATCH_DIR=$OUTPUT_ROOT/batch_runs/$SESSION
BATCH_LOG=$BATCH_DIR/batch.log

if (( INTERNAL_RUN )); then
    mkdir -p "$BATCH_DIR"
    exec > >(tee -a "$BATCH_LOG") 2>&1

    echo "[$(date -u +%FT%TZ)] NeuralUDF final batch started"
    echo "Shoes: ${SHOES[*]}"
    echo "Physical GPUs: ${GPUS[*]}"
    echo "Output: $OUTPUT_ROOT"

    declare -a WORKER_PIDS=()
    declare -a WORKER_LABELS=()
    for INDEX in "${!GPUS[@]}"; do
        GPU=${GPUS[$INDEX]}
        ASSIGNED=()
        for SHOE_INDEX in "${!SHOES[@]}"; do
            if (( SHOE_INDEX % ${#GPUS[@]} == INDEX )); then
                ASSIGNED+=("${SHOES[$SHOE_INDEX]}")
            fi
        done
        (( ${#ASSIGNED[@]} > 0 )) || continue

        WORKER_LOG=$BATCH_DIR/worker_${INDEX}_gpu${GPU}.log
        echo "Worker $INDEX -> GPU $GPU: ${ASSIGNED[*]}"
        (
            set -o pipefail
            NEURALUDF_BATCH_DIR=$BATCH_DIR \
            NEURALUDF_WORKER_ID=$INDEX \
            NEURALUDF_DATA_ROOT=$DATA_ROOT \
            NEURALUDF_SOURCE_SCENE_ROOT=$SOURCE_SCENE_ROOT \
            NEURALUDF_GROUND_TRUTH_ROOT=$GROUND_TRUTH_ROOT \
            NEURALUDF_OUTPUT_ROOT=$OUTPUT_ROOT \
            NEURALUDF_METRICS_ROOT=$METRICS_ROOT \
            NEURALUDF_VALIDATION_COMMAND=$VALIDATION_COMMAND \
                "$QUEUE_SCRIPT" "$GPU" "${ASSIGNED[@]}" 2>&1 | tee "$WORKER_LOG"
            exit "${PIPESTATUS[0]}"
        ) &
        WORKER_PIDS+=("$!")
        WORKER_LABELS+=("worker_${INDEX}_gpu${GPU}")
    done

    FINAL_STATUS=0
    for INDEX in "${!WORKER_PIDS[@]}"; do
        set +e
        wait "${WORKER_PIDS[$INDEX]}"
        STATUS=$?
        set -e
        echo "[$(date -u +%FT%TZ)] ${WORKER_LABELS[$INDEX]} status=$STATUS"
        if (( STATUS != 0 )); then
            FINAL_STATUS=1
        fi
    done

    if (( FINAL_STATUS == 0 )); then
        echo "[$(date -u +%FT%TZ)] NeuralUDF final batch completed successfully"
    else
        echo "[$(date -u +%FT%TZ)] NeuralUDF final batch failed"
    fi
    exit "$FINAL_STATUS"
fi

tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "tmux session already exists: $SESSION" >&2
    exit 1
}
[[ ! -e $BATCH_DIR ]] || { echo "Batch directory already exists: $BATCH_DIR" >&2; exit 1; }

for GPU in "${GPUS[@]}"; do
    FREE_MB=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits |
        head -n 1 | tr -dc '0-9')
    (( FREE_MB >= MIN_FREE_MB )) || {
        echo "GPU $GPU has ${FREE_MB} MiB free; ${MIN_FREE_MB} MiB is required" >&2
        exit 1
    }
done

for SHOE in "${SHOES[@]}"; do
    SHOE_ROOT=$OUTPUT_ROOT/$SHOE
    if [[ -d $SHOE_ROOT ]] && find "$SHOE_ROOT" -mindepth 1 -print -quit | grep -q .; then
        echo "Fresh-run output already exists: $SHOE_ROOT" >&2
        exit 1
    fi
done

mkdir -p "$BATCH_DIR"
AVAILABLE_MB=$(df --output=avail -BM "$OUTPUT_ROOT" | tail -n 1 | tr -dc '0-9')
if (( AVAILABLE_MB < MIN_STORAGE_MB )); then
    echo "Only ${AVAILABLE_MB} MiB remains; ${MIN_STORAGE_MB} MiB is required" >&2
    exit 1
fi

printf -v COMMAND 'env NEURALUDF_DATA_ROOT=%q NEURALUDF_SOURCE_SCENE_ROOT=%q NEURALUDF_GROUND_TRUTH_ROOT=%q NEURALUDF_OUTPUT_ROOT=%q NEURALUDF_METRICS_ROOT=%q NEURALUDF_VALIDATION_COMMAND=%q %q --internal-run --gpus %q --shoe-list %q --session %q' \
    "$DATA_ROOT" "$SOURCE_SCENE_ROOT" "$GROUND_TRUTH_ROOT" "$OUTPUT_ROOT" \
    "$METRICS_ROOT" "$VALIDATION_COMMAND" "$SCRIPT" "$GPUS_CSV" \
    "$SHOE_LIST" "$SESSION"
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "$COMMAND"

echo "Started tmux session: $SESSION"
echo "GPUs: $GPUS_CSV"
echo "Shoes: ${SHOES[*]}"
echo "Batch log: $BATCH_LOG"
echo "Worker logs: $BATCH_DIR/worker_*"
echo "Output: $OUTPUT_ROOT"
echo "Attach: tmux attach -t $SESSION"
