#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="/storage/Abhinay/Shell_Gaussian/baselines/SuGaR"
WORKER="$REPO_ROOT/training_scripts/train_fab_evaluation_colmap_masked_bounded_worker.sh"
RUNS_ROOT="$REPO_ROOT/output/fab_evaluation_colmap_masked_bounded/batch_runs"

usage() {
    cat <<'EOF'
Usage:
  launch_fab_evaluation_colmap_masked_bounded_batch_tmux.sh [gpu-0 gpu-1 [run-id]]

Launch exactly two tmux workers for the complete 20-scene masked COLMAP
evaluation set. If GPUs are omitted, the two GPUs with the least allocated
memory are selected. The resolved commands and GPU assignment are recorded in
the run directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 && $# -ne 2 && $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

if [[ $# -ge 2 ]]; then
    gpus=("$1" "$2")
else
    mapfile -t gpus < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
            sort -t, -k2,2n |
            head -n 2 |
            cut -d, -f1 |
            tr -d ' '
    )
fi
if [[ ${#gpus[@]} -ne 2 || "${gpus[0]}" == "${gpus[1]}" || \
      ! "${gpus[0]}" =~ ^[0-9]+$ || ! "${gpus[1]}" =~ ^[0-9]+$ ]]; then
    echo "Two distinct numeric GPU indices are required" >&2
    exit 2
fi

run_id=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir="$RUNS_ROOT/$run_id"
manifest="$run_dir/launch_manifest.tsv"
mkdir -p "$run_dir"
printf 'worker\tgpu\tsession\tlog\tcommand\n' > "$manifest"

for worker_index in 0 1; do
    gpu=${gpus[$worker_index]}
    session="sugar_eval_batch_${run_id}_w${worker_index}"
    log="$run_dir/${session}.log"
    command="$WORKER --worker-index $worker_index --worker-count 2 --gpu $gpu --run-id $run_id"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 2
    fi
    tmux new-session -d -s "$session" bash -lc \
        "cd '$REPO_ROOT'; set -o pipefail; $command 2>&1 | tee -a '$log'"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$worker_index" "$gpu" "$session" "$log" "$command" >> "$manifest"
    echo "Worker $worker_index: GPU $gpu"
    echo "  Session: $session"
    echo "  Log:     $log"
    echo "  Command: $command"
done

echo "Run ID:   $run_id"
echo "Manifest: $manifest"
