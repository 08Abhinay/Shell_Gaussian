#!/usr/bin/env bash
# Batched consistency check: all 16 shoes fitted in one batch, compared against
# the sequential run of the same configuration.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_batched
tmux kill-session -t "$SESSION" 2>/dev/null || true
# Batched and sequential at the SAME single-start setting, so any difference is
# attributable to batching rather than to a different search.
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --restarts 1 --batched \
     --run-name expI_batched 2>&1 | tee $FF_OUT/logs/batched.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --restarts 1 \
     --run-name expI_sequential 2>&1 | tee $FF_OUT/logs/sequential.log; exec bash"
echo "started tmux session '$SESSION'"
