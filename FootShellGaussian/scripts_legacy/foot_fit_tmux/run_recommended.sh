#!/usr/bin/env bash
# The recommended configuration: experiment D plus the sole safety offset that
# stops a soft penalty settling a few micrometres inside the footbed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_recommended
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=${1:-$FF_GPU_B} $FF_PY $FF_RUNNER \
   --variant full --support-safety-mm 0.15 --save-meshes \
   --run-name expD2_recommended 2>&1 | tee $FF_OUT/logs/expD2.log; exec bash"
echo "started tmux session '$SESSION'"
