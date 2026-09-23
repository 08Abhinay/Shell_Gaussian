#!/usr/bin/env bash
# The recommended configuration, re-run after the area accumulation was made
# deterministic. This is the reproducible headline result.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_deterministic
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=${1:-$FF_GPU_B} $FF_PY $FF_RUNNER \
   --variant full --support-safety-mm 0.15 --save-meshes \
   --run-name expD3_deterministic 2>&1 | tee $FF_OUT/logs/expD3.log; exec bash"
echo "started tmux session '$SESSION'"
