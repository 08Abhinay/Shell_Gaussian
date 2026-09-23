#!/usr/bin/env bash
# Both hard gates: SUPR autograd vs finite differences, and the shoe field.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_tests
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY -m pytest \
   FootShellGaussian/anatomical_coordinates/tests -v 2>&1 | tee $FF_OUT/logs/tests.log; exec bash"
echo "started tmux session '$SESSION'"
