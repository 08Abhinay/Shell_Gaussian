#!/usr/bin/env bash
# Experiment A: judge the existing NumPy fitter's stored output with the same
# evaluator the torch fits are judged by. Read-only over golden_set_evaluation.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_baseline
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY \
   FootShellGaussian/anatomical_coordinates/scripts/evaluate_baseline.py 2>&1 | tee $FF_OUT/logs/baseline.log; exec bash"
echo "started tmux session '$SESSION'"
