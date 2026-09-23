#!/usr/bin/env bash
# NumPy search fitter vs differentiable fitter, same shoes, same start.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SESSION=footfit_benchmark
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY \
   FootShellGaussian/anatomical_coordinates/scripts/benchmark_runtime.py \
   --shoes crocs 2>&1 | tee $FF_OUT/logs/benchmark.log; exec bash"
echo "started tmux session '$SESSION'"
