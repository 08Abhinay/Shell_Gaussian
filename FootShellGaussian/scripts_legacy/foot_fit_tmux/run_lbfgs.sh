#!/usr/bin/env bash
# Experiment G: Adam schedule followed by an L-BFGS polish, on the ablation
# subset so it is directly comparable to expD on the same shoes.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ABL_SHOES="crocs sneaker_vibe nike_air_jordan pb129_shoe_low birkenstock_arizona_sandal leather_boots canvas_shoe sandals_0001"
SESSION=footfit_lbfgs
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=${1:-$FF_GPU_A} $FF_PY $FF_RUNNER \
   --variant full --shoes $ABL_SHOES --lbfgs-steps 40 \
   --run-name ablK_adam_then_lbfgs 2>&1 | tee $FF_OUT/logs/ablK.log; exec bash"
echo "started tmux session '$SESSION'"
