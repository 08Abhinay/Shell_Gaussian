#!/usr/bin/env bash
# Prepare every golden-set shoe that lacks Checkpoint 4/5 artifacts, then fit
# the whole dataset with the recommended configuration.
set -euo pipefail
source "$(dirname "$0")/env.sh"
DATA=/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
ROOT=$FF_OUT/prepared_extra
HERE="$(cd "$(dirname "$0")" && pwd)"
MISSING=$(comm -23 <(ls "$DATA" | sort) \
  <(ls /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation | sort) | tr '\n' ' ')
SESSION=footfit_dataset
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd /storage/Abhinay/Shell_Gaussian && \
   bash $HERE/prepare_missing_shoes.sh $MISSING 2>&1 | tee $FF_OUT/logs/prepare_extra.log && \
   CUDA_VISIBLE_DEVICES=${1:-$FF_GPU_A} $FF_PY $FF_RUNNER --variant full \
     --golden-root $ROOT --support-safety-mm 0.15 --save-meshes \
     --run-name expL_extra_shoes 2>&1 | tee $FF_OUT/logs/expL.log; exec bash"
echo "started tmux session '$SESSION'"
echo "shoes to prepare: $MISSING"
