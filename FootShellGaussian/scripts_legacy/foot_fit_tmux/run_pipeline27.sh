#!/usr/bin/env bash
# All 27 shoes, one output root, prepared shoes -> Checkpoint 11-D.
# The foot fit runs on GPUs 2/4/6; every stage after it is the existing
# foot_prior implementation, unchanged.
set -u
source "$(dirname "$0")/env.sh"
GPU_PY=/home/ab5298/anaconda3/envs/Shell/bin/python
ROOT=/home/ab5298/Outputs/FootShellGaussian/pipeline27
SESSION=pipeline27
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $FF_REPO/FootShellGaussian && \
  $GPU_PY -m anatomical_coordinates.pipeline.run_to_11d \
    --root $ROOT --gpus 2 4 6 --jobs 8 --leg numpy ${*} \
    2>&1 | tee $ROOT/logs/pipeline27.log; \
  echo PIPELINE DONE; exec bash"
echo "started tmux session '$SESSION'"
echo "  tail -f $ROOT/logs/pipeline27.log"
