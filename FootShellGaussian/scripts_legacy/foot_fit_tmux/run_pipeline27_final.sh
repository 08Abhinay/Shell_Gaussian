#!/usr/bin/env bash
# All 27 shoes, prepared input -> Checkpoint 11-D, one output root.
#
# The anatomy budget ladder is the only per-shoe adaptation, and it is a rule
# rather than a table: fit with the largest budget, and refit tighter only the
# shoes Checkpoint 11-B refuses, because 11-B cannot deform the canonical
# tetrahedral volume onto anatomy that sits too far from canonical without
# inverting an element. A shoe that already works is never degraded.
set -u
source "$(dirname "$0")/env.sh"
P=/home/ab5298/Outputs/FootShellGaussian/pipeline27
tmux kill-session -t p27final 2>/dev/null || true
tmux new-session -d -s p27final "cd $FF_REPO/FootShellGaussian && \
  /home/ab5298/anaconda3/envs/Shell/bin/python -m anatomical_coordinates.pipeline.run_to_11d \
    --root $P --gpus 2 --jobs 32 --leg torch --from-stage fit \
    --beta-ladder 5.0 3.5 2.5 1.5 \
    2>&1 | tee $P/logs/pipeline27_final.log; echo DONE; exec bash"
echo "started p27final"
