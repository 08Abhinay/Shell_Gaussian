#!/usr/bin/env bash
# Full 27-shoe rerun with the two constraints that 11-B needs:
#   foot ||beta||_2 projected to <= 5.0 (was unbounded up to 9.49)
#   lower leg takes the gentlest pose within tolerance of the best collision
# Both keep the fitted anatomy close enough to canonical that the fixed
# tetrahedral connectivity can reach it without inverting an element.
set -u
source "$(dirname "$0")/env.sh"
P=/home/ab5298/Outputs/FootShellGaussian/pipeline27
GPUS="${1:-2}"
tmux kill-session -t p27v2 2>/dev/null || true
tmux new-session -d -s p27v2 "cd $FF_REPO/FootShellGaussian && \
  /home/ab5298/anaconda3/envs/Shell/bin/python -m anatomical_coordinates.pipeline.run_to_11d \
    --root $P --gpus $GPUS --jobs 32 --leg torch --from-stage fit \
    2>&1 | tee $P/logs/pipeline27_v2.log; echo DONE; exec bash"
echo "started p27v2 on GPUs $GPUS"
