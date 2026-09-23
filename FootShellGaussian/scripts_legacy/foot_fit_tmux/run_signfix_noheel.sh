#!/usr/bin/env bash
# The ablation showed w_heel=10 costs the golden set 7 clear fits once the
# collar exemption is geometric: the heel spring was compensating for the
# loophole that fix B closed. These runs drop it back to 0 and test the sign
# fix on top, on both shoe groups.
set -u
source "$(dirname "$0")/env.sh"
EXTRA=$FF_OUT/prepared_extra
SESSION=footfit_noheel
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $FF_REPO && \
  CUDA_VISIBLE_DEVICES=${FF_GPU_A} $FF_PY $FF_RUNNER \
    --support-safety-mm 0.15 --w-heel 0 --w-length 1 \
    --run-name s4_g_sign_nw > $FF_OUT/logs/s4_g_sign_nw.log 2>&1; \
  echo GOLDEN DONE; exec bash"
tmux new-window -t "$SESSION" "cd $FF_REPO && \
  CUDA_VISIBLE_DEVICES=${FF_GPU_B} $FF_PY $FF_RUNNER --golden-root $EXTRA \
    --support-safety-mm 0.15 --w-heel 0 --w-length 1 \
    --run-name s4_e_sign_nw > $FF_OUT/logs/s4_e_sign_nw.log 2>&1; \
  CUDA_VISIBLE_DEVICES=${FF_GPU_B} $FF_PY $FF_RUNNER --golden-root $EXTRA \
    --support-safety-mm 0.15 --w-heel 0 --w-length 1 --no-longitudinal-sign \
    --run-name s4_e_nosign_nw > $FF_OUT/logs/s4_e_nosign_nw.log 2>&1; \
  echo EXTRA DONE; exec bash"
echo "started tmux session '$SESSION' on GPUs $FF_GPU_A and $FF_GPU_B"
