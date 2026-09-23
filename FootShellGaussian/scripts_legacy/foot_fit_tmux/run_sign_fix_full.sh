#!/usr/bin/env bash
# The sign fix over the whole dataset: golden 16 (regression check) and the
# extra 12 (the hard cases). Same configuration as the headline
# expD3_deterministic run plus w_heel=10, so the only differences from the
# stored baselines are the two fixes under test. No weights are retuned.
set -u
source "$(dirname "$0")/env.sh"
EXTRA=$FF_OUT/prepared_extra
GPU=${1:-$FF_GPU_B}
SESSION=footfit_signfix_full
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $FF_REPO && \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER \
    --support-safety-mm 0.15 --w-heel 10 --w-length 1 \
    --run-name s2_signfix_golden16 > $FF_OUT/logs/s2_golden16.log 2>&1; \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER --golden-root $EXTRA \
    --support-safety-mm 0.15 --w-heel 10 --w-length 1 \
    --run-name s2_signfix_extra12 > $FF_OUT/logs/s2_extra12.log 2>&1; \
  echo ALL DONE; exec bash"
echo "started tmux session '$SESSION' on GPU $GPU"
