#!/usr/bin/env bash
# Disentangle the golden-16 regression. expD3_deterministic predates w_heel,
# so s2_signfix_golden16 differs from it by three things at once. These two
# runs separate them:
#   s3_g_nosign     w_heel=10, new collar exemption, sign fix OFF
#                   -> vs s2_signfix_golden16 isolates the sign fix alone
#   s3_g_nosign_nw  w_heel=0,  new collar exemption, sign fix OFF
#                   -> vs expD3_deterministic isolates the exemption change
set -u
source "$(dirname "$0")/env.sh"
GPU=${1:-$FF_GPU_A}
SESSION=footfit_ablation
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $FF_REPO && \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER \
    --support-safety-mm 0.15 --w-heel 10 --w-length 1 --no-longitudinal-sign \
    --run-name s3_g_nosign > $FF_OUT/logs/s3_g_nosign.log 2>&1; \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER \
    --support-safety-mm 0.15 --w-heel 0 --w-length 1 --no-longitudinal-sign \
    --run-name s3_g_nosign_nw > $FF_OUT/logs/s3_g_nosign_nw.log 2>&1; \
  echo ALL DONE; exec bash"
echo "started tmux session '$SESSION' on GPU $GPU"
