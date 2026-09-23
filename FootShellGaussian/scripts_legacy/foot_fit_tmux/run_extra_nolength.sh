#!/usr/bin/env bash
# Test: the extra shoes are narrower than the anchored 250 mm reference foot.
# SUPR's betas couple width and length, so narrowing the foot also shortens it
# and the length anchor fights the fix. This run drops the length term.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT=$FF_OUT/prepared_extra
SESSION=footfit_extra_nolength
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd $FF_REPO && CUDA_VISIBLE_DEVICES=${1:-$FF_GPU_B} $FF_PY $FF_RUNNER --variant full \
   --golden-root $ROOT --support-safety-mm 0.15 --w-length 0 \
   --run-name expM_extra_nolength 2>&1 | tee $FF_OUT/logs/expM.log; exec bash"
echo "started tmux session '$SESSION'"
