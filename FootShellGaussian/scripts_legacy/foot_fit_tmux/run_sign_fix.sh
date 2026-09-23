#!/usr/bin/env bash
# The longitudinal-sign + geometric-collar-exemption experiment.
#
# Three configurations, identical to the x_heel10_len1 baseline in every other
# respect (no weights are retuned here, deliberately):
#   s1_signfix      the fix, exemption on   - the candidate
#   s1_signfix_noex the fix, exemption off  - isolates the exemption
# The baseline to compare against is the stored x_heel10_len1.
set -u
source "$(dirname "$0")/env.sh"
ROOT=$FF_OUT/prepared_extra
SHOES="sneaker_1 sneaker_3 sneaker_b33 shoes_mockup_asset_vans_skate_old_skool_shoes adidas_substance sneaker_8"
GPU=${1:-$FF_GPU_A}
SESSION=footfit_signfix
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd $FF_REPO && \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER --shoes $SHOES --golden-root $ROOT \
    --support-safety-mm 0.15 --w-heel 10 --w-length 1 \
    --run-name s1_signfix > $FF_OUT/logs/s1_signfix.log 2>&1; \
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER --shoes $SHOES --golden-root $ROOT \
    --support-safety-mm 0.15 --w-heel 10 --w-length 1 --no-ankle-exemption \
    --run-name s1_signfix_noex > $FF_OUT/logs/s1_signfix_noex.log 2>&1; \
  echo ALL DONE; exec bash"
echo "started tmux session '$SESSION' on GPU $GPU"
