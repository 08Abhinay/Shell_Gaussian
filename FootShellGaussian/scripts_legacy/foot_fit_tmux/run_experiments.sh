#!/usr/bin/env bash
# Experiment matrix A-D plus ablations, split across the two free GPUs.
#   A = existing NumPy fitter (evaluated separately, see run_baseline.sh)
#   B = torch, translation only
#   C = torch, translation + ankle/midfoot pitch
#   D = torch, translation + pitch + betas   (the proposed fitter)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Ablations run on a diverse 8-shoe subset; the A-D comparison uses all 16.
ABL_SHOES="crocs sneaker_vibe nike_air_jordan pb129_shoe_low birkenstock_arizona_sandal leather_boots canvas_shoe sandals_0001"
SESSION=footfit_experiments
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n gpuA
tmux send-keys -t "$SESSION:gpuA" \
  "cd $FF_REPO && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY $FF_RUNNER --variant translation --run-name expB_translation 2>&1 | tee $FF_OUT/logs/expB.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY $FF_RUNNER --variant pose        --run-name expC_pose        2>&1 | tee $FF_OUT/logs/expC.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY $FF_RUNNER --variant full        --run-name expD_full --save-meshes 2>&1 | tee $FF_OUT/logs/expD.log" C-m
tmux new-window -t "$SESSION" -n gpuB
tmux send-keys -t "$SESSION:gpuB" \
  "cd $FF_REPO && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --shoes $ABL_SHOES --w-support 0 --run-name ablE_no_support 2>&1 | tee $FF_OUT/logs/ablE.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --shoes $ABL_SHOES --w-beta 0 --run-name ablF_no_beta_prior 2>&1 | tee $FF_OUT/logs/ablF.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --shoes $ABL_SHOES --no-ankle-exemption --run-name ablG_no_ankle_exempt 2>&1 | tee $FF_OUT/logs/ablG.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --shoes $ABL_SHOES --w-length 0 --run-name ablH_no_length 2>&1 | tee $FF_OUT/logs/ablH.log && \
   CUDA_VISIBLE_DEVICES=$FF_GPU_B $FF_PY $FF_RUNNER --variant full --shoes $ABL_SHOES --restarts 1 --run-name ablJ_single_start 2>&1 | tee $FF_OUT/logs/ablJ.log" C-m
echo "started tmux session '$SESSION' (windows: gpuA, gpuB)"
echo "  attach:  tmux attach -t $SESSION"
echo "  logs:    $FF_OUT/logs"
