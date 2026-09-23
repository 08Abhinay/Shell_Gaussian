#!/usr/bin/env bash
# Shared settings for every anatomical_coordinates tmux run.
export FF_REPO=/storage/Abhinay/Shell_Gaussian
export FF_PY=/home/ab5298/anaconda3/envs/Shell/bin/python
export FF_RUNNER=$FF_REPO/FootShellGaussian/anatomical_coordinates/scripts/run_torch_foot_fit.py
export FF_OUT=/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs
# Only these two GPUs are free on this host; the fitter is single-GPU by design
# (the model is tiny, so parallelism is across experiments, not within one).
export FF_GPU_A=4
export FF_GPU_B=6
mkdir -p "$FF_OUT/logs"
