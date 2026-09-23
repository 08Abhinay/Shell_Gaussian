#!/usr/bin/env bash
# Run the EXISTING Checkpoint 4 (preparation) and Checkpoint 5 (alignment)
# pipelines for golden-set shoes that have no prepared artifacts yet, writing
# into the foot_fit output tree rather than into golden_set_evaluation.
# Failures are recorded, not worked around.
set -u
source "$(dirname "$0")/env.sh"
DATA=/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
ROOT=$FF_OUT/prepared_extra
SUPR=/storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy
mkdir -p "$ROOT/shoe_preparation" "$ROOT/support_fit" "$FF_OUT/logs"
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

for s in "$@"; do
  echo "=== $s ==="
  if PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY scripts/run_shoe_preparation.py \
       --shoe-mesh "$DATA/$s/reference_mesh.ply" \
       --canonicalization "$DATA/$s/blender_canonicalization.json" \
       --output-dir "$ROOT/shoe_preparation/$s" --overwrite \
       > "$FF_OUT/logs/prep_$s.log" 2>&1; then
    echo "  prep OK"
  else
    echo "  PREP FAILED: $(tail -3 "$FF_OUT/logs/prep_$s.log" | tr '\n' ' ')"
    continue
  fi
  if PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES=$FF_GPU_A $FF_PY scripts/run_alignment.py \
       --preparation-dir "$ROOT/shoe_preparation/$s" \
       --supr-model "$SUPR" \
       --output-dir "$ROOT/support_fit/$s" --overwrite \
       > "$FF_OUT/logs/align_$s.log" 2>&1; then
    echo "  align OK"
  else
    echo "  ALIGN FAILED: $(tail -3 "$FF_OUT/logs/align_$s.log" | tr '\n' ' ')"
  fi
done
