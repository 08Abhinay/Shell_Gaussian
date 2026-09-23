#!/usr/bin/env bash
# Configuration search on the hard (previously unprepared) shoes.
set -u
source "$(dirname "$0")/env.sh"
ROOT=$FF_OUT/prepared_extra
SHOES="sneaker_1 sneaker_3 sneaker_b33 shoes_mockup_asset_vans_skate_old_skool_shoes adidas_substance sneaker_8"
GPU=$1; shift
for spec in "$@"; do
  name=${spec%%|*}; flags=${spec#*|}
  CUDA_VISIBLE_DEVICES=$GPU $FF_PY $FF_RUNNER --shoes $SHOES --golden-root $ROOT \
    --support-safety-mm 0.15 --run-name "x_$name" $flags > $FF_OUT/logs/x_$name.log 2>&1
  echo "done $name"
done
