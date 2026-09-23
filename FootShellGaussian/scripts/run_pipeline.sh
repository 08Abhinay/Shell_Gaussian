#!/usr/bin/env bash
# Shared anatomical coordinates, end to end.
#
#   ./scripts/run_pipeline.sh              # 2 GPUs (default)
#   ./scripts/run_pipeline.sh --gpus 3
#   ./scripts/run_pipeline.sh --from fit --to join
#
# Everything from the raw prepared dataset to anatomical addresses, in one
# command. Stages, in order: prepare, seat, fit, anatomy, leg, join,
# coordinates, address.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ab5298/anaconda3/envs/Shell/bin/python
cd "$REPO"
exec "$PY" -m anatomical_coordinates.pipeline.run_pipeline "$@"
