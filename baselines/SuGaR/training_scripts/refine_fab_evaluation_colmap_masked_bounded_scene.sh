#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="/storage/Abhinay/Shell_Gaussian/baselines/SuGaR"
EXPERIMENT_ROOT="$REPO_ROOT/output/fab_evaluation_colmap_masked_bounded"
CONDA_ROOT="/home/ab5298/anaconda3"
ENV_PREFIX="/home/ab5298/anaconda3/envs/SuGaR_env/SuGaR_env"
RUNNER="$REPO_ROOT/training_scripts/refine_masked_bounded_coarse.py"

usage() {
    cat <<'EOF'
Usage:
  refine_fab_evaluation_colmap_masked_bounded_scene.sh <scene-name> [gpu]

Resumes a completed, accepted coarse pilot. It does not repeat coarse SuGaR or
Poisson reconstruction. It runs the standard 15,000-step refinement and final
textured-mesh/PLY export.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 2
fi

scene_name=$1
gpu=${2:-}
output_root="$EXPERIMENT_ROOT/$scene_name"
coarse_result="$output_root/coarse_pilot_result.json"

if [[ ! -s "$coarse_result" ]]; then
    echo "Missing completed coarse result: $coarse_result" >&2
    exit 2
fi
if [[ -e "$output_root/refined" || -e "$output_root/refined_mesh" ]]; then
    echo "Refusing to overwrite an existing refinement output under: $output_root" >&2
    exit 2
fi

if [[ -z "$gpu" ]]; then
    gpu=$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
            sort -t, -k2,2n |
            head -n 1 |
            cut -d, -f1 |
            tr -d ' '
    )
fi
if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU must be a numeric physical GPU index; received: $gpu" >&2
    exit 2
fi

set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
cd "$REPO_ROOT"

echo "Scene:                    $scene_name"
echo "Coarse result manifest:   $coarse_result"
echo "GPU:                      $gpu"
echo "Refinement iterations:    15000"
echo "Masks:                    enabled"
echo "Repeat coarse training:   no"
echo "Repeat Poisson:           no"
echo "Final texture/PLY export: yes"

python "$RUNNER" --coarse-result "$coarse_result" --gpu "$gpu"
