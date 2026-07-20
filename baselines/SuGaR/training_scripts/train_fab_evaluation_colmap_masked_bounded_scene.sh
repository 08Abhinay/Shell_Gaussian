#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="/storage/Abhinay/Shell_Gaussian/baselines/SuGaR"
DATASET_ROOT="/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_sugar"
RGB_OUTPUT_ROOT="$REPO_ROOT/output/fab_evaluation_colmap_rgb_only"
EXPERIMENT_ROOT="fab_evaluation_colmap_masked_bounded"
CONDA_ROOT="/home/ab5298/anaconda3"
ENV_PREFIX="/home/ab5298/anaconda3/envs/SuGaR_env/SuGaR_env"
VALIDATOR="/storage/Abhinay/Shell_Gaussian/dataset_tools/golden_set_evaluation/validate_sugar.py"
RUNNER="$REPO_ROOT/training_scripts/train_coarse_masked_bounded.py"

usage() {
    cat <<'EOF'
Usage:
  train_fab_evaluation_colmap_masked_bounded_scene.sh <scene-name> [gpu]

This reuses the good RGB-only vanilla 3DGS checkpoint, runs masked and bounded
coarse SuGaR with a 5%-of-bbox-diagonal Gaussian scale cap, extracts one coarse
mesh, and stops before refinement.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 2
fi

scene_name=$1
gpu=${2:-}
scene_root="$DATASET_ROOT/$scene_name"
scene_path="$scene_root/undistorted"
manifest="$scene_root/masked_colmap_manifest.json"
gs_output="$RGB_OUTPUT_ROOT/$scene_name/vanilla_gs"
gs_checkpoint="$gs_output/point_cloud/iteration_7000/point_cloud.ply"
output_root="$REPO_ROOT/output/$EXPERIMENT_ROOT/$scene_name"

for required in \
    "$manifest" \
    "$gs_checkpoint" \
    "$gs_output/cameras.json" \
    "$scene_path/sparse/0/cameras.bin" \
    "$scene_path/sparse/0/images.bin" \
    "$scene_path/sparse/0/points3D.bin"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing required pilot input: $required" >&2
        exit 2
    fi
done

if [[ -e "$output_root" ]]; then
    echo "Refusing to overwrite existing pilot output: $output_root" >&2
    exit 2
fi

"$ENV_PREFIX/bin/python" "$VALIDATOR" --dataset-root "$DATASET_ROOT" --scene "$scene_name"

mapfile -t settings < <(
    "$ENV_PREFIX/bin/python" - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
box = manifest["foreground_bbox"]
contract = manifest["training_contract"]
for value in box["min"]:
    print(f"{value:.17g}")
for value in box["max"]:
    print(f"{value:.17g}")
print(f"{contract['max_gaussian_scale_ratio']:.17g}")
print(manifest["images"]["count"])
print(f"{manifest['images']['foreground_fraction_mean']:.17g}")
PY
)
if [[ ${#settings[@]} -ne 9 ]]; then
    echo "Could not parse masked COLMAP manifest: $manifest" >&2
    exit 2
fi
bbox_min=("${settings[0]}" "${settings[1]}" "${settings[2]}")
bbox_max=("${settings[3]}" "${settings[4]}" "${settings[5]}")
scale_ratio=${settings[6]}
image_count=${settings[7]}
mask_fraction=${settings[8]}

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
echo "Masked COLMAP dataset:    $scene_path"
echo "RGBA masks loaded:        $image_count/$image_count"
echo "Mean foreground coverage: $mask_fraction"
echo "Existing vanilla 3DGS:    $gs_output"
echo "Foreground bbox min:      ${bbox_min[*]}"
echo "Foreground bbox max:      ${bbox_max[*]}"
echo "Max scale ratio:          $scale_ratio"
echo "GPU:                      $gpu"
echo "Output:                   $output_root"
echo "Stopping after:           coarse mesh"

python "$RUNNER" \
    --scene-path "$scene_path" \
    --checkpoint-path "$gs_output" \
    --output-root "$output_root" \
    --bbox-min "${bbox_min[@]}" \
    --bbox-max "${bbox_max[@]}" \
    --max-gaussian-scale-ratio "$scale_ratio" \
    --gpu "$gpu"
