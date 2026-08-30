#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python}
PROJECT_ROOT=/storage/Abhinay/Shell_Gaussian/FootShellGaussian
INPUT_ROOT=${INPUT_ROOT:-/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ab5298/Outputs/FootShellGaussian/checkpoint3_normalization}

if [[ $# -gt 0 ]]; then
    SHOES=("$@")
else
    SHOES=(leather_boots ww_ii_german_jack_boots)
fi

EXTRA_ARGS=()
if [[ ${OVERWRITE:-0} == 1 ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

for shoe in "${SHOES[@]}"; do
    shoe_dir="$INPUT_ROOT/$shoe"
    shoe_mesh="$shoe_dir/reference_mesh.ply"
    canonicalization="$shoe_dir/blender_canonicalization.json"

    if [[ ! -f "$shoe_mesh" ]]; then
        echo "missing canonical shoe mesh: $shoe_mesh" >&2
        exit 1
    fi
    if [[ ! -f "$canonicalization" ]]; then
        echo "missing canonicalization metadata: $canonicalization" >&2
        exit 1
    fi

    echo "[prepare] $shoe"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_shoe_preparation.py" \
        --shoe-mesh "$shoe_mesh" \
        --canonicalization "$canonicalization" \
        --output-dir "$OUTPUT_ROOT/$shoe" \
        "${EXTRA_ARGS[@]}"
done
