#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python}
PROJECT_ROOT=/storage/Abhinay/Shell_Gaussian/FootShellGaussian
INPUT_ROOT=${INPUT_ROOT:-/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation}

if [[ $# -gt 0 ]]; then
    SHOES=("$@")
else
    SHOES=(
        aj_12_basketball_sneakers
        birkenstock_arizona_sandal
        canvas_shoe
        crocs
        crocs_by_speedyart_studio
        crocs_shoe
        duinn_shoes_womens_hiking_sandal_sport
        leather_boots
        nike_air_jordan
        pb129_shoe_low
        priest_karol_wojtyas_sports_shoes
        sandal_1
        sandals_0001
        shoes_mockup_asset_vans_skate_old_skool_shoes
        sneaker_vibe
        sneakers_seen
        ww_ii_german_jack_boots
    )
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
