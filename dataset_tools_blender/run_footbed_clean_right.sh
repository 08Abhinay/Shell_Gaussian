#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python}
BLENDER=${BLENDER:-/home/ab5298/anaconda3/envs/shellgaussianenv/bin/blender}
PIPELINE=${PIPELINE:-/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py}
MANIFEST=${MANIFEST:-/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/footbed_clean_right_manifest.json}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right}
AUDIT_ROOT=${AUDIT_ROOT:-/home/ab5298/Outputs/FootShellGaussian/canonicalization_audit}

ACTION=${1:-}
if [[ -z "$ACTION" ]]; then
    echo "usage: $0 {audit|build|validate|all} [shoe ...]" >&2
    exit 2
fi
shift

if [[ $# -gt 0 ]]; then
    SHOES=("$@")
else
    SHOES=(leather_boots ww_ii_german_jack_boots)
fi

GPU=${GPU:-0}
BUILD_ARGS=()
if [[ ${OVERWRITE:-0} == 1 ]]; then
    BUILD_ARGS+=(--overwrite)
fi

run_audit() {
    local shoe
    for shoe in "${SHOES[@]}"; do
        "$PYTHON" "$PIPELINE" audit \
            --shoe "$shoe" \
            --source-root "$SOURCE_ROOT" \
            --manifest "$MANIFEST" \
            --output-dir "$AUDIT_ROOT" \
            --blender "$BLENDER" \
            --gpu "$GPU"
    done
}

run_build() {
    local shoe
    for shoe in "${SHOES[@]}"; do
        "$PYTHON" "$PIPELINE" build \
            --shoe "$shoe" \
            --source-root "$SOURCE_ROOT" \
            --manifest "$MANIFEST" \
            --output-root "$OUTPUT_ROOT" \
            --blender "$BLENDER" \
            --gpu "$GPU" \
            "${BUILD_ARGS[@]}"
    done
}

run_validate() {
    local shoe
    for shoe in "${SHOES[@]}"; do
        "$PYTHON" "$PIPELINE" validate \
            --shoe "$shoe" \
            --source-root "$SOURCE_ROOT" \
            --manifest "$MANIFEST" \
            --output-root "$OUTPUT_ROOT"
    done
}

case "$ACTION" in
    audit)
        run_audit
        ;;
    build)
        run_build
        ;;
    validate)
        run_validate
        ;;
    all)
        run_audit
        run_build
        run_validate
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        echo "usage: $0 {audit|build|validate|all} [shoe ...]" >&2
        exit 2
        ;;
esac
