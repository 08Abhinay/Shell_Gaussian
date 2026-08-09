#!/usr/bin/env bash
set -euo pipefail

PYTHON="${NEUS2_TURNTABLE_PYTHON:-/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python}"
PIPELINE="${NEUS2_TURNTABLE_PIPELINE:-/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py}"
OUTPUT_ROOT="${NEUS2_TURNTABLE_OUTPUT_ROOT:-/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation_turntable}"

SHOES=(
    air_jordan_1
    female_gymnasts_shoes
    red_high_heel_shoes
    sandals_0001
    birkenstock_arizona_sandal
)

OVERWRITE_ARGS=()
if [[ "${NEUS2_TURNTABLE_OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS=(--overwrite)
fi

for shoe in "${SHOES[@]}"; do
    "${PYTHON}" "${PIPELINE}" prepare-neus2-turntable \
        --shoe "${shoe}" \
        --output-root "${OUTPUT_ROOT}" \
        "${OVERWRITE_ARGS[@]}"

    "${PYTHON}" "${PIPELINE}" validate-neus2-turntable \
        --shoe "${shoe}" \
        --output-root "${OUTPUT_ROOT}"
done

echo "NeuS2 turntable preparation and validation completed for ${#SHOES[@]} shoes."
