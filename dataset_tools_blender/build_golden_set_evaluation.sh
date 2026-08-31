#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Golden-set evaluation - 5 GPU dataset build
# ============================================================

SESSION_NAME="golden-set-evaluation-build"

PROJECT_ROOT="/storage/Abhinay/Shell_Gaussian"

PYTHON="/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python"

PIPELINE="/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py"

BLENDER="/home/ab5298/anaconda3/envs/shellgaussianenv/bin/blender"

SOURCE_ROOT="/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean"

MANIFEST="/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json"

OUTPUT_ROOT="/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation"

LOG_DIR="/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs"

LOG_FILE="${LOG_DIR}/dataset-build.log"

GPUS="0,1,2,3,4"


# ------------------------------------------------------------
# Create output/log directory
# ------------------------------------------------------------

mkdir -p "${LOG_DIR}"


# ------------------------------------------------------------
# Make sure another tmux session with this name is not running
# ------------------------------------------------------------

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION_NAME}' already exists."
    echo
    echo "Attach to it with:"
    echo "tmux attach -t ${SESSION_NAME}"
    echo
    echo "Or kill it with:"
    echo "tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi


# ------------------------------------------------------------
# Launch build
# ------------------------------------------------------------

echo "Starting dataset build..."
echo
echo "Source root : ${SOURCE_ROOT}"
echo "Manifest    : ${MANIFEST}"
echo "Output root : ${OUTPUT_ROOT}"
echo "GPUs        : ${GPUS}"
echo "Log         : ${LOG_FILE}"
echo


tmux new-session -d -s "${SESSION_NAME}" \
"bash -lc '
    set -o pipefail

    cd \"${PROJECT_ROOT}\"

    \"${PYTHON}\" \
    \"${PIPELINE}\" build \
    --all \
    --gpus \"${GPUS}\" \
    --source-root \"${SOURCE_ROOT}\" \
    --manifest \"${MANIFEST}\" \
    --output-root \"${OUTPUT_ROOT}\" \
    --blender \"${BLENDER}\" \
    2>&1 | tee \"${LOG_FILE}\"
'"


# ------------------------------------------------------------
# Confirmation
# ------------------------------------------------------------

echo "Build started successfully in tmux."
echo
echo "Attach:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Watch the log without attaching:"
echo "  tail -f ${LOG_FILE}"
echo
echo "Check whether the session is still running:"
echo "  tmux has-session -t ${SESSION_NAME} && echo RUNNING || echo FINISHED"
