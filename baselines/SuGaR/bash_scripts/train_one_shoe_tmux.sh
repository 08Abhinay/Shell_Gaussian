#!/usr/bin/env bash
set -euo pipefail

SHOE_NAME="${1:?Usage: $0 <shoe_name_or_scene_path> [gpu_id]}"
GPU_ID="${2:-${SUGAR_GPU:-0}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUGAR_ROOT="${SUGAR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_shoe.sh"

safe_name="$(printf '%s' "${SHOE_NAME}" | tr '/ :' '___' | tr -c '[:alnum:]_.-' '_')"
SESSION_NAME="${SUGAR_TMUX_SESSION:-sugar_${safe_name}_gpu${GPU_ID}}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available on PATH" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

cmd=(env "SUGAR_GPU=${GPU_ID}")
for name in \
    SUGAR_ROOT SUGAR_ENV SUGAR_DATA_ROOT SUGAR_OUTPUT_ROOT SUGAR_CACHE_ROOT \
    SUGAR_RUN_ID \
    SUGAR_GS_ITERATIONS SUGAR_GS_RESOLUTION SUGAR_REGULARIZATION SUGAR_SURFACE_LEVEL \
    SUGAR_ESTIMATION_FACTOR SUGAR_NORMAL_FACTOR SUGAR_NORMAL_CONSISTENCY \
    SUGAR_MESH_VERTICES SUGAR_GAUSSIANS_PER_TRIANGLE SUGAR_REFINEMENT_END_ITER \
    SUGAR_GS_OUTPUT_DIR SUGAR_SKIP_3DGS SUGAR_SKIP_EXISTING SUGAR_RENDER_PREVIEWS \
    SUGAR_EVAL SUGAR_WHITE_BACKGROUND SUGAR_SQUARE_SIZE SUGAR_PORT_BASE; do
    if [[ -n "${!name:-}" ]]; then
        cmd+=("${name}=${!name}")
    fi
done
cmd+=("bash" "${TRAIN_SCRIPT}" "${SHOE_NAME}" "${GPU_ID}")

printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_root '%q' "${SUGAR_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" "cd ${quoted_root} && ${quoted_cmd}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Shoe: ${SHOE_NAME}"
echo "GPU: ${GPU_ID}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
