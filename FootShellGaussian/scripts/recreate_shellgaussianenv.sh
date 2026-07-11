#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GSHELL_SCRIPT="$ROOT_DIR/../baselines/GShell/scripts/recreate_gshell_env.sh"
HOME_ROOT="${HOME:-/home/ab5298}"
CONDA_BASE="${CONDA_BASE:-$HOME_ROOT/anaconda3}"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/shellgaussianenv}"
DEFAULT_CACHE_ROOT="${DEFAULT_CACHE_ROOT:-$HOME_ROOT/.shell_gaussian_cache}"
LOCAL_CONDA_BIN="$CONDA_BASE/bin/conda"

if [[ ! -f "$GSHELL_SCRIPT" ]]; then
    echo "Could not find GShell rebuild script at: $GSHELL_SCRIPT" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ -x "$LOCAL_CONDA_BIN" ]]; then
    export PATH="$CONDA_BASE/bin:$PATH"
fi

export SHELL_GAUSSIAN_CACHE_ROOT="${SHELL_GAUSSIAN_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"

if [[ " $* " == *" --prefix "* ]]; then
    echo "This wrapper always builds: $ENV_PREFIX" >&2
    echo "Use baselines/GShell/scripts/recreate_gshell_env.sh directly if you want a custom prefix." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH." >&2
    echo "Install Anaconda at: $CONDA_BASE" >&2
    exit 1
fi

bash "$GSHELL_SCRIPT" --prefix "$ENV_PREFIX" "$@"
