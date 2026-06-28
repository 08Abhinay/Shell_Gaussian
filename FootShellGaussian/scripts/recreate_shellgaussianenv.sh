#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GSHELL_SCRIPT="$ROOT_DIR/../baselines/GShell/scripts/recreate_gshell_env.sh"
ENV_PREFIX="$ROOT_DIR/shellgaussianenv"

if [[ ! -f "$GSHELL_SCRIPT" ]]; then
    echo "Could not find GShell rebuild script at: $GSHELL_SCRIPT" >&2
    exit 1
fi

export SHELL_GAUSSIAN_CACHE_ROOT="${SHELL_GAUSSIAN_CACHE_ROOT:-/data/abelde}"

if [[ " $* " == *" --prefix "* ]]; then
    echo "This wrapper always builds: $ENV_PREFIX" >&2
    echo "Use baselines/GShell/scripts/recreate_gshell_env.sh directly if you want a custom prefix." >&2
    exit 1
fi

bash "$GSHELL_SCRIPT" --prefix "$ENV_PREFIX" "$@"
