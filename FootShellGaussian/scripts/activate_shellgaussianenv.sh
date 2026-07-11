#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOME_ROOT="${HOME:-/home/ab5298}"
CONDA_BASE="${CONDA_BASE:-$HOME_ROOT/anaconda3}"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_BASE/envs/shellgaussianenv}"
DEFAULT_CACHE_ROOT="${DEFAULT_CACHE_ROOT:-$HOME_ROOT/.shell_gaussian_cache}"

if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
    echo "Missing Anaconda/Conda base at: $CONDA_BASE" >&2
    exit 1
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Missing env at: $ENV_PREFIX" >&2
    echo "Run: $ROOT_DIR/scripts/recreate_shellgaussianenv.sh" >&2
    exit 1
fi

export PATH="$CONDA_BASE/bin:$PATH"
export SHELL_GAUSSIAN_CACHE_ROOT="${SHELL_GAUSSIAN_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"

echo "Run the following in your shell:"
echo "source \"$CONDA_BASE/etc/profile.d/conda.sh\""
echo "conda activate \"$ENV_PREFIX\""
