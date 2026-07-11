#!/usr/bin/env bash
set -euo pipefail

HOME_ROOT="${HOME:-/home/ab5298}"
CONDA_BASE="${CONDA_BASE:-$HOME_ROOT/anaconda3}"
CACHE_ROOT="${SHELL_GAUSSIAN_CACHE_ROOT:-$HOME_ROOT/.shell_gaussian_cache}"
INSTALLER_URL="${INSTALLER_URL:-https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh}"
INSTALLER_SHA256="${INSTALLER_SHA256:-57b2b48cc5b8665e25fce7011f0389d47c1288288007844b3b1ba482d4f39029}"

download_file() {
    local url="$1"
    local output_path="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L "$url" -o "$output_path"
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -O "$output_path" "$url"
        return
    fi

    echo "Neither curl nor wget is available to download: $url" >&2
    exit 1
}

mkdir -p "$CACHE_ROOT/installers"
INSTALLER_PATH="$CACHE_ROOT/installers/$(basename "$INSTALLER_URL")"

if [[ ! -f "$INSTALLER_PATH" ]]; then
    download_file "$INSTALLER_URL" "$INSTALLER_PATH"
fi

ACTUAL_SHA256=$(sha256sum "$INSTALLER_PATH" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$INSTALLER_SHA256" ]]; then
    echo "Checksum mismatch for $INSTALLER_PATH" >&2
    echo "Expected: $INSTALLER_SHA256" >&2
    echo "Actual:   $ACTUAL_SHA256" >&2
    exit 1
fi

if [[ -d "$CONDA_BASE" ]]; then
    echo "Conda base already exists at: $CONDA_BASE" >&2
    exit 1
fi

bash "$INSTALLER_PATH" -b -p "$CONDA_BASE"

"$CONDA_BASE/bin/conda" config --system --set auto_activate_base false

cat <<EOF
Anaconda installed at: $CONDA_BASE
Add it to PATH in your shell with:
  export PATH="$CONDA_BASE/bin:\$PATH"
Recommended Shell_Gaussian cache root:
  export SHELL_GAUSSIAN_CACHE_ROOT="$CACHE_ROOT"
EOF
