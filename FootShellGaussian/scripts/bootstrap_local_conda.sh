#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(cd "$ROOT_DIR/../.." && pwd)
DEFAULT_PREFIX="$WORKSPACE_ROOT/miniforge3"
DEFAULT_CACHE_ROOT="$WORKSPACE_ROOT/.shell_gaussian_cache"
INSTALLER_URL_DEFAULT="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

usage() {
    echo "Usage: $0 [--prefix PATH] [--installer-url URL]"
}

PREFIX="$DEFAULT_PREFIX"
INSTALLER_URL="$INSTALLER_URL_DEFAULT"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --installer-url)
            INSTALLER_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "$DEFAULT_CACHE_ROOT/installers"
INSTALLER_PATH="$DEFAULT_CACHE_ROOT/installers/$(basename "$INSTALLER_URL")"

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

if [[ -x "$PREFIX/bin/conda" ]]; then
    echo "Local conda already exists at: $PREFIX"
else
    download_file "$INSTALLER_URL" "$INSTALLER_PATH"
    bash "$INSTALLER_PATH" -b -p "$PREFIX"
fi

"$PREFIX/bin/conda" config --system --set auto_activate_base false

cat <<EOF
Local conda ready at: $PREFIX
Add it to PATH for this shell with:
  export PATH="$PREFIX/bin:\$PATH"

Recommended cache root for Shell_Gaussian:
  export SHELL_GAUSSIAN_CACHE_ROOT="$DEFAULT_CACHE_ROOT"
EOF
