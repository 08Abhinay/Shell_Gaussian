#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--prefix PATH] [--core-only]"
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="$ROOT_DIR/GShell_env"
INSTALL_EXTRAS=1
CACHE_ROOT="${SHELL_GAUSSIAN_CACHE_ROOT:-/data/abelde}"
BLENDER_VERSION="4.2.21"
BLENDER_DIR_NAME="blender-${BLENDER_VERSION}-linux-x64"
BLENDER_ARCHIVE="${BLENDER_DIR_NAME}.tar.xz"
BLENDER_URL="${BLENDER_URL:-https://download.blender.org/release/Blender4.2/${BLENDER_ARCHIVE}}"
BLENDER_ARCHIVE_SOURCE="${BLENDER_ARCHIVE_SOURCE:-$ROOT_DIR/GShell_env/opt/${BLENDER_ARCHIVE}}"
NVDR_TMP=""
TCNN_TMP=""

cleanup() {
    if [[ -n "$NVDR_TMP" && -d "$NVDR_TMP" ]]; then
        rm -rf "$NVDR_TMP"
    fi
    if [[ -n "$TCNN_TMP" && -d "$TCNN_TMP" ]]; then
        rm -rf "$TCNN_TMP"
    fi
}

run_conda() {
    local nounset_was_on=0
    local status=0
    if [[ -o nounset ]]; then
        nounset_was_on=1
        set +u
    fi

    if conda "$@"; then
        status=0
    else
        status=$?
    fi

    if [[ "$nounset_was_on" -eq 1 ]]; then
        set -u
    fi

    return "$status"
}

setup_build_environment() {
    mkdir -p \
        "$CACHE_ROOT/.conda/pkgs" \
        "$CACHE_ROOT/.cache/pip" \
        "$CACHE_ROOT/.cache/torch_extensions" \
        "$CACHE_ROOT/.cache/matplotlib" \
        "$CACHE_ROOT/.nv/ComputeCache" \
        "$CACHE_ROOT/tmp"

    export CONDA_PKGS_DIRS="$CACHE_ROOT/.conda/pkgs"
    export PIP_CACHE_DIR="$CACHE_ROOT/.cache/pip"
    export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
    export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/.cache/torch_extensions"
    export CUDA_CACHE_PATH="$CACHE_ROOT/.nv/ComputeCache"
    export MPLCONFIGDIR="$CACHE_ROOT/.cache/matplotlib"
    export TMPDIR="$CACHE_ROOT/tmp"
    export TMP="$CACHE_ROOT/tmp"
    export TEMP="$CACHE_ROOT/tmp"
    export CUDA_HOME="$ENV_PREFIX"
    export PATH="$ENV_PREFIX/bin:$PATH"
    export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -L$ENV_PREFIX/lib${LDFLAGS:+ $LDFLAGS}"
    export GSHELL_REPO_ROOT="$ROOT_DIR"
}

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

install_blender_bundle() {
    local opt_dir="$ENV_PREFIX/opt"
    local archive_path="$opt_dir/$BLENDER_ARCHIVE"
    local blender_root="$opt_dir/$BLENDER_DIR_NAME"

    mkdir -p "$opt_dir"

    if [[ ! -f "$archive_path" ]]; then
        if [[ -f "$BLENDER_ARCHIVE_SOURCE" ]]; then
            cp "$BLENDER_ARCHIVE_SOURCE" "$archive_path"
        else
            download_file "$BLENDER_URL" "$archive_path"
        fi
    fi

    if [[ ! -x "$blender_root/blender" ]]; then
        tar -xJf "$archive_path" -C "$opt_dir"
    fi

    ln -sfn "$blender_root/blender" "$ENV_PREFIX/bin/blender"
    ln -sfn "$blender_root/blender-launcher" "$ENV_PREFIX/bin/blender-launcher"
    ln -sfn "$blender_root/blender-softwaregl" "$ENV_PREFIX/bin/blender-softwaregl"
    ln -sfn "$blender_root/blender-thumbnailer" "$ENV_PREFIX/bin/blender-thumbnailer"
}

install_nvdiffrast() {
    local source_dir="$1"
    local python_bin="$2"
    local torch_lib_dir=""
    local build_lib_dir=""
    local build_so=""
    local -a object_files=()

    torch_lib_dir="$("$python_bin" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

    (
        cd "$source_dir"
        "$python_bin" setup.py build_ext --inplace

        build_lib_dir="$(find build -maxdepth 1 -type d -name 'lib.*' | head -n 1)"
        if [[ -z "$build_lib_dir" ]]; then
            echo "Could not find nvdiffrast build/lib directory" >&2
            exit 1
        fi

        build_so="$(find "$build_lib_dir" -maxdepth 1 -type f -name '_nvdiffrast_c*.so' | head -n 1)"
        if [[ -z "$build_so" ]]; then
            echo "Could not find built nvdiffrast extension under: $build_lib_dir" >&2
            exit 1
        fi

        mapfile -d '' object_files < <(find build -type f -name '*.o' -print0 | sort -z)
        if [[ "${#object_files[@]}" -eq 0 ]]; then
            echo "Could not find nvdiffrast object files to relink" >&2
            exit 1
        fi

        # Torch's default setuptools link step can resolve libcudart from the
        # system CUDA 12 install on this server. Re-linking from the already
        # built objects pins the extension back to the env-local CUDA 11.7
        # runtime that matches PyTorch 1.13.1.
        "$CXX" \
            -shared \
            -Wl,-rpath,"$ENV_PREFIX/lib" \
            -Wl,-rpath-link,"$ENV_PREFIX/lib" \
            -L"$torch_lib_dir" \
            -L"$ENV_PREFIX/lib" \
            "${object_files[@]}" \
            -lc10 \
            -ltorch \
            -ltorch_cpu \
            -ltorch_python \
            -lcudart \
            -lc10_cuda \
            -ltorch_cuda_cu \
            -ltorch_cuda_cpp \
            -o "$build_so"

        if readelf -d "$build_so" | grep -q 'libcudart.so.12'; then
            echo "nvdiffrast relink still resolved against CUDA 12 runtime" >&2
            exit 1
        fi

        "$python_bin" setup.py install --skip-build
    )
}

patch_mkl_activation_hooks() {
    local activate="$ENV_PREFIX/etc/conda/activate.d/libblas_mkl_activate.sh"
    local deactivate="$ENV_PREFIX/etc/conda/deactivate.d/libblas_mkl_deactivate.sh"

    if [[ -f "$activate" ]]; then
        cat > "$activate" <<'EOF'
export CONDA_MKL_INTERFACE_LAYER_BACKUP="${MKL_INTERFACE_LAYER-}"
export MKL_INTERFACE_LAYER=LP64,GNU
EOF
    fi

    if [[ -f "$deactivate" ]]; then
        cat > "$deactivate" <<'EOF'
if [ "${CONDA_MKL_INTERFACE_LAYER_BACKUP-}" = "" ]; then
    unset MKL_INTERFACE_LAYER
else
    export MKL_INTERFACE_LAYER="${CONDA_MKL_INTERFACE_LAYER_BACKUP}"
fi
unset CONDA_MKL_INTERFACE_LAYER_BACKUP
EOF
    fi
}

write_runtime_activation_hooks() {
    local act_dir="$ENV_PREFIX/etc/conda/activate.d"
    local deact_dir="$ENV_PREFIX/etc/conda/deactivate.d"

    mkdir -p "$act_dir" "$deact_dir"

    cat > "$act_dir/gshell_env_vars.sh" <<EOF
export _GSHELL_OLD_CONDA_PKGS_DIRS="\${CONDA_PKGS_DIRS-}"
export _GSHELL_OLD_PIP_CACHE_DIR="\${PIP_CACHE_DIR-}"
export _GSHELL_OLD_XDG_CACHE_HOME="\${XDG_CACHE_HOME-}"
export _GSHELL_OLD_TORCH_EXTENSIONS_DIR="\${TORCH_EXTENSIONS_DIR-}"
export _GSHELL_OLD_CUDA_CACHE_PATH="\${CUDA_CACHE_PATH-}"
export _GSHELL_OLD_MPLCONFIGDIR="\${MPLCONFIGDIR-}"
export _GSHELL_OLD_TMPDIR="\${TMPDIR-}"
export _GSHELL_OLD_TMP="\${TMP-}"
export _GSHELL_OLD_TEMP="\${TEMP-}"
export _GSHELL_OLD_CUDA_HOME="\${CUDA_HOME-}"
export _GSHELL_OLD_CC="\${CC-}"
export _GSHELL_OLD_CXX="\${CXX-}"
export _GSHELL_OLD_CUDAHOSTCXX="\${CUDAHOSTCXX-}"
export _GSHELL_OLD_LIBRARY_PATH="\${LIBRARY_PATH-}"
export _GSHELL_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
export _GSHELL_OLD_LDFLAGS="\${LDFLAGS-}"
export _GSHELL_OLD_BLENDER="\${BLENDER-}"

export GSHELL_ENV_PREFIX="$ENV_PREFIX"
export GSHELL_CACHE_ROOT="$CACHE_ROOT"
export CONDA_PKGS_DIRS="$CACHE_ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/.cache/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/.cache/torch_extensions"
export CUDA_CACHE_PATH="$CACHE_ROOT/.nv/ComputeCache"
export MPLCONFIGDIR="$CACHE_ROOT/.cache/matplotlib"
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export CUDA_HOME="$ENV_PREFIX"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$ENV_PREFIX/lib\${_GSHELL_OLD_LIBRARY_PATH:+:\$_GSHELL_OLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$ENV_PREFIX/lib\${_GSHELL_OLD_LD_LIBRARY_PATH:+:\$_GSHELL_OLD_LD_LIBRARY_PATH}"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -L$ENV_PREFIX/lib\${_GSHELL_OLD_LDFLAGS:+ \$_GSHELL_OLD_LDFLAGS}"
export BLENDER="$ENV_PREFIX/bin/blender"
EOF

    cat > "$deact_dir/gshell_env_vars.sh" <<'EOF'
restore_or_unset() {
    local name="$1"
    local backup_name="$2"
    local backup_value="${!backup_name-}"
    if [ -n "${backup_value}" ]; then
        export "$name=$backup_value"
    else
        unset "$name"
    fi
    unset "$backup_name"
}

restore_or_unset CONDA_PKGS_DIRS _GSHELL_OLD_CONDA_PKGS_DIRS
restore_or_unset PIP_CACHE_DIR _GSHELL_OLD_PIP_CACHE_DIR
restore_or_unset XDG_CACHE_HOME _GSHELL_OLD_XDG_CACHE_HOME
restore_or_unset TORCH_EXTENSIONS_DIR _GSHELL_OLD_TORCH_EXTENSIONS_DIR
restore_or_unset CUDA_CACHE_PATH _GSHELL_OLD_CUDA_CACHE_PATH
restore_or_unset MPLCONFIGDIR _GSHELL_OLD_MPLCONFIGDIR
restore_or_unset TMPDIR _GSHELL_OLD_TMPDIR
restore_or_unset TMP _GSHELL_OLD_TMP
restore_or_unset TEMP _GSHELL_OLD_TEMP
restore_or_unset CUDA_HOME _GSHELL_OLD_CUDA_HOME
restore_or_unset CC _GSHELL_OLD_CC
restore_or_unset CXX _GSHELL_OLD_CXX
restore_or_unset CUDAHOSTCXX _GSHELL_OLD_CUDAHOSTCXX
restore_or_unset LIBRARY_PATH _GSHELL_OLD_LIBRARY_PATH
restore_or_unset LD_LIBRARY_PATH _GSHELL_OLD_LD_LIBRARY_PATH
restore_or_unset LDFLAGS _GSHELL_OLD_LDFLAGS
restore_or_unset BLENDER _GSHELL_OLD_BLENDER

unset GSHELL_ENV_PREFIX
unset GSHELL_CACHE_ROOT
unset -f restore_or_unset
EOF
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            ENV_PREFIX="$2"
            shift 2
            ;;
        --core-only)
            INSTALL_EXTRAS=0
            shift
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

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH." >&2
    exit 1
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -e "$ENV_PREFIX" ]]; then
    echo "Environment prefix already exists: $ENV_PREFIX" >&2
    exit 1
fi

setup_build_environment

run_conda create --prefix "$ENV_PREFIX" python=3.10 -y

run_conda install --prefix "$ENV_PREFIX" -c pytorch -c nvidia \
    pytorch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    pytorch-cuda=11.7 \
    -y

run_conda install --prefix "$ENV_PREFIX" -c conda-forge mkl=2022.1.0 -y

run_conda install --prefix "$ENV_PREFIX" -c nvidia \
    cuda-nvcc=11.7 \
    cuda-cudart-dev=11.7 \
    cuda-libraries-dev=11.7 \
    -y

run_conda install --prefix "$ENV_PREFIX" -c conda-forge \
    gcc_linux-64=11.4.0 \
    gxx_linux-64=11.4.0 \
    -y

run_conda install --prefix "$ENV_PREFIX" -c conda-forge \
    "pandas>=1.2.0" \
    tectonic \
    -y

# Installing pandas later can pull CPU torch variants back in.
# Re-pin the CUDA torch stack exactly as seen in the live env.
run_conda install --prefix "$ENV_PREFIX" -c pytorch -c nvidia -c conda-forge -c defaults \
    pytorch=1.13.1=py3.10_cuda11.7_cudnn8.5.0_0 \
    pytorch-cuda=11.7 \
    torchvision=0.14.1=py310_cu117 \
    torchaudio=0.13.1=py310_cu117 \
    numpy=1.26.4 \
    -y

patch_mkl_activation_hooks
write_runtime_activation_hooks

PYTHON_BIN="$ENV_PREFIX/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "setuptools<81"

"$PYTHON_BIN" -m pip install \
    ConfigArgParse==1.7.5 \
    ImageIO==2.37.3 \
    PyOpenGL==3.1.0 \
    gdown==6.0.0 \
    glfw==2.10.0 \
    imageio-freeimage==0.1.0 \
    lpips==0.1.4 \
    matplotlib==3.8.4 \
    ninja==1.13.0 \
    opencv-python==4.13.0.92 \
    pytorch-msssim==1.0.0 \
    scipy==1.15.3 \
    tqdm==4.67.3 \
    xatlas==0.0.11

"$PYTHON_BIN" -m pip install \
    "kaolin==0.15.0" \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html

NVDR_TMP=$(mktemp -d "${TMPDIR:-/tmp}/nvdiffrast_v040.XXXXXX")
git clone --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git "$NVDR_TMP"
install_nvdiffrast "$NVDR_TMP" "$PYTHON_BIN"

TCNN_TMP=$(mktemp -d "${TMPDIR:-/tmp}/tiny_cuda_nn_v20.XXXXXX")
git clone --recursive --branch v2.0 --depth 1 https://github.com/NVlabs/tiny-cuda-nn.git "$TCNN_TMP"
(
    cd "$TCNN_TMP/bindings/torch"
    "$PYTHON_BIN" setup.py install --no-networks
)

install_blender_bundle

if [[ "$INSTALL_EXTRAS" -eq 1 ]]; then
    "$PYTHON_BIN" -m pip install \
        addict==2.4.0 \
        dash==4.2.0 \
        ipykernel==7.2.0 \
        ipywidgets==8.1.8 \
        open3d==0.19.0 \
        plotly==6.8.0 \
        pyquaternion==0.9.9 \
        pyrender==0.1.45 \
        scikit-learn==1.7.2 \
        trimesh==4.12.2
fi

# Keep the final numpy/scipy pair aligned with the tested working env.
"$PYTHON_BIN" -m pip install --force-reinstall --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.15.3

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import sys

import cv2
import kaolin
import numpy
import scipy
import torch
import nvdiffrast.torch  # noqa: F401
import tinycudann  # noqa: F401
import xatlas  # noqa: F401

repo_root = os.environ["GSHELL_REPO_ROOT"]
sys.path.insert(0, repo_root)
import render.renderutils  # noqa: F401,E402
import render.optixutils  # noqa: F401,E402

print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("cv2", cv2.__version__)
for pkg in ["kaolin", "nvdiffrast", "tinycudann", "xatlas"]:
    print(pkg, md.version(pkg))
print("render plugins import successfully")
PY

"$ENV_PREFIX/bin/blender" --version | head -n 2

echo
echo "GShell env created at: $ENV_PREFIX"
echo "After activation, repo-specific CUDA/cache vars are set automatically."
echo "Blender is available at: $ENV_PREFIX/bin/blender"
