# GShell Env Recreation

This repo does not contain a single pristine env file for the live `GShell_env`, so this guide is based on the environment that is currently installed at:

- `/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env`

## What We Recovered

The live env gives us three useful layers:

1. The Conda history:
   - Python `3.10`
   - PyTorch `1.13.1` with CUDA `11.7`
   - `mkl=2022.1.0`
   - CUDA dev packages for building extensions
   - GCC/G++ `11.4.0`
   - `pandas>=1.2.0`
   - `tectonic`
2. The current imported package set:
   - core training packages like `nvdiffrast`, `tinycudann`, `kaolin`, `xatlas`, `glfw`, `PyOpenGL`
   - analysis/debug packages like `open3d`, `plotly`, `dash`, `pyrender`, `trimesh`, `ipykernel`
3. The exact Conda lockfile:
   - [`environment_gshell_explicit.txt`](environment_gshell_explicit.txt)

## Important Caveat

Two packages in the live env were installed from local build directories that no longer exist:

- `nvdiffrast 0.4.0`
- `tinycudann 2.0`

So we cannot recreate those two from the original local folders. The clean replacement is:

- install `nvdiffrast` from the upstream `v0.4.0` tag
- install `tiny-cuda-nn` from the upstream `v2.0` tag

That gives us the same package versions, but not a byte-for-byte copy of the old local build tree.

## What Was Actually Broken

The straightforward pip commands for those two packages do not work cleanly in a fresh rebuild anymore. The working rebuild needs four fixes:

1. `setuptools<81`
   - newer setuptools drops the `pkg_resources` import path that `torch.utils.cpp_extension` still touches in this stack
2. force the Conda GCC 11 toolchain
   - otherwise CUDA 11.7 tries to build against the system `g++ 13.x`, which it rejects
3. add the system driver-library paths at link time
   - otherwise `tiny-cuda-nn` fails on `cannot find -lcuda`
4. install `nvdiffrast` and `tiny-cuda-nn` from cloned source trees
   - `nvdiffrast` needs the classic `setup.py install` path here
   - `tiny-cuda-nn` still expects `setup.py install --no-networks`, and modern pip no longer forwards that flag the old way

There is also one package-compatibility pin that matters in practice:

- final `numpy` must be `1.26.4`
- final `scipy` must be `1.15.3`

The rebuilt env initially drifted to `numpy 2.2.6`, which broke `kaolin 0.15.0`. The checked-in script now forces the final `numpy/scipy` pair back to the versions that were verified against the live working env.

## Recommended Rebuild Path

Use the checked-in script:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
./scripts/recreate_gshell_env.sh
```

By default it recreates the full working env, including the notebook/debug extras.

If you only want the training/runtime stack:

```bash
./scripts/recreate_gshell_env.sh --core-only
```

If you want to place the env somewhere else:

```bash
./scripts/recreate_gshell_env.sh --prefix /some/other/path/GShell_env
```

If you want the rebuilt env to reuse the existing caches under `/data/abelde`, leave the default alone.

If you want a different cache root, set:

```bash
export SHELL_GAUSSIAN_CACHE_ROOT=/some/cache/root
```

before running the script.

## Files In This Folder

- [`environment_gshell_rebuild.yml`](environment_gshell_rebuild.yml)
  - minimal Conda rebuild spec recovered from the live env history
- [`environment_gshell_explicit.txt`](environment_gshell_explicit.txt)
  - exact Conda package lock from the live env
- [`scripts/recreate_gshell_env.sh`](scripts/recreate_gshell_env.sh)
  - practical rebuild script that restores the Conda layer and the pip/source-built packages

## Validation

After rebuild, these should work:

```bash
/path/to/GShell_env/bin/python - <<'PY'
import importlib.metadata as md
import torch
import kaolin
import nvdiffrast.torch
import tinycudann
import xatlas

print("torch", torch.__version__, "cuda", torch.version.cuda)
for pkg in ["kaolin", "nvdiffrast", "tinycudann", "xatlas"]:
    print(pkg, md.version(pkg))
PY
```

And from the repo root, the local CUDA plugins should also import:

```bash
/path/to/GShell_env/bin/python - <<'PY'
import os
import sys

repo_root = "/data/abelde/projects/active/Shell_Gaussian/baselines/GShell"
sys.path.insert(0, repo_root)
import render.renderutils
import render.optixutils
print("render plugins import successfully")
PY
```

In the tested rebuilt env, the following also succeeded:

- `import kaolin`
- `import cv2`
- import of `dataset.dataset_nerf_colmap`
- import of `geometry.gshell_tets_geometry`
- `python train_gshelltet_polycam.py --help` from both `FootShellGaussian` and `baselines/GShell`

Expected core versions:

- `torch 1.13.1`
- CUDA runtime `11.7`
- `kaolin 0.15.0`
- `nvdiffrast 0.4.0`
- `tinycudann 2.0`
- `xatlas 0.0.11`

## Known Warning

The live env currently emits this warning when `pandas` imports `numexpr`:

```text
Pandas requires version '2.8.4' or newer of 'numexpr' (version '2.7.3' currently installed).
```

That warning is present in the current environment too. It does not block the core GShell training stack, so this rebuild guide preserves the live environment behavior instead of silently changing package versions.
