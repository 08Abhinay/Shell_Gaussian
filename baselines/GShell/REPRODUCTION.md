# GShell Reproduction Manual

This file is the practical reproduction guide for the shoe pipeline used in this repo.

It does **not** replace the original paper README. The original [README.md](README.md) explains G-Shell itself. This file explains the exact steps we use here for:

1. GShell training
2. watertight mesh export
3. foot-aware alignment
4. pseudo-last generation

It also shows the one-command full pipeline entry point.

## 1. Repository pieces

The pipeline is split across two parts of this repo:

- `baselines/GShell`: GShell training and watertight export
- `FootShellGaussian`: downstream shoe-specific processing

Main entry points:

- GShell single-shoe training: [scripts/train_shoe.sh](scripts/train_shoe.sh)
- GShell batch training in tmux: [scripts/train_all_shoes_tmux.sh](scripts/train_all_shoes_tmux.sh)
- Watertight export: [scripts/export_watertight_meshes.py](scripts/export_watertight_meshes.py)
- Foot-aware alignment: [../../FootShellGaussian/scripts/run_foot_aware_alignment_pipeline.py](../../FootShellGaussian/scripts/run_foot_aware_alignment_pipeline.py)
- Pseudo-last build: [../../FootShellGaussian/scripts/run_pseudo_last_builder.py](../../FootShellGaussian/scripts/run_pseudo_last_builder.py)
- Full 4-stage pipeline: [../../FootShellGaussian/scripts/run_final_pipeline.sh](../../FootShellGaussian/scripts/run_final_pipeline.sh)

## 2. Raw turntable dataset -> processed GShell dataset

Before GShell training, the raw turntable dataset has to be converted into the processed GShell layout.

The raw source we use is:

```text
/data/abelde/datasets/raw/golden_set
```

Per shoe, the raw layout is expected to be:

```text
<shoe_name>/
  images/
  masks/
  colmap/
    cameras.txt
    images.txt
    points3D.txt
```

The current one-step exporter is:

- [../../tools/gshell/export_turntable_to_gshell_canonical.py](../../tools/gshell/export_turntable_to_gshell_canonical.py)

That script replaced the older two-step flow:

- `export_shoes_to_gshell.py`
- `canonicalize_gshell_turntable_phase.py`

It does all of this in one pass:

- reads raw COLMAP cameras and poses
- converts them into GShell `transforms.json`
- applies the deterministic turntable phase alignment so `img01.jpg` lands at the chosen canonical angle
- symlinks `images/` to `image/`
- symlinks `masks/` to `mask/`
- writes `turntable_canonicalization.json`
- optionally attaches `invdepth/`
- optionally writes size-normalization metadata

## 3. Creating `/data/abelde/datasets/processed/gshell_shoes`

This is the standard processed turntable dataset:

```text
/data/abelde/datasets/processed/gshell_shoes
```

To create it from the raw `golden_set`:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

python tools/gshell/export_turntable_to_gshell_canonical.py \
  --input-dir /data/abelde/datasets/raw/golden_set \
  --output-dir /data/abelde/datasets/processed/gshell_shoes \
  --reference-frame img01.jpg \
  --target-angle-deg 90 \
  --overwrite
```

Notes:

- this does **not** modify the raw dataset
- `image/` and `mask/` in the processed dataset are symlinks back to the raw data
- the per-shoe transform canonicalization is recorded in `turntable_canonicalization.json`

If you want to process only a subset:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

python tools/gshell/export_turntable_to_gshell_canonical.py \
  --input-dir /data/abelde/datasets/raw/golden_set \
  --output-dir /data/abelde/datasets/processed/gshell_shoes \
  --shoe Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant \
  --shoe Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail \
  --reference-frame img01.jpg \
  --target-angle-deg 90 \
  --overwrite
```

If a shoe should ignore sparse COLMAP points and normalize from cameras only:

```bash
--camera-only-shoe <SHOE_NAME>
```

That option exists for difficult scenes where the sparse point cloud is unreliable.

## 4. Creating `/data/abelde/datasets/processed/gshell_shoes_size_metadata`

This dataset was created from the same raw `golden_set`, using the same exporter, but with the optional size metadata mode enabled:

```text
/data/abelde/datasets/processed/gshell_shoes_size_metadata
```

Important point: this mode is **metadata only**.

It does **not** rescale the images or physically rewrite the shoe scene into a new metric size. Instead, it computes a per-shoe uniform scale factor and records it in:

- each shoe’s `turntable_canonicalization.json`
- dataset-level `summary.json`
- dataset-level `summary.csv`

For normal shoes, the size rule is:

- `longest_dim`

For boots, the size rule is:

- `footprint_diag_xz`

The exact settings recorded in the current dataset are:

- `size_normalization = metadata_only`
- `target_size = 1.0`
- boots:
  - `Ugg-Bailey-Bow-Ii-Boot-Ribbon-Red-Kids`
  - `Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler`

Reproduction command:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

python tools/gshell/export_turntable_to_gshell_canonical.py \
  --input-dir /data/abelde/datasets/raw/golden_set \
  --output-dir /data/abelde/datasets/processed/gshell_shoes_size_metadata \
  --reference-frame img01.jpg \
  --target-angle-deg 90 \
  --size-normalization metadata_only \
  --target-size 1.0 \
  --boot-shoe Ugg-Bailey-Bow-Ii-Boot-Ribbon-Red-Kids \
  --boot-shoe Ugg-Classic-Short-Ii-Boot-Rock-Rose-Toddler \
  --overwrite
```

So, in plain terms:

- `gshell_shoes` = processed training dataset
- `gshell_shoes_size_metadata` = same kind of processed dataset, plus per-shoe size statistics and a recommended uniform scale factor

## 5. Optional inverse-depth attachment during export

If inverse-depth already exists somewhere else and you want to attach it while building the processed dataset, the exporter supports that too.

Example:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

python tools/gshell/export_turntable_to_gshell_canonical.py \
  --input-dir /data/abelde/datasets/raw/golden_set \
  --output-dir /data/abelde/datasets/processed/gshell_shoes \
  --invdepth-source-root /some/root/with/<shoe>/invdepth \
  --invdepth-mode symlink \
  --overwrite
```

This creates, per shoe:

```text
invdepth/
invdepth_summary.json
```

If `--invdepth-source-root` is omitted, no inverse-depth is attached at export time.

## 6. Environment

Project root:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
```

GShell root:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
```

GShell env used by the local scripts:

```text
/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env
```

The wrappers already activate that env internally, so in most cases you do **not** need to activate it by hand. If you want to work manually:

```bash
eval "$(conda shell.bash hook)"
conda activate /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env
```

## 7. Dataset layout expected by GShell

The current turntable dataset root is:

```text
/data/abelde/datasets/processed/gshell_shoes
```

Each shoe folder is expected to look like:

```text
<shoe_name>/
  image/
  mask/
  transforms.json
  turntable_canonicalization.json
```

If you use a depth-enabled config, the dataset must also contain:

```text
  invdepth/
```

If you use second-layer depth, it must also contain:

```text
  invdepth_second/
```

GShell will fail on startup if depth is enabled in the config but those files are missing.

## 8. Configs we currently use

Live shoe configs are in [configs](configs):

- `shoes_mc_normfix_512_768.json`
- `shoes_mc_normfix_512_768_depth.json`
- `shoes_mc_normfix_512_768_depth2.json`
- `shoes_mc_normfix_512_768_msdf_mlp.json`
- `shoes_mc_normfix_512_768_depth_msdf_mlp.json`

Typical meaning:

- `shoes_mc_normfix_512_768.json`: RGB + masks only
- `shoes_mc_normfix_512_768_depth.json`: RGB + masks + inverse depth
- `shoes_mc_normfix_512_768_depth2.json`: RGB + masks + inverse depth + second-layer depth
- `shoes_mc_normfix_512_768_msdf_mlp.json`: RGB + masks with mSDF MLP enabled
- `shoes_mc_normfix_512_768_depth_msdf_mlp.json`: depth + mSDF MLP

## 9. Stage 1: Train one shoe

Example:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
bash scripts/train_shoe.sh Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant 0
```

What this does:

- reads the shoe from `/data/abelde/datasets/processed/gshell_shoes`
- uses the default config from `GSHELL_CONFIG` if you override it, otherwise the script default
- writes output under `baselines/GShell/output`

Useful overrides:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes \
GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768_depth.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
bash scripts/train_shoe.sh Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant 0
```

The training log is written to:

```text
<output_root>/<shoe_name><suffix>/logs/train.log
```

## 10. Stage 1: Train many shoes in tmux

Run the whole dataset:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes \
GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
bash scripts/train_all_shoes_tmux.sh final_gshell_batch
```

Run only selected shoes:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes \
GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
bash scripts/train_all_shoes_tmux.sh final_gshell_batch \
  Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant \
  Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail
```

Useful scheduler controls:

```bash
MIN_FREE_MB=51200
MAX_PARALLEL_JOBS=0
ALLOWED_GPUS=0,1
SKIP_EXISTING=1
```

Example:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
MIN_FREE_MB=45000 \
MAX_PARALLEL_JOBS=2 \
ALLOWED_GPUS=0,1 \
GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes \
GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768_depth.json \
GSHELL_OUT_SUFFIX=_turntable_depth \
GSHELL_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
bash scripts/train_all_shoes_tmux.sh final_gshell_depth_batch
```

The launcher will print the tmux session name and batch log path. Attach with:

```bash
tmux attach -t final_gshell_batch
```

## 11. Stage 2: Export watertight meshes

After training is complete:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
eval "$(conda shell.bash hook)"
conda activate /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env

python scripts/export_watertight_meshes.py \
  --output-root /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
  --config /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json \
  --overwrite
```

Export only one trained scene:

```bash
python scripts/export_watertight_meshes.py \
  --output-root /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
  --config /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json \
  --scene Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant_turntable \
  --overwrite
```

This writes:

```text
<scene_dir>/mesh_watertight/mesh.obj
```

and a summary:

```text
<output_root>/watertight_export_summary.json
```

## 12. Stage 3: Foot-aware alignment

This stage needs both:

- the open GShell mesh: `mesh/mesh.obj`
- the watertight GShell mesh: `mesh_watertight/mesh.obj`

Example:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
eval "$(conda shell.bash hook)"
conda activate /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env

python FootShellGaussian/scripts/run_foot_aware_alignment_pipeline.py \
  --mesh-root /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
  --output-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/foot_aware_alignment \
  --device cuda \
  --overwrite
```

Run only one shoe:

```bash
python FootShellGaussian/scripts/run_foot_aware_alignment_pipeline.py \
  --mesh-root /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/output/final/gshell \
  --output-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/foot_aware_alignment \
  --shoe-name Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant \
  --device cuda \
  --overwrite
```

Main output per shoe:

```text
foot_aligned_initial.obj
foot_aligned_optimized.obj
fit_metrics.json
support/
```

## 13. Stage 4: Pseudo-last generation

Example:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
eval "$(conda shell.bash hook)"
conda activate /data/abelde/projects/active/Shell_Gaussian/baselines/GShell/GShell_env

python FootShellGaussian/scripts/run_pseudo_last_builder.py \
  --alignment-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/foot_aware_alignment \
  --output-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/pseudo_last \
  --device cuda \
  --allow-non-normal \
  --overwrite
```

Run only one shoe:

```bash
python FootShellGaussian/scripts/run_pseudo_last_builder.py \
  --alignment-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/foot_aware_alignment \
  --output-root /data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/pseudo_last \
  --shoe-name Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant \
  --device cuda \
  --allow-non-normal \
  --overwrite
```

Main output per shoe:

```text
pseudo_last.obj
pseudo_last_metrics.json
pseudo_last_sdf.npz
```

## 14. One-command full 4-stage pipeline

If you want the full run from training through pseudo-last:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
bash FootShellGaussian/scripts/run_final_pipeline.sh
```

Important defaults in that script:

- `DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes`
- `FINAL_ROOT=/data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final`
- `GSHELL_OUTPUT_ROOT=${FINAL_ROOT}/gshell`
- `ALIGNMENT_ROOT=${FINAL_ROOT}/foot_aware_alignment`
- `PSEUDO_LAST_ROOT=${FINAL_ROOT}/pseudo_last`
- `GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json`
- `GSHELL_OUT_SUFFIX=_turntable`

Run a subset of shoes:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
bash FootShellGaussian/scripts/run_final_pipeline.sh \
  Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant \
  Nike-Cortez-Se-Suede-Pacific-Moss-Infinite-Gold-Muslin-Sail
```

Run only later stages:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
START_STAGE=2 END_STAGE=4 \
bash FootShellGaussian/scripts/run_final_pipeline.sh
```

Stage numbering in that script:

- `1`: GShell training
- `2`: watertight mesh export
- `3`: foot-aware alignment
- `4`: pseudo-last build

## 15. Logs and outputs

### GShell single-shoe training

```text
baselines/GShell/output/<shoe_name><suffix>/logs/train.log
```

### GShell tmux batch training

```text
<GSHELL_OUTPUT_ROOT>/batch_runs/<session_name>_<timestamp>/batch.log
```

### Final 4-stage pipeline

```text
FootShellGaussian/output/final/logs/final_pipeline_<timestamp>.log
```

### Final output roots

```text
FootShellGaussian/output/final/gshell
FootShellGaussian/output/final/foot_aware_alignment
FootShellGaussian/output/final/pseudo_last
```

## 16. Reproducing the current turntable run

This is the closest compact recipe for the current turntable pipeline:

```bash
cd /data/abelde/projects/active/Shell_Gaussian

bash FootShellGaussian/scripts/run_final_pipeline.sh
```

If you want to run it in pieces:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell

GSHELL_DATASET_ROOT=/data/abelde/datasets/processed/gshell_shoes \
GSHELL_CONFIG=/data/abelde/projects/active/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768.json \
GSHELL_OUT_SUFFIX=_turntable \
GSHELL_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/FootShellGaussian/output/final/gshell \
bash scripts/train_all_shoes_tmux.sh final_gshell_batch
```

Then:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
START_STAGE=2 END_STAGE=4 \
bash FootShellGaussian/scripts/run_final_pipeline.sh
```

## 17. Common failure points

### Missing depth targets

If you use a depth config but the dataset has no `invdepth/`, training will fail immediately.

### Wrong config for dataset

Make sure the config matches the dataset contents:

- RGB-only dataset -> non-depth config
- dataset with `invdepth/` -> depth config
- dataset with `invdepth_second/` -> depth2 config

### No `ninja`

The training wrapper checks for `ninja` before training. If it is missing, local CUDA/PyTorch extensions will not build.

### No free GPU

The batch launcher waits until `nvidia-smi` shows a GPU above `MIN_FREE_MB`.

### Missing watertight mesh at alignment time

Alignment needs both:

- `mesh/mesh.obj`
- `mesh_watertight/mesh.obj`

so do not skip Stage 2 if you plan to run Stage 3.

## 18. Quick command index

Train one shoe:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
bash scripts/train_shoe.sh <SHOE_NAME> <GPU_ID>
```

Train many shoes in tmux:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
bash scripts/train_all_shoes_tmux.sh <SESSION_NAME> [SHOE_NAME ...]
```

Export watertight:

```bash
cd /data/abelde/projects/active/Shell_Gaussian/baselines/GShell
python scripts/export_watertight_meshes.py --output-root <ROOT> --config <CONFIG> --overwrite
```

Run alignment:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
python FootShellGaussian/scripts/run_foot_aware_alignment_pipeline.py --mesh-root <ROOT> --output-root <ROOT> --device cuda --overwrite
```

Run pseudo-last:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
python FootShellGaussian/scripts/run_pseudo_last_builder.py --alignment-root <ROOT> --output-root <ROOT> --device cuda --allow-non-normal --overwrite
```

Run all 4 stages:

```bash
cd /data/abelde/projects/active/Shell_Gaussian
bash FootShellGaussian/scripts/run_final_pipeline.sh
```
