# Golden Set Evaluation Dataset

This directory is the single supported preprocessing location for the external
GLB evaluation set. Use `pipeline.py`; the other Python files are internal
workers or explicit trainer adapters.

## Dataset Contract

```text
external/golden_set_eval_glb/
  <shoe>.glb

raw/golden_set_evaluation/
  <shoe>/
    images/img001.jpg ... img180.jpg
    masks/img001.png  ... img180.png

processed/golden_set_evaluation_colmap/
  <shoe>/
    undistorted/images/
    undistorted/masks/
    undistorted/sparse/0/
    logs/
    conversion_manifest.json
```

The raw dataset is the only owner of the original RGB images and masks. Blender
camera transforms, depth, train/test splits, canonical meshes, and summaries
are deliberately not written there. COLMAP reads the raw RGB images directly
and estimates camera intrinsics, camera poses, and sparse points. Masks do not
influence reconstruction; they are geometrically undistorted after COLMAP
finishes.

The processed dataset contains only derived, training-ready outputs. The
distorted COLMAP model is retained temporarily while masks are aligned, then
removed together with COLMAP's unused `undistorted/stereo` directory. Original
RGB images and masks are never duplicated under the processed root.

## Commands

```bash
cd /storage/Abhinay/Shell_Gaussian
PYTHON=/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python
PIPELINE=dataset_tools/golden_set_evaluation/pipeline.py

# Render one GLB into exactly 180 RGB/mask pairs.
$PYTHON $PIPELINE render --shoe air_jordan_1

# Validate one raw scene without changing it.
$PYTHON $PIPELINE validate-render --shoe air_jordan_1

# Estimate fresh poses with COLMAP, then align that scene's masks.
$PYTHON $PIPELINE colmap --scene air_jordan_1 --gpu-index 0 --overwrite

# Require all 180 images to be registered before accepting the scene.
$PYTHON $PIPELINE validate-colmap \
  --scene air_jordan_1 --require-all-registered
```

Use `--all` instead of `--scene` for COLMAP only after a pilot scene succeeds.
Use `render` without `--shoe` to render every GLB. Existing outputs are never
replaced unless `--overwrite` is passed.

## Trainer Adapters

The common COLMAP dataset is the source of truth. Trainer adapters consume it;
they never use Blender camera metadata.

```bash
# Physical-copy GShell layout plus COLMAP-derived transforms.json.
$PYTHON $PIPELINE prepare-gshell --shoe air_jordan_1 --overwrite

# Optional SuGaR RGBA and foreground-bounding-box dataset.
$PYTHON $PIPELINE prepare-sugar --scene air_jordan_1 --overwrite
```

`prepare-gshell` is strict about 180 registered frames when called through this
pipeline. The underlying `gshell_adapter.py` remains frame-count agnostic so it
also supports the existing 36-view turntable dataset.

## File Responsibilities

- `pipeline.py`: only public entry point.
- `blender_renderer.py`: Blender worker; writes RGB and masks only.
- `colmap_pipeline.py`: fresh RGB-only COLMAP reconstruction, reading raw RGB in place.
- `align_colmap_masks.py`: aligns raw masks and compacts the processed scene.
- `validate_colmap.py`: validates the common processed dataset.
- `gshell_adapter.py`: converts COLMAP cameras to GShell input.
- `sugar_adapter.py` and `validate_sugar.py`: optional SuGaR-specific RGBA view without repeating mask undistortion.
