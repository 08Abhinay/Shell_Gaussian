# Direct Blender Evaluation Dataset Pipeline

This directory builds the reviewed evaluation GLBs directly into a GShell- and
FootShellGaussian-ready dataset. Blender supplies the exact camera poses. The
optional SuGaR export uses COLMAP only to match features and triangulate sparse
points; it never estimates or modifies the camera orbit.

The existing `dataset_tools/`, datasets, loaders, and baseline code are not
modified by this pipeline.

## Inputs And Outputs

Default input:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb
```

Default output:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender
```

Optional derived SuGaR output:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender_sugar
```

Each published shoe contains one shared copy of every asset:

```text
<shoe>/
  image/img001.jpg ... img180.jpg
  mask/img001.png ... img180.png
  invdepth/img001.npy ... img180.npy
  transforms.json
  transforms_train.json
  transforms_test.json
  reference_mesh.ply
  blender_canonicalization.json
```

`transforms_train.json` uses 150 frames. `transforms_test.json` uses every sixth
frame, so its 30 frames are distributed equally across the five elevation
rings. Both files reference the shared `image/`, `mask/`, and `invdepth/`
directories.

## Reviewed Manifest

[`evaluation_manifest.json`](evaluation_manifest.json) is the production source
of truth for semantic orientation and pair selection. Each GLB has an explicit
heel-to-toe, width, and physical-up axis plus a SHA256 checksum. `build` and
`validate` reject an unreviewed, missing, additional, or changed GLB.

When replacing or adding a download, add its checksum and provisional explicit
axes with `reviewed: false`, render temporary cardinal views with `audit`, then
correct the entry and deliberately set `reviewed: true`. `build` and `validate`
will continue to reject it until that final step. Do not bypass the checksum
gate.

## Camera Contract

- Resolution: 1536 x 1024.
- Horizontal field of view: 21 degrees.
- Radius: exactly 1.0 for every camera.
- Elevations: 0, -25, 20, 45, and 65 degrees.
- Views: 36 per elevation, 180 total.
- Azimuth: `-90 + 10 * index` degrees within each ring.
- `img001`: level reference-side view with the toe pointing right.
- Canonical Blender geometry: +X heel-to-toe, +Y width, +Z physical up.

The object is centered once and uniformly scaled once for the whole scene. The
scale targets 84% horizontal occupancy in `img001`, but is reduced when needed
to keep every elevated and rotated view away from the image border. Cameras are
never moved to compensate for object shape.

The current GShell loader right-multiplies model-view by `Rx(-90 degrees)`.
Accordingly, the JSON stores:

```text
saved_c2w = Rx(-180 degrees) @ blender_c2w
```

The loader therefore recovers the intended effective pose:

```text
effective_c2w = Rx(-90 degrees) @ blender_c2w
```

The aligned `reference_mesh.ply` is exported as binary little-endian PLY in
that same effective world frame. RGB rendering always uses the unmodified
Blender camera pose.

## Commands

Use the project environment for the public pipeline command:

```bash
PYTHON=/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python
PIPELINE=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py
```

Render temporary semantic audit views for all reviewed assets:

```bash
$PYTHON $PIPELINE audit --all --gpus 2,3
```

Build one shoe:

```bash
$PYTHON $PIPELINE build --shoe birkenstock_arizona_sandal --gpu 2
```

Build all shoes with two workers:

```bash
$PYTHON $PIPELINE build --all --gpus 2,3
```

Existing valid scenes are validated and skipped by default; invalid existing
scenes fail instead of being replaced. Use `--overwrite` only when deliberately
regenerating a scene. Each new scene is first written into a hidden temporary
directory, fully validated, and atomically renamed into place.

Validate one or all published scenes without rebuilding:

```bash
$PYTHON $PIPELINE validate --shoe birkenstock_arizona_sandal
$PYTHON $PIPELINE validate --all
```

Useful optional overrides are `--source-root`, `--output-root`, `--manifest`,
and `--blender`.

## SuGaR Export

SuGaR needs a COLMAP model and an initial sparse point cloud. The exporter
reads the already validated effective cameras from `transforms.json`, converts
them to COLMAP's OpenCV camera axes, and writes them into a fixed seed model.
It then runs masked SIFT extraction, exhaustive matching, and
`point_triangulator`. It deliberately never runs `mapper` or bundle adjustment.

The SuGaR dataset is stored separately so the GShell dataset remains unchanged:

```text
<shoe>/
  undistorted/
    images/img001.png ... img180.png
    masks/img001.png ... img180.png
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt
      cameras.bin
      images.bin
      points3D.bin
  masked_colmap_manifest.json
```

The PNG images contain the Blender mask as their alpha channel. Inverse depth
and `reference_mesh.ply` are neither copied nor read, so SuGaR receives no
ground-truth geometry.

Prepare and validate one pilot scene:

```bash
$PYTHON $PIPELINE prepare-sugar --shoe air_jordan_1 --gpu 2
$PYTHON $PIPELINE validate-sugar --shoe air_jordan_1
```

Prepare all scenes with two workers, then validate the published dataset:

```bash
$PYTHON $PIPELINE prepare-sugar --all --gpus 2,3
$PYTHON $PIPELINE validate-sugar --all
```

Existing valid SuGaR scenes are validated and skipped. `--overwrite` rebuilds
one deliberately. Preparation happens in a hidden temporary directory and is
published only after image, mask, camera, sparse-point, and bounding-box checks
pass. Optional overrides are `--input-root`, `--output-root`, and
`--colmap-bin`.

## GShell Polycam Smoke Test

After the Air Jordan and Birkenstock scenes validate, run the existing Polycam
trainer without changing baseline code:

```bash
GSHELL_DATASET_ROOT=/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender \
GSHELL_CONFIG=/storage/Abhinay/Shell_Gaussian/baselines/GShell/configs/shoes_mc_normfix_512_768_depth.json \
GSHELL_CONDA_ENV=/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv \
GSHELL_OUTPUT_ROOT=/storage/Abhinay/Shell_Gaussian/baselines/GShell/output/golden_set_evaluation_blender_polycam_smoke \
GSHELL_OUT_SUFFIX=_depth_polycam_blender \
GSHELL_TRAINER=polycam \
SKIP_EXISTING=0 \
ALLOWED_GPUS=2,3 \
MAX_PARALLEL_JOBS=2 \
MIN_FREE_MB=40000 \
bash /storage/Abhinay/Shell_Gaussian/baselines/GShell/scripts/train_all_shoes_tmux.sh \
  golden_eval_blender_polycam_smoke \
  air_jordan_1 \
  birkenstock_arizona_sandal
```

Check the GT/prediction panels for matching orientation before starting a full
22-shoe build. This smoke test consumes the generated split JSON files and
first-surface inverse depth.

## Validation Guarantees

Before publication, `build` verifies:

- exactly 180 RGB images, masks, inverse-depth arrays, and all-pose entries;
- the deterministic radius, FOV, elevation, and azimuth schedule;
- valid camera rotations and exact recovery through the current GShell loader;
- nonempty masks that do not touch an image border;
- inverse-depth/mask IoU of at least 0.98;
- a nonempty reference mesh and sampled mesh-to-mask projection consistency;
- 150/30 train/test split sizes without duplicating image files.

Before publication, `prepare-sugar` additionally verifies:

- exactly 180 RGBA images, masks, and registered COLMAP cameras;
- exact camera and intrinsic recovery after OpenGL-to-OpenCV conversion;
- unchanged radius, orbit ordering, and orientation;
- nonempty finite sparse points and reprojection errors;
- a robust sparse-point bounding box containing at least 95% of points;
- absence of inverse depth and ground-truth mesh assets.

Second-surface depth remains outside this pipeline.

Inverse-depth files remain ordinary dense-shape float32 `.npy` arrays to the
loader. Their zero-valued background pages are allocated as sparse filesystem
holes, reducing physical disk use without compression, quantization, or a
loader change. Use sparse-aware copy tools (for example, `cp --sparse=always`)
when relocating the generated dataset.
