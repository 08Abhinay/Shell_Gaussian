# Direct Blender Evaluation Dataset Pipeline

This directory builds the reviewed evaluation GLBs directly into a GShell- and
FootShellGaussian-ready dataset. Blender supplies the exact camera poses. The
optional SuGaR export uses COLMAP only to match features and triangulate sparse
points; it never estimates or modifies the camera orbit.

The existing `dataset_tools/`, datasets, loaders, and baseline code are not
modified by this pipeline.

The public CLI remains `pipeline.py`, while implementation details are split by
consumer:

```text
dataset_tools_blender/
  pipeline.py          # command-line parser and command dispatch
  core.py              # shared Blender dataset contracts and utilities
  sugar/pipeline.py    # SuGaR conversion and validation
  neuraludf/pipeline.py
  neus2/pipeline.py
```

Existing commands and imports from `dataset_tools_blender.pipeline` remain
compatible. New baseline-specific code should be added to its corresponding
subpackage rather than to the CLI module.

## Inputs And Outputs

Default input:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb
```

Default output:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

Optional derived SuGaR output:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/sugar/golden_set_evaluation
```

Optional derived NeuralUDF output:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/neuraludf/golden_set_evaluation
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

[`manifests/baseline_evaluation_manifest.json`](manifests/baseline_evaluation_manifest.json)
is the source of truth for the broader baseline-export inventory used by Milo,
SuGaR, and the public pipeline defaults. The FootShellGaussian golden-set
workflow has its own manifest described below.

Each manifest entry has an explicit heel-to-toe, width, and physical-up axis
plus a SHA256 checksum. `build` and `validate` reject an unreviewed, missing,
additional, or changed GLB.

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

## Golden-Set Evaluation Workflow

The active FootShellGaussian dataset has one stable manifest and one stable
processed location:

```text
raw GLBs:
/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean

manifest:
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json

processed dataset:
/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

[`golden_set_evaluation_manifest.json`](golden_set_evaluation_manifest.json)
is the reviewed source of truth for these 17 normal right shoes. It uses
`inventory_policy = "listed_subset"`: every listed GLB must exist and match its
checksum, while unrelated future files in the raw directory do not invalidate
the reviewed set. The two high-heel assets remain deferred.

After source-axis mapping and right-shoe selection, the worker estimates the
remaining top-view heading from lower-shoe geometry and rotates the dominant
length direction onto `+X`. It then centers and uniformly scales the complete
selected shoe. Footbed detection is not part of dataset orientation.

For a new or changed GLB:

1. Add its checksum and provisional reviewed axes to the manifest.
2. Record whether the source contains one shoe or a pair, including the rule
   that selects the right shoe.
3. Run the Blender audit and inspect side, toe, heel, top, and bottom views.
4. Correct the manifest until physical up, heading, and handedness are right.
5. Set `reviewed: true`, build, and validate transactionally.

Build the complete dataset in `tmux` on five GPUs:

```bash
cd /storage/Abhinay/Shell_Gaussian
dataset_tools_blender/build_golden_set_evaluation.sh
```

Monitor it with:

```bash
tmux attach -t golden-set-evaluation-build
tail -f /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/dataset-build.log
```

Validate the published dataset without rebuilding:

```bash
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  dataset_tools_blender/pipeline.py validate --all \
  --source-root /home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean \
  --manifest dataset_tools_blender/golden_set_evaluation_manifest.json \
  --output-root /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

The manifest and dataset names remain stable. Future revisions are built into
a temporary directory, validated, and then published at the same canonical
location; Git history records how the dataset changed.

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

## MILo Export

The MILo exporter consumes the validated G-Shell images, masks, and exact
Blender cameras. It writes MILo's Blender-style JSON layout and triangulates
the initial sparse point cloud from training images only. Test images are
available for evaluation but never appear in COLMAP point tracks. The output
has no top-level `sparse/` directory, so stock MILo honors the explicit split.

```bash
$PYTHON $PIPELINE prepare-milo --shoe air_jordan_1 --gpu 0
$PYTHON $PIPELINE validate-milo --shoe air_jordan_1
$PYTHON $PIPELINE prepare-milo-turntable --shoe air_jordan_1 --gpu 0
$PYTHON $PIPELINE validate-milo-turntable --shoe air_jordan_1
```

The pilot-first five-shoe workflow runs both variants in a detached tmux
session and records persistent validation logs under the MILo output root:

```bash
bash dataset_tools_blender/milo/prepare_five_tmux.sh --gpus 0,1
```

## NeuralUDF Export

NeuralUDF expects the IDR/NeuS camera archive convention. The exporter uses the
150 frames selected by `transforms_train.json`, converts the exact effective
Blender cameras to OpenCV projection matrices, and writes the official
`camera_mat_*`, `world_mat_*`, and `scale_mat_*` entries. COLMAP is not used.
The normalization sphere is estimated from masks and exact cameras; the
reference mesh and inverse depth are never read or copied.

The normalization matrix scales scene positions into the unit sphere. It is
part of the projective camera matrix, but it is not a physical camera rotation.
The NeuralUDF loader therefore removes the uniform normalization scale while
recovering each pose. Camera positions remain normalized, while rotation axes
and ray directions remain unit length. Nonuniform scale, non-rigid rotation,
or invalid matrices are rejected.

```text
<shoe>/
  image/000.png ... 149.png
  mask/000.png ... 149.png
  cameras_sphere.npz
```

Prepare and validate the two pilot shoes:

```bash
$PYTHON $PIPELINE prepare-neuraludf --shoe air_jordan_1
$PYTHON $PIPELINE prepare-neuraludf --shoe birkenstock_arizona_sandal
$PYTHON $PIPELINE validate-neuraludf --shoe air_jordan_1
$PYTHON $PIPELINE validate-neuraludf --shoe birkenstock_arizona_sandal
```

Validation reads the existing prepared scenes; it does not regenerate images,
masks, or `cameras_sphere.npz`. The strict smoke and full configurations are respectively
`baselines/NeuralUDF/confs/udf_shoes_smoke.conf` and
`baselines/NeuralUDF/confs/udf_shoes.conf`. These shoe-specific configurations
use a fixed white renderer background, foreground-masked RGB loss, the existing
silhouette loss with weight `0.1`, and no outside NeRF. The network, Eikonal,
sparse, and color implementations remain unchanged; the original upstream DTU
and DeepFashion configurations are untouched.

From the NeuralUDF directory, launch the corrected pilot runs through the
versioned tmux launcher:

```bash
scripts/launch_train_shoe_tmux.sh \
  air_jordan_1 2 masked_white_smoke confs/udf_shoes_smoke.conf 256

scripts/launch_train_shoe_tmux.sh \
  birkenstock_arizona_sandal 3 masked_white_smoke confs/udf_shoes_smoke.conf 256
```

The outputs are written below
`baselines/NeuralUDF/output/golden_set_evaluation_blender_masked_white_smoke/`.
The smoke config runs 5,000 iterations. Accept it only when both meshes resemble
upright shoes, do not touch the `[-1, 1]` extraction boundary, and no large
planar sheets remain. After that check, replace `udf_shoes_smoke.conf` with
`udf_shoes.conf` and use `--resolution 512` for the 300,000-iteration baseline.

To diagnose topology-dependent behavior before a full run, use the reproducible
25,000-iteration DTU and open-garment probes. Both use seed `0`, save checkpoints
and meshes every 5,000 iterations, and consume the same prepared images, masks,
and cameras:

```bash
scripts/launch_train_shoe_tmux.sh \
  air_jordan_1 2 masked_white_dtu_25k confs/udf_shoes_dtu_probe.conf 256

scripts/launch_train_shoe_tmux.sh \
  birkenstock_arizona_sandal 3 masked_white_dtu_25k \
  confs/udf_shoes_dtu_probe.conf 256
```

The DTU pair should be run first, followed by the garment pair on the same
physical GPUs. This isolates the training recipe as the only experimental
variable; it does not regenerate or alter the prepared NeuralUDF dataset.

## NeuS2 Export

NeuS2 uses the same 150/30 split and exact cameras as the GShell evaluation
dataset. The exporter writes one shared set of 180 RGBA PNGs and does not read
or copy inverse depth or `reference_mesh.ply`.

```text
<shoe>/
  images/img001.png ... img180.png
  transform_train.json
  transform_test.json
  conversion_manifest.json
```

The camera stored in each frame is:

```text
(Rx(+90 degrees) @ saved_gshell_c2w) @ diag(1, -1, -1, 1)
```

This is an OpenCV-style camera-to-world matrix. `from_na=true` tells NeuS2 not
to apply its usual NeRF axis permutation. Intrinsics remain exactly
`1536x1024` with a 21-degree horizontal field of view.

The 150 training masks and exact cameras define a conservative visual hull.
Its bounding sphere is mapped into NeuS2's unit cube with one uniform scale
and one translation. The held-out masks, inverse depth, and reference mesh do
not influence this normalization.

Prepare and validate a scene:

```bash
PYTHON=/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python
PIPELINE=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py

$PYTHON $PIPELINE prepare-neus2 --shoe adidas_yeezy_boost_350_v2_zyon
$PYTHON $PIPELINE validate-neus2 --shoe adidas_yeezy_boost_350_v2_zyon
```

The default output root is:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation
```

Preparation is transactional. An existing valid scene is checked and skipped;
`--overwrite` is required to replace it. Validation checks all 180 alpha
masks, both disjoint splits, exact camera matrices, rigid rotations, unit
rays, intrinsics, and the visual-hull normalization.

### NeuS2 Turntable-Only Export

The turntable-only export matches the established `neus2_shoes` protocol. It
uses only the first, level camera ring from the 180-view Blender dataset:

```text
<shoe>/
  images/img001.png ... img036.png
  transform.json
  transform_train.json
  transform_test.json
  conversion_manifest.json
```

`transform.json` contains all 36 level views. Training uses 30 views, while
source indices `0, 6, 12, 18, 24, 30` (images `001, 007, 013, 019, 025, 031`)
form the six-view test set. The visual-hull normalization uses only the 30
training masks. No elevated views, inverse depth, or reference mesh influence
the prepared turntable training scene.

Prepare or validate one shoe:

```bash
$PYTHON $PIPELINE prepare-neus2-turntable --shoe air_jordan_1
$PYTHON $PIPELINE validate-neus2-turntable --shoe air_jordan_1
```

Prepare and validate the five paper shoes:

```bash
bash /storage/Abhinay/Shell_Gaussian/dataset_tools_blender/neus2/prepare_turntable_five.sh
```

The default output root is:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation_turntable
```

Derived NeuS2 and SuGaR manifests use portable source identity fields rather
than absolute filesystem paths:

```json
{
  "version": 2,
  "source_dataset": "gshell/golden_set_evaluation",
  "scene": "air_jordan_1",
  "source_transforms_sha256": "..."
}
```

The scene name and transforms hash identify the exact source. This avoids
false validation failures when the same dataset is reached through
`/storage/Abhinay/home_ab5298` or its resolved `/home/ab5298` path.

Corrected probe outputs use the
`golden_set_evaluation_blender_masked_white_probe_dtu` and
`golden_set_evaluation_blender_masked_white_probe_garment` roots. Older output
roots were produced with stretched camera rays and are not valid baseline
results.

## GShell Polycam Smoke Test

After the Air Jordan and Birkenstock scenes validate, run the existing Polycam
trainer without changing baseline code:

```bash
GSHELL_DATASET_ROOT=/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation \
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

`validate-neuraludf` additionally requires rigid rotations with determinant
one, unit-length camera rays, and valid near/far intervals around the normalized
unit sphere. It also checks the unchanged images, masks, and all 150 IDR camera
entries.

Second-surface depth remains outside this pipeline.

Inverse-depth files remain ordinary dense-shape float32 `.npy` arrays to the
loader. Their zero-valued background pages are allocated as sparse filesystem
holes, reducing physical disk use without compression, quantization, or a
loader change. Use sparse-aware copy tools (for example, `cp --sparse=always`)
when relocating the generated dataset.

## Shared Turntable Exports

The G-Shell, NeuralUDF, NeuS2, and SuGaR turntable datasets share the first
level camera ring. Images `001, 007, 013, 019, 025, 031` are held out; the
remaining 30 images are used for training. No elevated image is included.

The centralized commands for the three additional baseline formats are:

```bash
$PYTHON $PIPELINE prepare-gshell-turntable --shoe air_jordan_1
$PYTHON $PIPELINE validate-gshell-turntable --shoe air_jordan_1

$PYTHON $PIPELINE prepare-neuraludf-turntable --shoe air_jordan_1
$PYTHON $PIPELINE validate-neuraludf-turntable --shoe air_jordan_1

$PYTHON $PIPELINE prepare-sugar-turntable --shoe air_jordan_1 --gpu 2
$PYTHON $PIPELINE validate-sugar-turntable --shoe air_jordan_1
```

G-Shell retains physical copies of first-surface inverse depth, but omits the
reference mesh. NeuralUDF stores only its 30 author-compatible training assets
and recomputes normalization from those masks. SuGaR exports all 36 cameras,
but feature matching and sparse triangulation use only the 30 training images;
the six held-out cameras have no sparse-point observations.

Prepare the five paper shoes for all three formats in one tmux session:

```bash
bash /storage/Abhinay/Shell_Gaussian/dataset_tools_blender/turntable/prepare_other_baselines_five.sh \
  --gpus 2,3 \
  --session turntable_other_baselines_five
```

The three output roots are:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/gshell/golden_set_evaluation_turntable
/storage/Abhinay/home_ab5298/dataset/datasets/processed/neuraludf/golden_set_evaluation_turntable
/storage/Abhinay/home_ab5298/dataset/datasets/processed/sugar/golden_set_evaluation_turntable
```

The launcher requires exactly two GPUs with at least 20 GiB free, publishes
each scene transactionally, skips existing valid scenes, and records one batch
log plus separate CPU and SuGaR worker logs. Dataset manifests contain portable
source identifiers and hashes rather than resolved `/home/ab5298` paths.
