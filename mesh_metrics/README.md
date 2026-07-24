# Mesh Metrics

This package aligns reconstructed shoe meshes to their Blender reference meshes before geometry metrics are calculated. Alignment is evaluation-only: it never changes training inputs, baseline code, or the original mesh files.

## Alignment contract

The recovered transform maps original prediction vertices into ground-truth coordinates:

```text
x_aligned = scale * rotation * x_prediction + translation
```

The optimizer permits translation, rotation, and one positive uniform scale. It rejects reflections, nonuniform scaling, shear, and deformation.

Alignment uses deterministic area-uniform surface samples, the existing pose plus right-handed PCA orientation candidates, and robust bidirectional similarity ICP. The worst correspondence distances are trimmed during optimization so missing parts and floating artifacts do not determine the placement. FPFH, RANSAC, manual landmarks, smoothing, hole filling, and largest-component filtering are not used.

## Environment

Use the existing Shell Gaussian environment:

```bash
PYTHON=/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python
cd /storage/Abhinay/Shell_Gaussian
```

Run the tests:

```bash
$PYTHON -m unittest discover -s mesh_metrics/tests
```

## Align one mesh

```bash
$PYTHON -m mesh_metrics.align_mesh \
  --prediction /path/to/reconstructed_mesh.ply \
  --ground-truth /storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender/air_jordan_1/reference_mesh.ply \
  --output mesh_metrics/output/neuraludf/air_jordan_1 \
  --save-aligned
```

The command writes `alignment.json`. The optional `--save-aligned` flag also writes `aligned_prediction.ply` for visual inspection. Generated output is ignored by Git.

Important options:

```text
--samples             Dense refinement sample count; default 50000
--coarse-samples      Coarse search sample count; default 5000
--candidates          Number of orientation candidates refined; default 4
--inlier-fraction     Fraction retained during robust fitting; default 0.8
--seed                Deterministic sampling seed; default 0
```

## Visual inspection

Open `notebooks/inspect_alignment.ipynb`, set the prediction and ground-truth paths in the first code cell, and run all cells. The notebook calls the package implementation directly and overlays sampled surfaces before and after alignment. It does not contain a separate ICP implementation.

For a complete single-shoe analysis, open `notebooks/evaluate_mesh.ipynb`. It includes presets for the available G-Shell and NeuralUDF outputs, runs the production evaluator, and displays the alignment, classified metric table, directional surface-error maps, distance histograms, mesh diagnostics, and one exact held-out camera comparison. Select a reconstruction using `PRESET`; G-Shell Air Jordan is the default. Full 30-camera metrics are available through the notebook's `RUN_FULL_HELDOUT_METRICS` option. Notebook files are tracked by Git; only generated results under `mesh_metrics/output/` are ignored.

## Evaluate one mesh

The unified command aligns a prediction, computes point-to-triangle geometry metrics, validates the exact test-camera convention, and computes held-out silhouette and depth metrics:

```bash
SCENE=/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender/air_jordan_1

$PYTHON -m mesh_metrics.evaluate_mesh \
  --prediction /path/to/reconstructed_mesh.ply \
  --scene "$SCENE" \
  --output mesh_metrics/output/evaluations/neuraludf/air_jordan_1 \
  --training-view-set train \
  --save-aligned
```

Use `--geometry-only` while debugging to skip the 30-camera rendering stage. `--training-view-set train` declares that the method used only the 150 training frames. Runs trained on all 180 frames may still be evaluated, but their view metrics are not labelled held out.

Outputs are:

```text
alignment.json
geometry_metrics.json
reference_render_validation.json
view_metrics.json
aligned_prediction.ply       # only with --save-aligned
```

Geometry distances are normalized by the ground-truth bounding-box diagonal. The headline values are accuracy, completeness, Chamfer-L1, F-score at 1%, orientation-invariant normal consistency, and bidirectional P95 distance. The JSON also includes F-scores at 0.5% and 2%, directional values, topology diagnostics, and raw distances.

View metrics use the 30 frames in `transforms_test.json`: silhouette IoU, boundary F-score at two pixels, camera-Z depth MAE, depth overlap coverage, underside depth MAE, and top-view depth MAE. The evaluator first requires `reference_mesh.ply` to reproduce the Blender masks and inverse depth before prediction metrics are accepted.

## Aggregate the core benchmark

Store evaluations as `<input-root>/<method>/<shoe>/`. Once every method has all eight shoes:

```bash
$PYTHON -m mesh_metrics.aggregate_results \
  --input-root mesh_metrics/output/evaluations \
  --output mesh_metrics/output/summary
```

This uses `configs/core_eight.json` and writes `per_shoe.csv`, `method_summary.csv`, and `tables.tex`. Incomplete development runs can be summarized with `--allow-incomplete`, but final comparisons require all eight shoes for every method.
