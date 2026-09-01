# FootShellGaussian: End-to-End Steps So Far

This is the practical runbook for reproducing the project from a raw GLB to
the latest accepted stage. Update this file whenever the pipeline gains a new
stage. Keep `README.md` as the technical explanation of the code and keep this
file focused on what to run, what to inspect, and where the results appear.

## Current endpoint

The active dataset contains canonical right shoes with this effective frame:

```text
+X = heel to toe
+Y = downward toward the sole
+Z = shoe width
```

The current FootShell preparation behavior is:

```text
normal
  -> detect the interior footbed
  -> calculate reversible functional-length normalization
  -> write footbed review and normalized-shoe artifacts

high_heel
  -> detect the inclined interior support
  -> calculate heel/forefoot support diagnostics
  -> write footbed review artifacts
  -> stop before normalization
```

High-heel normalization and high-heel SUPR fitting are not implemented yet.

## Active locations

```text
Raw GLBs:
/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean

Dataset manifest:
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json

Dataset tools:
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender

Processed dataset:
/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation

FootShell project:
/storage/Abhinay/Shell_Gaussian/FootShellGaussian

FootShell outputs:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation
```

The manifest and processed dataset paths remain stable as shoes are added. Do
not create a new manifest or a new dataset directory merely because the number
of shoes increases.

## Step 1: Add a raw GLB

Copy or place the new file in the raw GLB directory. Use a stable filename
that will also become the shoe name in the processed dataset.

Example:

```text
/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean/new_shoe.glb
```

## Step 2: Calculate its checksum

```bash
sha256sum \
  /home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean/new_shoe.glb
```

Copy the checksum into the manifest entry. The pipeline rejects the file if
its content later changes without a matching manifest update.

## Step 3: Add the manifest entry

Edit:

```text
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json
```

Start a new entry with `reviewed: false`.

```json
{
  "name": "new_shoe",
  "model": "new_shoe.glb",
  "sha256": "CHECKSUM_FROM_SHA256SUM",
  "reviewed": false,
  "shoe_profile": "normal",
  "source_axes": {
    "length": "X",
    "width": "Y",
    "up": "Z"
  },
  "selection": {
    "mode": "all"
  },
  "mirror_width": false
}
```

Use exactly one profile:

```json
"shoe_profile": "normal"
```

or:

```json
"shoe_profile": "high_heel"
```

The profile does not rotate, scale, mirror, or otherwise change the dataset
geometry. It tells FootShell which support rules and downstream stages are
approved for the shoe.

Do not copy the example `source_axes` without checking the raw model. The
three entries tell the pipeline which raw direction represents length, width,
and physical up. A leading minus sign reverses that direction, for example
`"length": "-Y"`.

### Selecting one shoe from a pair

Use:

```json
"selection": {
  "mode": "axis-side",
  "axis": "Y",
  "side": "min",
  "separate_loose_parts": true
}
```

only when the raw asset contains a pair or unrelated components that must be
removed.

Selection happens after `source_axes` has mapped the imported geometry into
the canonical Blender frame:

```text
Blender X = shoe length
Blender Y = shoe width
Blender Z = physical up
```

The selection code does not cut a mesh in half. It keeps or removes complete
mesh components according to the center of each component's bounding box.

The chosen axis means:

- `axis: "Y"`: compare components from one width side to the other. This is
  normally appropriate when the left and right shoes stand beside each other.
- `axis: "X"`: compare components from the heel side to the toe side. Use this
  only if two complete shoes are arranged along the length direction.
- `axis: "Z"`: compare lower and upper components. This is rarely appropriate
  for selecting the right shoe.

For the chosen axis, the pipeline calculates one middle divider:

```text
pivot = (lowest coordinate + highest coordinate) / 2
```

Then:

- `side: "min"` keeps components whose centers are on the lower-coordinate
  side of the divider.
- `side: "max"` keeps components whose centers are on the higher-coordinate
  side of the divider.

For example, with `axis: "Y"`, `min` keeps components centered toward
negative/lower Y and `max` keeps components centered toward positive/higher Y.
Neither value inherently means "right shoe". Which side contains the right
shoe depends on the asset and must be confirmed in the audit images.

`separate_loose_parts: true` first joins the imported mesh objects and then
separates every disconnected piece into its own component. This is needed when
both shoes were imported as one Blender object but are physically disconnected.

If the GLB already contains only the desired right shoe, use:

```json
"selection": {
  "mode": "all"
}
```

## Step 4: Audit the new shoe

Set reusable shell variables:

```bash
TASK_PY=/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python
TASK_PIPELINE=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py
TASK_BLENDER=/home/ab5298/anaconda3/envs/shellgaussianenv/bin/blender
TASK_GLBS=/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean
TASK_MANIFEST=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json
TASK_AUDIT=/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/audit
TASK_SHOE=new_shoe
```

Run one-shoe audit:

```bash
"$TASK_PY" "$TASK_PIPELINE" audit \
  --shoe "$TASK_SHOE" \
  --gpu 0 \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-dir "$TASK_AUDIT" \
  --blender "$TASK_BLENDER"
```

Inspect the side, toe, heel, top, and bottom views. Confirm:

1. The correct right shoe was retained.
2. The shoe is physically upright.
3. The heel is at `-X`.
4. The toe points toward `+X`.
5. The width orientation is correct and not mirrored.
6. No required shoe component was deleted.

If the result is wrong, correct `source_axes`, `selection`, or `mirror_width`
and audit again. Do not proceed simply because the command succeeded.

After visual acceptance, change the entry to:

```json
"reviewed": true
```

## Step 5: Build the processed dataset

### Recommended complete build

The existing launcher uses five GPUs in tmux:

```bash
cd /storage/Abhinay/Shell_Gaussian
dataset_tools_blender/build_golden_set_evaluation.sh
```

It builds from the stable manifest into the stable processed dataset. Existing
valid shoes are validated and skipped; newly listed shoes are built.

Monitor it with either:

```bash
tmux attach -t golden-set-evaluation-build
```

or:

```bash
tail -f \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/dataset-build.log
```

Check whether it is still running:

```bash
tmux has-session -t golden-set-evaluation-build \
  && echo RUNNING \
  || echo FINISHED
```

### Build only the new shoe

```bash
"$TASK_PY" "$TASK_PIPELINE" build \
  --shoe "$TASK_SHOE" \
  --gpu 0 \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-root /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation \
  --blender "$TASK_BLENDER"
```

Existing valid outputs are not overwritten by default. Use `--overwrite` only
when deliberately replacing that shoe's published processed scene.

## Step 6: Validate the processed shoe

```bash
"$TASK_PY" "$TASK_PIPELINE" validate \
  --shoe "$TASK_SHOE" \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-root /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

The processed shoe directory should contain at least:

```text
golden_set_evaluation/new_shoe/
├── reference_mesh.ply
├── blender_canonicalization.json
├── transforms.json
├── image/
├── mask/
└── invdepth/
```

The canonicalization JSON must contain the selected `shoe_profile` and the
effective GShell coordinate convention.

## Step 7: Run FootShell preparation

The FootShell wrapper accepts explicit shoe names. Pass every newly added name
because its no-argument list currently contains the original reviewed shoes.

### One shoe

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

scripts/prepare_golden_set_evaluation_shoes.sh \
  new_shoe
```

### Several shoes

```bash
scripts/prepare_golden_set_evaluation_shoes.sh \
  new_normal_shoe \
  new_high_heel
```

### Run preparation in tmux

```bash
mkdir -p \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs

tmux new-session -d -s prepare-new-shoes \
  'cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian && \
  scripts/prepare_golden_set_evaluation_shoes.sh \
    new_normal_shoe \
    new_high_heel \
  2>&1 | tee \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/new-shoes-preparation.log'
```

Monitor it with:

```bash
tmux attach -t prepare-new-shoes
```

or:

```bash
tail -f \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/new-shoes-preparation.log
```

## Step 8: Inspect a normal-shoe result

The output appears at:

```text
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/new_normal_shoe
```

Expected files:

```text
shoe_preparation.json
footbed_surface.ply
footbed_overlay.ply
shoe_normalized.ply
```

Inspect in this order:

1. Open `footbed_overlay.ply`. Green must be the interior surface on which the
   foot stands, not the outsole, upper, toe panel, or shaft.
2. Open `shoe_normalized.ply`. It must preserve the shoe's axis directions and
   shape.
3. Check `shoe_preparation.json`. The normalization matrices and measurements
   must be finite.

For an accepted normal shoe, the current pipeline endpoint is functional-length
normalization.

## Step 9: Inspect a high-heel result

The output appears at:

```text
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/new_high_heel
```

Expected files:

```text
shoe_preparation.json
footbed_surface.ply
footbed_overlay.ply
```

It must not contain:

```text
shoe_normalized.ply
```

The JSON should contain:

```json
{
  "shoe_profile": "high_heel",
  "preparation_status": "support_detected_normalization_deferred",
  "normalization": null
}
```

Open `footbed_overlay.ply`. Green must follow the inclined interior support and
exclude the outsole bottom, heel column, straps, and upper panels.

For an accepted high heel, the current pipeline endpoint is support detection.
Do not treat it as normalized and do not use the current neutral-foot alignment
on it.

## Step 10: Rerun an existing FootShell result

Preparation refuses to replace existing artifacts unless explicitly asked.

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

OVERWRITE=1 scripts/prepare_golden_set_evaluation_shoes.sh \
  new_shoe
```

`OVERWRITE=1` replaces only the known artifacts for that profile and preserves
unrelated files in the output directory.

## Failure rules

- If audit orientation is wrong, fix the manifest and rerun the audit.
- If the wrong shoe from a pair is selected, change `selection.axis` or
  `selection.side` based on the audited component positions.
- If dataset validation fails, do not run FootShell on that shoe.
- If the green support overlay is wrong, stop and diagnose that mesh. Do not
  weaken thresholds or add a per-shoe exception without testing the complete
  accepted set.
- If a `high_heel` produces no support, inspect its diagnostics. Do not process
  it as `normal` merely to bypass the heel rules.

## What comes next

The next major project stage is foot fitting:

- Normal shoes: place and deform the SUPR foot against the accepted support.
- High heels: first add heel-specific normalization and deterministic SUPR
  plantarflexion using the recorded heel and forefoot support measurements.

When those stages are implemented, append their exact commands, outputs, and
visual checks to this file.
