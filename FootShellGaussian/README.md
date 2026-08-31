# FootShellGaussian

`FootShellGaussian` is a small, deterministic first-stage geometry project for
preparing a canonical right shoe and placing a neutral right SUPR foot inside
it. Shoe preparation validates the coordinate frame, detects the interior
support, and constructs a reversible functional-length normalization. The
initial foot milestone uses a fixed coordinate remap, uniform length scaling,
horizontal bounding-box centering, detected-footbed first contact, and explicit
forward and inverse coordinate transforms.

The project intentionally does not include pose fitting, PCA alignment, SDFs,
CUDA, learned optimization, or shoe reconstruction. The archived prototype is
available separately at `../GShellFootPriorPrototype/` for reference only.

## Development setup

```bash
python -m pip install -e '.[test]'
pytest
```

## Coordinate contract

This project accepts canonical **right shoes only**. The evaluation shoe frame
is `+X = heel-to-toe`, `+Y = down toward the sole`, and `+Z = width`. The
canonicalization metadata must declare
`effective_gshell_x_length_y_down_z_width`; incompatible or missing metadata is
rejected instead of inferring a frame with PCA. It must also declare exactly
one `shoe_profile`: `normal` or `high_heel`. A metadata `mirror_width` value
records how the source asset was canonicalized and does not trigger additional
runtime mirroring.

The current support detector and normalization are approved only for the
`normal` profile. A `high_heel` input is reported as `[deferred-support]` before
the mesh is loaded and produces no preparation artifacts. A missing or unknown
profile is an error. This is routing metadata only; it never changes the
canonical shoe geometry.

Raw SUPR is interpreted as `X = width`, `Y = anatomical height`, and
`Z = heel-to-toe`. The fixed remap is therefore:

```text
shoe X =  SUPR Z
shoe Y = -SUPR Y
shoe Z =  SUPR X
```

No PCA, data-dependent rotation, or runtime left/right mirroring is used.

## Interior support detection

Footbed detection does not assume that the support surface is a disconnected
mesh component. It first keeps nondegenerate triangles whose normals are within
60 degrees of either vertical direction. Using the absolute normal direction
makes this step insensitive to globally reversed face winding while excluding
vertical sidewalls.

The projected triangles are indexed on an adaptive X/Z grid with 256 cells
along shoe length and square cells across shoe width. Exact barycentric
intersections are evaluated at the grid samples. Support patches are grouped
across 8-neighbour cells when their adjacent heights differ by no more than 2%
of shoe length. Broad patches are kept separate so a valid footbed cannot be
absorbed into an upper or outsole through a chain of locally shallow faces.

A candidate must cover at least 65% of shoe length and 40% of shoe width, lie
within the lower 60% of the shoe, and have consistent face orientation. It must
also contain support within the middle 20% of shoe width along at least 65% of
shoe length. This central-support check prevents separated upper panels from
qualifying merely because their outermost points span a broad rectangle.

The component-layer detector remains the primary path. A local-height tracing
fallback runs only when that primary path would return a consistently
downward-facing outsole while an opening-facing layer above it passes every
rule except central support. The fallback joins measurements in neighbouring
grid cells only when they are mutual closest-height matches, the local height
step is at most 2% of shoe length, and the complete traced height range remains
within 15% of shoe length. It then reapplies the same strict qualification
thresholds; none are lowered.

If the fallback cannot find one unambiguous opening-facing support above the
suspicious outsole, detection fails instead of guessing. A traced output uses
only the original source faces responsible for its grid measurements. JSON
diagnostics record whether selection used `component_layers` or
`local_height_trace`, which primary layer was rejected, and why tracing ran.

`footprint_fill_fraction` is retained in diagnostics for compatibility. It is
the projected surface completeness: the fraction of grid cells occupied inside
the candidate's own X/Z bounding rectangle. The opening-facing candidate with
the smallest median Y is selected after all qualification checks. Its compact
output consists only of contributing original shoe faces: the detector does
not repair topology, fill holes, invent heights, or use a fixed-Y fallback.
Height queries use those exact source triangles, so uncovered regions remain
invalid.

This detector has been run deterministically on all 17 `normal` assets in
`golden_set_evaluation`. The two `high_heel` assets,
`red_high_heel_shoes` and `plateau_sandal_heels`, are explicitly deferred by
the profile gate.

## Functional-length normalization

Shoe normalization is derived from the selected support grid rather than the
outer shoe bounding box. The largest 8-neighbour footprint is retained, and
the functional heel-to-toe range is the longest run of columns whose support
width is at least 10% of the median non-empty width. This trims narrow decorative
extensions while preserving real holes and disconnected noise as absent data.

The origin uses the functional heel X coordinate, the median rear centerline Z,
and the median rear support height within the central half of local width. A
matching front statistic defines the toe landmark. The transform translates the
heel origin to zero and scales every axis uniformly by the reciprocal functional
length. It performs no rotation, mirroring, anisotropic scaling, topology repair,
or mesh modification.

Run shoe preparation with:

```bash
python scripts/run_shoe_preparation.py \
  --shoe-mesh /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation/canvas_shoe/reference_mesh.ply \
  --canonicalization /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation/canvas_shoe/blender_canonicalization.json \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/canvas_shoe
```

The preparation output contains:

```text
shoe_preparation.json
footbed_surface.ply
footbed_overlay.ply
shoe_normalized.ply
```

The footbed and gray-shoe/green-footbed overlay remain in the original frame.
The normalized shoe has functional heel `X=0`, functional toe `X=1`, unchanged
axis signs, and an exact inverse transform recorded in the JSON. Pass
`--overwrite` to replace only these four artifacts; unrelated output files are
preserved.

### Prepare newly canonicalized shoes through Checkpoints 2 and 3

A new raw GLB must first be audited, built, and validated by
`../dataset_tools_blender/`. This produces the two inputs used here:

```text
<processed-root>/<shoe>/reference_mesh.ply
<processed-root>/<shoe>/blender_canonicalization.json
```

Residual top-view heading is corrected in the Blender dataset stage, before
this project detects a footbed. The canonical input root is:

```text
/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

Footbed detection must not be used to rotate a shoe. It consumes the already
oriented `reference_mesh.ply` and supplies only the support geometry needed by
functional normalization.

Shoe preparation then performs the following operations in a fixed order:

1. Validate the explicit shoe profile and canonical right-shoe coordinate
   frame. Stop cleanly with no output for `high_heel`.
2. For `normal`, load the canonical `reference_mesh.ply` without repairing its
   topology.
3. Detect the interior support surface. This is Checkpoint 2.
4. Build the functional heel, toe, origin, length, and reversible transforms
   from that detected support. This is Checkpoint 3.
5. Write the footbed inspection meshes, normalized shoe, and JSON diagnostics.

Checkpoint 2 is therefore always calculated before Checkpoint 3. They do not
need separate commands: `run_shoe_preparation.py` executes them sequentially
and stops if either calculation fails. It writes artifacts only after all
calculations succeed.

The reusable batch wrapper visits all 19 reviewed shoes when no shoe names are
supplied. It prepares the 17 `normal` shoes and reports the two `high_heel`
shoes as deferred without creating their output directories. Explicit names
can be supplied to process a subset:

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian
scripts/prepare_golden_set_evaluation_shoes.sh \
  leather_boots \
  ww_ii_german_jack_boots
```

By default, existing preparation artifacts are not replaced. Set `OVERWRITE=1`
only when deliberately regenerating the four known artifacts for each shoe.

To route all 19 shoes in `tmux` with a persistent log:

```bash
mkdir -p /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs
tmux new-session -d -s golden-set-shoe-preparation \
  'cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian && \
  scripts/prepare_golden_set_evaluation_shoes.sh \
  2>&1 | tee /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/shoe-preparation.log'
```

Monitor the job without interrupting it:

```bash
tmux attach -t golden-set-shoe-preparation
tail -f /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/shoe-preparation.log
```

After the job finishes, inspect `footbed_overlay.ply` first. The green geometry
must be the interior surface on which the foot stands, not the outsole, upper,
toe panel, or shaft. Only accept `shoe_normalized.ply` and the normalization in
`shoe_preparation.json` after that support surface is visually accepted. If the
footbed is wrong, the derived normalization is also invalid and must not be used
for SUPR placement.

This command runs Checkpoint 2 followed by Checkpoint 3 for each shoe. A
failure or incorrect green support overlay is a reason to stop and diagnose
that shoe; it is not permission to change footbed thresholds.

## Run the canvas example

```bash
python scripts/run_alignment.py \
  --shoe-mesh /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation/canvas_shoe/reference_mesh.ply \
  --supr-model ../baselines/SUPR/data/supr_male_right_foot.npy \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/alignment/canvas_shoe
```

The output directory receives exactly these artifacts:

```text
alignment.json
foot_aligned.ply
footbed_surface.ply
alignment_overlay.ply
```

The runner refuses to replace an existing artifact unless `--overwrite` is
passed. That flag replaces only these four known files and preserves every
other file in the output directory.

The gray-shoe/blue-foot overlay is intended for inspection. The selected
footbed is saved separately in green so the overlay does not duplicate its
triangles. Neutral SUPR first contact does not conform the foot to toe spring
or the full shoe cavity; those remaining gaps are measurements, not fitting
errors corrected by this milestone.

## Verified canvas result and limitations

The checked canvas run selects an interior sheet with 207 vertices and 350
faces. It covers 94.92% of the shoe length and 86.46% of its width. The initial
alignment covers 103 of 107 plantar samples (96.26%), reaches a zero minimum
gap at first contact, and leaves a maximum covered plantar gap of approximately
0.02137 shoe-frame units.

Visual inspection confirms heel-to-toe orientation along positive X, upright
placement, plausible horizontal scale, and no catastrophic intersection in
this example. Canvas canonicalization records `mirror_width: true` in the
reviewed right-shoe dataset; no independent medial/lateral landmark is present
for a stronger handedness assertion.

The selected canvas footbed contains one small rectangular hole in the source
topology. The implementation reports projection misses rather than repairing
that hole. It also does not pose the neutral foot, conform it to toe spring,
optimize cavity clearance, or provide a general collision-free fitting
guarantee.
