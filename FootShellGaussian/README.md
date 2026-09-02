# FootShellGaussian

`FootShellGaussian` is a deterministic geometry project for preparing a
canonical right shoe and fitting a right SUPR foot to its interior support.
Shoe preparation validates the coordinate frame, detects the interior support,
and constructs a reversible functional-length normalization. The current
normal-shoe fit uses a fixed coordinate remap, uniform length scaling,
functional-heel anchoring, footbed-centerline lateral placement, and a small
CUDA search over SUPR ankle and midfoot pitch.

The project intentionally does not yet include SUPR shape fitting, toe
articulation, full shoe-cavity collision fitting, SDFs, learned optimization,
or shoe reconstruction. The archived prototype is available separately at
`../GShellFootPriorPrototype/` for reference only.

## Development setup

```bash
python -m pip install -e ../baselines/SUPR
python -m pip install -e '.[fitting,test]'
pytest
```

The official SUPR implementation used by articulated fitting creates CUDA
buffers, so this stage requires a CUDA-capable PyTorch environment. Neutral
SUPR loading and all earlier shoe-preparation stages remain NumPy-only.

## Coordinate contract

This project accepts canonical **right shoes only**. The evaluation shoe frame
is `+X = heel-to-toe`, `+Y = down toward the sole`, and `+Z = width`. The
canonicalization metadata must declare
`effective_gshell_x_length_y_down_z_width`; incompatible or missing metadata is
rejected instead of inferring a frame with PCA. It must also declare exactly
one `shoe_profile`: `normal` or `high_heel`. A metadata `mirror_width` value
records how the source asset was canonicalized and does not trigger additional
runtime mirroring.

The `normal` profile uses the approved normal-shoe detector and functional
normalization. The `high_heel` profile uses a separate steep-support detector,
then uses the same functional normalization while preserving the incline. A
missing or unknown profile is an error. The profile selects the support rules;
it never changes the canonical shoe geometry or normalization mathematics.

Raw SUPR is interpreted as `X = width`, `Y = anatomical height`, and
`Z = heel-to-toe`. The fixed remap is therefore:

```text
shoe X =  SUPR Z
shoe Y = -SUPR Y
shoe Z =  SUPR X
```

No PCA, data-dependent rotation, or runtime left/right mirroring is used.

## Interior support detection

### Normal shoes

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

This path has been run deterministically on all 17 `normal` assets in
`golden_set_evaluation`.

### High heels

High heels use the same exact triangle intersections and 256-column grid, but
they do not use the normal-shoe limit that restricts the complete support
height range to 15% of shoe length. A heel may rise substantially from
forefoot to rear while remaining a smooth surface. The existing 2%-of-length
local height-step rule remains active when support patches are grouped.

A heel candidate must retain the normal length, width, central-support, and
projected-completeness checks. In addition, at least 65% of its occupied
central X columns must contain another support-like intersection farther in
`+Y`. This is evidence of physical shoe material beneath the candidate. It
rejects the outsole bottom, while straps and upper panels fail the broad
central-coverage checks.

Canonical opening-facing triangles are evaluated first. Opposite winding is
tried only when that direction yields no valid support. All qualifying layers
from the accepted winding form an upper envelope: at each grid position, the
opening-nearest valid height is retained. This uses an inset footbed where it
exists and the sole's upper surface where the insert ends. It does not fill
holes, interpolate missing support, create faces, or modify source geometry.

The detector records rear-heel and forefoot landmarks, heel elevation, and a
diagnostic support angle for later plantarflexion work. These measurements do
not fit SUPR. After detection, the selected support is passed to the common
functional normalizer without flattening its incline. The reviewed red
stiletto and plateau sandal selections are locked by source-face digests in
the test suite.

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

For either shoe profile, the preparation output contains:

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

For a `high_heel`, the JSON additionally records
`preparation_status="support_detected_and_normalized"` and the detected heel
support measurements. Its normalized mesh preserves the original support
angle; normalization does not flatten or otherwise rotate the shoe.

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
   frame.
2. Load the canonical `reference_mesh.ply` without repairing its topology.
3. Detect the interior support with the rules selected by `shoe_profile`.
4. For either profile, build the functional heel, toe, origin, length, and
   reversible transforms from the accepted support.
5. For `high_heel`, also record the inclined-support diagnostics.
6. Write the four preparation artifacts.

Checkpoint 2 is therefore always calculated before Checkpoint 3. They do not
need separate commands: `run_shoe_preparation.py` executes them sequentially
and stops if either calculation fails. It writes artifacts only after all
calculations succeed.

The reusable batch wrapper visits all 19 reviewed shoes when no shoe names are
supplied. It uses the profile-specific detector and normalizes all accepted
shoes. Explicit names can be supplied to process a subset:

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian
scripts/prepare_golden_set_evaluation_shoes.sh \
  leather_boots \
  ww_ii_german_jack_boots
```

By default, existing preparation artifacts are not replaced. Set `OVERWRITE=1`
only when deliberately regenerating the four preparation artifacts.

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
toe panel, strap, heel column, or shaft. For normal shoes, only accept
`shoe_normalized.ply` after that support is accepted. For high heels, also
confirm that the normalized mesh preserves the inclined support shape.

This command runs Checkpoint 2 followed by Checkpoint 3 for each shoe. A
failure or incorrect green support overlay is a reason to stop and diagnose
that shoe; it is not permission to change footbed thresholds.

## Articulated SUPR support fit for a normal shoe

The fit consumes the completed preparation artifacts. It loads the normalized
shoe, transforms the already-saved original-frame footbed with the recorded
`shoe_to_normalized` matrix, and never runs footbed detection again. Only
`shoe_profile="normal"` is accepted in this checkpoint.

```bash
python scripts/run_alignment.py \
  --preparation-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/canvas_shoe \
  --supr-model ../baselines/SUPR/data/supr_male_right_foot.npy \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit/canvas_shoe
```

The output directory receives exactly these artifacts:

```text
support_fit.json
foot_support_fitted.ply
footbed_normalized.ply
support_fit_overlay.ply
```

The runner refuses to replace an existing artifact unless `--overwrite` is
passed. That flag replaces only these four known files and preserves every
other file in the output directory.

The 266-vertex neutral template defines fixed heel, arch, forefoot, and toe
contact regions. Only SUPR pose entries 3 and 6 are changed: ankle pitch and
midfoot pitch. Root motion, toe joints, and ten shape values remain zero.

For each pose, the complete heel-to-longest-toe span is scaled to reserve a
physical toe allowance. The default represents a 250 mm foot with 12.5 mm in
front, so the foot occupies `250 / 262.5 = 0.95238095` of normalized functional
length. `--toe-allowance-mm` accepts 10 through 15 mm. The rear-most posed foot
point is anchored at `X=0`, and the longest toe ends at the resulting ratio.

A single lateral translation fits plantar-face centroids to the saved footbed
centerline using projected face area. The foot then moves vertically until the
first covered plantar point touches the exact support without sampled plantar
penetration. Candidate poses must retain broad support beneath the complete
plantar area, heel, forefoot, and toes. The arch is measured but is allowed to
remain naturally elevated.

The deterministic search first checks ankle and midfoot pitch from -20 to +20
degrees in 2-degree steps. It then searches at 0.25-degree resolution within 2
degrees of the best coarse result. The score balances the worse of heel and
forefoot RMS gap. Poses whose scores differ by less than one normalized support
grid cell are treated as equivalent, and the pose closest to neutral wins.
Degenerate, reversed, excessively distorted, insufficiently supported, or
penetrating candidates are rejected.

The gray-shoe/blue-foot overlay is intended for inspection. The transformed
saved footbed is written separately in green so the overlay does not duplicate
its triangles. `support_fit.json` stores the complete SUPR pose and zero betas,
fixed contact indices, search diagnostics, contact gaps, and mappings from the
posed SUPR frame to normalized and original shoe frames. SUPR articulation is
non-rigid, so the JSON does not claim that a matrix maps the neutral template
to the posed foot.

## Current limitations

The selected canvas footbed contains one genuine rectangular source-topology
hole. The implementation reports missing support there rather than repairing
it. This checkpoint balances contact against the detected support only. It does
not yet measure the heel cup, toe wall, sidewalls, or upper, and therefore does
not claim full cavity containment or a collision-free fit. Shape fitting, toe
curl, and complete shoe-volume clearance belong to later checkpoints.
