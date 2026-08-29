# FootShellGaussian

`FootShellGaussian` is a small, deterministic first-stage geometry project for
placing a neutral right SUPR foot inside an evaluation shoe mesh. The initial
milestone uses a fixed coordinate remap, uniform length scaling, horizontal
bounding-box centering, detected-footbed first contact, and explicit forward
and inverse coordinate transforms.

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
rejected instead of inferring a frame with PCA. A metadata `mirror_width` value
records how the source asset was canonicalized and does not trigger additional
runtime mirroring.

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

`footprint_fill_fraction` is retained in diagnostics for compatibility. It is
the projected surface completeness: the fraction of grid cells occupied inside
the candidate's own X/Z bounding rectangle. The opening-facing candidate with
the smallest median Y is selected after all qualification checks. Its compact
output consists only of contributing original shoe faces: the detector does
not repair topology, fill holes, invent heights, or use a fixed-Y fallback.
Height queries use those exact source triangles, so uncovered regions remain
invalid.

This detector has been run deterministically on the 15 normal-shoe assets in
`footbed_clean_right`. The two heel assets, `red_high_heel_shoes` and
`plateau_sandal_heels`, remain explicitly deferred.

## Run the canvas example

```bash
python scripts/run_alignment.py \
  --shoe-mesh /home/ab5298/dataset/datasets/processed/gshell/footbed_clean_right/canvas_shoe/reference_mesh.ply \
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
faces. It covers 94.99% of the shoe length and 86.42% of its width. The initial
alignment covers 104 of 107 plantar samples (97.20%), reaches a zero minimum
gap at first contact, and leaves a maximum covered plantar gap of approximately
0.02129 shoe-frame units.

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
