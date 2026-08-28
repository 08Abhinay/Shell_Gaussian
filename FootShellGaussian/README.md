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

The evaluation shoe frame is `X = heel-to-toe`, `Y = down toward the sole`,
and `Z = width`. Raw SUPR is interpreted as `X = width`, `Y = anatomical
height`, and `Z = heel-to-toe`. The fixed remap is therefore:

```text
shoe X =  SUPR Z
shoe Y = -SUPR Y
shoe Z =  SUPR X
```

No PCA or data-dependent rotation is used.

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
