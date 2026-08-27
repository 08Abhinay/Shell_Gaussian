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

The command-line runner and its output contract will be documented when the
alignment implementation is complete.
