# Legacy External Asset Preprocessing

These scripts and manifests supported the earlier mixed OBJ/FBX/DAE source
tree. They normalized material and texture layouts before Blender rendering.

They are archived because the current evaluation source contract is one
self-contained GLB per shoe. The supported workflow is now:

`dataset_tools/golden_set_evaluation/pipeline.py`

Do not use these files for `golden_set_eval_glb`; they are retained only for
historical reproducibility of the old external-source experiments.
