# FootShellGaussian Scripts

This folder is intentionally kept small around the current shoe pipeline.

Primary entry points:

- `run_final_pipeline.sh`: orchestrate the downstream shoe pipeline stages from GShell training through alignment and pseudo-last generation.
- `render_obj_top_bottom_evaluation.py`: render external shoe assets into the legacy evaluation dataset layout from Blender.

Core helpers kept in this folder:

- `run_foot_aware_alignment_pipeline.py`
- `run_foot_fit_optimization.py`
- `run_pseudo_last_builder.py`
- `run_support_footbed_analysis.py`
- `build_foot_sdf.py`

Older debug and experiment scripts were moved to `../archive/cleanup_20260622/scripts`.
