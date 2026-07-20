# FootShellGaussian Scripts

This folder is intentionally kept small around the current shoe pipeline.

Primary entry points:

- `run_final_pipeline.sh`: orchestrate the downstream shoe pipeline stages from GShell training through alignment and pseudo-last generation.

External evaluation-dataset preprocessing is intentionally centralized at
`../../dataset_tools/golden_set_evaluation/pipeline.py`; it no longer lives in
this training/application scripts directory.

Core helpers kept in this folder:

- `run_foot_aware_alignment_pipeline.py`
- `run_foot_fit_optimization.py`
- `run_pseudo_last_builder.py`
- `run_support_footbed_analysis.py`
- `build_foot_sdf.py`

Older debug and experiment scripts were moved to `../archive/cleanup_20260622/scripts`.
