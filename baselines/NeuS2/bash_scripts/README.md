# NeuS2 Shoe Experiments

NeuS2 uses the derived exact-camera dataset at:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation
```

The official `configs/nerf/dtu.json` is candidate A and remains unchanged.
Candidate B is `configs/nerf/dtu_shoes_masked.json`; its only recipe change is
`mask_loss_weight: 0.1`.

Both candidates must first run for 15,000 steps on the unreported development
shoe `adidas_yeezy_boost_350_v2_zyon`. Select lower Chamfer-L1 unless the
relative difference is below 2%, in which case retain official DTU. Do not
tune on the five reportable shoes.

## One-Shoe Pilot

Official DTU:

```bash
NEUS2_ENV=/home/ab5298/anaconda3/envs/neus2 \
NEUS2_DATA_ROOT=/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation \
NEUS2_OUTPUT_ROOT=/storage/Abhinay/Shell_Gaussian/baselines/NeuS2/output/golden_set_evaluation_blender_pilot_dtu \
NEUS2_SHOE_LIST=/storage/Abhinay/Shell_Gaussian/baselines/NeuS2/bash_scripts/development_shoes.txt \
NEUS2_CONFIG=dtu.json \
NEUS2_N_STEPS=15000 \
NEUS2_MARCHING_CUBES_RES=512 \
NEUS2_GPUS=2 \
NEUS2_RUN_EVAL=1 \
NEUS2_RUN_MESH_METRICS=1 \
NEUS2_MESH_METRICS_ROOT=/storage/Abhinay/Shell_Gaussian/mesh_metrics/output/evaluations/neus2_pilot_dtu \
NEUS2_TMUX_SESSION=neus2_yeezy_dtu_gpu2 \
bash bash_scripts/train_all_shoes.sh
```

Run candidate B with a different output root, metric root, and tmux name:

```bash
NEUS2_CONFIG=dtu_shoes_masked.json
NEUS2_OUTPUT_ROOT=/storage/Abhinay/Shell_Gaussian/baselines/NeuS2/output/golden_set_evaluation_blender_pilot_masked
NEUS2_MESH_METRICS_ROOT=/storage/Abhinay/Shell_Gaussian/mesh_metrics/output/evaluations/neus2_pilot_masked
NEUS2_TMUX_SESSION=neus2_yeezy_masked_gpu2
```

## Final Five-Shoe Queue

After freezing the selected config:

```bash
NEUS2_ENV=/home/ab5298/anaconda3/envs/neus2 \
NEUS2_DATA_ROOT=/storage/Abhinay/home_ab5298/dataset/datasets/processed/neus2/golden_set_evaluation \
NEUS2_OUTPUT_ROOT=/storage/Abhinay/Shell_Gaussian/baselines/NeuS2/output/golden_set_evaluation_blender_final \
NEUS2_SHOE_LIST=/storage/Abhinay/Shell_Gaussian/baselines/NeuS2/bash_scripts/evaluation_shoes.txt \
NEUS2_CONFIG=<selected-config> \
NEUS2_N_STEPS=15000 \
NEUS2_MARCHING_CUBES_RES=512 \
NEUS2_GPUS=2 \
NEUS2_RUN_EVAL=1 \
NEUS2_RUN_MESH_METRICS=1 \
NEUS2_MESH_METRICS_ROOT=/storage/Abhinay/Shell_Gaussian/mesh_metrics/output/evaluations/neus2 \
NEUS2_TMUX_SESSION=neus2_final_five_gpu2 \
bash bash_scripts/train_all_shoes.sh
```

Monitor:

```bash
tmux ls
tmux capture-pane -pt neus2_final_five_gpu2 | tail -40
tail -f baselines/NeuS2/output/batch_runs/<run-id>/batch.log
nvidia-smi -i 2
```

Each output scene contains a 15k checkpoint, a 512-cubed mesh, 30 held-out
prediction/GT pairs, evaluation logs, resource usage, and an experiment
manifest. Mesh metrics are written separately under `mesh_metrics/output`.
The queue stops on training, evaluation, boundary-validation, or metric
failure.

For reporting, PSNR, SSIM, and LPIPS come directly from NeuS2's
`scripts/render_utils.py` and are aggregated from the 30 held-out views in
`eval_log.txt`. No project-level image scorer replaces these repository-native
metrics. The shared `mesh_metrics` package is used only for similarity-aligned
geometry and mesh-derived silhouette/depth analysis.
