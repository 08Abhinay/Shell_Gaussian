# SuGaR Shoe Mesh Scripts

These scripts mirror the NeuS2 batch style and keep caches under `/data/abelde`.

## One Shoe In Tmux

```sh
cd /data/abelde/projects/active/Shell_Gaussian/baselines/SuGaR
SUGAR_GPU=1 bash bash_scripts/train_one_shoe_tmux.sh Air-Jordan-1-Mid-Wear-Away-Chicago-Gs
tmux attach -t sugar_Air-Jordan-1-Mid-Wear-Away-Chicago-Gs_gpu1
```

For the existing 15k 3DGS checkpoint from this conversation:

```sh
SUGAR_GPU=0 \
SUGAR_SKIP_3DGS=1 \
SUGAR_GS_OUTPUT_DIR=/data/abelde/projects/active/Shell_Gaussian/baselines/SuGaR/output/air_jordan_1_mid_3dgs_15k \
SUGAR_RUN_ID=air_jordan_1_mid_from_existing_3dgs \
bash bash_scripts/train_one_shoe_tmux.sh Air-Jordan-1-Mid-Wear-Away-Chicago-Gs 0
```

## One Shoe Directly

```sh
bash bash_scripts/train_shoe.sh Nike-Calm-Slide-Cinnamon-Monarch 2
```

## All Shoes

```sh
SUGAR_GPUS="0 1 2 3" bash bash_scripts/train_all_shoes.sh
tmux attach -t sugar_all_shoes
```

The default list is `bash_scripts/shoes.txt`.

## Important Defaults

```sh
SUGAR_DATA_ROOT=/data/abelde/datasets/raw/golden_set
SUGAR_ENV=/data/abelde/projects/active/Shell_Gaussian/baselines/SuGaR/SuGaR_env
SUGAR_OUTPUT_ROOT=/data/abelde/projects/active/Shell_Gaussian/baselines/SuGaR/output/sugar_runs
SUGAR_CACHE_ROOT=/data/abelde/.cache
SUGAR_GS_ITERATIONS=15000
SUGAR_GS_RESOLUTION=2
SUGAR_REGULARIZATION=density
SUGAR_MESH_VERTICES=200000
SUGAR_GAUSSIANS_PER_TRIANGLE=6
SUGAR_REFINEMENT_END_ITER=9000
```

`SUGAR_REFINEMENT_END_ITER=9000` means roughly 2k real refinement iterations in this clone, because the trainer starts its internal counter around 7k.

Each run writes:

```text
output/sugar_runs/<run_id>/vanilla_3dgs/
output/sugar_runs/<run_id>/coarse/
output/sugar_runs/<run_id>/coarse_mesh/
output/sugar_runs/<run_id>/refined/
output/sugar_runs/<run_id>/refined_mesh/
output/sugar_runs/<run_id>/previews/
output/sugar_runs/<run_id>/logs/pipeline.log
```

The quick visuals are saved as PNGs and a GIF under `previews/`.
