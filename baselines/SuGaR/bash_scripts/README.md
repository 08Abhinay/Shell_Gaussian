# SuGaR Shoe Mesh Scripts

These scripts mirror the NeuS2 batch style and keep caches under `/data/abelde`.

## Prepare Blender Evaluation Data

Use the permanent launcher below to convert exact-camera Blender scenes into
masked COLMAP/SuGaR scenes. It launches tmux, distributes shoes across the
requested physical GPUs, skips existing valid scenes, records the resolved shoe
list, and validates every selected scene.

Prepare specific shoes:

```sh
bash bash_scripts/prepare_evaluation_dataset_tmux.sh \
  --shoe air_jordan_1 \
  --shoe birkenstock_arizona_sandal \
  --gpus 2,3
```

Prepare a text file of shoes:

```sh
bash bash_scripts/prepare_evaluation_dataset_tmux.sh \
  --shoe-list /path/to/evaluation_shoes.txt \
  --gpus 2,3
```

Prepare all reviewed GLBs:

```sh
bash bash_scripts/prepare_evaluation_dataset_tmux.sh \
  --all \
  --gpus 2,3
```

The default derived dataset is:

```text
/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender_sugar
```

## Blender Evaluation Training

Use the evaluation launcher for a reproducible, resumable paper run:

```sh
bash bash_scripts/train_evaluation_shoe_tmux.sh \
  --shoe adidas_yeezy_boost_350_v2_zyon \
  --gpu 2 \
  --session sugar_adidas_yeezy_pilot
```

The launcher prepares a missing SuGaR scene, validates the exact 150/30 split,
and then runs 7k vanilla 3DGS, 15k bounded `dn_consistency` SuGaR, a
one-million-vertex level-0.3 mesh, and 15k refinement with one Gaussian per
triangle. It uses RGBA masks, white backgrounds, the sparse foreground bounds,
and a 5% Gaussian scale cap. It finishes with the repository-native PSNR, SSIM,
and LPIPS metrics on exactly 30 held-out views plus shared aligned mesh metrics.

Monitor without attaching:

```sh
tail -f output/golden_set_evaluation_blender_pilot/adidas_yeezy_boost_350_v2_zyon/logs/batch.log
nvidia-smi -i 2
```

The final textured OBJ path is recorded in `refinement_run_result.json`. A
texture-free `final_geometry_mesh.ply`, stage manifests, native image metrics,
runtime, GPU-memory samples, and logs are stored in the same shoe output.

Run the fixed five-shoe benchmark as two sequential GPU queues:

```sh
bash bash_scripts/train_evaluation_batch_tmux.sh \
  --gpus 2,3 \
  --session sugar_final_five_gpu23
```

The queue uses `bash_scripts/evaluation_shoes.txt`. Each GPU processes one shoe
at a time. Prepared scenes that do not satisfy the current 150/30 split contract
are rebuilt and validated automatically. A failed shoe is retried once by
default, then recorded in `batch_runs/<session>/status.tsv`; the worker continues
with later shoes instead of abandoning its queue. Use `--retries N` to change
the retry count and `--overwrite-data` only when every derived scene should be
rebuilt deliberately.

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
