# NeuS2 Shoe Training Scripts

The normal entry point is `train_all_shoes.sh`. It starts a detached tmux session, so the batch keeps running after you disconnect SSH.

```sh
cd /data/abelde/projects/active/Shell_Gaussian/baselines/NeuS2
bash bash_scripts/train_all_shoes.sh
```

Attach to the running batch:

```sh
tmux attach -t neus2_all_shoes
```

Detach again with `Ctrl-b d`.

Defaults:

```sh
NEUS2_SHOE_LIST=bash_scripts/shoes.txt
NEUS2_N_STEPS=10000
NEUS2_TMUX_SESSION=neus2_all_shoes
```

If `NEUS2_GPUS` is not set, the script uses all GPUs listed by `nvidia-smi`. To choose GPUs yourself:

```sh
NEUS2_GPUS="0 1 2 3" bash bash_scripts/train_all_shoes.sh
```

Useful overrides:

```sh
NEUS2_DATA_ROOT=/data/abelde/datasets/processed/neus2_shoes
NEUS2_ENV=/data/abelde/projects/active/Shell_Gaussian/baselines/NeuS2/neus2_env
NEUS2_CONFIG=dtu.json
NEUS2_TRAIN_TRANSFORM=transform_train.json
NEUS2_CACHE_ROOT=/data/abelde/.cache
NEUS2_N_STEPS=10000
NEUS2_GPUS="0 1 2 3"
```

Logs are printed when the tmux session starts. Batch logs go under `output/batch_runs/<run_id>/`, and each shoe also writes `output/<shoe_name>_neus2_<steps>/logs/train.log`.

For debugging one shoe directly:

```sh
bash bash_scripts/train_shoe.sh Adidas-Yeezy-Boost-350-V2-Desert-Sage-Infant 0 10000
```

`train_shoe.sh` only sets cache directories, activates the NeuS2 conda env, sets `PYTHONPATH=build`, and runs:

```sh
python -u scripts/run.py \
  --scene <dataset>/<shoe>/transform_train.json \
  --name <shoe>_neus2_<steps> \
  --network dtu.json \
  --n_steps <steps>
```
