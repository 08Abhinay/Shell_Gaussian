"""Run every (variant, split, seed) of the Stage 1 comparison, a queue per GPU.

    python -m anatomical_coordinates.stage1.run_all --gpu-ids 1 2 3 --seeds 0 1

Each job is its own process, so a crash loses one job, not the queue; a job
whose result file already exists is skipped, so the command can be re-run to
fill gaps.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
import threading

from . import common
from .train import SPLITS, VARIANTS

PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=12000)
    args = parser.parse_args()

    jobs = [
        (v, s, seed) for seed, s, v in itertools.product(args.seeds, args.splits, args.variants)
        if not (common.STAGE1_OUTPUT / "runs" / s / f"{v}_seed{seed}.json").is_file()
    ]
    logs = common.STAGE1_OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if not jobs:
                    return
                variant, split, seed = jobs.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTHONPATH=f"{PACKAGE_PARENT}:{os.environ.get('PYTHONPATH', '')}")
            with open(logs / f"train.{split}.{variant}.seed{seed}.log", "w") as handle:
                code = subprocess.call(
                    [sys.executable, "-m", "anatomical_coordinates.stage1.train",
                     "--variant", variant, "--split", split, "--seed", str(seed),
                     "--steps", str(args.steps)],
                    cwd=str(PACKAGE_PARENT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            print(f"gpu {gpu}: {split}/{variant}/seed{seed} exit {code}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
