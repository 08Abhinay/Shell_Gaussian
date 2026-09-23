"""One command: raw prepared shoes to anatomical addresses.

    python -m anatomical_coordinates.pipeline.run_pipeline --gpus 2

Every stage writes into one output root, one directory per stage, named for
what the stage does. A shoe that fails a stage stops advancing and stays
visible in ``status.json``; the batch continues.

Two shared artifacts are copied rather than recomputed - the canonical
tetrahedral volume and the canonical semantic field - because the downstream
topology digests must keep matching the validated originals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from .stages import BY_KEY, ORDER, STAGES, slice_stages


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
LEGACY = PROJECT_ROOT / "scripts_legacy"

GPU_PYTHON = Path("/home/ab5298/anaconda3/envs/Shell/bin/python")
CPU_PYTHON = Path("/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python")
FOOT_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
BODY_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male.npy"

DATASET = Path("/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation")
DEFAULT_OUTPUT = Path("/home/ab5298/Outputs/FootShellGaussian/anatomical_coordinates")

#: Reference artifacts that are shared by every shoe and must stay identical.
VALIDATED = Path("/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation")
CANONICAL_VOLUME = VALIDATED / "anatomical_volume/reference"
CANONICAL_FIELD = VALIDATED / "anatomical_fibers/converged/reference"
#: The outward direction field solved for during the original fiber work. It is
#: what makes fibers traceable and is read unchanged, never recomputed.
CANONICAL_DIRECTIONS = (
    VALIDATED / "anatomical_fibers/fiber_audit_directionfix_20260919_014849"
    / "reference" / "fiber_field.npz"
)

#: Source geometry that is inconsistent with the accepted normalized set.
EXCLUDED = ("sneaker_vibe",)

SINGLE_THREAD = {
    key: "1"
    for key in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pipeline:
    def __init__(self, root: Path, shoes: Sequence[str], gpus: Sequence[int], jobs: int):
        self.root = Path(root)
        self.shoes = list(shoes)
        self.alive = list(shoes)
        self.gpus = list(gpus)
        self.jobs = int(jobs)
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, Any] = {
            "started": now(), "root": str(self.root), "requested": list(shoes),
            "gpus": list(gpus), "stages": [], "failures": {},
        }

    def dir(self, key: str) -> Path:
        path = self.root / BY_KEY[key].directory
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state: str) -> None:
        self.status["state"] = state
        self.status["updated"] = now()
        self.status["alive"] = list(self.alive)
        (self.root / "status.json").write_text(json.dumps(self.status, indent=2) + "\n")

    def drop(self, name: str, stage: str, reason: str) -> None:
        if name in self.alive:
            self.alive.remove(name)
        self.status["failures"].setdefault(name, []).append(
            {"stage": stage, "reason": reason[:400]}
        )
        print(f"    ! {name}: {reason[:150]}", flush=True)

    def _run(self, label: str, command: list[str], gpu: int | None = None) -> tuple[int, str]:
        env = dict(os.environ)
        env.update(SINGLE_THREAD)
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH','')}"
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        result = subprocess.run(
            command, cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        (self.logs / f"{label}.log").write_text(result.stdout or "")
        return result.returncode, result.stdout or ""

    def shard(self, command_for: Callable[[int, list[str]], list[str]], label: str) -> None:
        """Split the surviving shoes across the GPUs and run them in parallel."""

        shards: list[list[str]] = [[] for _ in self.gpus]
        for index, name in enumerate(self.alive):
            shards[index % len(self.gpus)].append(name)
        running = []
        for gpu, shard in zip(self.gpus, shards):
            if not shard:
                continue
            env = dict(os.environ)
            env.update(SINGLE_THREAD)
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH','')}"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = open(self.logs / f"{label}.gpu{gpu}.log", "w")
            running.append((
                gpu,
                subprocess.Popen(command_for(gpu, shard), cwd=str(PROJECT_ROOT),
                                 env=env, stdout=handle, stderr=subprocess.STDOUT),
                handle,
            ))
        for gpu, process, handle in running:
            code = process.wait()
            handle.close()
            if code != 0:
                print(f"    (GPU {gpu} shard exited {code})", flush=True)

    def per_shoe(self, key: str, argv: Callable[[str], list[str]],
                 python: Path, parallel_per_gpu: int = 1) -> None:
        """Run one independent command per shoe, spread over the GPUs.

        ``prepare`` and ``seat`` are per-shoe and independent, but were written
        as a blocking loop that merely *assigned* a GPU round-robin - so the
        assignment was decorative and 15 minutes of the run was serial on one
        device. They shard like every other per-shoe stage now.
        """

        queue = list(self.alive)
        capacity = max(1, len(self.gpus) * max(1, parallel_per_gpu))
        running: list[tuple[str, subprocess.Popen, Any]] = []
        failures: dict[str, str] = {}
        cycle = 0

        def launch(name: str) -> None:
            nonlocal cycle
            gpu = self.gpus[cycle % len(self.gpus)]
            cycle += 1
            env = dict(os.environ)
            env.update(SINGLE_THREAD)
            env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH','')}"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = open(self.logs / f"{key}.{name}.log", "w")
            running.append((
                name,
                subprocess.Popen([str(python), *argv(name)], cwd=str(PROJECT_ROOT),
                                 env=env, stdout=handle, stderr=subprocess.STDOUT),
                handle,
            ))

        while queue or running:
            while queue and len(running) < capacity:
                launch(queue.pop(0))
            time.sleep(0.5)
            for entry in [item for item in running if item[1].poll() is not None]:
                name, process, handle = entry
                handle.close()
                running.remove(entry)
                if process.returncode != 0:
                    lines = (self.logs / f"{key}.{name}.log").read_text().strip().splitlines()
                    failures[name] = lines[-1] if lines else f"exit {process.returncode}"
        for name, reason in sorted(failures.items()):
            self.drop(name, key, reason)

    def check(self, key: str, artifact: Callable[[str], Path],
              verify: Callable[[Path, str], None] | None = None) -> None:
        for name in list(self.alive):
            path = artifact(name)
            if not path.is_file():
                self.drop(name, key, f"missing {path.name}")
                continue
            if verify is not None:
                try:
                    verify(path, name)
                except Exception as error:  # noqa: BLE001
                    self.drop(name, key, f"{type(error).__name__}: {error}")

    @staticmethod
    def expect(path: Path, name: str, **fields: Any) -> dict:
        payload = json.loads(path.read_text())
        for field, wanted in fields.items():
            value = payload.get(field)
            if isinstance(wanted, tuple):
                if value not in wanted:
                    raise ValueError(f"{field}={value!r} not in {wanted}")
            elif value != wanted:
                raise ValueError(f"{field}={value!r} expected {wanted!r}")
        return payload

    def begin(self, stage) -> None:
        print(f"\n[{stage.key}] {stage.summary}  ({len(self.alive)} shoes)", flush=True)
        self._started = now()

    def end(self, stage) -> None:
        self.status["stages"].append({
            "stage": stage.key, "directory": stage.directory,
            "started": self._started, "finished": now(), "surviving": len(self.alive),
        })
        self.save("running")
        print(f"[{stage.key}] -> {len(self.alive)} shoes", flush=True)


# --------------------------------------------------------------------------


def seed_canonical(root: Path) -> None:
    target = root / "canonical"
    if (target / "volume").is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CANONICAL_VOLUME, target / "volume")
    if CANONICAL_FIELD.is_dir():
        shutil.copytree(CANONICAL_FIELD, target / "semantic_field")
    if CANONICAL_DIRECTIONS.is_file():
        shutil.copyfile(CANONICAL_DIRECTIONS, target / "semantic_field" / "fiber_field.npz")
    print(f"[canonical] copied the shared reference volume and field", flush=True)


def discover(dataset: Path) -> list[str]:
    return sorted(
        item.name for item in dataset.iterdir()
        if item.is_dir()
        and item.name not in EXCLUDED
        and (item / "reference_mesh.ply").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument(
        "--gpus", type=int, default=2,
        help="how many GPUs to use, or list explicit ids with --gpu-ids",
    )
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument(
        "--per-gpu", type=int, default=2,
        help="concurrent per-shoe jobs on each GPU; these stages are small",
    )
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--from", dest="first", default="prepare", choices=ORDER)
    parser.add_argument("--to", dest="last", default="address", choices=ORDER)
    parser.add_argument("--samples", type=int, default=16384)
    args = parser.parse_args()

    gpus = args.gpu_ids if args.gpu_ids else list(range(args.gpus))
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    shoes = args.shoes or discover(args.dataset)
    wanted = {stage.key for stage in slice_stages(args.first, args.last)}

    print(f"anatomical coordinates: {len(shoes)} shoes -> {root}")
    print(f"GPUs {gpus}   stages {args.first} .. {args.last}")
    pipe = Pipeline(root, shoes, gpus, args.jobs)
    seed_canonical(root)

    # -- prepare ----------------------------------------------------------
    if "prepare" in wanted:
        stage = BY_KEY["prepare"]; pipe.begin(stage)
        out = pipe.dir("prepare")
        pipe.per_shoe("prepare", lambda n: [
            str(LEGACY / "run_shoe_preparation.py"),
            "--shoe-mesh", str(args.dataset / n / "reference_mesh.ply"),
            "--canonicalization", str(args.dataset / n / "blender_canonicalization.json"),
            "--output-dir", str(out / n), "--overwrite",
        ], GPU_PYTHON, parallel_per_gpu=args.per_gpu)
        pipe.check("prepare", lambda n: pipe.dir("prepare") / n / "shoe_preparation.json")
        pipe.end(stage)

    # -- seat -------------------------------------------------------------
    if "seat" in wanted:
        stage = BY_KEY["seat"]; pipe.begin(stage)
        out = pipe.dir("seat")
        pipe.per_shoe("seat", lambda n: [
            str(LEGACY / "run_alignment.py"),
            "--preparation-dir", str(pipe.dir("prepare") / n),
            "--supr-model", str(FOOT_MODEL),
            "--output-dir", str(out / n), "--overwrite",
        ], GPU_PYTHON, parallel_per_gpu=args.per_gpu)
        pipe.check("seat", lambda n: pipe.dir("seat") / n / "support_fit.json")
        pipe.end(stage)

    # The later stages read a single root holding both prepared directories.
    inputs = root / "inputs"
    for kind, source in (("shoe_preparation", "prepare"), ("support_fit", "seat")):
        (inputs / kind).mkdir(parents=True, exist_ok=True)
        for name in pipe.alive:
            link = inputs / kind / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(pipe.dir(source) / name)

    # -- fit --------------------------------------------------------------
    if "fit" in wanted:
        stage = BY_KEY["fit"]; pipe.begin(stage)
        pipe.shard(
            lambda gpu, shard: [
                str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.fit_stage",
                "--output-root", str(root), "--input-root", str(inputs),
                "--restarts", str(args.restarts),
                "--containment-dir", str(pipe.dir("fit")),
                "--shoes", *shard,
            ],
            "fit",
        )
        pipe.check(
            "fit", lambda n: pipe.dir("fit") / n / "containment_fit.json",
            lambda p, n: Pipeline.expect(p, n, schema_version=5, shoe_profile="normal"),
        )
        pipe.end(stage)

    # -- anatomy ----------------------------------------------------------
    if "anatomy" in wanted:
        stage = BY_KEY["anatomy"]; pipe.begin(stage)
        pipe._run("anatomy", [
            str(CPU_PYTHON), "-m", "scripts_legacy.run_anatomical_surface",
            "--containment-root", str(pipe.dir("fit")),
            "--supr-model", str(FOOT_MODEL),
            "--output-root", str(pipe.dir("anatomy")),
            "--overwrite", *pipe.alive,
        ], gpus[0])
        pipe.check(
            "anatomy", lambda n: pipe.dir("anatomy") / n / "anatomical_surface.json",
            lambda p, n: Pipeline.expect(p, n, schema_version=1),
        )
        pipe.end(stage)

    # -- leg --------------------------------------------------------------
    if "leg" in wanted:
        stage = BY_KEY["leg"]; pipe.begin(stage)
        pipe.shard(
            lambda gpu, shard: [
                str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.leg_stage",
                "--anatomical-surface-root", str(pipe.dir("anatomy")),
                "--output-root", str(pipe.dir("leg")),
                "--full-body-supr-model", str(BODY_MODEL),
                "--input-root", str(inputs),
                "--shoes", *shard,
            ],
            "leg",
        )
        pipe.check(
            "leg", lambda n: pipe.dir("leg") / n / "lower_leg_attachment.json",
            lambda p, n: Pipeline.expect(p, n, schema_version=2),
        )
        pipe.end(stage)

    # -- join -------------------------------------------------------------
    if "join" in wanted:
        stage = BY_KEY["join"]; pipe.begin(stage)
        pipe._run("join", [
            str(CPU_PYTHON), "-m", "scripts_legacy.run_extended_anatomical_surface",
            "--anatomical-surface-root", str(pipe.dir("anatomy")),
            "--lower-leg-root", str(pipe.dir("leg")),
            "--full-body-supr-model", str(BODY_MODEL),
            "--output-root", str(pipe.dir("join")),
            "--overwrite", *pipe.alive,
        ], gpus[0])
        pipe.check(
            "join", lambda n: pipe.dir("join") / n / "extended_anatomical_surface.json",
            lambda p, n: Pipeline.expect(p, n, schema_version=1),
        )
        pipe.end(stage)

    # -- coordinates ------------------------------------------------------
    if "coordinates" in wanted:
        stage = BY_KEY["coordinates"]; pipe.begin(stage)
        code, _ = pipe._run("coordinates", [
            str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.coordinate_stage",
            "--root", str(root), "--shoes", *pipe.alive,
        ], gpus[0])
        pipe.check("coordinates", lambda n: pipe.dir("coordinates") / "summary.json")
        pipe.end(stage)

    # -- address ----------------------------------------------------------
    if "address" in wanted:
        stage = BY_KEY["address"]; pipe.begin(stage)
        # Each shoe's addresses are independent once the correspondence table
        # exists, and the table is written on first use, so the shards share it.
        pipe.shard(lambda gpu, shard: [
            str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.address_stage",
            "--root", str(root), "--samples", str(args.samples), "--shoes", *shard,
        ], "address")
        pipe.check("address", lambda n: pipe.dir("address") / n / "addresses.npz")
        pipe.end(stage)

        # An address that round trips within its own shoe is not yet a shared
        # coordinate. This sends one set of addresses through every shoe.
        if len(pipe.alive) > 1:
            pipe._run("address_check", [
                str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.address_check",
                "--root", str(root),
            ], gpus[0])
            report = pipe.dir("address") / "cross_shoe.json"
            if report.is_file():
                levels = json.loads(report.read_text())["levels"]
                inner = [x for x in levels if x["level"] > 0.0]
                if inner:
                    drift = max(x["drift_p99_mm"] or 0.0 for x in inner)
                    print(f"    shared across shoes: address drift p99 {drift:.4f} mm")

    done = len(pipe.alive) == len(shoes)
    pipe.save("complete" if done else "partial")
    print(f"\n{'='*68}\n{len(pipe.alive)}/{len(shoes)} shoes completed\n"
          f"output: {root}\nstatus: {root / 'status.json'}\n{'='*68}")
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
