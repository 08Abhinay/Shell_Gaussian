"""One run: prepared shoes -> Checkpoint 11-D, all 27 in one output root.

The differentiable stages run here on the GPU. Every CPU stage is the existing
``foot_prior`` script, invoked unchanged through ``shellgaussianenv``. A shoe
that fails a stage stops advancing and stays visible in ``status.json``; a
leftover file from an earlier attempt is never taken as proof of success.

Two shared inputs are copied rather than recomputed, because the downstream
topology digests must keep matching: the canonical tetrahedral volume and the
canonical 11-D scalar field.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCRIPTS = PROJECT_ROOT / "scripts"
CPU_PYTHON = Path("/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python")
GPU_PYTHON = Path("/home/ab5298/anaconda3/envs/Shell/bin/python")
FOOT_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
BODY_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male.npy"

#: Canonical artifacts that are shared by every instance and must stay
#: byte-identical to the validated originals.
STABLE_ROOT = Path("/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation")
CANONICAL_VOLUME = STABLE_ROOT / "anatomical_volume/reference"
CANONICAL_SCALAR_FIELD = STABLE_ROOT / "anatomical_fibers/converged/reference"

DEFAULT_GPUS = (2, 4, 6)

#: Stages that refuse a --jobs value above their own ceiling.
BATCH_MAX_JOBS = {"run_anatomical_fibers.py": 8}

THREAD_ENV = {
    key: "1"
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


def _stage_command(script: str, argv: list[str]) -> list[str]:
    """Invoke a stage as ``python -m scripts.<name>``.

    Running it by file path puts the *script's* directory on ``sys.path``
    rather than the project root, so ``run_anatomical_fibers`` fails at import
    with "No module named 'scripts'". The module form puts the working
    directory first, which is what the repository's own runner does.
    """

    return [str(CPU_PYTHON), "-m", f"scripts.{Path(script).stem}", *argv]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pipeline:
    """Drives the stages and records which shoes are still alive."""

    def __init__(
        self,
        root: Path,
        shoes: Sequence[str],
        gpus: Sequence[int],
        log_dir: Path,
        jobs: int,
    ) -> None:
        self.root = Path(root)
        self.shoes = list(shoes)
        self.alive = list(shoes)
        self.gpus = list(gpus)
        self.jobs = int(jobs)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, Any] = {
            "started": _now(),
            "root": str(self.root),
            "requested": list(shoes),
            "gpus": list(gpus),
            "stages": [],
            "failures": {},
        }

    # -- bookkeeping ------------------------------------------------------
    def _save(self, state: str) -> None:
        self.status["state"] = state
        self.status["updated"] = _now()
        self.status["alive"] = list(self.alive)
        (self.root / "status.json").write_text(
            json.dumps(self.status, indent=2) + "\n"
        )

    def _drop(self, name: str, stage: str, reason: str) -> None:
        if name in self.alive:
            self.alive.remove(name)
        self.status["failures"].setdefault(name, []).append(
            {"stage": stage, "reason": reason[:400]}
        )
        print(f"  ! {name}: {reason[:160]}", flush=True)

    # -- process helpers ---------------------------------------------------
    def _run(self, label: str, command: list[str], env_extra: dict | None = None) -> tuple[int, str]:
        env = dict(os.environ)
        env.update(THREAD_ENV)
        if env_extra:
            env.update(env_extra)
        log = self.log_dir / f"{label}.log"
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write_text(result.stdout or "")
        return result.returncode, result.stdout or ""

    def cpu_stage(
        self,
        script: str,
        argv: Callable[[list[str]], list[str]],
        artifact: Callable[[str], Path],
        verify: Callable[[Path, str], None] | None = None,
        per_shoe: bool = False,
        reset: Callable[[str], Path] | None = None,
        reset_root: Path | None = None,
    ) -> None:
        """Run one foot_prior stage over the surviving shoes.

        ``reset`` names a per-shoe directory to remove first. Two stages -
        ``run_instance_volume_batch`` and ``run_anatomical_fibers`` - have no
        ``--overwrite`` and silently skip a shoe whose artifact already exists.
        On a rerun that pairs a stale volume with freshly fitted anatomy, and
        11-C then correctly refuses it with "normalized fitted anatomy and B3
        boundary disagree". Clearing only the shoes about to be reprocessed
        keeps the rerun honest without touching anything else.
        """

        if not self.alive:
            return
        # These two stages refuse to start at all when their output root
        # exists, so the whole root goes rather than the per-shoe directories.
        if reset_root is not None and Path(reset_root).is_dir():
            shutil.rmtree(reset_root)
        if reset is not None:
            for name in self.alive:
                stale = reset(name)
                if stale.is_dir():
                    shutil.rmtree(stale)
        print(f"[stage] {script}  ({len(self.alive)} shoes)", flush=True)
        started = _now()
        targets = list(self.alive)
        # Several foot_prior stages build CUDA buffers even though their work
        # is on the CPU. Without a pinned device they land on GPU 0, which on
        # this host is usually someone else's. Pin them to ours.
        env_extra = {"CUDA_VISIBLE_DEVICES": str(self.gpus[0])} if self.gpus else None
        if per_shoe:
            for name in targets:
                code, output = self._run(
                    f"{script}.{name}",
                    _stage_command(script, argv([name])),
                    env_extra,
                )
                if code != 0:
                    self._drop(name, script, output.strip().splitlines()[-1] if output.strip() else f"exit {code}")
        else:
            code, output = self._run(
                script,
                _stage_command(script, argv(targets)),
                env_extra,
            )
            if code != 0:
                print(f"  (batch returned {code}; checking artifacts per shoe)", flush=True)
        for name in list(self.alive):
            path = artifact(name)
            if not path.is_file():
                self._drop(name, script, f"missing {path.name}")
                continue
            if verify is not None:
                try:
                    verify(path, name)
                except Exception as error:  # noqa: BLE001
                    self._drop(name, script, f"{type(error).__name__}: {error}")
        self.status["stages"].append(
            {
                "stage": script,
                "started": started,
                "finished": _now(),
                "surviving": len(self.alive),
            }
        )
        self._save("running")
        print(f"[stage] {script} -> {len(self.alive)} alive", flush=True)

    # -- verifiers ---------------------------------------------------------
    @staticmethod
    def _expect(path: Path, name: str, **fields: Any) -> dict:
        payload = json.loads(path.read_text())
        for key, wanted in fields.items():
            value = payload.get(key)
            if isinstance(wanted, tuple):
                if value not in wanted:
                    raise ValueError(f"{key}={value!r} not in {wanted}")
            elif value != wanted:
                raise ValueError(f"{key}={value!r} expected {wanted!r}")
        return payload

    # -- GPU stage ---------------------------------------------------------
    def gpu_fit_stage(
        self, restarts: int, input_root: Path, max_beta_norm: float | None = None
    ) -> None:
        """Fit every shoe's foot, sharded across the free GPUs."""

        print(f"[stage] torch foot fit  ({len(self.alive)} shoes, GPUs {self.gpus})", flush=True)
        started = _now()
        shards: list[list[str]] = [[] for _ in self.gpus]
        for index, name in enumerate(self.alive):
            shards[index % len(self.gpus)].append(name)

        processes = []
        for gpu, shard in zip(self.gpus, shards):
            if not shard:
                continue
            report = self.root / "logs" / f"fit_gpu{gpu}.json"
            command = [
                str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.fit_stage",
                "--output-root", str(self.root),
                "--input-root", str(input_root),
                "--restarts", str(restarts),
                "--report", str(report),
                *(["--max-beta-norm", str(max_beta_norm)]
                  if max_beta_norm is not None else []),
                "--shoes", *shard,
            ]
            env = dict(os.environ)
            env.update(THREAD_ENV)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(self.log_dir / f"fit_gpu{gpu}.log", "w")
            processes.append(
                (gpu, subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log)
            )
        for gpu, process, log in processes:
            code = process.wait()
            log.close()
            if code != 0:
                print(f"  ! GPU {gpu} fit shard exited {code}", flush=True)

        for name in list(self.alive):
            path = self.root / "containment_fit" / name / "containment_fit.json"
            if not path.is_file():
                self._drop(name, "torch_foot_fit", "no containment_fit.json")
                continue
            try:
                self._expect(
                    path, name, schema_version=5, shoe_profile="normal",
                    status=("contained_target_fit", "residual_target_fit"),
                )
                if not (path.parent / "foot_containment_fitted.ply").is_file():
                    raise FileNotFoundError("foot_containment_fitted.ply")
            except Exception as error:  # noqa: BLE001
                self._drop(name, "torch_foot_fit", f"{type(error).__name__}: {error}")
        self.status["stages"].append(
            {"stage": "torch_foot_fit", "started": started, "finished": _now(),
             "surviving": len(self.alive)}
        )
        self._save("running")
        print(f"[stage] torch foot fit -> {len(self.alive)} alive", flush=True)


def _gpu_leg_stage(self, input_root: Path, steps: int) -> None:
    """Fit every shank on the GPU, sharded like the foot fit."""

    print(f"[stage] torch lower leg  ({len(self.alive)} shoes, GPUs {self.gpus})", flush=True)
    started = _now()
    shards: list[list[str]] = [[] for _ in self.gpus]
    for index, name in enumerate(self.alive):
        shards[index % len(self.gpus)].append(name)

    processes = []
    for gpu, shard in zip(self.gpus, shards):
        if not shard:
            continue
        command = [
            str(GPU_PYTHON), "-m", "anatomical_coordinates.pipeline.leg_stage",
            "--anatomical-surface-root", str(self.root / "anatomical_surface"),
            "--output-root", str(self.root / "lower_leg_attachment"),
            "--full-body-supr-model", str(BODY_MODEL),
            "--input-root", str(input_root),
            "--steps", str(steps),
            "--shoes", *shard,
        ]
        env = dict(os.environ)
        env.update(THREAD_ENV)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = open(self.log_dir / f"leg_gpu{gpu}.log", "w")
        processes.append(
            (gpu, subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env,
                                   stdout=log, stderr=subprocess.STDOUT), log)
        )
    for gpu, process, log in processes:
        code = process.wait()
        log.close()
        if code != 0:
            print(f"  ! GPU {gpu} leg shard exited {code}", flush=True)

    for name in list(self.alive):
        path = self.root / "lower_leg_attachment" / name / "lower_leg_attachment.json"
        if not path.is_file():
            self._drop(name, "torch_lower_leg", "no lower_leg_attachment.json")
            continue
        try:
            Pipeline._expect(
                path, name, schema_version=2,
                stage="fitted_foot_natural_lower_leg_collar_fit",
            )
            if not (path.parent / "foot_lower_leg.ply").is_file():
                raise FileNotFoundError("foot_lower_leg.ply")
        except Exception as error:  # noqa: BLE001
            self._drop(name, "torch_lower_leg", f"{type(error).__name__}: {error}")
    self.status["stages"].append(
        {"stage": "torch_lower_leg", "started": started, "finished": _now(),
         "surviving": len(self.alive)}
    )
    self._save("running")
    print(f"[stage] torch lower leg -> {len(self.alive)} alive", flush=True)


Pipeline.gpu_leg_stage = _gpu_leg_stage


def seed_canonical(root: Path) -> None:
    """Copy the shared canonical artifacts. Never recompute them."""

    volume = root / "anatomical_volume" / "reference"
    if not volume.is_dir():
        volume.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CANONICAL_VOLUME, volume)
        print(f"[seed] canonical volume -> {volume}", flush=True)
    field = root / "anatomical_fibers" / "converged" / "reference"
    if not field.is_dir() and CANONICAL_SCALAR_FIELD.is_dir():
        field.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CANONICAL_SCALAR_FIELD, field)
        print(f"[seed] canonical scalar field -> {field}", flush=True)


def main() -> None:
    from .shoe_set import PIPELINE_ROOT, UNIFIED_INPUT_ROOT, all_shoes, build_unified_input_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--shoes", nargs="*", default=None)
    parser.add_argument("--gpus", type=int, nargs="*", default=list(DEFAULT_GPUS))
    parser.add_argument("--restarts", type=int, default=8)
    # 11-B/11-C/11-D are CPU/FEM and scale with --jobs. This host has 256
    # cores and typically sits near a load of 17, so a default of 8 left the
    # long pole of the pipeline almost entirely serial.
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--leg-steps", type=int, default=220)
    parser.add_argument(
        "--beta-ladder", type=float, nargs="*", default=None,
        help=(
            "anatomy budgets to try in order. A shoe keeps the largest budget "
            "the shared volume accepts; only shoes 11-B refuses are refitted "
            "tighter, so a shoe that already works is never degraded."
        ),
    )
    parser.add_argument(
        "--leg", choices=("torch", "numpy"), default="numpy",
        help="which lower-leg fitter to use",
    )
    parser.add_argument(
        "--from-stage", default="fit",
        choices=("fit", "surface", "leg", "extended", "11a", "11b", "11c", "11d"),
    )
    parser.add_argument(
        "--to-stage", default="11d",
        choices=("fit", "surface", "leg", "extended", "11a", "11b", "11c", "11d"),
        help="last stage to run; the flow path only needs the chain up to 'extended'",
    )
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    input_root = root / "inputs"
    build_unified_input_root(input_root)
    shoes = args.shoes or all_shoes(input_root)
    seed_canonical(root)

    volume_root = root / "anatomical_volume"
    # One B3 output root per anatomy budget, because the batch runner refuses
    # to start when its root exists and must not be handed a stale volume.
    # Accepted volumes are linked into a single root for 11-C and 11-D.
    ladder = list(args.beta_ladder) if args.beta_ladder else [None]
    b3_root = root / "instance_anatomical_volume" / "batch"
    accepted_root = root / "instance_anatomical_volume" / "accepted"
    mapping_root = root / "instance_volume_mapping" / "batch"
    audit_root = root / "anatomical_fibers" / "audit"
    for path in (b3_root, mapping_root, audit_root):
        path.parent.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(root, shoes, args.gpus, root / "logs", args.jobs)
    order = ("fit", "surface", "leg", "extended", "11a", "11b", "11c", "11d")
    start = order.index(args.from_stage)

    stop = order.index(args.to_stage)

    def wanted(stage: str) -> bool:
        return start <= order.index(stage) <= stop

    accepted: list[str] = []
    pending = list(shoes)
    budgets: dict[str, float | None] = {}
    accepted_rung: dict[str, int] = {}

    # Only run the ladder when the stages it drives are actually wanted.
    # Resuming at 11-C must not re-accept every shoe at rung 0: the ladder's
    # bookkeeping would then link each shoe's *failed* first attempt.
    # The ladder drives the fit..11-B stages, so it runs whenever the start is
    # at or before 11-B. It must NOT be disabled by stopping early: a run that
    # ends at 'extended' still needs the fit to happen.
    runs_ladder = order.index(args.from_stage) <= order.index("11b")
    for rung, cap in enumerate(ladder if runs_ladder else []):
        if not pending:
            break
        if len(ladder) > 1:
            b3_root = root / "instance_anatomical_volume" / f"batch_rung{rung}"
            pipe.alive = list(pending)
            pipe.shoes = list(pending)
            print(f"\n[ladder] anatomy budget {cap}: {len(pending)} shoe(s)", flush=True)

        if wanted("fit"):
            pipe.gpu_fit_stage(args.restarts, input_root, max_beta_norm=cap)

        if wanted("surface"):
            pipe.cpu_stage(
                "run_anatomical_surface.py",
                lambda shoes: [
                    "--containment-root", str(root / "containment_fit"),
                    "--supr-model", str(FOOT_MODEL),
                    "--output-root", str(root / "anatomical_surface"),
                    "--overwrite", *shoes,
                ],
                lambda n: root / "anatomical_surface" / n / "anatomical_surface.json",
                lambda p, n: Pipeline._expect(
                    p, n, schema_version=1,
                    stage="canonical_dense_supr_anatomical_surface",
                ),
            )

        if wanted("leg"):
            if args.leg == "torch":
                pipe.gpu_leg_stage(input_root, args.leg_steps)
            else:
                pipe.cpu_stage(
                    "run_lower_leg_attachment.py",
                    lambda shoes: [
                        "--anatomical-surface-root", str(root / "anatomical_surface"),
                        "--preparation-root", str(input_root / "shoe_preparation"),
                        "--support-fit-root", str(input_root / "support_fit"),
                        "--full-body-supr-model", str(BODY_MODEL),
                        "--output-root", str(root / "lower_leg_attachment"),
                        "--overwrite", *shoes,
                    ],
                    lambda n: root / "lower_leg_attachment" / n / "lower_leg_attachment.json",
                    lambda p, n: Pipeline._expect(
                        p, n, schema_version=2,
                        stage="fitted_foot_natural_lower_leg_collar_fit",
                    ),
                    per_shoe=True,
                )

        if wanted("extended"):
            pipe.cpu_stage(
                "run_extended_anatomical_surface.py",
                lambda shoes: [
                    "--anatomical-surface-root", str(root / "anatomical_surface"),
                    "--lower-leg-root", str(root / "lower_leg_attachment"),
                    "--full-body-supr-model", str(BODY_MODEL),
                    "--output-root", str(root / "extended_anatomical_surface"),
                    "--overwrite", *shoes,
                ],
                lambda n: root / "extended_anatomical_surface" / n / "extended_anatomical_surface.json",
                lambda p, n: Pipeline._expect(
                    p, n, schema_version=1,
                    stage="extended_canonical_supr_anatomical_surface",
                ),
            )

        if wanted("11a"):
            pipe.cpu_stage(
                "run_instance_anatomical_volume.py",
                lambda shoes: [
                    "--anatomical-volume-root", str(volume_root),
                    "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                    "--overwrite", *shoes,
                ],
                lambda n: volume_root / n / "boundary_target.json",
                lambda p, n: Pipeline._expect(
                    p, n, status=("ready", "ready_requires_untangling")
                ),
            )

        if wanted("11b"):
            current_b3 = b3_root
            pipe.cpu_stage(
                "run_instance_volume_batch.py",
                lambda shoes: [
                    "--anatomical-volume-root", str(volume_root),
                    "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                    "--output-root", str(current_b3),
                    "--jobs", str(args.jobs), *shoes,
                ],
                lambda n: current_b3 / n / "instance_volume.json",
                lambda p, n: Pipeline._expect(
                    p, n, status=("final_exact_target", "final_corrected_target")
                ),
                reset_root=current_b3,
            )

    # A shoe keeps the largest budget the shared volume accepts. Only the
        # shoes 11-B refused move to the next rung, so nothing that already
        # works is refitted tighter.
        for name in pipe.alive:
            budgets[name] = cap
            accepted_rung[name] = rung
        accepted.extend(pipe.alive)
        pending = [n for n in pending if n not in set(pipe.alive)]
        # Without 11-B there is no accept/reject signal, so there is nothing for
        # a second rung to act on; whatever survived the stages that did run is
        # the result.
        if len(ladder) == 1 or not wanted("11b"):
            accepted = list(pipe.alive)
            break

    # Gather the accepted volumes into one root for the remaining stages.
    if len(ladder) > 1 and runs_ladder:
        if accepted_root.is_dir():
            shutil.rmtree(accepted_root)
        accepted_root.mkdir(parents=True, exist_ok=True)
        # Link each shoe from the rung that actually accepted it. An earlier
        # rung usually left a directory behind for the same shoe - the attempt
        # that failed - and linking that one hands 11-C a volume with no npz.
        for name in accepted:
            source = (
                root / "instance_anatomical_volume"
                / f"batch_rung{accepted_rung[name]}" / name
            )
            if source.is_dir():
                (accepted_root / name).symlink_to(source)
        for rung in sorted(set(accepted_rung.values())):
            source = root / "instance_anatomical_volume" / f"batch_rung{rung}"
            for entry in source.iterdir():
                if entry.is_file() and not (accepted_root / entry.name).exists():
                    shutil.copyfile(entry, accepted_root / entry.name)
        b3_root = accepted_root
    if len(ladder) > 1 and not accepted and accepted_root.is_dir():
        # Resuming at a later stage: the ladder already ran, so take the
        # accepted set from disk rather than rebuilding it.
        accepted = sorted(
            e.name for e in accepted_root.iterdir() if e.is_dir() or e.is_symlink()
        )
        b3_root = accepted_root
    pipe.alive = list(accepted)
    pipe.shoes = list(shoes)
    pipe.status["anatomy_budgets"] = budgets

    if wanted("11c"):
        pipe.cpu_stage(
            "run_instance_volume_mapping.py",
            lambda shoes: [
                "--anatomical-volume-root", str(volume_root),
                "--instance-volume-batch-root", str(b3_root),
                "--containment-fit-root", str(root / "containment_fit"),
                "--output-root", str(mapping_root),
                "--jobs", str(args.jobs), "--overwrite", *shoes,
            ],
            lambda n: mapping_root / n / "mapping_validation.json",
            lambda p, n: Pipeline._expect(p, n, status="mapping_valid"),
        )

    if wanted("11d"):
        pipe.cpu_stage(
            "run_anatomical_fibers.py",
            lambda shoes: [
                "fibers",
                "--anatomical-volume-root", str(volume_root),
                "--scalar-field-root", str(root / "anatomical_fibers" / "converged" / "reference"),
                "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                "--instance-volume-batch-root", str(b3_root),
                "--containment-fit-root", str(root / "containment_fit"),
                "--output-root", str(audit_root),
                "--jobs", str(min(args.jobs, BATCH_MAX_JOBS["run_anatomical_fibers.py"])),
                "--shoes", *shoes,
            ],
            lambda n: audit_root / n / "fiber_coverage.json",
            None,
            reset_root=audit_root,
        )

    complete = len(pipe.alive) == len(shoes)
    pipe._save("complete" if complete else "partial_failure")
    print(
        f"\n{'='*66}\n{len(pipe.alive)}/{len(shoes)} shoes reached the end\n"
        f"root: {root}\nstatus: {root / 'status.json'}\n{'='*66}"
    )
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
