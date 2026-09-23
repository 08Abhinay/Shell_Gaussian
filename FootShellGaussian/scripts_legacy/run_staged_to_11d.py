#!/usr/bin/env python3
"""Carry verified containment fits through the lower-leg and 11-D stages.

Each shoe must exit successfully and produce a valid stage record before it
advances. Failures remain visible in the status file and make the final exit
nonzero; a leftover JSON file is never treated as proof of success.

Two inputs the later stages need are assembled first. Shoe preparations are
linked into the run root, because they may legitimately live under more than
one earlier run while every stage script takes a single ``--preparation-root``.
The canonical tetrahedral volume is copied from the validated reference: it is
shared by every instance and must stay byte-identical for the topology digests
to keep matching, so it is never recomputed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import BoundedSemaphore
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PYTHON = Path("/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python")
STABLE_ROOT = Path("/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation")
FOOT_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
BODY_MODEL = WORKSPACE_ROOT / "baselines/SUPR/data/supr_male.npy"
PREPARATION_ROOTS = (
    STABLE_ROOT / "shoe_preparation",
    STABLE_ROOT / "new_shoes_to_11d_framefix_20260919_124101/shoe_preparation",
)
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

EXCLUDED = ("sneaker_vibe",)
STAGE_SCRIPTS = (
    "run_anatomical_surface.py",
    "run_lower_leg_attachment.py",
    "run_extended_anatomical_surface.py",
    "run_instance_anatomical_volume.py",
    "run_instance_volume_batch.py",
    "run_instance_volume_mapping.py",
    "run_anatomical_fibers.py",
)
# The fiber runner caps process workers at eight, independently of the
# per-GPU shoe-worker limit used by the earlier stages.
BATCH_MAX_JOBS = {"run_anatomical_fibers.py": 8}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class Pipeline:
    """Run verified post-containment stages, retaining independent successes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.run_root
        self.jobs = args.jobs
        self.gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
        if not self.gpus or len(set(self.gpus)) != len(self.gpus) or any(
            not gpu.isdigit() for gpu in self.gpus
        ):
            raise ValueError("--gpus must list distinct numeric GPU IDs")
        if self.jobs > 8 * len(self.gpus):
            raise ValueError("at most eight shoe workers are allowed per GPU")
        self.gpu_slots = {gpu: BoundedSemaphore(8) for gpu in self.gpus}
        self.logs = self.root / "validation_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status_path = self.logs / "to_11d_status.json"
        self.shoes = sorted(
            directory.name
            for directory in (self.root / "containment_fit").iterdir()
            if (directory / "containment_fit.json").is_file()
        )
        if not self.shoes:
            raise ValueError(f"no containment fits under {self.root}/containment_fit")
        self.alive = [name for name in self.shoes if name not in EXCLUDED]
        self.stages: list[dict] = []
        self.dropped: dict[str, str] = {}
        self.excluded = sorted(set(self.shoes).intersection(EXCLUDED))

    # -- bookkeeping --------------------------------------------------------

    def _save(self, state: str) -> None:
        _write_json(
            self.status_path,
            {
                "status": state,
                "updated_at": _now(),
                "run_root": str(self.root),
                "gpus": self.gpus,
                "jobs": self.jobs,
                "shoes_total": len(self.shoes),
                "shoes_alive": len(self.alive),
                "alive": self.alive,
                "dropped": self.dropped,
                "excluded": self.excluded,
                "stages": self.stages,
            },
        )

    def _say(self, message: str) -> None:
        print(f"[{_now()}] {message}", flush=True)

    def _survivors(self, stage: str, failures: dict[str, str]) -> None:
        """Advance only shoes with a successful process and verified artifacts."""

        kept, lost = [], {}
        for name in self.alive:
            if name in failures:
                lost[name] = failures[name]
                self.dropped.setdefault(name, f"{stage}: {failures[name]}")
            else:
                kept.append(name)
        self.alive = kept
        self.stages.append(
            {
                "stage": stage,
                "finished_at": _now(),
                "kept": len(kept),
                "dropped": lost,
            }
        )
        if lost:
            for name, reason in sorted(lost.items()):
                self._say(f"  [failed] {name} at {stage}: {reason}")
        self._say(f"  {stage}: {len(kept)}/{len(self.shoes)} shoes continue")
        self._save("running")

    # -- execution ----------------------------------------------------------

    def _environment(self, gpu: str | None) -> dict[str, str]:
        environment = {**os.environ, **THREAD_ENV}
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu
        return environment

    def _batch_jobs(self, stage: str) -> int:
        return min(self.jobs, BATCH_MAX_JOBS.get(stage, self.jobs))

    def _check_stage_entrypoints(self) -> None:
        """Catch broken module imports before any expensive shoe processing."""

        for stage in STAGE_SCRIPTS:
            command = [str(PYTHON), "-m", f"scripts.{Path(stage).stem}", "--help"]
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self._environment(None),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{stage}: entry-point preflight failed (exit {result.returncode}): "
                    f"{result.stdout[-2000:]}"
                )
            if stage in BATCH_MAX_JOBS and "--jobs" not in result.stdout:
                raise RuntimeError(f"{stage}: expected --jobs option is unavailable")

    def _invoke(self, stage: str, argv: list[str], log: Path, gpu: str | None) -> int:
        script = PROJECT_ROOT / "scripts" / stage
        if not script.is_file():
            raise FileNotFoundError(f"{stage}: script not found: {script}")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"$ {' '.join(argv)}\n\n")
            stream.flush()
            result = subprocess.run(
                argv,
                cwd=PROJECT_ROOT,
                env=self._environment(gpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode

    @staticmethod
    def _record(path: Path, *, stage: str, name: str | None = None,
                statuses: tuple[str, ...] | None = None) -> dict:
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("stage") != stage:
            raise ValueError(f"{path}: wrong or missing stage")
        if name is not None and record.get("shoe_name") != name:
            raise ValueError(f"{path}: wrong shoe name")
        if statuses is not None and record.get("status") not in statuses:
            raise ValueError(f"{path}: unacceptable status {record.get('status')!r}")
        return record

    def _verify_lower_leg(self, name: str, path: Path) -> None:
        record = self._record(path, stage="fitted_foot_natural_lower_leg_collar_fit",
                              name=name, statuses=("clear_exit", "residual_collar_intersections"))
        fit = record.get("fit", {})
        baseline, selected, search = (fit.get(key, {}) for key in ("baseline", "selected", "search"))
        baseline_hits = baseline.get("exact_pair_count")
        selected_hits = selected.get("exact_pair_count")
        candidates = search.get("evaluated_candidate_count")
        reason = search.get("stopping_reason")
        joined_pairs = search.get("joined_nonfoot_intersection_pairs")
        if (not isinstance(baseline_hits, int) or not isinstance(selected_hits, int)
                or not isinstance(candidates, int) or candidates < 1
                or not isinstance(reason, str)
                or (baseline_hits > 0 and candidates <= 1)
                or (record["status"] == "clear_exit") != (selected_hits == 0)
                or fit.get("collar_intersections", {}).get("exact_pair_count") != selected_hits
                or (joined_pairs is not None and joined_pairs != 0)):
            raise ValueError(f"{name}: lower-leg pose/shape search record is incomplete")
        for artifact in ("foot_lower_leg.ply", "lower_leg_collar_colored.ply",
                         "lower_leg_collar_overlay.ply"):
            if not (path.parent / artifact).is_file():
                raise FileNotFoundError(path.parent / artifact)
        self._say(f"  [leg fit] {name}: {candidates} candidates; "
                  f"collisions {baseline_hits} -> {selected_hits}; "
                  f"start={search.get('selected_start')}; {reason}")

    def per_shoe(self, stage: str, arguments, produced,
                 verify: Callable[[str, Path], None]) -> None:
        """Run independent shoe jobs, with at most eight active on each GPU."""

        if not self.alive:
            return

        self._say(f"{stage}: {len(self.alive)} shoes, {self.jobs} workers, GPUs {self.gpus}")
        log_dir = self.logs / Path(stage).stem
        assignment = {
            name: self.gpus[index % len(self.gpus)]
            for index, name in enumerate(self.alive)
        }

        def one(name: str) -> tuple[str, int]:
            argv = [str(PYTHON), "-m", f"scripts.{Path(stage).stem}", *arguments(name)]
            gpu = assignment[name]
            with self.gpu_slots[gpu]:
                code = self._invoke(stage, argv, log_dir / f"{name}.log", gpu)
            return name, code

        failures = {}
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            future_to_name = {pool.submit(one, n): n for n in self.alive}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    _, code = future.result()
                    if code != 0:
                        raise RuntimeError(f"exit {code}; see {log_dir / (name + '.log')}")
                    verify(name, produced(name))
                except Exception as error:
                    failures[name] = str(error)
        self._survivors(stage, failures)

    def batch(self, stage: str, arguments, produced,
              verify: Callable[[str, Path], None]) -> None:
        """Run a shared-reference stage and verify each selected output."""

        if not self.alive:
            return
        self._say(f"{stage}: batch over {len(self.alive)} shoes, "
                  f"up to {self._batch_jobs(stage)} workers, GPU {self.gpus[0]}")
        existing = [produced(name) for name in self.alive if produced(name).exists()]
        if existing:
            raise FileExistsError(f"{stage}: existing output requires an explicit recovery plan: {existing[0]}")
        log_dir = self.logs / Path(stage).stem
        argv = [str(PYTHON), "-m", f"scripts.{Path(stage).stem}", *arguments(self.alive)]
        code = self._invoke(stage, argv, log_dir / "batch.log", self.gpus[0])
        failures = {}
        for name in self.alive:
            try:
                if code != 0 and stage not in {
                    "run_instance_volume_batch.py", "run_instance_volume_mapping.py",
                    "run_anatomical_fibers.py",
                }:
                    raise RuntimeError(f"batch exited {code}")
                verify(name, produced(name))
            except Exception as error:
                failures[name] = f"{error}; batch exit {code}; see {log_dir / 'batch.log'}"
        self._survivors(stage, failures)

    # -- prerequisites ------------------------------------------------------

    def link_preparations(self) -> None:
        """Link every shoe's preparation under one root the stages can take."""

        target = self.root / "shoe_preparation"
        target.mkdir(exist_ok=True)
        linked, absent = 0, []
        for name in self.alive:
            destination = target / name
            if destination.exists() or destination.is_symlink():
                if not (destination / "shoe_preparation.json").is_file():
                    absent.append(name)
                    continue
                linked += 1
                continue
            source = next(
                (
                    candidate / name
                    for candidate in PREPARATION_ROOTS
                    if (candidate / name / "shoe_preparation.json").is_file()
                ),
                None,
            )
            if source is None:
                absent.append(name)
                continue
            destination.symlink_to(source, target_is_directory=True)
            linked += 1
        self._say(f"shoe_preparation: {linked} linked into the run root")
        if absent:
            for name in absent:
                self.dropped[name] = "shoe_preparation_missing"
            self.alive = [n for n in self.alive if n not in set(absent)]
            self._say(f"  no preparation found for: {', '.join(absent)}")

    def copy_canonical_volume(self) -> Path:
        """Copy the validated canonical volume; never recompute it."""

        volume_root = self.root / "anatomical_volume"
        if not (volume_root / "reference").is_dir():
            shutil.copytree(
                STABLE_ROOT / "anatomical_volume/reference", volume_root / "reference"
            )
            self._say("anatomical_volume/reference: copied from the validated canonical set")
        else:
            for filename in ("canonical_volume.json", "canonical_volume.npz"):
                source = STABLE_ROOT / "anatomical_volume/reference" / filename
                target = volume_root / "reference" / filename
                if not target.is_file() or target.read_bytes() != source.read_bytes():
                    raise ValueError(f"canonical reference differs from validated source: {target}")
            self._say("anatomical_volume/reference: already present, left untouched")
        return volume_root

    def _preflight_11_b(self, volume_root: Path, surface_root: Path) -> None:
        """Isolate a bad 11-A target before the all-shoe B3 preflight."""

        if not self.alive:
            return
        from foot_prior.anatomical_volume import (
            _load_extended_reference,
            load_canonical_anatomical_volume,
            load_instance_volume_problem,
        )

        canonical = load_canonical_anatomical_volume(volume_root)
        reference = _load_extended_reference(surface_root)
        failures = {}
        for name in self.alive:
            try:
                load_instance_volume_problem(
                    volume_root, surface_root, name,
                    canonical_volume=canonical, extended_reference=reference,
                )
            except Exception as error:
                failures[name] = f"11-B1 preflight: {type(error).__name__}: {error}"
        self._survivors("checkpoint_11_b1_preflight", failures)

    def _verify_b3(self, name: str, path: Path) -> None:
        summary = json.loads((path.parents[1] / "batch_summary.json").read_text())
        result = summary.get("results", {}).get(name, {})
        if (result.get("exit_code") != 0 or result.get("post_validation") != "passed"
                or result.get("status") not in {"final_exact_target", "final_corrected_target"}):
            raise ValueError(f"B3 was not independently post-validated: {result}")
        self._record(path, stage="instance_volume_optimization", name=name,
                     statuses=("final_exact_target", "final_corrected_target"))
        for artifact in ("instance_volume.npz", "instance_volume.vtk"):
            if not (path.parent / artifact).is_file():
                raise FileNotFoundError(path.parent / artifact)

    def _verify_mapping(self, name: str, path: Path) -> None:
        summary = json.loads((path.parents[1] / "mapping_summary.json").read_text())
        if summary.get("results", {}).get(name, {}).get("status") != "mapping_valid":
            raise ValueError(f"11-C mapping failed: {name}")
        record = self._record(path, stage="instance_volume_mapping_validation",
                              name=name, statuses=("mapping_valid",))
        if record.get("schema_version") != 2:
            raise ValueError(f"{name}: expected corrected 11-C schema version 2")

    def _verify_fibers(self, name: str, path: Path) -> None:
        summary = json.loads((path.parents[1] / "fiber_summary.json").read_text())
        if summary.get("results", {}).get(name, {}).get("status") != "coverage_review_required":
            raise ValueError(f"11-D audit failed: {name}")
        self._record(path, stage="anatomical_fiber_coverage", name=name,
                     statuses=("coverage_review_required",))

    # -- the sequence -------------------------------------------------------

    def run(self) -> int:
        root = self.root
        self._say(f"run root: {root}")
        self._say(f"{len(self.shoes)} shoes with a containment fit")
        for required in (
            PYTHON, FOOT_MODEL, BODY_MODEL,
            STABLE_ROOT / "anatomical_volume/reference/canonical_volume.json",
            STABLE_ROOT / "anatomical_volume/reference/canonical_volume.npz",
            STABLE_ROOT / "anatomical_fibers/converged/reference/semantic_field.json",
            STABLE_ROOT / "anatomical_fibers/converged/reference/semantic_field.npz",
            *(PROJECT_ROOT / "scripts" / stage for stage in STAGE_SCRIPTS),
        ):
            if not required.is_file():
                raise FileNotFoundError(f"pipeline prerequisite missing: {required}")
        self._check_stage_entrypoints()
        self._say("all stage entry points and worker limits passed preflight")
        self._save("running")
        self.link_preparations()
        volume_root = self.copy_canonical_volume()

        def verify_surface(name: str, path: Path) -> None:
            self._record(path, stage="canonical_dense_supr_anatomical_surface")
            if not (path.parent / "foot_dense.ply").is_file():
                raise FileNotFoundError(path.parent / "foot_dense.ply")

        def verify_extended(name: str, path: Path) -> None:
            record = self._record(path, stage="extended_canonical_supr_anatomical_surface",
                                  name=name)
            leg_status = self._record(
                root / "lower_leg_attachment" / name / "lower_leg_attachment.json",
                stage="fitted_foot_natural_lower_leg_collar_fit", name=name,
            ).get("status")
            if record.get("lower_leg_fit", {}).get("status") != leg_status:
                raise ValueError(f"{name}: extended anatomy did not preserve lower-leg fit status")
            if not (path.parent / "foot_lower_leg.ply").is_file():
                raise FileNotFoundError(path.parent / "foot_lower_leg.ply")

        def verify_target(name: str, path: Path) -> None:
            self._record(path, stage="fitted_anatomical_boundary_target", name=name,
                         statuses=("ready", "ready_requires_untangling"))
            if not (path.parent / "boundary_target.npz").is_file():
                raise FileNotFoundError(path.parent / "boundary_target.npz")

        self.batch(
            "run_anatomical_surface.py",
            lambda shoes: [
                "--containment-root", str(root / "containment_fit"),
                "--supr-model", str(FOOT_MODEL),
                "--output-root", str(root / "anatomical_surface"),
                *shoes,
            ],
            lambda n: root / "anatomical_surface" / n / "anatomical_surface.json",
            verify_surface,
        )
        self.per_shoe(
            "run_lower_leg_attachment.py",
            lambda n: [
                "--anatomical-surface-root", str(root / "anatomical_surface"),
                "--preparation-root", str(root / "shoe_preparation"),
                "--support-fit-root", str(root / "support_fit"),
                "--full-body-supr-model", str(BODY_MODEL),
                "--warm-start-root", str(STABLE_ROOT / "lower_leg_attachment"),
                "--output-root", str(root / "lower_leg_attachment"),
                n,
            ],
            lambda n: root / "lower_leg_attachment" / n / "lower_leg_attachment.json",
            self._verify_lower_leg,
        )
        self.batch(
            "run_extended_anatomical_surface.py",
            lambda shoes: [
                "--anatomical-surface-root", str(root / "anatomical_surface"),
                "--lower-leg-root", str(root / "lower_leg_attachment"),
                "--full-body-supr-model", str(BODY_MODEL),
                "--output-root", str(root / "extended_anatomical_surface"),
                *shoes,
            ],
            lambda n: root / "extended_anatomical_surface" / n / "extended_anatomical_surface.json",
            verify_extended,
        )
        self.per_shoe(
            "run_instance_anatomical_volume.py",
            lambda n: [
                "--anatomical-volume-root", str(volume_root),
                "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                n,
            ],
            lambda n: volume_root / n / "boundary_target.json",
            verify_target,
        )
        self._preflight_11_b(volume_root, root / "extended_anatomical_surface")

        b3_root = root / "instance_anatomical_volume/batch"
        b3_root.parent.mkdir(parents=True, exist_ok=True)
        self.batch(
            "run_instance_volume_batch.py",
            lambda shoes: [
                "--anatomical-volume-root", str(volume_root),
                "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                "--output-root", str(b3_root),
                "--jobs", str(self._batch_jobs("run_instance_volume_batch.py")),
                *shoes,
            ],
            lambda n: b3_root / n / "instance_volume.json",
            self._verify_b3,
        )

        mapping_root = root / "instance_volume_mapping/batch"
        mapping_root.parent.mkdir(parents=True, exist_ok=True)
        self.batch(
            "run_instance_volume_mapping.py",
            lambda shoes: [
                "--anatomical-volume-root", str(volume_root),
                "--instance-volume-batch-root", str(b3_root),
                "--containment-fit-root", str(root / "containment_fit"),
                "--output-root", str(mapping_root),
                "--jobs", str(self._batch_jobs("run_instance_volume_mapping.py")),
                *shoes,
            ],
            lambda n: mapping_root / n / "mapping_validation.json",
            self._verify_mapping,
        )

        audit_root = root / "anatomical_fibers/audit"
        audit_root.parent.mkdir(parents=True, exist_ok=True)
        self.batch(
            "run_anatomical_fibers.py",
            lambda shoes: [
                "fibers",
                "--anatomical-volume-root", str(volume_root),
                "--scalar-field-root", str(STABLE_ROOT / "anatomical_fibers/converged/reference"),
                "--extended-anatomical-surface-root", str(root / "extended_anatomical_surface"),
                "--instance-volume-batch-root", str(b3_root),
                "--containment-fit-root", str(root / "containment_fit"),
                "--output-root", str(audit_root),
                "--jobs", str(self._batch_jobs("run_anatomical_fibers.py")),
                "--shoes", *shoes,
            ],
            lambda n: audit_root / n / "fiber_coverage.json",
            self._verify_fibers,
        )

        finished = len(self.alive) == len(self.shoes) - len(self.excluded)
        self._save("coverage_review_required" if finished else "partial_failure")
        self._say(
            f"DONE: {len(self.alive)}/{len(self.shoes)} shoes reached the 11-D audit"
        )
        if self.dropped:
            self._say("dropped along the way:")
            for name, stage in sorted(self.dropped.items()):
                self._say(f"  {name}: {stage}")
        self._say(f"status: {self.status_path}")
        return 0 if finished else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gpus", default="3,6,7")
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve(strict=True)
    if not 1 <= args.jobs <= 24:
        parser.error("--jobs must be in [1, 24]")
    pipeline = Pipeline(args)
    try:
        code = pipeline.run()
    except Exception as error:
        pipeline._say(f"STOPPED: {type(error).__name__}: {error}")
        pipeline._save("failed")
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
