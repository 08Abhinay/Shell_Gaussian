#!/usr/bin/env python3
"""Run independent instance-volume jobs concurrently with bounded CPU threads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


NUMERICAL_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SUCCESSFUL_B3_STATUSES = {"final_exact_target", "final_corrected_target"}
SOURCE_PATHS = (
    "foot_prior/anatomical_volume.py",
    "foot_prior/cavity.py",
    "foot_prior/instance_volume_optimization.py",
    "scripts/run_instance_volume_batch.py",
    "scripts/run_instance_volume_deformation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and run independent Checkpoint 11-B3 shoe volumes in "
            "parallel, with one numerical-library thread per child process."
        )
    )
    parser.add_argument("--anatomical-volume-root", required=True, type=Path)
    parser.add_argument(
        "--extended-anatomical-surface-root", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SHOE",
    )
    parser.add_argument("shoes", nargs="*")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_metadata(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _read_child_status(directory: Path) -> str:
    path = directory / "instance_volume.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload["status"]
    except (KeyError, OSError, TypeError, ValueError):
        return "missing_or_invalid_final_record"
    return str(status)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name, value in NUMERICAL_THREAD_ENVIRONMENT.items():
        os.environ[name] = value

    # Import numerical modules only after fixing the parent and child thread policy.
    from foot_prior.anatomical_volume import (
        _load_extended_reference,
        load_canonical_anatomical_volume,
        load_instance_volume_problem,
    )
    from foot_prior.instance_volume_optimization import (
        build_instance_optimization_system,
        optimization_configuration,
    )
    from scripts.run_instance_volume_deformation import (
        _preflight_resume_state,
        _validated_names,
        instance_continuation_configuration,
    )

    repository = Path(__file__).resolve().parents[1]
    volume_root = args.anatomical_volume_root.expanduser().resolve(strict=True)
    surface_root = (
        args.extended_anatomical_surface_root.expanduser().resolve(strict=True)
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.parent.resolve(strict=True)
    if not volume_root.is_dir() or not surface_root.is_dir():
        raise NotADirectoryError(
            "anatomical volume and surface roots must be directories"
        )
    if output_root.exists():
        raise FileExistsError(f"batch output root already exists: {output_root}")

    names = _validated_names(volume_root, list(args.shoes), list(args.exclude))
    if "sneaker_vibe" in names:
        raise ValueError("sneaker_vibe is excluded from the accepted B3 scope")

    canonical_volume = load_canonical_anatomical_volume(volume_root)
    extended_reference = _load_extended_reference(surface_root)
    problems = {}
    for name in names:
        problems[name] = load_instance_volume_problem(
            volume_root,
            surface_root,
            name,
            canonical_volume=canonical_volume,
            extended_reference=extended_reference,
        )
        print(f"[preflight] PASS {name}", flush=True)

    output_root.mkdir()
    logs_root = output_root / "logs"
    logs_root.mkdir()
    manifest = {
        "stage": "instance_volume_optimization_batch",
        "shoe_count": len(names),
        "shoes": names,
        "excluded": sorted(args.exclude),
        "jobs": min(args.jobs, len(names)),
        "numerical_thread_environment": NUMERICAL_THREAD_ENVIRONMENT,
        "continuation_configuration": instance_continuation_configuration(),
        "optimization_configuration": optimization_configuration(),
        "git": _git_metadata(repository),
        "source_sha256": {
            relative: _file_digest(repository / relative)
            for relative in SOURCE_PATHS
        },
    }
    _write_json_atomic(output_root / "batch_manifest.json", manifest)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    pending = list(names)
    running: dict[str, tuple[subprocess.Popen[bytes], float]] = {}
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    def save_summary(*, audit_complete: bool = False) -> dict[str, Any]:
        successful = sum(
            result.get("post_validation") == "passed"
            if audit_complete
            else result.get("status") in SUCCESSFUL_B3_STATUSES
            and result.get("exit_code") == 0
            for result in results.values()
        )
        summary = {
            "stage": "instance_volume_optimization_batch",
            "completed": len(results),
            "total": len(names),
            "successful": successful,
            "failed": len(results) - successful,
            "audit_complete": audit_complete,
            "batch_wall_seconds": time.perf_counter() - started,
            "results": dict(sorted(results.items())),
        }
        _write_json_atomic(output_root / "batch_summary.json", summary)
        return summary

    try:
        while pending or running:
            while pending and len(running) < args.jobs:
                name = pending.pop(0)
                command = (
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.run_instance_volume_deformation",
                    "--anatomical-volume-root",
                    str(volume_root),
                    "--extended-anatomical-surface-root",
                    str(surface_root),
                    "--output-root",
                    str(output_root),
                    "--stop-after",
                    "11-b3",
                    "--exclude",
                    "sneaker_vibe",
                    name,
                )
                log_path = logs_root / f"{name}.log"
                with log_path.open("xb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=repository,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                running[name] = (process, time.perf_counter())
                print(f"[started] {name} pid={process.pid}", flush=True)

            finished = [
                name for name, (process, _) in running.items()
                if process.poll() is not None
            ]
            if not finished:
                time.sleep(0.2)
                continue
            for name in sorted(finished):
                process, shoe_started = running.pop(name)
                status = _read_child_status(output_root / name)
                results[name] = {
                    "exit_code": int(process.returncode),
                    "status": status,
                    "wall_seconds": time.perf_counter() - shoe_started,
                    "post_validation": "pending",
                }
                save_summary()
                print(
                    f"[done {len(results)}/{len(names)}] {name}: {status} "
                    f"EXIT={process.returncode}",
                    flush=True,
                )
    except BaseException:
        for process, _ in running.values():
            process.terminate()
        for process, _ in running.values():
            process.wait()
        raise

    optimization_system = build_instance_optimization_system(canonical_volume)
    for name in names:
        result = results[name]
        if (
            result["exit_code"] != 0
            or result["status"] not in SUCCESSFUL_B3_STATUSES
        ):
            result["post_validation"] = "not_applicable"
            continue
        try:
            resumed = _preflight_resume_state(
                output_root / name,
                problems[name],
                optimization_system,
            )
            if resumed.final_status != result["status"]:
                raise ValueError("reloaded final status differs from child result")
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            result["post_validation"] = "failed"
            result["post_validation_error"] = str(error)
        else:
            result["post_validation"] = "passed"
            print(f"[post-validation] PASS {name}", flush=True)

    summary = save_summary(audit_complete=True)
    print(
        f"[batch finished] passed={summary['successful']} "
        f"failed={summary['failed']}",
        flush=True,
    )
    return summary


def main() -> None:
    try:
        result = run(parse_args())
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"instance volume batch failed: {error}") from error
    if result["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
