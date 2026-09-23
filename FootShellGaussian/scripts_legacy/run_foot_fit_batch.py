"""Fit many prepared shoes in parallel, one process per shoe.

Each shoe is independent -- it reads its own preparation and writes its own
directory -- so the only reason the sequential path exists is that a single
fit is already a few minutes of single-threaded NumPy. Running them together
turns half an hour into a few minutes.

A shoe that cannot be seated is recorded as a failure and the batch continues.
That is deliberate: ``failed_no_seating`` is information, not an accident, and
finding out which shoes it applies to is the point of running the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


THREAD_LIMITS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_foot_fit.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit every prepared shoe with the single-stage foot fitter."
    )
    parser.add_argument("--preparation-root", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shoes", nargs="*", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--compression-allowance-mm", type=float, default=None)
    parser.add_argument(
        "--gpus",
        nargs="*",
        default=[],
        help=(
            "GPU indices to spread shoes across, round-robin. Each fit uses "
            "well under a gigabyte, so this is about not crowding one device "
            "rather than about capacity."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Concurrent shoes; 0 starts every selected shoe at once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _selected_shoes(
    preparation_root: Path, requested: list[str], excluded: list[str]
) -> list[str]:
    available = sorted(
        directory.name
        for directory in preparation_root.iterdir()
        if (directory / "shoe_preparation.json").is_file()
    )
    if not available:
        raise ValueError(f"no prepared shoes found under {preparation_root}")
    selected = requested if requested else available
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"unknown shoes: {unknown}")
    chosen = [name for name in selected if name not in set(excluded)]
    if not chosen:
        raise ValueError("every selected shoe was excluded")
    return sorted(chosen)


def _json_atomic(path: Path, payload: dict[str, Any]) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _command(args: argparse.Namespace, name: str) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--preparation-dir",
        str(args.preparation_root / name),
        "--supr-model",
        str(args.supr_model),
        "--output-dir",
        str(args.output_root / name),
    ]
    if args.compression_allowance_mm is not None:
        command += [
            "--compression-allowance-mm",
            str(args.compression_allowance_mm),
        ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def _summarize(output_dir: Path) -> dict[str, Any]:
    record = json.loads(
        (output_dir / "foot_fit.json").read_text(encoding="utf-8")
    )
    acceptance = record["seating"]["acceptance"]
    return {
        "seating_status": record["seating_status"],
        "legacy_status": record["status"],
        "worst_region_name": acceptance["worst_region_name"],
        "worst_region_rms_gap": acceptance["worst_region_rms_gap"],
        "worst_region_rms_gap_mm": acceptance["worst_region_rms_gap_mm"],
        "overall_median_gap": acceptance["overall_median_gap"],
        "minimum_region_coverage": acceptance["minimum_region_coverage"],
        "compression_depth_mm": acceptance["compression_depth_mm"],
    }


def run(args: argparse.Namespace) -> int:
    preparation_root = args.preparation_root.expanduser().resolve(strict=True)
    supr_model = args.supr_model.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    names = _selected_shoes(preparation_root, args.shoes, args.exclude)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)

    base_environment = {**os.environ, **THREAD_LIMITS, "PYTHONPATH": ".:scripts"}
    gpus = [str(index) for index in args.gpus]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "foot_fit_batch",
        "shoes": names,
        "completed": 0,
        "failed": 0,
        "results": {},
    }
    summary_path = output_root / "foot_fit_summary.json"
    _json_atomic(summary_path, summary)

    limit = args.max_parallel if args.max_parallel > 0 else len(names)
    queue = list(names)
    running: dict[str, tuple[subprocess.Popen[bytes], Any, float]] = {}
    started = time.monotonic()
    while queue or running:
        while queue and len(running) < limit:
            name = queue.pop(0)
            log = (output_root / "logs" / f"{name}.log").open("wb")
            environment = dict(base_environment)
            device = ""
            if gpus:
                device = gpus[names.index(name) % len(gpus)]
                environment["CUDA_VISIBLE_DEVICES"] = device
            process = subprocess.Popen(
                _command(args, name),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[name] = (process, log, time.monotonic())
            where = f" gpu={device}" if device else ""
            print(f"[started] {name} pid={process.pid}{where}", flush=True)
        time.sleep(1.0)
        for name in [
            name
            for name, (process, _, _) in running.items()
            if process.poll() is not None
        ]:
            process, log, begin = running.pop(name)
            log.close()
            elapsed = time.monotonic() - begin
            if process.returncode == 0:
                summary["results"][name] = {
                    "exit_code": 0,
                    "wall_seconds": elapsed,
                    **_summarize(output_root / name),
                }
                summary["completed"] += 1
                status = summary["results"][name]["seating_status"]
            else:
                tail = (
                    (output_root / "logs" / f"{name}.log")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                    .splitlines()
                )
                summary["results"][name] = {
                    "exit_code": process.returncode,
                    "wall_seconds": elapsed,
                    "seating_status": "run_failed",
                    "error": tail[-1] if tail else "",
                }
                summary["failed"] += 1
                status = "run_failed"
            _json_atomic(summary_path, summary)
            print(
                f"[done {summary['completed'] + summary['failed']}/{len(names)}]"
                f" {name}: {status} WALL={elapsed:.1f}s",
                flush=True,
            )
    print(
        f"[batch finished] completed={summary['completed']} "
        f"failed={summary['failed']} WALL={time.monotonic() - started:.1f}s",
        flush=True,
    )
    print(f"summary: {summary_path}")
    return int(summary["failed"] > 0)


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"foot-fit batch failed: {error}") from error


if __name__ == "__main__":
    main()
