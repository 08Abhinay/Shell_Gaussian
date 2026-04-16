#!/usr/bin/env python3
"""Batch-export multiple shoe folders into NeuS2 static-scene format."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Directory containing per-shoe folders.")
    parser.add_argument("--output-root", required=True, help="Directory where exported NeuS2 scenes will be written.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Folder names to skip. Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported scene folders.",
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=6,
        help="Hold out every N-th sorted view for test.",
    )
    parser.add_argument(
        "--test-offset",
        type=int,
        default=0,
        help="Offset for the every-N test split.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Scene margin inside the NeuS2 unit cube.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    exporter = Path(__file__).resolve().parent / "export_shoe_to_neus2.py"

    exclude = set(args.exclude)
    shoe_dirs = sorted(
        path for path in input_root.iterdir() if path.is_dir() and path.name not in exclude
    )

    if not shoe_dirs:
        raise SystemExit(f"No shoe directories found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for idx, shoe_dir in enumerate(shoe_dirs, start=1):
        out_dir = output_root / shoe_dir.name
        cmd = [
            sys.executable,
            str(exporter),
            "--shoe-dir",
            str(shoe_dir),
            "--output-dir",
            str(out_dir),
            "--test-stride",
            str(args.test_stride),
            "--test-offset",
            str(args.test_offset),
            "--margin",
            str(args.margin),
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"[{idx}/{len(shoe_dirs)}] Exporting {shoe_dir.name}")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            failures.append(shoe_dir.name)
            print(result.stdout.strip())
            print(result.stderr.strip())

    print()
    print(f"Completed: {len(shoe_dirs) - len(failures)}/{len(shoe_dirs)} succeeded")
    if failures:
        print("Failures:")
        for name in failures:
            print(f"- {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
