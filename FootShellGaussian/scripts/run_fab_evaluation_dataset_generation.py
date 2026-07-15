#!/usr/bin/env python3
"""Render FAB shoe assets into the standardized FootShellGaussian evaluation layout.

Typical usage:

    /storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python \
      FootShellGaussian/scripts/run_fab_evaluation_dataset_generation.py \
      --source-root /path/to/fab/source \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
FOOTSHELL_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = FOOTSHELL_ROOT.parent
DEFAULT_RENDERER_SCRIPT = SCRIPTS_DIR / "render_obj_top_bottom_evaluation.py"
DEFAULT_PREPARE_SCRIPT = SCRIPTS_DIR / "prepare_external_shoe_assets.py"
DEFAULT_MANIFEST = FOOTSHELL_ROOT / "configs" / "external_source_normalized_all_shoes_render_manifest.json"
DEFAULT_OUTPUT_ROOT = Path("/storage/Abhinay/home_ab5298/dataset/datasets/processed/fab_evaluation")
DEFAULT_PREPARED_DATASET_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/external_source_preprocessed"
)
DEFAULT_PREPARED_MANIFEST_NAME = "external_source_preprocessed_render_manifest.json"
DEFAULT_BLENDER_PATH = PROJECT_ROOT / "baselines/GShell/GShell_env/opt/blender-4.2.21-linux-x64/blender"
EXPECTED_COUNTS = {
    "all_images": 180,
    "all_masks": 180,
    "all_invdepth": 180,
    "train_images": 150,
    "train_masks": 150,
    "train_invdepth": 150,
    "val_images": 30,
    "val_masks": 30,
    "val_invdepth": 30,
    "transforms_train_frames": 150,
    "transforms_test_frames": 30,
}


def default_blender() -> str:
    env_value = os.environ.get("BLENDER")
    if env_value:
        return env_value
    if DEFAULT_BLENDER_PATH.is_file():
        return str(DEFAULT_BLENDER_PATH)
    return "blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Root holding FAB shoe assets referenced by the manifest.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--renderer-script", type=Path, default=DEFAULT_RENDERER_SCRIPT)
    parser.add_argument("--prepare-script", type=Path, default=DEFAULT_PREPARE_SCRIPT)
    parser.add_argument("--prepared-dataset-root", type=Path, default=DEFAULT_PREPARED_DATASET_ROOT)
    parser.add_argument(
        "--prepare-copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    parser.add_argument(
        "--prepare-source",
        action="store_true",
        help="Discover models inside the raw external source tree, preprocess them, then render.",
    )
    parser.add_argument("--blender", default=default_blender())
    parser.add_argument(
        "--selection-debug-dir",
        type=Path,
        default=None,
        help="Optional directory for keeping renderer auto-selection probe outputs.",
    )
    parser.add_argument("--shoe", action="append", default=None, help="Render only this shoe name. May be repeated.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def resolve_executable(candidate: str) -> str:
    candidate_path = Path(candidate)
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return str(candidate_path)
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"Blender executable not found or not executable: {candidate}")


def build_renderer_cmd(
    args: argparse.Namespace,
    blender_executable: str,
    manifest_path: Path,
    source_root: Path,
) -> list[str]:
    cmd = [
        blender_executable,
        "--background",
        "--python",
        str(args.renderer_script),
        "--",
        "--manifest",
        str(manifest_path),
        "--source-root",
        str(source_root),
        "--output-root",
        str(args.output_root),
        "--mode",
        "multi_elevation_360",
        "--render-invdepth",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.selection_debug_dir is not None:
        cmd.extend(["--selection-debug-dir", str(args.selection_debug_dir)])
    for shoe_name in args.shoe or []:
        cmd.extend(["--shoe", shoe_name])
    return cmd


def prepare_output_manifest_path(prepared_dataset_root: Path) -> Path:
    return prepared_dataset_root / "manifests" / DEFAULT_PREPARED_MANIFEST_NAME


def build_prepare_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(args.prepare_script),
        "--source-root",
        str(args.source_root),
        "--input-manifest",
        str(args.manifest),
        "--output-root",
        str(args.prepared_dataset_root),
        "--output-manifest",
        str(prepare_output_manifest_path(args.prepared_dataset_root)),
        "--copy-mode",
        args.prepare_copy_mode,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    for shoe_name in args.shoe or []:
        cmd.extend(["--shoe", shoe_name])
    return cmd


def resolve_render_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.prepare_source:
        return args.source_root, args.manifest

    if not args.prepare_script.is_file():
        raise FileNotFoundError(f"Prepare script not found: {args.prepare_script}")

    prepare_cmd = build_prepare_cmd(args)
    print("Running source preparation:")
    print(" ".join(prepare_cmd))
    subprocess.run(prepare_cmd, check=True)

    prepared_source_root = args.prepared_dataset_root / "source"
    prepared_manifest = prepare_output_manifest_path(args.prepared_dataset_root)
    return prepared_source_root, prepared_manifest


def count_glob(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def transforms_frame_count(path: Path) -> int:
    payload = load_json(path)
    return len(payload.get("frames", []))


def validate_rendered_output(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing renderer summary: {summary_path}")

    payload = load_json(summary_path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Renderer summary rows must be a list: {summary_path}")

    errors: list[str] = []
    rendered_shoes: dict[str, Any] = {}
    for row in rows:
        shoe_name = row.get("shoe")
        if row.get("status") != "ok":
            errors.append(f"{shoe_name}: render status {row.get('status')} ({row.get('error')})")
            continue

        shoe_dir = output_root / str(shoe_name)
        multi_root = shoe_dir / "multi_elevation_360"
        all_dir = multi_root / "all"
        train_dir = multi_root / "train"
        val_dir = multi_root / "val"
        counts = {
            "all_images": count_glob(all_dir / "image", "*.jpg"),
            "all_masks": count_glob(all_dir / "mask", "*.png"),
            "all_invdepth": count_glob(all_dir / "invdepth", "*.npy"),
            "train_images": count_glob(train_dir / "image", "*.jpg"),
            "train_masks": count_glob(train_dir / "mask", "*.png"),
            "train_invdepth": count_glob(train_dir / "invdepth", "*.npy"),
            "val_images": count_glob(val_dir / "image", "*.jpg"),
            "val_masks": count_glob(val_dir / "mask", "*.png"),
            "val_invdepth": count_glob(val_dir / "invdepth", "*.npy"),
            "transforms_train_frames": transforms_frame_count(shoe_dir / "transforms_train.json"),
            "transforms_test_frames": transforms_frame_count(shoe_dir / "transforms_test.json"),
        }
        rendered_shoes[str(shoe_name)] = counts
        for key, expected in EXPECTED_COUNTS.items():
            if counts[key] != expected:
                errors.append(f"{shoe_name} {key}: expected {expected}, got {counts[key]}")

    return {
        "status": "failed" if errors else "ok",
        "expected": EXPECTED_COUNTS,
        "rendered_shoes": rendered_shoes,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"FAB source root not found: {args.source_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"FAB manifest not found: {args.manifest}")
    if not args.renderer_script.is_file():
        raise FileNotFoundError(f"Renderer script not found: {args.renderer_script}")

    blender_executable = resolve_executable(args.blender)
    render_source_root, render_manifest = resolve_render_inputs(args)
    if not render_source_root.is_dir():
        raise FileNotFoundError(f"Prepared source root not found: {render_source_root}")
    if not render_manifest.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {render_manifest}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    renderer_cmd = build_renderer_cmd(
        args,
        blender_executable=blender_executable,
        manifest_path=render_manifest,
        source_root=render_source_root,
    )
    print("Running renderer:")
    print(" ".join(renderer_cmd))
    subprocess.run(renderer_cmd, check=True)

    validation = validate_rendered_output(args.output_root)
    summary_payload = {
        "manifest": str(render_manifest),
        "source_root": str(render_source_root),
        "requested_manifest": str(args.manifest),
        "requested_source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "renderer_script": str(args.renderer_script),
        "blender": blender_executable,
        "prepare_source": args.prepare_source,
        "prepare_script": str(args.prepare_script) if args.prepare_source else None,
        "prepared_dataset_root": str(args.prepared_dataset_root) if args.prepare_source else None,
        "prepare_copy_mode": args.prepare_copy_mode if args.prepare_source else None,
        "selected_shoes": list(args.shoe or []),
        **validation,
    }
    write_json(args.output_root / "generation_summary.json", summary_payload)
    if validation["errors"]:
        raise SystemExit("Validation failed: " + "; ".join(validation["errors"]))

    print(
        f"Validation ok: {len(validation['rendered_shoes'])} shoe(s) written to {args.output_root}"
    )


if __name__ == "__main__":
    main()
