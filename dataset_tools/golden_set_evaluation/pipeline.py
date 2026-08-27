#!/usr/bin/env python3
"""Single public entry point for the synthetic golden-set evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GLB_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb"
)
DEFAULT_RAW_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/raw/golden_set_evaluation"
)
DEFAULT_COLMAP_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_colmap"
)
DEFAULT_GSHELL_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_gshell"
)
DEFAULT_SUGAR_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_sugar"
)
DEFAULT_BLENDER = Path(
    "/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/blender"
)
DEFAULT_COLMAP_PYTHON = Path("/storage/Abhinay/conda_envs/colmap/bin/python")
DEFAULT_IMAGE_PYTHON = Path(
    "/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/python"
)
EXPECTED_VIEW_COUNT = 180
RENDER_OVERRIDES: dict[str, dict[str, Any]] = {
    "duinn_shoes_womens_hiking_sandal_sport": {
        "source_axes": {"length": "-Y", "width": "X", "up": "Z"},
    },
    "nike_air_jordan": {
        "source_axes": {"length": "-X", "width": "Y", "up": "Z"},
        "selection": {
            "mode": "axis-side",
            "axis": "Y",
            "side": "min",
            "separate_loose_parts": True,
        },
    },
    "sneaker_vibe": {
        "source_axes": {"length": "X", "width": "Y", "up": "Z"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render 180 RGB images and masks per GLB.")
    render.add_argument("--source-root", type=Path, default=DEFAULT_GLB_ROOT)
    render.add_argument("--output-root", type=Path, default=DEFAULT_RAW_ROOT)
    render.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    render.add_argument("--shoe", action="append", default=None)
    render.add_argument("--selection-debug-dir", type=Path, default=None)
    render.add_argument("--overwrite", action="store_true")

    validate_render = subparsers.add_parser(
        "validate-render", help="Validate the raw RGB/mask layout."
    )
    validate_render.add_argument("--dataset-root", type=Path, default=DEFAULT_RAW_ROOT)
    validate_render.add_argument("--shoe", action="append", default=None)

    colmap = subparsers.add_parser("colmap", help="Run fresh COLMAP reconstruction.")
    colmap.add_argument("--source-root", type=Path, default=DEFAULT_RAW_ROOT)
    colmap.add_argument("--output-root", type=Path, default=DEFAULT_COLMAP_ROOT)
    colmap.add_argument("--colmap-python", type=Path, default=DEFAULT_COLMAP_PYTHON)
    colmap.add_argument("--image-python", type=Path, default=DEFAULT_IMAGE_PYTHON)
    selection = colmap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", action="append")
    selection.add_argument("--all", action="store_true")
    colmap.add_argument("--gpu-index", type=int, default=0)
    colmap.add_argument(
        "--reconstruction-profile",
        choices=("default", "synthetic-low-texture"),
        default="default",
    )
    colmap.add_argument("--min-registered-images", type=int, default=1)
    colmap.add_argument("--keep-workspace", action="store_true")
    colmap.add_argument("--overwrite", action="store_true")

    validate_colmap = subparsers.add_parser(
        "validate-colmap", help="Validate processed COLMAP scenes."
    )
    validate_colmap.add_argument("--dataset-root", type=Path, default=DEFAULT_COLMAP_ROOT)
    selection = validate_colmap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", action="append")
    selection.add_argument("--all", action="store_true")
    validate_colmap.add_argument("--require-all-registered", action="store_true")

    sugar = subparsers.add_parser(
        "prepare-sugar", help="Create the optional RGBA/bounded SuGaR adapter dataset."
    )
    sugar.add_argument("--input-dir", type=Path, default=DEFAULT_COLMAP_ROOT)
    sugar.add_argument("--output-dir", type=Path, default=DEFAULT_SUGAR_ROOT)
    sugar.add_argument("--image-python", type=Path, default=DEFAULT_IMAGE_PYTHON)
    sugar.add_argument(
        "--colmap-bin",
        type=Path,
        default=Path("/storage/Abhinay/conda_envs/colmap/bin/colmap"),
    )
    selection = sugar.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", action="append")
    selection.add_argument("--all", action="store_true")
    sugar.add_argument("--overwrite", action="store_true")

    gshell = subparsers.add_parser(
        "prepare-gshell", help="Convert COLMAP scenes to canonical GShell input."
    )
    gshell.add_argument("--input-dir", type=Path, default=DEFAULT_COLMAP_ROOT)
    gshell.add_argument("--output-dir", type=Path, default=DEFAULT_GSHELL_ROOT)
    gshell.add_argument("--shoe", action="append", default=None)
    gshell.add_argument("--overwrite", action="store_true")
    gshell.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def shoe_name(path: Path) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", path.stem.strip().lower()).strip("_")
    if not name:
        raise ValueError(f"Cannot derive a shoe name from {path.name!r}")
    return name


def discover_glbs(source_root: Path) -> list[dict[str, Any]]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    records: list[dict[str, Any]] = []
    names: dict[str, Path] = {}
    for model in sorted(path for path in source_root.rglob("*.glb") if path.is_file()):
        name = shoe_name(model)
        if name in names:
            raise ValueError(
                f"GLB name collision after normalization: {names[name].name!r} and {model.name!r}"
            )
        names[name] = model
        record: dict[str, Any] = {
            "name": name,
            "model": model.relative_to(source_root).as_posix(),
            "source_axes": "auto",
        }
        record.update(RENDER_OVERRIDES.get(name, {}))
        records.append(record)
    if not records:
        raise ValueError(f"No GLB files found in {source_root}")
    return records


def selected_records(records: list[dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return records
    requested = set(selected)
    available = {str(record["name"]) for record in records}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Unknown shoe names: {', '.join(missing)}")
    return [record for record in records if record["name"] in requested]


def validate_raw_scene(scene: Path) -> list[str]:
    errors: list[str] = []
    expected_images = {f"img{index:03d}.jpg" for index in range(1, EXPECTED_VIEW_COUNT + 1)}
    expected_masks = {f"img{index:03d}.png" for index in range(1, EXPECTED_VIEW_COUNT + 1)}
    images = {path.name for path in (scene / "images").glob("*.jpg")}
    masks = {path.name for path in (scene / "masks").glob("*.png")}
    if images != expected_images:
        errors.append(f"{scene.name}: expected {EXPECTED_VIEW_COUNT} numbered JPEGs, found {len(images)}")
    if masks != expected_masks:
        errors.append(f"{scene.name}: expected {EXPECTED_VIEW_COUNT} numbered PNG masks, found {len(masks)}")
    unexpected = sorted(path.name for path in scene.iterdir() if path.name not in {"images", "masks"})
    if unexpected:
        errors.append(f"{scene.name}: unexpected raw artifacts: {unexpected}")
    return errors


def run_render(args: argparse.Namespace) -> None:
    blender = args.blender.resolve()
    if not blender.is_file() or not os.access(blender, os.X_OK):
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    records = selected_records(discover_glbs(args.source_root.resolve()), args.shoe)
    args.output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="golden_eval_manifest_") as temporary:
        manifest = Path(temporary) / "manifest.json"
        manifest.write_text(json.dumps({"shoes": records}, indent=2) + "\n", encoding="utf-8")
        command = [
            str(blender),
            "--background",
            "--python",
            str(SCRIPT_DIR / "blender_renderer.py"),
            "--",
            "--manifest",
            str(manifest),
            "--source-root",
            str(args.source_root.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.selection_debug_dir is not None:
            command.extend(["--selection-debug-dir", str(args.selection_debug_dir.resolve())])
        subprocess.run(command, check=True)

    errors = [error for record in records for error in validate_raw_scene(args.output_root / record["name"])]
    if errors:
        raise RuntimeError("Raw render validation failed:\n" + "\n".join(errors))
    print(f"Validated {len(records)} scene(s) with {EXPECTED_VIEW_COUNT} image/mask pairs each")


def run_validate_render(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    names = args.shoe or sorted(path.name for path in root.iterdir() if path.is_dir())
    errors = [error for name in names for error in validate_raw_scene(root / name)]
    if errors:
        raise RuntimeError("Raw render validation failed:\n" + "\n".join(errors))
    print(f"Validated {len(names)} raw scene(s)")


def run_internal(script_name: str, python: Path, arguments: list[str]) -> None:
    executable = python.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Python executable not found: {executable}")
    subprocess.run([str(executable), str(SCRIPT_DIR / script_name), *arguments], check=True)


def run_colmap(args: argparse.Namespace) -> None:
    arguments = [
        "--source-root", str(args.source_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--gpu-index", str(args.gpu_index),
        "--reconstruction-profile", args.reconstruction_profile,
        "--min-registered-images", str(args.min_registered_images),
    ]
    for scene in args.scene or []:
        arguments.extend(["--scene", scene])
    if args.all:
        arguments.append("--all")
    if args.keep_workspace:
        arguments.append("--keep-workspace")
    if args.overwrite:
        arguments.append("--overwrite")
    run_internal("colmap_pipeline.py", args.colmap_python, arguments)

    mask_arguments = [
        "--dataset-root", str(args.output_root.resolve()),
        "--source-root", str(args.source_root.resolve()),
    ]
    for scene in args.scene or []:
        mask_arguments.extend(["--scene", scene])
    if args.all:
        mask_arguments.append("--all")
    if args.overwrite:
        mask_arguments.append("--overwrite")
    run_internal("align_colmap_masks.py", args.image_python, mask_arguments)


def run_validate_colmap(args: argparse.Namespace) -> None:
    arguments = ["--dataset-root", str(args.dataset_root.resolve())]
    for scene in args.scene or []:
        arguments.extend(["--scene", scene])
    if args.all:
        arguments.append("--all")
    if args.require_all_registered:
        arguments.append("--require-all-registered")
    run_internal("validate_colmap.py", Path(sys.executable), arguments)


def run_prepare_gshell(args: argparse.Namespace) -> None:
    arguments = ["--input-dir", str(args.input_dir.resolve()), "--output-dir", str(args.output_dir.resolve())]
    for shoe in args.shoe or []:
        arguments.extend(["--shoe", shoe])
    if args.overwrite:
        arguments.append("--overwrite")
    if args.dry_run:
        arguments.append("--dry-run")
    arguments.extend(["--expected-frame-count", str(EXPECTED_VIEW_COUNT)])
    run_internal("gshell_adapter.py", Path(sys.executable), arguments)


def run_prepare_sugar(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    names = (
        sorted(path.name for path in input_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
        if args.all
        else args.scene
    )
    for name in names:
        arguments = [
            "--source-root", str(input_dir),
            "--output-root", str(args.output_dir.resolve()),
            "--scene", name,
            "--colmap-bin", str(args.colmap_bin.resolve()),
        ]
        if args.overwrite:
            arguments.append("--overwrite")
        run_internal("sugar_adapter.py", args.image_python, arguments)
        run_internal(
            "validate_sugar.py",
            args.image_python,
            ["--dataset-root", str(args.output_dir.resolve()), "--scene", name],
        )


def main() -> None:
    args = parse_args()
    handlers = {
        "render": run_render,
        "validate-render": run_validate_render,
        "colmap": run_colmap,
        "validate-colmap": run_validate_colmap,
        "prepare-sugar": run_prepare_sugar,
        "prepare-gshell": run_prepare_gshell,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
