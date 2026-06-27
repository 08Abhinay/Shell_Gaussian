#!/usr/bin/env python3
"""Build a single GShell dataset from separately rendered view folders.

GShell's DatasetNERF loader expects one folder containing:

    image/imgXXX.jpg
    mask/imgXXX.png
    transforms.json

This utility combines the synthetic external-shoe turntable and top-to-bottom
renders into that layout without deleting the original folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data/abelde/datasets/processed/external_shoes_canonical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shoe", required=True, help="Shoe folder name under --dataset-root.")
    parser.add_argument(
        "--output-name",
        default="combined_turntable_top_to_bottom",
        help="Output subfolder to create under the shoe folder.",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=("turntable", "top_to_bottom"),
        default=None,
        help="Source subfolder to include. Repeat to control order. Defaults to turntable then top_to_bottom.",
    )
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


def source_transforms_path(source_dir: Path) -> Path:
    path = source_dir / "transforms.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing transforms file: {path}")
    return path


def copy_frame(
    source_dir: Path,
    output_dir: Path,
    frame: dict[str, Any],
    output_index: int,
    source_name: str,
) -> dict[str, Any]:
    src_image = source_dir / frame["file_path"]
    mask_rel = Path(frame["file_path"])
    mask_rel = Path("mask") / mask_rel.name.replace(".jpg", ".png")
    src_mask = source_dir / mask_rel
    if not src_image.exists():
        raise FileNotFoundError(f"Missing image: {src_image}")
    if not src_mask.exists():
        raise FileNotFoundError(f"Missing mask: {src_mask}")

    basename = f"img{output_index:03d}"
    dst_image = output_dir / "image" / f"{basename}.jpg"
    dst_mask = output_dir / "mask" / f"{basename}.png"
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_mask, dst_mask)

    return {
        "file_path": f"image/{basename}.jpg",
        "camera_angle_x": frame["camera_angle_x"],
        "transform_matrix": frame["transform_matrix"],
        "source_view_set": source_name,
        "source_file_path": frame["file_path"],
    }


def build_combined_dataset(
    shoe_dir: Path,
    output_name: str,
    source_names: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = shoe_dir / output_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    next_index = 1
    for source_name in source_names:
        source_dir = shoe_dir / source_name
        payload = load_json(source_transforms_path(source_dir))
        frames = payload.get("frames", [])
        if not frames:
            raise ValueError(f"No frames found in {source_dir / 'transforms.json'}")
        for frame in frames:
            combined_frames.append(copy_frame(source_dir, output_dir, frame, next_index, source_name))
            next_index += 1
        source_counts[source_name] = len(frames)

    write_json(output_dir / "transforms.json", {"frames": combined_frames})
    write_json(
        output_dir / "combined_dataset_summary.json",
        {
            "shoe": shoe_dir.name,
            "output_dir": str(output_dir),
            "sources": source_names,
            "source_counts": source_counts,
            "total_frames": len(combined_frames),
        },
    )
    return {
        "shoe": shoe_dir.name,
        "output_dir": str(output_dir),
        "total_frames": len(combined_frames),
        "source_counts": source_counts,
    }


def main() -> None:
    args = parse_args()
    shoe_dir = args.dataset_root / args.shoe
    if not shoe_dir.exists():
        raise FileNotFoundError(f"Shoe folder not found: {shoe_dir}")
    source_names = args.source or ["turntable", "top_to_bottom"]
    summary = build_combined_dataset(shoe_dir, args.output_name, source_names, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
