#!/usr/bin/env python3
"""Align raw masks to COLMAP images, then compact the processed scene."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Raw dataset root containing <scene>/masks.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_camera(path: Path) -> tuple[str, int, int, list[float]]:
    rows = [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if len(rows) != 1:
        raise ValueError(f"Expected one camera in {path}, found {len(rows)}")
    row = rows[0]
    return row[1], int(row[2]), int(row[3]), [float(value) for value in row[4:]]


def camera_matrix(model: str, parameters: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if model == "SIMPLE_RADIAL":
        focal, cx, cy, radial = parameters
        distortion = np.asarray([radial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        matrix = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    elif model == "SIMPLE_PINHOLE":
        focal, cx, cy = parameters
        distortion = np.zeros(5, dtype=np.float64)
        matrix = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    elif model == "PINHOLE":
        fx, fy, cx, cy = parameters
        distortion = np.zeros(5, dtype=np.float64)
        matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Unsupported camera model for mask alignment: {model}")
    return matrix.astype(np.float64), distortion


def update_manifest(scene: Path, source_scene: Path, mask_count: int) -> None:
    manifest_path = scene / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.setdefault("inputs", {})
    if "masks" not in inputs and "masks_copied_after_colmap" in inputs:
        inputs["masks"] = inputs.pop("masks_copied_after_colmap")
    manifest["source_scene"] = str(source_scene.resolve())
    manifest["mask_alignment"] = {
        "method": "OpenCV initUndistortRectifyMap with nearest-neighbor remap",
        "source": str((source_scene / "masks").resolve()),
        "output": "undistorted/masks",
        "count": mask_count,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["storage_contract"] = {
        "raw_rgb_and_masks_are_external": True,
        "duplicates_raw_inputs": False,
        "final_layout": [
            "undistorted/images",
            "undistorted/masks",
            "undistorted/sparse/0",
            "logs",
            "conversion_manifest.json",
        ],
        "raw_colmap_model_is_transient_for_mask_alignment": True,
        "raw_model_retained": False,
        "compaction_complete": True,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def compact_scene(scene: Path) -> None:
    for redundant in (scene / "images", scene / "masks", scene / "colmap"):
        if redundant.is_dir():
            shutil.rmtree(redundant)
        elif redundant.exists():
            redundant.unlink()
    stereo = scene / "undistorted" / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)
    points_ply = scene / "undistorted" / "sparse" / "0" / "points3D.ply"
    if points_ply.exists():
        points_ply.unlink()


def align_scene(scene: Path, source_scene: Path, overwrite: bool) -> int:
    raw_model = scene / "colmap" / "cameras.txt"
    output_model = scene / "undistorted" / "sparse" / "0" / "cameras.txt"
    raw_masks = source_scene / "masks"
    output_images = scene / "undistorted" / "images"
    for required in (raw_model, output_model, raw_masks, output_images):
        if not required.exists():
            raise FileNotFoundError(required)

    raw_name, _, _, raw_parameters = read_camera(raw_model)
    output_name, output_width, output_height, output_parameters = read_camera(output_model)
    raw_matrix, distortion = camera_matrix(raw_name, raw_parameters)
    output_matrix, _ = camera_matrix(output_name, output_parameters)
    map_x, map_y = cv2.initUndistortRectifyMap(
        raw_matrix,
        distortion,
        np.eye(3, dtype=np.float64),
        output_matrix,
        (output_width, output_height),
        cv2.CV_32FC1,
    )

    destination = scene / "undistorted" / "masks"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Undistorted masks already exist: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix="masks_", dir=destination.parent))
    try:
        image_paths = sorted(output_images.glob("*.jpg"))
        for image_path in image_paths:
            source = raw_masks / f"{image_path.stem}.png"
            mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(source)
            aligned = cv2.remap(
                mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            aligned = np.where(aligned >= 128, 255, 0).astype(np.uint8)
            if not np.any(aligned):
                raise ValueError(f"Aligned mask is empty: {source}")
            if not cv2.imwrite(str(temporary / source.name), aligned):
                raise RuntimeError(f"Could not write aligned mask for {source.name}")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        mask_count = len(image_paths)
        compact_scene(scene)
        update_manifest(scene, source_scene, mask_count)
        return mask_count
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    source_root = args.source_root.resolve()
    names = (
        sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        if args.all
        else args.scene
    )
    for name in names:
        count = align_scene(root / name, source_root / name, args.overwrite)
        print(f"{name}: aligned {count} masks and compacted the processed scene")


if __name__ == "__main__":
    main()
