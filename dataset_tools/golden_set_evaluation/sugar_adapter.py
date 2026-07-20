#!/usr/bin/env python3
"""Create an optional SuGaR RGBA/bounding-box view of a common COLMAP scene.

The source fresh-COLMAP scene is never modified.  The output keeps the same
camera poses and sparse points, converts the undistorted RGB images to RGBA
PNG files, and changes only the corresponding image-name extensions in the
undistorted COLMAP model.  Blender cameras, Blender depth, and ground-truth
meshes are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TOOL_VERSION = "1.1.0"
EXPECTED_IMAGE_COUNT = 180
DEFAULT_COLMAP = Path("/storage/Abhinay/conda_envs/colmap/bin/colmap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-overwriting, mask-aligned copy of a fresh-COLMAP scene."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--colmap-bin", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--bbox-low-quantile", type=float, default=0.01)
    parser.add_argument("--bbox-high-quantile", type=float, default=0.99)
    parser.add_argument(
        "--bbox-margin-fraction",
        type=float,
        default=0.25,
        help="Expand each side of the robust COLMAP point box by this fraction of its extent.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_single_camera(path: Path) -> dict[str, object]:
    records: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                records.append(stripped.split())
    if len(records) != 1:
        raise ValueError(f"Expected exactly one shared camera in {path}; found {len(records)}")
    values = records[0]
    return {
        "camera_id": int(values[0]),
        "model": values[1],
        "width": int(values[2]),
        "height": int(values[3]),
        "params": [float(value) for value in values[4:]],
    }


def read_colmap_points(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not points:
        raise ValueError(f"No COLMAP points found in {path}")
    return np.asarray(points, dtype=np.float64)


def robust_bbox(
    points: np.ndarray, low_quantile: float, high_quantile: float, margin: float
) -> dict[str, object]:
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("Bounding-box quantiles must satisfy 0 <= low < high <= 1")
    if margin < 0:
        raise ValueError("Bounding-box margin must be non-negative")
    robust_min = np.quantile(points, low_quantile, axis=0)
    robust_max = np.quantile(points, high_quantile, axis=0)
    extent = robust_max - robust_min
    minimum = robust_min - margin * extent
    maximum = robust_max + margin * extent
    inside = np.logical_and(points >= minimum, points <= maximum).all(axis=1)
    return {
        "source": "undistorted/sparse/0/points3D.txt",
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "margin_fraction_per_side": margin,
        "robust_min": robust_min.tolist(),
        "robust_max": robust_max.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "diagonal": float(np.linalg.norm(maximum - minimum)),
        "point_count": int(points.shape[0]),
        "points_inside": int(inside.sum()),
        "points_outside_fraction": float((~inside).mean()),
    }


def rewrite_image_extensions(source: Path, destination: Path) -> int:
    changed = 0
    output_lines: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            fields = stripped.split()
            is_image_record = (
                len(fields) == 10
                and fields[0].isdigit()
                and fields[8].isdigit()
                and fields[9].lower().endswith((".jpg", ".jpeg"))
            )
            if is_image_record:
                fields[9] = str(Path(fields[9]).with_suffix(".png"))
                line = " ".join(fields) + "\n"
                changed += 1
            output_lines.append(line)
    destination.write_text("".join(output_lines), encoding="utf-8")
    return changed


def create_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    overlay = rgb.copy()
    overlay[edges] = np.asarray([0, 255, 0], dtype=np.uint8)
    return overlay


def write_montage(records: list[tuple[str, np.ndarray, np.ndarray]], destination: Path) -> None:
    selected_indices = np.linspace(0, len(records) - 1, num=6, dtype=int)
    tiles: list[np.ndarray] = []
    for index in selected_indices:
        name, rgb, mask = records[index]
        overlay = create_overlay(rgb, mask)
        tile = cv2.resize(overlay, (510, 340), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (510, 32), (0, 0, 0), thickness=-1)
        cv2.putText(
            tile,
            name,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    montage = np.concatenate(
        [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(montage, mode="RGB").save(destination, quality=95)


def atomic_install(temporary: Path, destination: Path, overwrite: bool) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> int:
    args = parse_args()
    source_scene = (args.source_root / args.scene).resolve()
    destination = (args.output_root / args.scene).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = args.output_root / f".{args.scene}.tmp-{uuid.uuid4().hex}"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {destination}")

    source_manifest = source_scene / "conversion_manifest.json"
    sparse_source = source_scene / "undistorted" / "sparse" / "0"
    output_camera_path = sparse_source / "cameras.txt"
    points_path = sparse_source / "points3D.txt"
    aligned_masks = source_scene / "undistorted" / "masks"
    for required in (source_manifest, output_camera_path, points_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not aligned_masks.is_dir():
        raise FileNotFoundError(aligned_masks)

    output_camera = read_single_camera(output_camera_path)
    points = read_colmap_points(points_path)
    box = robust_bbox(
        points,
        args.bbox_low_quantile,
        args.bbox_high_quantile,
        args.bbox_margin_fraction,
    )

    started = time.time()
    temporary.mkdir(parents=True)
    try:
        sparse_destination = temporary / "undistorted" / "sparse" / "0"
        sparse_destination.mkdir(parents=True)
        for name in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
        ):
            source = sparse_source / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, sparse_destination / name)
        output_images = temporary / "undistorted" / "images"
        output_masks = temporary / "undistorted" / "masks"
        output_images.mkdir(parents=True)
        output_masks.mkdir(parents=True)

        rgb_paths = sorted((source_scene / "undistorted" / "images").glob("*.jpg"))
        if len(rgb_paths) != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"Expected {EXPECTED_IMAGE_COUNT} undistorted RGB images; found {len(rgb_paths)}")

        coverage: list[float] = []
        file_records: list[dict[str, object]] = []
        montage_records: list[tuple[str, np.ndarray, np.ndarray]] = []
        expected_size = (int(output_camera["width"]), int(output_camera["height"]))

        for rgb_path in rgb_paths:
            aligned_mask_path = aligned_masks / f"{rgb_path.stem}.png"
            undistorted_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            mapped_mask = cv2.imread(str(aligned_mask_path), cv2.IMREAD_GRAYSCALE)
            if mapped_mask is None or undistorted_bgr is None:
                raise ValueError(f"Could not decode RGB/mask files for {rgb_path.name}")
            if (undistorted_bgr.shape[1], undistorted_bgr.shape[0]) != expected_size:
                raise ValueError(f"Unexpected undistorted RGB size for {rgb_path.name}")
            if mapped_mask.shape != undistorted_bgr.shape[:2]:
                raise ValueError(f"Aligned mask/RGB size mismatch for {rgb_path.name}")
            mapped_mask = np.where(mapped_mask >= 128, 255, 0).astype(np.uint8)
            mask_fraction = float((mapped_mask > 0).mean())
            if not 0.001 < mask_fraction < 0.95:
                raise ValueError(f"Implausible mask coverage {mask_fraction:.6f} for {rgb_path.name}")
            coverage.append(mask_fraction)

            rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
            rgba = np.dstack((rgb, mapped_mask))
            png_path = output_images / f"{rgb_path.stem}.png"
            mask_path = output_masks / f"{rgb_path.stem}.png"
            Image.fromarray(rgba, mode="RGBA").save(png_path, compress_level=6)
            Image.fromarray(mapped_mask, mode="L").save(mask_path, compress_level=6)
            file_records.append(
                {
                    "name": png_path.name,
                    "source_undistorted_rgb": rgb_path.name,
                    "source_aligned_mask": aligned_mask_path.name,
                    "rgba_sha256": sha256(png_path),
                    "mask_sha256": sha256(mask_path),
                    "foreground_fraction": mask_fraction,
                }
            )
            montage_records.append((png_path.name, rgb, mapped_mask))

        changed = rewrite_image_extensions(
            sparse_source / "images.txt", sparse_destination / "images.txt"
        )
        if changed != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"Changed {changed} COLMAP image names; expected {EXPECTED_IMAGE_COUNT}")

        text_model = temporary / ".text_model"
        binary_model = temporary / ".binary_model"
        text_model.mkdir()
        binary_model.mkdir()
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            shutil.copy2(sparse_destination / name, text_model / name)
        completed = subprocess.run(
            [
                str(args.colmap_bin.resolve()),
                "model_converter",
                "--input_path",
                str(text_model),
                "--output_path",
                str(binary_model),
                "--output_type",
                "BIN",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"COLMAP model conversion failed:\n{completed.stdout}")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(binary_model / name, sparse_destination / name)
        shutil.rmtree(text_model)
        shutil.rmtree(binary_model)

        montage_path = temporary / "validation" / "mask_overlay_montage.jpg"
        write_montage(montage_records, montage_path)

        source_protocol = json.loads(source_manifest.read_text(encoding="utf-8"))
        manifest = {
            "tool": "golden_set_evaluation/sugar_adapter.py",
            "tool_version": TOOL_VERSION,
            "scene": args.scene,
            "protocol": "compact_fresh_colmap_with_rgba_masks",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_rgb_only_scene": str(source_scene),
            "source_manifest_sha256": sha256(source_manifest),
            "source_colmap_protocol": source_protocol.get("protocol"),
            "camera_invariance": {
                "poses_and_intrinsics": "copied from the compact fresh-COLMAP model; image extensions only changed from .jpg to .png",
                "undistorted_camera": output_camera,
                "registered_images": changed,
            },
            "mask_alignment": {
                "method": "reuse masks already aligned by the common COLMAP pipeline",
                "source": "undistorted/masks",
                "threshold": 128,
            },
            "storage_contract": {
                "duplicates_common_raw_inputs": False,
                "contains_rgba_training_images": True,
                "contains_colmap_model_with_png_names": True,
            },
            "images": {
                "count": len(file_records),
                "width": output_camera["width"],
                "height": output_camera["height"],
                "foreground_fraction_min": float(np.min(coverage)),
                "foreground_fraction_mean": float(np.mean(coverage)),
                "foreground_fraction_max": float(np.max(coverage)),
                "files": file_records,
            },
            "foreground_bbox": box,
            "training_contract": {
                "use_masks": True,
                "white_background": True,
                "center_bbox": False,
                "foreground_only": True,
                "constrain_points_to_bbox": True,
                "filter_gaussians_by_bbox": True,
                "max_gaussian_scale_ratio": 0.05,
                "uses_blender_camera_transforms": False,
                "uses_inverse_depth": False,
                "uses_ground_truth_mesh": False,
            },
            "validation_montage": "validation/mask_overlay_montage.jpg",
            "elapsed_seconds": time.time() - started,
        }
        (temporary / "masked_colmap_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        atomic_install(temporary, destination, args.overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"Created: {destination}")
    print(f"RGBA images: {EXPECTED_IMAGE_COUNT}")
    print(f"Mean foreground coverage: {np.mean(coverage) * 100:.3f}%")
    print(f"Foreground bbox min: {box['min']}")
    print(f"Foreground bbox max: {box['max']}")
    print(f"Foreground bbox diagonal: {box['diagonal']:.6f}")
    print(f"Mask montage: {destination / 'validation' / 'mask_overlay_montage.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
