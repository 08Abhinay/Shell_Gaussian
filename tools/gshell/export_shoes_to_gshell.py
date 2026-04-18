#!/usr/bin/env python3
"""Export raw shoe folders into GShell-compatible format.

Expected input layout (per shoe):
    <shoe_dir>/
      images/          # *.jpg
      masks/           # *.png
      colmap/
        cameras.txt
        images.txt
        points3D.txt

Output layout (per shoe):
    <output_dir>/
      image/           # *.jpg  (symlinks to raw)
      mask/            # *.png  (symlinks to raw)
      transforms.json  # GShell-compatible transforms
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP parsing (reused from neus2 exporter)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass(frozen=True)
class ImageEntry:
    image_id: int
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str


def parse_colmap_cameras(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(float(v) for v in parts[4:])
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def parse_colmap_images(path: Path) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    with path.open() as f:
        raw_lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Unexpected COLMAP image line: {line}")
        entry = ImageEntry(
            image_id=int(parts[0]),
            qvec=tuple(float(v) for v in parts[1:5]),
            tvec=tuple(float(v) for v in parts[5:8]),
            camera_id=int(parts[8]),
            name=parts[9],
        )
        entries.append(entry)
        if i < len(raw_lines):
            i += 1  # skip points2D line
    if not entries:
        raise ValueError(f"No images found in {path}")
    return sorted(entries, key=lambda e: e.name)


def parse_points_xyz(path: Path) -> np.ndarray:
    points = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not points:
        raise ValueError(f"No 3D points found in {path}")
    return np.asarray(points, dtype=np.float64)


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def colmap_image_to_c2w(entry: ImageEntry) -> np.ndarray:
    rot_wc = qvec_to_rotmat(entry.qvec)
    t_wc = np.asarray(entry.tvec, dtype=np.float64)
    rot_cw = rot_wc.T
    center = -rot_cw @ t_wc
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_cw
    c2w[:3, 3] = center
    return c2w


def get_focal(camera: Camera) -> float:
    """Extract focal length from COLMAP camera."""
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return camera.params[0]
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return camera.params[0]  # fx
    raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")


def focal_to_fovx(focal: float, width: int) -> float:
    """Convert focal length to horizontal field-of-view in radians."""
    return 2.0 * math.atan(width / (2.0 * focal))


# ---------------------------------------------------------------------------
# Normalization: center scene in [-1, 1] box for GShell
# ---------------------------------------------------------------------------

def compute_normalization(
    points: np.ndarray, cameras_c2w: List[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """Return (center, scale) so that the scene fits in a unit sphere."""
    cam_centers = np.array([c2w[:3, 3] for c2w in cameras_c2w])
    all_pts = np.concatenate([points, cam_centers], axis=0)
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    radius = float(np.linalg.norm(all_pts - center, axis=1).max())
    if radius <= 0:
        radius = 1.0
    return center, radius


def normalize_c2w(c2w: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Translate and scale a camera-to-world matrix."""
    result = c2w.copy()
    result[:3, 3] = (c2w[:3, 3] - center) / scale
    return result


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_shoe(shoe_dir: Path, output_dir: Path) -> None:
    images_dir = shoe_dir / "images"
    masks_dir = shoe_dir / "masks"
    colmap_dir = shoe_dir / "colmap"

    cameras = parse_colmap_cameras(colmap_dir / "cameras.txt")
    entries = parse_colmap_images(colmap_dir / "images.txt")

    points_path = colmap_dir / "points3D.txt"
    if points_path.exists():
        scene_points = parse_points_xyz(points_path)
    else:
        scene_points = np.zeros((0, 3), dtype=np.float64)

    # Compute camera-to-world for each image
    c2ws = [colmap_image_to_c2w(entry) for entry in entries]

    # Compute normalization
    if scene_points.shape[0] > 0:
        center, radius = compute_normalization(scene_points, c2ws)
    else:
        cam_centers = np.array([c[:3, 3] for c in c2ws])
        center = cam_centers.mean(axis=0)
        radius = float(np.linalg.norm(cam_centers - center, axis=1).max())
        if radius <= 0:
            radius = 1.0

    # Create output directories
    out_image_dir = output_dir / "image"
    out_mask_dir = output_dir / "mask"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # Build frames
    frames = []
    for entry, c2w in zip(entries, c2ws):
        camera = cameras[entry.camera_id]
        focal = get_focal(camera)
        fovx = focal_to_fovx(focal, camera.width)

        # Normalize the c2w
        norm_c2w = normalize_c2w(c2w, center, radius)

        # Symlink image
        src_image = images_dir / entry.name
        stem = Path(entry.name).stem
        dst_image = out_image_dir / entry.name
        if not dst_image.exists():
            os.symlink(src_image.resolve(), dst_image)

        # Symlink mask
        mask_name = f"{stem}.png"
        src_mask = masks_dir / mask_name
        dst_mask = out_mask_dir / mask_name
        if not dst_mask.exists() and src_mask.exists():
            os.symlink(src_mask.resolve(), dst_mask)

        frames.append({
            "camera_angle_x": fovx,
            "file_path": f"image/{entry.name}",
            "transform_matrix": norm_c2w.tolist(),
        })

    transforms = {"frames": frames}
    with (output_dir / "transforms.json").open("w") as f:
        json.dump(transforms, f, indent=4)

    print(f"  {shoe_dir.name}: {len(frames)} frames, "
          f"center=[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}], "
          f"radius={radius:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/abelde/datasets/raw/golden_set",
        help="Path to raw golden_set directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/abelde/datasets/processed/gshell_shoes",
        help="Path to output processed directory.",
    )
    parser.add_argument(
        "--shoes",
        nargs="*",
        default=None,
        help="Specific shoe names to export. If not provided, exports all.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.shoes:
        shoe_names = args.shoes
    else:
        shoe_names = sorted([
            d.name for d in input_dir.iterdir()
            if d.is_dir() and (d / "colmap" / "cameras.txt").exists()
        ])

    print(f"Exporting {len(shoe_names)} shoes from {input_dir} -> {output_dir}")

    for name in shoe_names:
        shoe_path = input_dir / name
        out_path = output_dir / name
        out_path.mkdir(parents=True, exist_ok=True)
        try:
            export_shoe(shoe_path, out_path)
        except Exception as e:
            print(f"  ERROR exporting {name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
