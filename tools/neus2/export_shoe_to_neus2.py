#!/usr/bin/env python3
"""Export one shoe folder into NeuS2 static-scene format.

Expected input layout:
    <shoe_dir>/
      images/
      masks/
      colmap/
        cameras.txt
        images.txt
        points3D.txt

Output layout:
    <output_dir>/
      images/
        *.png   # RGBA images (mask stored in alpha)
      transform.json
      transform_train.json
      transform_test.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe-dir", required=True, help="Path to one shoe folder.")
    parser.add_argument("--output-dir", required=True, help="Where to write the NeuS2 export.")
    parser.add_argument(
        "--points-path",
        default="",
        help="Optional override for COLMAP points3D.txt used for normalization.",
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=6,
        help="Hold out every N-th sorted view for transform_test.json.",
    )
    parser.add_argument(
        "--test-offset",
        type=int,
        default=0,
        help="Offset applied before the every-N split.",
    )
    parser.add_argument(
        "--test-indices",
        default="",
        help="Optional comma-separated 0-based indices to use as test views instead of stride splitting.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Scene margin inside the NeuS2 unit cube when computing scale/offset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before exporting.",
    )
    return parser.parse_args()


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
            raise ValueError(f"Unexpected COLMAP image line in {path}: {line}")

        entry = ImageEntry(
            image_id=int(parts[0]),
            qvec=tuple(float(v) for v in parts[1:5]),
            tvec=tuple(float(v) for v in parts[5:8]),
            camera_id=int(parts[8]),
            name=parts[9],
        )
        entries.append(entry)

        # Skip the following points2D line if present.
        if i < len(raw_lines):
            i += 1

    if not entries:
        raise ValueError(f"No images found in {path}")
    return sorted(entries, key=lambda e: e.name)


def parse_points_xyz(path: Path) -> np.ndarray:
    points: List[Tuple[float, float, float]] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            points.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not points:
        raise ValueError(f"No 3D points found in {path}")
    return np.asarray(points, dtype=np.float64)


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def colmap_image_to_c2w(entry: ImageEntry) -> np.ndarray:
    rot_wc = qvec_to_rotmat(entry.qvec)
    t_wc = np.asarray(entry.tvec, dtype=np.float64)
    rot_cw = rot_wc.T
    center = -rot_cw @ t_wc

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_cw
    c2w[:3, 3] = center
    return c2w


def camera_to_intrinsic(camera: Camera) -> np.ndarray:
    intrinsic = np.eye(4, dtype=np.float64)

    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = camera.params[:3]
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        return intrinsic

    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = camera.params[:4]
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        return intrinsic

    raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")


def camera_distortion(camera: Camera) -> Dict[str, float]:
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return {}
    if camera.model == "SIMPLE_RADIAL":
        return {"k1": float(camera.params[3])}
    if camera.model == "RADIAL":
        return {"k1": float(camera.params[3]), "k2": float(camera.params[4])}
    if camera.model == "OPENCV":
        return {
            "k1": float(camera.params[4]),
            "k2": float(camera.params[5]),
            "p1": float(camera.params[6]),
            "p2": float(camera.params[7]),
        }
    raise NotImplementedError(
        "Raw-image export currently supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, and OPENCV."
    )


def consistent_distortion(cameras: Iterable[Camera]) -> Dict[str, float]:
    cameras = list(cameras)
    distortions = [camera_distortion(cam) for cam in cameras]
    first = distortions[0]
    for other in distortions[1:]:
        if other != first:
            raise ValueError(
                "Multiple distinct camera distortion settings found. "
                "This exporter currently expects a shared camera calibration per shoe."
            )
    return first


def compute_normalization(points: np.ndarray, margin: float) -> Tuple[float, List[float]]:
    if not (0.0 <= margin < 0.5):
        raise ValueError("--margin must be in [0, 0.5).")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Could not compute a valid scene radius for normalization.")

    target_radius = 0.5 - margin
    scale = target_radius / radius
    offset = (np.array([0.5, 0.5, 0.5], dtype=np.float64) - center * scale).tolist()
    return float(scale), [float(v) for v in offset]


def parse_test_indices(text: str) -> List[int]:
    if not text.strip():
        return []
    return sorted({int(part.strip()) for part in text.split(",") if part.strip()})


def select_test_indices(num_frames: int, explicit: Sequence[int], stride: int, offset: int) -> List[int]:
    if explicit:
        for idx in explicit:
            if idx < 0 or idx >= num_frames:
                raise ValueError(f"Test index {idx} is out of range for {num_frames} frames.")
        return sorted(set(explicit))

    if stride <= 0:
        raise ValueError("--test-stride must be > 0 when --test-indices is not provided.")

    start = offset % stride
    indices = list(range(start, num_frames, stride))
    if not indices and num_frames > 0:
        indices = [num_frames - 1]
    return indices


def export_rgba_images(
    entries: Sequence[ImageEntry],
    cameras: Dict[int, Camera],
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
) -> Dict[str, str]:
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    for entry in entries:
        stem = Path(entry.name).stem
        image_path = images_dir / entry.name
        mask_path = masks_dir / f"{stem}.png"
        out_name = f"{stem}.png"
        out_path = out_images_dir / out_name
        camera = cameras[entry.camera_id]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {entry.name}: {mask_path}")

        rgb = Image.open(image_path).convert("RGB")
        alpha = Image.open(mask_path).convert("L")
        if rgb.size != alpha.size:
            raise ValueError(f"Image/mask size mismatch for {entry.name}: {rgb.size} vs {alpha.size}")

        target_size = (camera.width, camera.height)
        if rgb.size != target_size:
            rgb = rgb.resize(target_size, resample=Image.Resampling.LANCZOS)
            alpha = alpha.resize(target_size, resample=Image.Resampling.NEAREST)

        rgba = rgb.copy()
        rgba.putalpha(alpha)
        rgba.save(out_path)
        mapping[entry.name] = f"images/{out_name}"

    return mapping


def make_frame(entry: ImageEntry, camera: Camera, file_path: str) -> Dict[str, object]:
    return {
        "file_path": file_path,
        "transform_matrix": colmap_image_to_c2w(entry).tolist(),
        "intrinsic_matrix": camera_to_intrinsic(camera).tolist(),
    }


def write_transform_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:
    args = parse_args()
    shoe_dir = Path(args.shoe_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    images_dir = shoe_dir / "images"
    masks_dir = shoe_dir / "masks"
    colmap_dir = shoe_dir / "colmap"
    cameras_path = colmap_dir / "cameras.txt"
    images_path = colmap_dir / "images.txt"
    points_path = Path(args.points_path).resolve() if args.points_path else colmap_dir / "points3D.txt"

    required_paths = [images_dir, masks_dir, cameras_path, images_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = parse_colmap_cameras(cameras_path)
    entries = parse_colmap_images(images_path)
    image_paths = export_rgba_images(entries, cameras, images_dir, masks_dir, output_dir)

    sizes = {(cam.width, cam.height) for cam in cameras.values()}
    if len(sizes) != 1:
        raise ValueError("Multiple image resolutions found across cameras; this exporter expects one shared resolution.")
    width, height = next(iter(sizes))

    if points_path.exists():
        norm_points = parse_points_xyz(points_path)
    else:
        norm_points = np.asarray([colmap_image_to_c2w(entry)[:3, 3] for entry in entries], dtype=np.float64)

    scale, offset = compute_normalization(norm_points, margin=args.margin)
    distortion = consistent_distortion(cameras.values())

    payload_base: Dict[str, object] = {
        "w": width,
        "h": height,
        "aabb_scale": 1.0,
        "scale": scale,
        "offset": offset,
        "from_na": True,
    }
    payload_base.update(distortion)

    frames = [make_frame(entry, cameras[entry.camera_id], image_paths[entry.name]) for entry in entries]
    test_indices = set(select_test_indices(len(frames), parse_test_indices(args.test_indices), args.test_stride, args.test_offset))

    all_payload = dict(payload_base)
    all_payload["frames"] = frames
    write_transform_json(output_dir / "transform.json", all_payload)

    train_payload = dict(payload_base)
    train_payload["frames"] = [frame for idx, frame in enumerate(frames) if idx not in test_indices]
    write_transform_json(output_dir / "transform_train.json", train_payload)

    test_payload = dict(payload_base)
    test_payload["frames"] = [frame for idx, frame in enumerate(frames) if idx in test_indices]
    write_transform_json(output_dir / "transform_test.json", test_payload)

    print(f"Exported {shoe_dir.name} -> {output_dir}")
    print(f"Views: total={len(frames)} train={len(train_payload['frames'])} test={len(test_payload['frames'])}")
    print(f"Image size: {width}x{height}")
    print(f"Normalization: scale={scale:.8f} offset={offset}")
    if distortion:
        print(f"Distortion: {distortion}")
    else:
        print("Distortion: none")


if __name__ == "__main__":
    main()
