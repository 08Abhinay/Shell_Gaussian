#!/usr/bin/env python3
"""Export legacy or compact COLMAP shoe scenes to canonical GShell format.

Supported input layouts per shoe:
    <input_root>/<shoe>/
      images/
      masks/
      colmap/
        cameras.txt
        images.txt
        points3D.txt

or the compact evaluation layout:
    <input_root>/<shoe>/
      undistorted/images/
      undistorted/masks/
      undistorted/sparse/0/{cameras,images,points3D}.txt

Output layout per shoe:
    <output_root>/<shoe>/
      image/  (physical copy of input images)
      mask/   (physical copy of input masks)
      transforms.json
      turntable_canonicalization.json
      invdepth/                                           (optional copy)
      invdepth_summary.json                              (optional)

This combines the old two-step path:
    export_shoes_to_gshell.py
    canonicalize_gshell_turntable_phase.py

without writing an intermediate processed dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


OPENCV_TO_OPENGL_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImageEntry:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class InitialSceneData:
    payload: dict[str, Any]
    export_info: dict[str, Any]
    canonical_points_pre_yaw: np.ndarray


@dataclass(frozen=True)
class SceneLayout:
    name: str
    images: Path
    masks: Path
    model: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/abelde/datasets/raw/golden_set"),
        help="Raw golden_set-like root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes"),
        help="Canonical GShell output root.",
    )
    parser.add_argument(
        "--shoe",
        action="append",
        default=None,
        help="Process only this shoe name. May be repeated.",
    )
    parser.add_argument(
        "--camera-only-shoe",
        action="append",
        default=None,
        help="Ignore sparse points for normalization for this shoe. May be repeated.",
    )
    parser.add_argument(
        "--reference-frame",
        default=None,
        help="Frame used to fix turntable phase; defaults to the first registered frame.",
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=None,
        help="Optional strict registered-frame count (180 for the evaluation set).",
    )
    parser.add_argument("--target-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--invdepth-source-root",
        type=Path,
        default=None,
        help="Optional root containing <shoe>/invdepth to attach to output scenes.",
    )
    parser.add_argument(
        "--invdepth-mode",
        choices=("copy",),
        default="copy",
        help="How to attach optional invdepth folders.",
    )
    parser.add_argument(
        "--size-normalization",
        choices=("off", "metadata_only"),
        default="off",
        help="Optional category-aware size normalization metadata export.",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=1.0,
        help="Target canonical size for the selected size-normalization basis.",
    )
    parser.add_argument(
        "--boot-shoe",
        action="append",
        default=None,
        help="Treat this shoe as a boot for category-aware size normalization. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def parse_colmap_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=tuple(float(v) for v in parts[4:]),
            )
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def parse_colmap_images(path: Path) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
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
        entries.append(
            ImageEntry(
                image_id=int(parts[0]),
                qvec=tuple(float(v) for v in parts[1:5]),
                tvec=tuple(float(v) for v in parts[5:8]),
                camera_id=int(parts[8]),
                name=parts[9],
            )
        )
        if i < len(raw_lines):
            i += 1
    if not entries:
        raise ValueError(f"No images found in {path}")
    return sorted(entries, key=lambda entry: entry.name)


def parse_points_xyz(path: Path) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
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


def get_focal(camera: Camera) -> float:
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return camera.params[0]
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return camera.params[0]
    raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")


def focal_to_fovx(focal: float, width: int) -> float:
    return 2.0 * math.atan(width / (2.0 * focal))


def rotate_x_matrix(angle: float) -> np.ndarray:
    s, c = math.sin(angle), math.cos(angle)
    mat = np.eye(4, dtype=np.float64)
    mat[1, 1] = c
    mat[1, 2] = s
    mat[2, 1] = -s
    mat[2, 2] = c
    return mat


def normalize_c2w(c2w: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    result = c2w.copy()
    result[:3, 3] = (c2w[:3, 3] - center) / scale
    return result


def compute_normalization(points: np.ndarray, cameras_c2w: list[np.ndarray]) -> tuple[np.ndarray, float]:
    cam_centers = np.asarray([c2w[:3, 3] for c2w in cameras_c2w])
    if points.size:
        all_pts = np.concatenate([points, cam_centers], axis=0)
        center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
        radius = float(np.linalg.norm(all_pts - center, axis=1).max())
    else:
        center = cam_centers.mean(axis=0)
        radius = float(np.linalg.norm(cam_centers - center, axis=1).max())
    if radius <= 0:
        radius = 1.0
    return center, radius


def colmap_c2w_to_gshell_transform(c2w: np.ndarray) -> np.ndarray:
    return gshell_dataset_rotation_matrix() @ c2w @ OPENCV_TO_OPENGL_CAMERA


def gshell_dataset_rotation_matrix() -> np.ndarray:
    return rotate_x_matrix(-math.pi / 2.0)


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (points - center) / scale


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return points @ rotation.T + translation


def angle_delta_deg(target: float, current: float) -> float:
    return normalize_angle_deg(target - current)


def raw_xy_orbit_angle_deg(matrix: np.ndarray) -> float:
    center = matrix[:3, 3]
    return normalize_angle_deg(math.degrees(math.atan2(float(center[1]), float(center[0]))))


def raw_z_rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    s, c = math.sin(angle_rad), math.cos(angle_rad)
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot


def frame_basename(frame: dict[str, Any]) -> str:
    return Path(str(frame["file_path"])).name


def matrix_for_frame(frame: dict[str, Any], scene_name: str) -> np.ndarray:
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{scene_name}: transform_matrix must be 4x4")
    return matrix


def find_reference_frame(frames: list[dict[str, Any]], reference_name: str) -> dict[str, Any]:
    for frame in frames:
        if frame_basename(frame) == reference_name or frame["file_path"] == reference_name:
            return frame
    available = ", ".join(frame_basename(frame) for frame in frames[:8])
    raise ValueError(f"Reference frame {reference_name!r} not found. First frames: {available}")


def angle_map(frames: list[dict[str, Any]], scene_name: str) -> dict[str, float]:
    return {
        frame_basename(frame): raw_xy_orbit_angle_deg(matrix_for_frame(frame, scene_name))
        for frame in frames
    }


def key_frame_names(frames: list[dict[str, Any]]) -> tuple[str, ...]:
    if not frames:
        return ()
    indices = sorted({0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4})
    return tuple(frame_basename(frames[index]) for index in indices)


def key_angle_map(
    angles: dict[str, float], frames: list[dict[str, Any]]
) -> dict[str, float | None]:
    return {name: angles.get(name) for name in key_frame_names(frames)}


def median_step_deg(angles: dict[str, float], frames: list[dict[str, Any]]) -> float:
    ordered = [angles[frame_basename(frame)] for frame in frames]
    unwrapped = np.unwrap(np.radians(ordered))
    steps = np.degrees(np.diff(unwrapped))
    return float(np.median(steps)) if steps.size else 0.0


def rotation_stats(frames: list[dict[str, Any]], scene_name: str) -> dict[str, float | bool]:
    dets: list[float] = []
    ortho_errors: list[float] = []
    bottom_row_errors: list[float] = []
    for frame in frames:
        matrix = matrix_for_frame(frame, scene_name)
        rot = matrix[:3, :3]
        dets.append(float(np.linalg.det(rot)))
        ortho_errors.append(float(np.linalg.norm(rot.T @ rot - np.eye(3), ord="fro")))
        bottom_row_errors.append(float(np.linalg.norm(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0]))))
    return {
        "rotation_det_min": min(dets),
        "rotation_det_max": max(dets),
        "rotation_orthonormal_error_max": max(ortho_errors),
        "bottom_row_error_max": max(bottom_row_errors),
        "rotations_passed": bool(
            min(dets) > 0.999
            and max(dets) < 1.001
            and max(ortho_errors) < 1e-5
            and max(bottom_row_errors) < 1e-8
        ),
    }


def copy_or_link_dir(source: Path, target: Path, mode: str, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {target}")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    if mode == "copy":
        shutil.copytree(source, target)
    else:
        raise ValueError(f"Unknown copy/link mode: {mode}")


def resolve_scene_layout(shoe_dir: Path) -> SceneLayout:
    layouts = (
        SceneLayout(
            name="compact_undistorted",
            images=shoe_dir / "undistorted" / "images",
            masks=shoe_dir / "undistorted" / "masks",
            model=shoe_dir / "undistorted" / "sparse" / "0",
        ),
        SceneLayout(
            name="legacy_raw_colmap",
            images=shoe_dir / "images",
            masks=shoe_dir / "masks",
            model=shoe_dir / "colmap",
        ),
    )
    for layout in layouts:
        if (
            layout.images.is_dir()
            and layout.masks.is_dir()
            and (layout.model / "cameras.txt").is_file()
            and (layout.model / "images.txt").is_file()
        ):
            return layout
    raise FileNotFoundError(f"No supported COLMAP scene layout found: {shoe_dir}")


def resolve_shoe_dirs(input_dir: Path, shoe_names: list[str] | None) -> list[Path]:
    if shoe_names:
        shoe_dirs = [input_dir / name for name in shoe_names]
    else:
        shoe_dirs = []
        for path in sorted(candidate for candidate in input_dir.iterdir() if candidate.is_dir()):
            try:
                resolve_scene_layout(path)
            except FileNotFoundError:
                continue
            shoe_dirs.append(path)
    missing = []
    for path in shoe_dirs:
        try:
            resolve_scene_layout(path)
        except FileNotFoundError:
            missing.append(path)
    if missing:
        raise FileNotFoundError("Invalid COLMAP shoe dirs: " + ", ".join(str(path) for path in missing))
    return shoe_dirs


def validate_target_size(target_size: float) -> None:
    if not math.isfinite(target_size) or target_size <= 0:
        raise ValueError(f"target_size must be a positive finite number, got {target_size}")


def build_initial_payload(
    shoe_dir: Path,
    camera_only_shoes: set[str],
) -> InitialSceneData:
    layout = resolve_scene_layout(shoe_dir)
    cameras = parse_colmap_cameras(layout.model / "cameras.txt")
    entries = parse_colmap_images(layout.model / "images.txt")
    points_path = layout.model / "points3D.txt"
    scene_points = parse_points_xyz(points_path) if points_path.exists() else np.zeros((0, 3), dtype=np.float64)
    c2ws = [colmap_image_to_c2w(entry) for entry in entries]
    points_for_norm = np.zeros((0, 3), dtype=np.float64) if shoe_dir.name in camera_only_shoes else scene_points
    center, radius = compute_normalization(points_for_norm, c2ws)

    frames: list[dict[str, Any]] = []
    for entry, c2w in zip(entries, c2ws):
        camera = cameras[entry.camera_id]
        norm_c2w = normalize_c2w(c2w, center, radius)
        gshell_c2w = colmap_c2w_to_gshell_transform(norm_c2w)
        frames.append(
            {
                "camera_angle_x": focal_to_fovx(get_focal(camera), camera.width),
                "file_path": f"image/{entry.name}",
                "transform_matrix": gshell_c2w.tolist(),
            }
        )

    export_info = {
        "source_layout": layout.name,
        "colmap_camera_count": len(cameras),
        "frame_count": len(frames),
        "normalization_source": "cameras" if shoe_dir.name in camera_only_shoes else "points+cameras",
        "normalization_center": center,
        "normalization_radius": radius,
        "sparse_point_count": int(scene_points.shape[0]),
    }
    normalized_points = normalize_points(scene_points, center, radius)
    canonical_points_pre_yaw = transform_points(normalized_points, gshell_dataset_rotation_matrix())
    return InitialSceneData(
        payload={"frames": frames},
        export_info=export_info,
        canonical_points_pre_yaw=canonical_points_pre_yaw,
    )


def canonicalize_payload(
    payload: dict[str, Any],
    scene_name: str,
    reference_frame: str,
    target_angle_deg: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frames = payload["frames"]
    before_angles = angle_map(frames, scene_name)
    reference = find_reference_frame(frames, reference_frame)
    before_reference_angle = raw_xy_orbit_angle_deg(matrix_for_frame(reference, scene_name))
    delta = angle_delta_deg(target_angle_deg, before_reference_angle)
    yaw = raw_z_rotation_matrix(delta)

    output_payload = json.loads(json.dumps(payload))
    for frame in output_payload["frames"]:
        matrix = matrix_for_frame(frame, scene_name)
        frame["transform_matrix"] = (yaw @ matrix).tolist()

    after_frames = output_payload["frames"]
    after_angles = angle_map(after_frames, scene_name)
    after_reference = find_reference_frame(after_frames, reference_frame)
    after_reference_angle = raw_xy_orbit_angle_deg(matrix_for_frame(after_reference, scene_name))
    target_error = abs(angle_delta_deg(target_angle_deg, after_reference_angle))
    validation = {
        "target_reference_angle_error_deg": target_error,
        "reference_angle_passed": target_error < 1e-6,
        "before_rotation_stats": rotation_stats(frames, scene_name),
        "after_rotation_stats": rotation_stats(after_frames, scene_name),
    }
    metadata = {
        "scene": scene_name,
        "method": "raw_colmap_export_plus_raw_xy_turntable_phase_alignment",
        "settings": {
            "reference_frame": reference_frame,
            "target_angle_deg": float(target_angle_deg),
            "orbit_plane": "raw_xy",
            "rotation_axis": "raw_z",
            "changed_fields": ["frames[*].transform_matrix"],
            "unchanged_fields": ["frames[*].camera_angle_x", "frames[*].file_path"],
        },
        "phase_correction": {
            "before_reference_angle_deg": before_reference_angle,
            "target_reference_angle_deg": float(target_angle_deg),
            "delta_yaw_deg": delta,
            "after_reference_angle_deg": after_reference_angle,
            "raw_z_rotation_matrix": yaw,
        },
        "before": {
            "key_frame_angles_deg": key_angle_map(before_angles, frames),
            "median_frame_step_deg": median_step_deg(before_angles, frames),
        },
        "after": {
            "key_frame_angles_deg": key_angle_map(after_angles, after_frames),
            "median_frame_step_deg": median_step_deg(after_angles, after_frames),
        },
        "validation": validation,
    }
    row_bits = {
        "before_reference_angle_deg": before_reference_angle,
        "after_reference_angle_deg": after_reference_angle,
        "delta_yaw_deg": delta,
        "target_error_deg": target_error,
        "before_median_step_deg": metadata["before"]["median_frame_step_deg"],
        "after_median_step_deg": metadata["after"]["median_frame_step_deg"],
        "rotations_passed": validation["after_rotation_stats"]["rotations_passed"],
        "rotation_det_min": validation["after_rotation_stats"]["rotation_det_min"],
        "rotation_det_max": validation["after_rotation_stats"]["rotation_det_max"],
        "rotation_orthonormal_error_max": validation["after_rotation_stats"]["rotation_orthonormal_error_max"],
    }
    for name in key_frame_names(frames):
        stem = Path(name).stem
        row_bits[f"before_{stem}_angle_deg"] = before_angles.get(name)
        row_bits[f"after_{stem}_angle_deg"] = after_angles.get(name)
    return output_payload, metadata, row_bits


def scene_extent_metrics(points: np.ndarray, scene_name: str) -> dict[str, float]:
    if points.size == 0:
        raise ValueError(f"{scene_name}: sparse COLMAP points are required for size normalization")
    if not np.isfinite(points).all():
        raise ValueError(f"{scene_name}: sparse COLMAP points contain non-finite values")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    x_extent, y_extent, z_extent = (float(value) for value in extents)
    longest_dim = max(x_extent, y_extent, z_extent)
    footprint_diag_xz = math.hypot(x_extent, z_extent)
    return {
        "x_extent": x_extent,
        "y_extent": y_extent,
        "z_extent": z_extent,
        "longest_dim": float(longest_dim),
        "footprint_diag_xz": float(footprint_diag_xz),
    }


def resolve_size_category(scene_name: str, boot_shoes: set[str]) -> str:
    return "boot" if scene_name in boot_shoes else "default"


def uniform_scale_payload(payload: dict[str, Any], scene_name: str, scale_factor: float) -> dict[str, Any]:
    scaled_payload = json.loads(json.dumps(payload))
    for frame in scaled_payload["frames"]:
        matrix = matrix_for_frame(frame, scene_name)
        matrix[:3, 3] *= scale_factor
        frame["transform_matrix"] = matrix.tolist()
    return scaled_payload


def size_normalization_defaults(args: argparse.Namespace) -> dict[str, Any]:
    enabled = args.size_normalization != "off"
    return {
        "size_normalization_enabled": enabled,
        "size_category": None,
        "size_rule": None,
        "size_target": float(args.target_size) if enabled else None,
        "size_scale_factor": None,
        "x_extent": None,
        "y_extent": None,
        "z_extent": None,
        "longest_dim": None,
        "footprint_diag_xz": None,
    }


def compute_category_aware_size_normalization(
    metadata: dict[str, Any],
    scene_name: str,
    canonical_points_pre_yaw: np.ndarray,
    args: argparse.Namespace,
    boot_shoes: set[str],
) -> dict[str, Any]:
    defaults = size_normalization_defaults(args)
    if args.size_normalization == "off":
        metadata["size_normalization"] = {
            "enabled": False,
            "mode": "off",
            "category": None,
            "rule": None,
            "target_size": None,
            "scale_factor": None,
            "uniform_scaling": True,
        }
        return defaults

    validate_target_size(float(args.target_size))
    yaw = np.asarray(metadata["phase_correction"]["raw_z_rotation_matrix"], dtype=np.float64)
    canonical_points = transform_points(canonical_points_pre_yaw, yaw)
    metrics = scene_extent_metrics(canonical_points, scene_name)
    category = resolve_size_category(scene_name, boot_shoes)
    rule = "footprint_diag_xz" if category == "boot" else "longest_dim"
    scale_basis = metrics[rule]
    if not math.isfinite(scale_basis) or scale_basis <= 0:
        raise ValueError(f"{scene_name}: invalid size-normalization basis {scale_basis!r} for rule {rule}")
    scale_factor = float(args.target_size) / scale_basis

    metadata["size_normalization"] = {
        "enabled": True,
        "mode": args.size_normalization,
        "category": category,
        "rule": rule,
        "target_size": float(args.target_size),
        "scale_factor": scale_factor,
        "x_extent": metrics["x_extent"],
        "y_extent": metrics["y_extent"],
        "z_extent": metrics["z_extent"],
        "longest_dim": metrics["longest_dim"],
        "footprint_diag_xz": metrics["footprint_diag_xz"],
        "sparse_point_count_used": int(canonical_points.shape[0]),
        "points_frame": "canonical_gshell",
        "uniform_scaling": True,
    }
    row_bits = {
        "size_normalization_enabled": True,
        "size_category": category,
        "size_rule": rule,
        "size_target": float(args.target_size),
        "size_scale_factor": scale_factor,
        "x_extent": metrics["x_extent"],
        "y_extent": metrics["y_extent"],
        "z_extent": metrics["z_extent"],
        "longest_dim": metrics["longest_dim"],
        "footprint_diag_xz": metrics["footprint_diag_xz"],
    }
    return row_bits


def validate_output_scene(
    output_scene: Path,
    payload: dict[str, Any],
    expected_frame_count: int | None,
) -> dict[str, Any]:
    frames = payload["frames"]
    image_ok = (output_scene / "image").exists()
    mask_ok = (output_scene / "mask").exists()
    paths_ok = True
    for frame in frames:
        image_path = output_scene / frame["file_path"]
        mask_path = output_scene / str(frame["file_path"]).replace("image/", "mask/").replace(".jpg", ".png")
        paths_ok = paths_ok and image_path.exists() and mask_path.exists()
    return {
        "frame_count": len(frames),
        "frame_count_is_expected": (
            expected_frame_count is None or len(frames) == expected_frame_count
        ),
        "image_dir_valid": image_ok,
        "mask_dir_valid": mask_ok,
        "frame_image_mask_paths_valid": paths_ok,
    }


def attach_invdepth(
    scene_name: str,
    output_scene: Path,
    source_root: Path | None,
    mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    if source_root is None:
        return {"status": "not_requested", "attached": False}

    source_scene = source_root / scene_name
    source_invdepth = source_scene / "invdepth"
    if not source_invdepth.is_dir():
        return {"status": "missing_source", "attached": False, "source_invdepth": str(source_invdepth)}

    output_invdepth = output_scene / "invdepth"
    copy_or_link_dir(source_invdepth, output_invdepth, mode, overwrite=overwrite)
    npy_files = sorted(source_invdepth.glob("*.npy"))
    summary = {
        "shoe": scene_name,
        "dataset_dir": str(output_scene),
        "source_invdepth_dir": str(source_invdepth),
        "output_invdepth_dir": str(output_invdepth),
        "mode": mode,
        "frame_count": len(npy_files),
        "attached": True,
    }
    if npy_files:
        sample = np.load(npy_files[0], mmap_mode="r")
        summary["sample_shape"] = list(sample.shape)
        summary["sample_dtype"] = str(sample.dtype)
    with (output_scene / "invdepth_summary.json").open("w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
        f.write("\n")
    return {"status": "ok", **summary}


def write_scene(
    shoe_dir: Path,
    output_scene: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if output_scene.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output scene exists; pass --overwrite: {output_scene}")
        shutil.rmtree(output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)
    layout = resolve_scene_layout(shoe_dir)
    copy_or_link_dir(layout.images, output_scene / "image", "copy", overwrite=True)
    copy_or_link_dir(layout.masks, output_scene / "mask", "copy", overwrite=True)

    metadata["source_scene"] = str(shoe_dir)
    metadata["source_layout"] = layout.name
    metadata["output_scene"] = str(output_scene)
    with (output_scene / "transforms.json").open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
        f.write("\n")

    validation = validate_output_scene(output_scene, payload, args.expected_frame_count)
    metadata["validation"]["postwrite"] = validation
    with (output_scene / "turntable_canonicalization.json").open("w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
        f.write("\n")

    return validation


def export_scene(
    shoe_dir: Path,
    args: argparse.Namespace,
    camera_only_shoes: set[str],
    boot_shoes: set[str],
) -> dict[str, Any]:
    initial_scene = build_initial_payload(shoe_dir, camera_only_shoes)
    reference_frame = args.reference_frame or frame_basename(initial_scene.payload["frames"][0])
    payload, metadata, row_bits = canonicalize_payload(
        initial_scene.payload,
        shoe_dir.name,
        reference_frame,
        float(args.target_angle_deg),
    )
    size_row_bits = compute_category_aware_size_normalization(
        metadata,
        shoe_dir.name,
        initial_scene.canonical_points_pre_yaw,
        args,
        boot_shoes,
    )
    row_bits.update(size_row_bits)
    metadata["raw_export"] = {
        **initial_scene.export_info,
        "input_scene": str(shoe_dir),
    }
    output_scene = args.output_dir / shoe_dir.name

    if args.dry_run:
        postwrite = {
            "frame_count": len(payload["frames"]),
            "frame_count_is_expected": (
                args.expected_frame_count is None
                or len(payload["frames"]) == args.expected_frame_count
            ),
            "image_dir_valid": None,
            "mask_dir_valid": None,
            "frame_image_mask_paths_valid": None,
        }
        invdepth = {"status": "dry_run", "attached": False}
    else:
        postwrite = write_scene(shoe_dir, output_scene, payload, metadata, args)
        invdepth = attach_invdepth(
            shoe_dir.name,
            output_scene,
            args.invdepth_source_root,
            args.invdepth_mode,
            overwrite=True,
        )

    status = "ok"
    if not row_bits["rotations_passed"] or row_bits["target_error_deg"] >= 1e-6:
        status = "failed"
    if not args.dry_run and not all(
        bool(postwrite[key])
        for key in ("frame_count_is_expected", "image_dir_valid", "mask_dir_valid", "frame_image_mask_paths_valid")
    ):
        status = "failed"

    return {
        "scene": shoe_dir.name,
        "status": status,
        "frames": len(payload["frames"]),
        "source_scene": str(shoe_dir),
        "output_scene": str(output_scene),
        "transforms_json": str(output_scene / "transforms.json"),
        "turntable_canonicalization_json": str(output_scene / "turntable_canonicalization.json"),
        "reference_frame": reference_frame,
        "target_angle_deg": float(args.target_angle_deg),
        **row_bits,
        "frame_count_is_expected": postwrite["frame_count_is_expected"],
        "image_dir_valid": postwrite["image_dir_valid"],
        "mask_dir_valid": postwrite["mask_dir_valid"],
        "frame_image_mask_paths_valid": postwrite["frame_image_mask_paths_valid"],
        "invdepth_status": invdepth["status"],
        "invdepth_attached": invdepth["attached"],
        "invdepth_frame_count": invdepth.get("frame_count"),
        "normalization_source": initial_scene.export_info["normalization_source"],
        "normalization_radius": initial_scene.export_info["normalization_radius"],
        "sparse_point_count": initial_scene.export_info["sparse_point_count"],
    }


def aggregate_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "scene_count": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
        "settings": {
            "reference_frame": args.reference_frame,
            "expected_frame_count": args.expected_frame_count,
            "target_angle_deg": float(args.target_angle_deg),
            "invdepth_source_root": str(args.invdepth_source_root) if args.invdepth_source_root else None,
            "invdepth_mode": args.invdepth_mode,
            "size_normalization": args.size_normalization,
            "target_size": float(args.target_size),
            "boot_shoes": sorted(args.boot_shoe or []),
        },
        "all_rotations_passed": all(bool(row.get("rotations_passed")) for row in rows),
        "all_frame_counts_expected": all(bool(row.get("frame_count_is_expected")) for row in rows),
        "all_invdepth_attached_when_requested": (
            all(bool(row.get("invdepth_attached")) for row in rows)
            if args.invdepth_source_root
            else None
        ),
        "original_dataset_modified": False,
        "rows": to_jsonable(rows),
    }


def write_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(to_jsonable(aggregate_summary(rows, args)), f, indent=2)
        f.write("\n")

    if args.invdepth_source_root:
        inv_summary = {
            "source_root": str(args.invdepth_source_root),
            "output_root": str(args.output_dir),
            "mode": args.invdepth_mode,
            "requested": True,
            "attached": sum(1 for row in rows if row["invdepth_attached"]),
            "missing": [row["scene"] for row in rows if not row["invdepth_attached"]],
        }
        with (args.output_dir / "invdepth_batch_summary.json").open("w") as f:
            json.dump(to_jsonable(inv_summary), f, indent=2)
            f.write("\n")


def main() -> None:
    args = parse_args()
    validate_target_size(float(args.target_size))
    if args.output_dir.resolve() == args.input_dir.resolve():
        raise ValueError("Output dir must not be the same as input dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty; pass --overwrite: {args.output_dir}")

    shoe_dirs = resolve_shoe_dirs(args.input_dir, args.shoe)
    camera_only_shoes = set(args.camera_only_shoe or [])
    boot_shoes = set(args.boot_shoe or [])
    print(f"Exporting {len(shoe_dirs)} shoes from {args.input_dir} -> {args.output_dir}")

    rows: list[dict[str, Any]] = []
    for idx, shoe_dir in enumerate(shoe_dirs, start=1):
        print(f"[{idx}/{len(shoe_dirs)}] {shoe_dir.name}")
        try:
            row = export_scene(shoe_dir, args, camera_only_shoes, boot_shoes)
        except Exception as exc:
            row = {
                "scene": shoe_dir.name,
                "status": "error",
                "source_scene": str(shoe_dir),
                "output_scene": str(args.output_dir / shoe_dir.name),
                "error": str(exc),
            }
        rows.append(row)
        if row["status"] == "ok":
            print(
                "  ok "
                f"img001 {float(row['before_reference_angle_deg']):.2f}"
                f" -> {float(row['after_reference_angle_deg']):.2f} deg "
                f"(yaw {float(row['delta_yaw_deg']):+.2f}); "
                f"invdepth={row['invdepth_status']}"
            )
        else:
            print(f"  {row['status']}: {row.get('error', 'validation failed')}")

    if not args.dry_run:
        write_summary(rows, args)
        print(f"Wrote {args.output_dir / 'summary.csv'}")
        print(f"Wrote {args.output_dir / 'summary.json'}")

    errors = [row for row in rows if row["status"] not in {"ok"}]
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
