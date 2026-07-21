#!/usr/bin/env python3
"""Build GShell-, SuGaR-, and NeuralUDF-ready data from reviewed GLB assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/external/golden_set_eval_glb"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/golden_set_evaluation_blender"
)
DEFAULT_SUGAR_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "golden_set_evaluation_blender_sugar"
)
DEFAULT_NEURALUDF_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "golden_set_evaluation_neuraludf"
)
DEFAULT_BLENDER = Path(
    "/storage/Abhinay/home_ab5298/anaconda3/envs/shellgaussianenv/bin/blender"
)
DEFAULT_COLMAP = Path("/storage/Abhinay/conda_envs/colmap/bin/colmap")
DEFAULT_MANIFEST = SCRIPT_DIR / "evaluation_manifest.json"

RESOLUTION = (1536, 1024)
FOV_X_DEG = 21.0
CAMERA_RADIUS = 1.0
ELEVATIONS_DEG = (0.0, -25.0, 20.0, 45.0, 65.0)
VIEWS_PER_RING = 36
VIEW_COUNT = len(ELEVATIONS_DEG) * VIEWS_PER_RING
TEST_STRIDE = 6
TEST_INDICES = tuple(range(0, VIEW_COUNT, TEST_STRIDE))
TRAIN_INDICES = tuple(index for index in range(VIEW_COUNT) if index not in TEST_INDICES)
MIN_INVDEPTH_MASK_IOU = 0.98
SUGAR_PROTOCOL = "exact_blender_cameras_colmap_triangulation_v1"
SUGAR_CAMERA_ATOL = 1e-6
SUGAR_BBOX_LOW_QUANTILE = 0.01
SUGAR_BBOX_HIGH_QUANTILE = 0.99
SUGAR_BBOX_MARGIN = 0.25
NEURALUDF_GRID_RESOLUTION = 96
NEURALUDF_SEARCH_HALF_EXTENT = 0.25
NEURALUDF_SCALE_MARGIN = 1.1
NEURALUDF_CAMERA_ATOL = 5e-6
OPENGL_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def rotation_x(angle_rad: float) -> np.ndarray:
    sine, cosine = math.sin(angle_rad), math.cos(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, sine, 0.0],
            [0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


BLENDER_TO_EFFECTIVE_GSHELL = rotation_x(-math.pi / 2.0)
GSHELL_LOADER_LEFT_ROTATION = rotation_x(math.pi / 2.0)
BLENDER_TO_SAVED_GSHELL = rotation_x(-math.pi)


def orbit_eye(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return np.array(
        [
            radius * math.cos(azimuth) * cos_elevation,
            radius * math.sin(azimuth) * cos_elevation,
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def c2w_from_eye(eye: np.ndarray) -> np.ndarray:
    forward = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def expected_frame(index: int) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    ring_index, azimuth_index = divmod(index, VIEWS_PER_RING)
    elevation_deg = ELEVATIONS_DEG[ring_index]
    azimuth_deg = -90.0 + 10.0 * azimuth_index
    blender_c2w = c2w_from_eye(orbit_eye(CAMERA_RADIUS, azimuth_deg, elevation_deg))
    saved_c2w = BLENDER_TO_SAVED_GSHELL @ blender_c2w
    effective_c2w = BLENDER_TO_EFFECTIVE_GSHELL @ blender_c2w
    metadata: dict[str, float | int] = {
        "ring_index": ring_index,
        "azimuth_index": azimuth_index,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
    }
    return saved_c2w, effective_c2w, metadata


def normalized_name(model_name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", Path(model_name).stem.lower()).strip("_")
    if not name:
        raise ValueError(f"Cannot derive a scene name from {model_name!r}")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(
    path: Path,
    source_root: Path,
    verify_hashes: bool = True,
    require_reviewed: bool = True,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("shoes"), list):
        raise ValueError(f"Unsupported manifest schema: {path}")

    records: list[dict[str, Any]] = []
    names: set[str] = set()
    models: set[str] = set()
    axis_tokens = {"X", "Y", "Z", "-X", "-Y", "-Z"}
    for record in payload["shoes"]:
        name = str(record.get("name", ""))
        model = str(record.get("model", ""))
        if name != normalized_name(model):
            raise ValueError(f"Manifest name/model mismatch: {name!r}, {model!r}")
        if name in names or model in models:
            raise ValueError(f"Duplicate manifest entry: {name}")
        if require_reviewed and record.get("reviewed") is not True:
            raise ValueError(f"Production entry is not reviewed: {name}")
        axes = record.get("source_axes")
        if not isinstance(axes, dict) or set(axes) != {"length", "width", "up"}:
            raise ValueError(f"Invalid source_axes for {name}")
        tokens = [str(axes[key]) for key in ("length", "width", "up")]
        if any(token not in axis_tokens for token in tokens):
            raise ValueError(f"Invalid source axis token for {name}: {tokens}")
        if len({token[-1] for token in tokens}) != 3:
            raise ValueError(f"Source axes are not orthogonal for {name}: {tokens}")
        selection = record.get("selection", {"mode": "all"})
        if selection.get("mode") not in {"all", "axis-side"}:
            raise ValueError(f"Invalid selection mode for {name}")
        if selection.get("mode") == "axis-side":
            if selection.get("axis") not in {"X", "Y", "Z"}:
                raise ValueError(f"Invalid selection axis for {name}")
            if selection.get("side") not in {"min", "max"}:
                raise ValueError(f"Invalid selection side for {name}")
        model_path = source_root / model
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if verify_hashes:
            actual_hash = sha256_file(model_path)
            if actual_hash != record.get("sha256"):
                raise ValueError(
                    f"GLB checksum changed for {name}: expected {record.get('sha256')}, got {actual_hash}"
                )
        names.add(name)
        models.add(model)
        records.append(record)

    source_models = {path.name for path in source_root.glob("*.glb")}
    if source_models != models:
        missing = sorted(models - source_models)
        unreviewed = sorted(source_models - models)
        raise ValueError(f"Manifest/source mismatch; missing={missing}, unreviewed={unreviewed}")
    return sorted(records, key=lambda record: str(record["name"]))


def selected_records(records: list[dict[str, Any]], shoe: str | None, all_shoes: bool) -> list[dict[str, Any]]:
    if all_shoes:
        return records
    available = {str(record["name"]): record for record in records}
    if shoe not in available:
        raise ValueError(f"Unknown shoe {shoe!r}; available: {', '.join(sorted(available))}")
    return [available[str(shoe)]]


def numbered_names(folder: str, extension: str) -> set[str]:
    return {f"img{index:03d}.{extension}" for index in range(1, VIEW_COUNT + 1)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mask_array(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"))
    return image > 127


def inverse_depth_mask_iou(mask: np.ndarray, inverse_depth: np.ndarray) -> float:
    depth_mask = np.isfinite(inverse_depth) & (inverse_depth > 0.0)
    intersection = np.logical_and(mask, depth_mask).sum()
    union = np.logical_or(mask, depth_mask).sum()
    return float(intersection / union) if union else 1.0


def ply_counts(path: Path) -> tuple[int, int]:
    vertices = faces = None
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertices = int(line.rsplit(" ", 1)[1])
            elif line.startswith("element face "):
                faces = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
    if not vertices or not faces:
        raise ValueError(f"Invalid or empty PLY mesh: {path}")
    return vertices, faces


def validate_frame_payload(
    frames: list[dict[str, Any]], expected_indices: tuple[int, ...], scene: Path
) -> list[str]:
    errors: list[str] = []
    if len(frames) != len(expected_indices):
        return [f"expected {len(expected_indices)} frames, found {len(frames)}"]
    for frame, index in zip(frames, expected_indices):
        saved_expected, effective_expected, metadata = expected_frame(index)
        expected_name = f"img{index + 1:03d}.jpg"
        if frame.get("file_path") != f"image/{expected_name}":
            errors.append(f"frame {index}: unexpected file_path {frame.get('file_path')!r}")
        if frame.get("invdepth_path") != f"invdepth/img{index + 1:03d}.npy":
            errors.append(f"frame {index}: unexpected invdepth_path")
        if not math.isclose(float(frame.get("camera_angle_x", -1.0)), math.radians(FOV_X_DEG), abs_tol=1e-10):
            errors.append(f"frame {index}: incorrect horizontal FOV")
        for key, expected_value in metadata.items():
            actual = frame.get(key)
            if isinstance(expected_value, float):
                if not math.isclose(float(actual), expected_value, abs_tol=1e-9):
                    errors.append(f"frame {index}: incorrect {key}")
            elif actual != expected_value:
                errors.append(f"frame {index}: incorrect {key}")
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append(f"frame {index}: transform_matrix is not finite 4x4")
            continue
        if not np.allclose(matrix, saved_expected, atol=1e-7):
            errors.append(f"frame {index}: saved camera does not match the deterministic orbit")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            errors.append(f"frame {index}: camera rotation is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            errors.append(f"frame {index}: camera rotation determinant is not one")
        if not math.isclose(float(np.linalg.norm(matrix[:3, 3])), CAMERA_RADIUS, abs_tol=1e-7):
            errors.append(f"frame {index}: camera radius is not one")
        loader_effective = GSHELL_LOADER_LEFT_ROTATION @ matrix
        if not np.allclose(loader_effective, effective_expected, atol=1e-7):
            errors.append(f"frame {index}: GShell loader would recover the wrong camera")
    return errors


def validate_scene(scene: Path, validate_pixels: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    expected_files = {
        "image": numbered_names("image", "jpg"),
        "mask": numbered_names("mask", "png"),
        "invdepth": numbered_names("invdepth", "npy"),
    }
    for folder, expected in expected_files.items():
        directory = scene / folder
        actual = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
        if actual != expected:
            errors.append(
                f"{folder}: missing={sorted(expected - actual)[:5]}, unexpected={sorted(actual - expected)[:5]}"
            )

    payload_specs = (
        ("transforms.json", tuple(range(VIEW_COUNT))),
        ("transforms_train.json", TRAIN_INDICES),
        ("transforms_test.json", TEST_INDICES),
    )
    for filename, indices in payload_specs:
        path = scene / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        payload = read_json(path)
        if payload.get("pose_convention") != "legacy_gshell_saved_c2w_for_fixed_loader":
            errors.append(f"{filename}: incorrect pose_convention")
        errors.extend(
            f"{filename}: {error}"
            for error in validate_frame_payload(payload.get("frames", []), indices, scene)
        )

    metadata_path = scene / "blender_canonicalization.json"
    if not metadata_path.is_file():
        errors.append("missing blender_canonicalization.json")
        metadata: dict[str, Any] = {}
    else:
        metadata = read_json(metadata_path)
        camera = metadata.get("camera_contract", {})
        if camera.get("view_count") != VIEW_COUNT:
            errors.append("canonicalization metadata has the wrong view count")
        if camera.get("radius") != CAMERA_RADIUS:
            errors.append("canonicalization metadata has the wrong radius")
        projection = metadata.get("reference_mesh_projection", {})
        if not projection.get("passed", False):
            errors.append("reference mesh projection validation did not pass")

    mesh_path = scene / "reference_mesh.ply"
    if not mesh_path.is_file():
        errors.append("missing reference_mesh.ply")
    else:
        try:
            vertex_count, face_count = ply_counts(mesh_path)
            mesh_metadata = metadata.get("reference_mesh", {})
            if vertex_count != mesh_metadata.get("vertices") or face_count != mesh_metadata.get("faces"):
                errors.append("reference mesh counts do not match canonicalization metadata")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    minimum_iou = 1.0
    if validate_pixels and not errors:
        for index in range(1, VIEW_COUNT + 1):
            image_path = scene / "image" / f"img{index:03d}.jpg"
            mask_path = scene / "mask" / f"img{index:03d}.png"
            inverse_depth_path = scene / "invdepth" / f"img{index:03d}.npy"
            with Image.open(image_path) as image:
                if image.size != RESOLUTION:
                    errors.append(f"img{index:03d}: image resolution is {image.size}")
                    continue
            mask = mask_array(mask_path)
            if mask.shape != (RESOLUTION[1], RESOLUTION[0]):
                errors.append(f"img{index:03d}: mask shape is {mask.shape}")
                continue
            if not mask.any():
                errors.append(f"img{index:03d}: mask is empty")
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                errors.append(f"img{index:03d}: mask touches an image border")
            inverse_depth = np.load(inverse_depth_path)
            if inverse_depth.shape != mask.shape or inverse_depth.dtype != np.float32:
                errors.append(f"img{index:03d}: invalid inverse-depth shape or dtype")
                continue
            iou = inverse_depth_mask_iou(mask, inverse_depth)
            minimum_iou = min(minimum_iou, iou)
            if iou < MIN_INVDEPTH_MASK_IOU:
                errors.append(f"img{index:03d}: inverse-depth/mask IoU is {iou:.6f}")

    if errors:
        raise RuntimeError(f"Validation failed for {scene}:\n" + "\n".join(errors[:100]))
    return {
        "scene": scene.name,
        "view_count": VIEW_COUNT,
        "train_count": len(TRAIN_INDICES),
        "test_count": len(TEST_INDICES),
        "minimum_invdepth_mask_iou": minimum_iou,
    }


def sugar_focal_length() -> float:
    return 0.5 * RESOLUTION[0] / math.tan(0.5 * math.radians(FOV_X_DEG))


def neuraludf_intrinsic() -> np.ndarray:
    focal = sugar_focal_length()
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[0, 2] = RESOLUTION[0] / 2.0
    intrinsic[1, 2] = RESOLUTION[1] / 2.0
    return intrinsic


def effective_neuraludf_frames(
    source_scene: Path,
) -> list[tuple[str, str, np.ndarray]]:
    payload = read_json(source_scene / "transforms_train.json")
    frames = payload.get("frames", [])
    if len(frames) != len(TRAIN_INDICES):
        raise ValueError(f"Expected {len(TRAIN_INDICES)} training poses, found {len(frames)}")
    result: list[tuple[str, str, np.ndarray]] = []
    for output_index, (source_index, frame) in enumerate(zip(TRAIN_INDICES, frames)):
        source_name = f"img{source_index + 1:03d}.jpg"
        if frame.get("file_path") != f"image/{source_name}":
            raise ValueError(f"Unexpected NeuralUDF source frame: {frame.get('file_path')!r}")
        saved_c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if saved_c2w.shape != (4, 4) or not np.isfinite(saved_c2w).all():
            raise ValueError(f"Invalid saved pose for {source_name}")
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        result.append((f"{output_index:03d}.png", source_name, effective_c2w))
    return result


def neuraludf_scale_matrix(
    source_scene: Path,
    frames: list[tuple[str, str, np.ndarray]],
    grid_resolution: int = NEURALUDF_GRID_RESOLUTION,
) -> np.ndarray:
    if grid_resolution < 8:
        raise ValueError("NeuralUDF visual-hull grid resolution must be at least eight")
    axis = np.linspace(
        -NEURALUDF_SEARCH_HALF_EXTENT,
        NEURALUDF_SEARCH_HALF_EXTENT,
        grid_resolution,
        dtype=np.float64,
    )
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    candidates = grid
    intrinsic = neuraludf_intrinsic()
    for _, source_name, effective_c2w in frames:
        mask_path = source_scene / "mask" / f"{Path(source_name).stem}.png"
        with Image.open(mask_path) as mask_handle:
            dilated = mask_handle.convert("L").filter(ImageFilter.MaxFilter(3))
            mask = np.asarray(dilated, dtype=np.uint8) >= 128
        world_to_camera = effective_to_colmap_w2c(effective_c2w)
        camera_points = candidates @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = camera_points[:, 2]
        valid = depth > 1e-8
        indices = np.flatnonzero(valid)
        if len(indices):
            projected = camera_points[indices, :2] / depth[indices, None]
            pixels = projected * np.array(
                [intrinsic[0, 0], intrinsic[1, 1]], dtype=np.float64
            ) + np.array([intrinsic[0, 2], intrinsic[1, 2]], dtype=np.float64)
            pixel_x = np.rint(pixels[:, 0]).astype(np.int64)
            pixel_y = np.rint(pixels[:, 1]).astype(np.int64)
            inside = (
                (pixel_x >= 0)
                & (pixel_x < RESOLUTION[0])
                & (pixel_y >= 0)
                & (pixel_y < RESOLUTION[1])
            )
            foreground = np.zeros(len(indices), dtype=bool)
            visible_indices = np.flatnonzero(inside)
            foreground[visible_indices] = mask[
                pixel_y[visible_indices], pixel_x[visible_indices]
            ]
            valid[indices] = foreground
        candidates = candidates[valid]
        if not len(candidates):
            raise ValueError(f"Visual hull became empty at {source_name}")

    spacing = axis[1] - axis[0]
    boundary = NEURALUDF_SEARCH_HALF_EXTENT - 0.5 * spacing
    if np.any(np.abs(candidates) >= boundary):
        raise ValueError("Visual hull touches the conservative search boundary")
    minimum = candidates.min(axis=0)
    maximum = candidates.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = float(np.linalg.norm(candidates - center, axis=1).max())
    radius = (radius + spacing) * NEURALUDF_SCALE_MARGIN
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid NeuralUDF normalization radius: {radius}")
    scale = np.diag([radius, radius, radius, 1.0]).astype(np.float64)
    scale[:3, 3] = center
    return scale


def neuraludf_camera_matrices(
    effective_c2w: np.ndarray, scale_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    intrinsic = neuraludf_intrinsic()
    world_to_camera = effective_to_colmap_w2c(effective_c2w)
    world_matrix = intrinsic @ world_to_camera
    return {
        "camera_mat": intrinsic.astype(np.float32),
        "camera_mat_inv": np.linalg.inv(intrinsic).astype(np.float32),
        "world_mat": world_matrix.astype(np.float32),
        "world_mat_inv": np.linalg.inv(world_matrix).astype(np.float32),
        "scale_mat": scale_matrix.astype(np.float32),
        "scale_mat_inv": np.linalg.inv(scale_matrix).astype(np.float32),
    }


def normalized_neuraludf_pose(
    effective_c2w: np.ndarray, scale_matrix: np.ndarray
) -> np.ndarray:
    """Express a rigid OpenCV camera pose in the normalized NeuralUDF world."""
    effective_c2w = np.asarray(effective_c2w, dtype=np.float64)
    scale_matrix = np.asarray(scale_matrix, dtype=np.float64)
    camera_pose = effective_c2w @ OPENGL_TO_OPENCV_CAMERA
    linear_scale = scale_matrix[:3, :3]
    axis_scales = np.linalg.norm(linear_scale, axis=0)
    uniform_scale = float(axis_scales.mean())
    if uniform_scale <= 0.0 or not np.allclose(
        axis_scales, uniform_scale, rtol=1e-7, atol=1e-9
    ):
        raise ValueError(f"NeuralUDF normalization must be uniform, found {axis_scales}")
    pose = camera_pose.copy()
    pose[:3, 3] = np.linalg.solve(
        linear_scale, camera_pose[:3, 3] - scale_matrix[:3, 3]
    )
    return pose


def recover_neuraludf_pose(
    camera_matrix: np.ndarray,
    world_matrix: np.ndarray,
    scale_matrix: np.ndarray,
) -> np.ndarray:
    """Recover the rigid pose consumed by NeuralUDF from IDR matrices."""
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    world_matrix = np.asarray(world_matrix, dtype=np.float64)
    scale_matrix = np.asarray(scale_matrix, dtype=np.float64)
    projective_w2c = np.linalg.inv(camera_matrix) @ world_matrix @ scale_matrix
    row_scales = np.linalg.norm(projective_w2c[:3, :3], axis=1)
    uniform_scale = float(row_scales.mean())
    if uniform_scale <= 0.0 or not np.allclose(
        row_scales, uniform_scale, rtol=1e-4, atol=1e-7
    ):
        raise ValueError(f"NeuralUDF camera scale is not uniform: {row_scales}")
    rigid_w2c = np.eye(4, dtype=np.float64)
    rigid_w2c[:3, :4] = projective_w2c[:3, :4] / uniform_scale
    return np.linalg.inv(rigid_w2c)


def write_neuraludf_scene(source_scene: Path, destination: Path) -> None:
    frames = effective_neuraludf_frames(source_scene)
    scale_matrix = neuraludf_scale_matrix(source_scene, frames)
    image_dir = destination / "image"
    mask_dir = destination / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    camera_payload: dict[str, np.ndarray] = {}
    for index, (output_name, source_name, effective_c2w) in enumerate(frames):
        with Image.open(source_scene / "image" / source_name) as image_handle:
            image_handle.convert("RGB").save(image_dir / output_name, compress_level=6)
        source_mask = source_scene / "mask" / f"{Path(source_name).stem}.png"
        with Image.open(source_mask) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(mask_dir / output_name, compress_level=6)
        matrices = neuraludf_camera_matrices(effective_c2w, scale_matrix)
        for key, matrix in matrices.items():
            camera_payload[f"{key}_{index}"] = matrix
    np.savez(destination / "cameras_sphere.npz", **camera_payload)


def validate_neuraludf_scene(scene: Path, source_scene: Path) -> dict[str, Any]:
    errors: list[str] = []
    expected_names = {f"{index:03d}.png" for index in range(len(TRAIN_INDICES))}
    image_dir = scene / "image"
    mask_dir = scene / "mask"
    actual_images = {path.name for path in image_dir.glob("*.png")}
    actual_masks = {path.name for path in mask_dir.glob("*.png")}
    if actual_images != expected_names:
        errors.append("NeuralUDF image set is incomplete")
    if actual_masks != expected_names:
        errors.append("NeuralUDF mask set is incomplete")
    camera_path = scene / "cameras_sphere.npz"
    if not camera_path.is_file():
        errors.append("missing cameras_sphere.npz")
    if (scene / "reference_mesh.ply").exists() or (scene / "invdepth").exists():
        errors.append("NeuralUDF output contains forbidden ground-truth geometry or inverse depth")
    if errors:
        raise RuntimeError(f"NeuralUDF validation failed for {scene}:\n" + "\n".join(errors))

    frames = effective_neuraludf_frames(source_scene)
    expected_scale = neuraludf_scale_matrix(source_scene, frames).astype(np.float32)
    matrix_names = (
        "camera_mat",
        "camera_mat_inv",
        "world_mat",
        "world_mat_inv",
        "scale_mat",
        "scale_mat_inv",
    )
    expected_keys = {
        f"{matrix_name}_{index}"
        for index in range(len(frames))
        for matrix_name in matrix_names
    }
    maximum_camera_error = 0.0
    maximum_rotation_error = 0.0
    maximum_ray_norm_error = 0.0
    minimum_rotation_determinant = math.inf
    minimum_foreground_fraction = 1.0
    with np.load(camera_path) as cameras:
        if set(cameras.files) != expected_keys:
            errors.append("cameras_sphere.npz has missing or unexpected matrix entries")
        for index, (output_name, source_name, effective_c2w) in enumerate(frames):
            with Image.open(image_dir / output_name) as image_handle:
                if image_handle.mode != "RGB" or image_handle.size != RESOLUTION:
                    errors.append(f"invalid NeuralUDF image {output_name}")
            with Image.open(mask_dir / output_name) as mask_handle:
                mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
            if mask.shape != (RESOLUTION[1], RESOLUTION[0]) or not np.any(mask >= 128):
                errors.append(f"invalid NeuralUDF mask {output_name}")
            else:
                minimum_foreground_fraction = min(
                    minimum_foreground_fraction, float((mask >= 128).mean())
                )
            source_mask = mask_array(source_scene / "mask" / f"{Path(source_name).stem}.png")
            if mask.shape == source_mask.shape and not np.array_equal(mask >= 128, source_mask):
                errors.append(f"NeuralUDF mask changed for {output_name}")
            expected = neuraludf_camera_matrices(effective_c2w, expected_scale)
            for matrix_name, expected_matrix in expected.items():
                key = f"{matrix_name}_{index}"
                if key not in cameras:
                    continue
                actual = cameras[key]
                error = float(np.max(np.abs(actual - expected_matrix)))
                maximum_camera_error = max(maximum_camera_error, error)
                if error > NEURALUDF_CAMERA_ATOL:
                    errors.append(f"camera matrix changed for {output_name}/{matrix_name}: {error:.3g}")
            if all(f"{name}_{index}" in cameras for name in ("camera_mat", "world_mat", "scale_mat")):
                recovered_pose = recover_neuraludf_pose(
                    cameras[f"camera_mat_{index}"],
                    cameras[f"world_mat_{index}"],
                    cameras[f"scale_mat_{index}"],
                )
                expected_pose = normalized_neuraludf_pose(effective_c2w, expected_scale)
                pose_error = float(np.max(np.abs(recovered_pose - expected_pose)))
                maximum_camera_error = max(maximum_camera_error, pose_error)
                if pose_error > NEURALUDF_CAMERA_ATOL:
                    errors.append(f"normalized camera pose changed for {output_name}: {pose_error:.3g}")

                rotation = recovered_pose[:3, :3]
                rotation_error = float(
                    np.max(np.abs(rotation.T @ rotation - np.eye(3)))
                )
                determinant = float(np.linalg.det(rotation))
                maximum_rotation_error = max(maximum_rotation_error, rotation_error)
                minimum_rotation_determinant = min(minimum_rotation_determinant, determinant)
                if rotation_error > NEURALUDF_CAMERA_ATOL or not np.isclose(
                    determinant, 1.0, rtol=1e-5, atol=NEURALUDF_CAMERA_ATOL
                ):
                    errors.append(f"non-rigid camera pose for {output_name}")

                intrinsic = cameras[f"camera_mat_{index}"][:3, :3]
                center_pixel = np.array(
                    [intrinsic[0, 2], intrinsic[1, 2], 1.0], dtype=np.float64
                )
                local_ray = np.linalg.inv(intrinsic) @ center_pixel
                local_ray /= np.linalg.norm(local_ray)
                world_ray = rotation @ local_ray
                ray_norm_error = abs(float(np.linalg.norm(world_ray)) - 1.0)
                maximum_ray_norm_error = max(maximum_ray_norm_error, ray_norm_error)
                if ray_norm_error > NEURALUDF_CAMERA_ATOL:
                    errors.append(f"non-unit camera ray for {output_name}")

                ray_origin = recovered_pose[:3, 3]
                ray_a = float(world_ray @ world_ray)
                ray_b = float(2.0 * (ray_origin @ world_ray))
                midpoint = -0.5 * ray_b / ray_a
                near = midpoint - 1.0
                far = midpoint + 1.0
                closest_radius = float(np.linalg.norm(ray_origin + midpoint * world_ray))
                near_radius = float(np.linalg.norm(ray_origin + near * world_ray))
                far_radius = float(np.linalg.norm(ray_origin + far * world_ray))
                if not (
                    near > 0.0
                    and far > near
                    and closest_radius < 1.0
                    and near_radius >= 1.0 - 1e-5
                    and far_radius >= 1.0 - 1e-5
                ):
                    errors.append(f"camera ray does not bracket the unit sphere for {output_name}")

    if errors:
        raise RuntimeError(f"NeuralUDF validation failed for {scene}:\n" + "\n".join(errors[:100]))
    return {
        "scene": scene.name,
        "train_count": len(frames),
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "normalization_center": expected_scale[:3, 3].tolist(),
        "normalization_radius": float(expected_scale[0, 0]),
        "minimum_foreground_fraction": minimum_foreground_fraction,
        "maximum_camera_matrix_error": maximum_camera_error,
        "maximum_rotation_error": maximum_rotation_error,
        "minimum_rotation_determinant": minimum_rotation_determinant,
        "maximum_ray_norm_error": maximum_ray_norm_error,
    }


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    quaternion = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def qvec_to_rotmat(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qw * qz,
                2.0 * qz * qx + 2.0 * qw * qy,
            ],
            [
                2.0 * qx * qy + 2.0 * qw * qz,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qw * qx,
            ],
            [
                2.0 * qz * qx - 2.0 * qw * qy,
                2.0 * qy * qz + 2.0 * qw * qx,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def effective_sugar_frames(source_scene: Path) -> list[tuple[str, np.ndarray]]:
    payload = read_json(source_scene / "transforms.json")
    frames = payload.get("frames", [])
    if len(frames) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} source poses, found {len(frames)}")
    result: list[tuple[str, np.ndarray]] = []
    for index, frame in enumerate(frames):
        source_name = Path(str(frame.get("file_path", ""))).name
        expected_name = f"img{index + 1:03d}.jpg"
        if source_name != expected_name:
            raise ValueError(f"Unexpected source image order: {source_name!r} != {expected_name!r}")
        saved_c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        result.append((source_name, effective_c2w))
    return result


def effective_to_colmap_w2c(effective_c2w: np.ndarray) -> np.ndarray:
    opencv_c2w = effective_c2w @ OPENGL_TO_OPENCV_CAMERA
    return np.linalg.inv(opencv_c2w)


def colmap_w2c_to_effective(world_to_camera: np.ndarray) -> np.ndarray:
    opencv_c2w = np.linalg.inv(world_to_camera)
    return opencv_c2w @ OPENGL_TO_OPENCV_CAMERA


def write_seed_colmap_model(model_dir: Path, frames: list[tuple[str, np.ndarray]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    width, height = RESOLUTION
    focal = sugar_focal_length()
    (model_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {focal:.17g} {focal:.17g} "
        f"{width / 2.0:.17g} {height / 2.0:.17g}\n",
        encoding="utf-8",
    )
    image_lines = [
        "# Image list with two lines of data per image:\n",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
    ]
    for image_id, (name, effective_c2w) in enumerate(frames, start=1):
        world_to_camera = effective_to_colmap_w2c(effective_c2w)
        quaternion = rotmat_to_qvec(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        values = [*quaternion.tolist(), *translation.tolist()]
        image_lines.append(
            f"{image_id} " + " ".join(f"{value:.17g}" for value in values) + f" 1 {name}\n\n"
        )
    (model_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
    (model_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n",
        encoding="utf-8",
    )


def parse_colmap_images(path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) != 10 or not fields[0].isdigit() or not fields[8].isdigit():
                continue
            quaternion = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            translation = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = qvec_to_rotmat(quaternion)
            world_to_camera[:3, 3] = translation
            name = fields[9]
            if name in poses:
                raise ValueError(f"Duplicate COLMAP image name: {name}")
            poses[name] = colmap_w2c_to_effective(world_to_camera)
    return poses


def parse_colmap_camera(path: Path) -> dict[str, Any]:
    records = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                records.append(stripped.split())
    if len(records) != 1:
        raise ValueError(f"Expected one COLMAP camera, found {len(records)}")
    fields = records[0]
    return {
        "id": int(fields[0]),
        "model": fields[1],
        "width": int(fields[2]),
        "height": int(fields[3]),
        "params": [float(value) for value in fields[4:]],
    }


def parse_colmap_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    errors: list[float] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 8:
                raise ValueError(f"Malformed COLMAP point record in {path}")
            points.append([float(value) for value in fields[1:4]])
            errors.append(float(fields[7]))
    return np.asarray(points, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def robust_sparse_bbox(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("Cannot compute a bounding box without finite sparse points")
    if not np.isfinite(points).all():
        raise ValueError("Sparse points contain non-finite coordinates")
    robust_min = np.quantile(points, SUGAR_BBOX_LOW_QUANTILE, axis=0)
    robust_max = np.quantile(points, SUGAR_BBOX_HIGH_QUANTILE, axis=0)
    extent = robust_max - robust_min
    minimum = robust_min - SUGAR_BBOX_MARGIN * extent
    maximum = robust_max + SUGAR_BBOX_MARGIN * extent
    inside = np.logical_and(points >= minimum, points <= maximum).all(axis=1)
    return {
        "source": "undistorted/sparse/0/points3D.txt",
        "low_quantile": SUGAR_BBOX_LOW_QUANTILE,
        "high_quantile": SUGAR_BBOX_HIGH_QUANTILE,
        "margin_fraction_per_side": SUGAR_BBOX_MARGIN,
        "robust_min": robust_min.tolist(),
        "robust_max": robust_max.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "diagonal": float(np.linalg.norm(maximum - minimum)),
        "point_count": int(len(points)),
        "points_inside": int(inside.sum()),
        "points_outside_fraction": float((~inside).mean()),
    }


def rewrite_colmap_image_extensions(source: Path, destination: Path) -> int:
    changed = 0
    output_lines: list[str] = []
    with source.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 10 and fields[0].isdigit() and fields[8].isdigit():
                if not fields[9].lower().endswith((".jpg", ".jpeg")):
                    raise ValueError(f"Unexpected COLMAP image extension: {fields[9]}")
                fields[9] = str(Path(fields[9]).with_suffix(".png"))
                line = " ".join(fields) + "\n"
                changed += 1
            output_lines.append(line)
    destination.write_text("".join(output_lines), encoding="utf-8")
    return changed


def run_colmap_stage(
    name: str,
    command: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLMAP stage {name} failed with exit code {completed.returncode}:\n{completed.stdout}"
        )
    return {
        "name": name,
        "command": command,
        "elapsed_seconds": time.time() - started,
        "last_output_lines": completed.stdout.splitlines()[-20:],
    }


def write_sugar_images(source_scene: Path, destination: Path) -> dict[str, float | int]:
    output_images = destination / "undistorted" / "images"
    output_masks = destination / "undistorted" / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    foreground_fractions: list[float] = []
    for index in range(1, VIEW_COUNT + 1):
        basename = f"img{index:03d}"
        with Image.open(source_scene / "image" / f"{basename}.jpg") as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(source_scene / "mask" / f"{basename}.png") as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape or rgb.shape[1::-1] != RESOLUTION:
            raise ValueError(f"RGB/mask dimensions do not match the camera for {basename}")
        if not np.any(mask):
            raise ValueError(f"Mask is empty for {basename}")
        rgba = np.dstack((rgb, mask))
        Image.fromarray(rgba, mode="RGBA").save(output_images / f"{basename}.png", compress_level=6)
        Image.fromarray(mask, mode="L").save(output_masks / f"{basename}.png", compress_level=6)
        foreground_fractions.append(float((mask > 0).mean()))
    return {
        "count": VIEW_COUNT,
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "foreground_fraction_min": float(np.min(foreground_fractions)),
        "foreground_fraction_mean": float(np.mean(foreground_fractions)),
        "foreground_fraction_max": float(np.max(foreground_fractions)),
    }


def install_transactionally(temporary: Path, target: Path, overwrite: bool) -> None:
    backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def validate_sugar_scene(scene: Path, source_scene: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = scene / "masked_colmap_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != SUGAR_PROTOCOL:
        errors.append("incorrect SuGaR protocol")
    if Path(str(manifest.get("source_scene", ""))).resolve() != source_scene.resolve():
        errors.append("source scene path does not match")
    source_transforms = source_scene / "transforms.json"
    if not source_transforms.is_file():
        errors.append("source transforms.json is missing")
    elif manifest.get("source_transforms_sha256") != sha256_file(source_transforms):
        errors.append("source transforms.json changed after SuGaR preparation")

    image_dir = scene / "undistorted" / "images"
    mask_dir = scene / "undistorted" / "masks"
    sparse_dir = scene / "undistorted" / "sparse" / "0"
    expected_pngs = numbered_names("images", "png")
    actual_images = {path.name for path in image_dir.glob("*.png")}
    actual_masks = {path.name for path in mask_dir.glob("*.png")}
    if actual_images != expected_pngs:
        errors.append("SuGaR RGBA image set is incomplete")
    if actual_masks != expected_pngs:
        errors.append("SuGaR mask set is incomplete")
    for name in ("cameras.txt", "images.txt", "points3D.txt", "cameras.bin", "images.bin", "points3D.bin"):
        path = sparse_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty sparse/0/{name}")

    expected_frames = effective_sugar_frames(source_scene)
    if not errors:
        camera = parse_colmap_camera(sparse_dir / "cameras.txt")
        focal = sugar_focal_length()
        expected_params = np.asarray(
            [focal, focal, RESOLUTION[0] / 2.0, RESOLUTION[1] / 2.0], dtype=np.float64
        )
        if camera["model"] != "PINHOLE" or (camera["width"], camera["height"]) != RESOLUTION:
            errors.append("COLMAP camera model or resolution changed")
        elif not np.allclose(camera["params"], expected_params, atol=1e-7):
            errors.append("COLMAP intrinsics changed")

        colmap_poses = parse_colmap_images(sparse_dir / "images.txt")
        if len(colmap_poses) != VIEW_COUNT:
            errors.append(f"registered camera count {len(colmap_poses)} != {VIEW_COUNT}")
        maximum_pose_error = 0.0
        for jpg_name, expected_pose in expected_frames:
            png_name = str(Path(jpg_name).with_suffix(".png"))
            actual_pose = colmap_poses.get(png_name)
            if actual_pose is None:
                errors.append(f"missing COLMAP camera {png_name}")
                continue
            pose_error = float(np.max(np.abs(actual_pose - expected_pose)))
            maximum_pose_error = max(maximum_pose_error, pose_error)
            if pose_error > SUGAR_CAMERA_ATOL:
                errors.append(f"COLMAP camera changed for {png_name}: {pose_error:.3g}")
            if not math.isclose(float(np.linalg.norm(actual_pose[:3, 3])), CAMERA_RADIUS, abs_tol=1e-6):
                errors.append(f"COLMAP camera radius changed for {png_name}")

        points, reprojection_errors = parse_colmap_points(sparse_dir / "points3D.txt")
        if not len(points):
            errors.append("COLMAP did not triangulate any sparse points")
        elif not np.isfinite(points).all() or not np.isfinite(reprojection_errors).all():
            errors.append("sparse points or reprojection errors are non-finite")
        else:
            computed_box = robust_sparse_bbox(points)
            recorded_box = manifest.get("foreground_bbox", {})
            if not np.allclose(recorded_box.get("min", []), computed_box["min"], atol=1e-9):
                errors.append("recorded foreground bounding-box minimum changed")
            if not np.allclose(recorded_box.get("max", []), computed_box["max"], atol=1e-9):
                errors.append("recorded foreground bounding-box maximum changed")
            if computed_box["points_outside_fraction"] > 0.05 + 1e-12:
                errors.append("foreground bounding box contains fewer than 95% of sparse points")

        for index in range(1, VIEW_COUNT + 1):
            basename = f"img{index:03d}.png"
            with Image.open(image_dir / basename) as rgba_handle, Image.open(mask_dir / basename) as mask_handle:
                if rgba_handle.mode != "RGBA" or rgba_handle.size != RESOLUTION:
                    errors.append(f"invalid RGBA image {basename}")
                    continue
                alpha = np.asarray(rgba_handle.getchannel("A"), dtype=np.uint8)
                mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
                if not np.array_equal(alpha, mask):
                    errors.append(f"RGBA alpha and mask differ for {basename}")
    else:
        maximum_pose_error = float("inf")
        points = np.empty((0, 3), dtype=np.float64)
        reprojection_errors = np.empty((0,), dtype=np.float64)

    if (scene / "reference_mesh.ply").exists() or (scene / "invdepth").exists():
        errors.append("SuGaR output contains forbidden ground-truth geometry or inverse depth")
    if errors:
        raise RuntimeError(f"SuGaR validation failed for {scene}:\n" + "\n".join(errors[:100]))
    return {
        "scene": scene.name,
        "view_count": VIEW_COUNT,
        "sparse_point_count": int(len(points)),
        "mean_reprojection_error": float(np.mean(reprojection_errors)),
        "maximum_camera_matrix_error": maximum_pose_error,
    }


def prepare_sugar_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    gpu_pool: queue.Queue[int],
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.resolve() / name
    target = args.output_root.resolve() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_sugar_scene(target, source_scene)
        print(f"[skip-sugar] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve()))
    workspace = temporary / ".workspace"
    seed_model = workspace / "seed_model"
    triangulated_model = workspace / "triangulated_model"
    triangulated_text = workspace / "triangulated_text"
    colmap_masks = workspace / "colmap_masks"
    database = workspace / "database.db"
    gpu = gpu_pool.get()
    started = time.time()
    try:
        print(f"[prepare-sugar] {name} on physical GPU {gpu}", flush=True)
        frames = effective_sugar_frames(source_scene)
        write_seed_colmap_model(seed_model, frames)
        colmap_masks.mkdir(parents=True)
        for image_name, _ in frames:
            source_mask = source_scene / "mask" / f"{Path(image_name).stem}.png"
            shutil.copy2(source_mask, colmap_masks / f"{image_name}.png")

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        colmap = str(args.colmap_bin.resolve())
        focal = sugar_focal_length()
        camera_params = (
            f"{focal:.17g},{focal:.17g},{RESOLUTION[0] / 2.0:.17g},"
            f"{RESOLUTION[1] / 2.0:.17g}"
        )
        stages = [
            run_colmap_stage(
                "feature_extractor",
                [
                    colmap,
                    "feature_extractor",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(source_scene / "image"),
                    "--ImageReader.mask_path",
                    str(colmap_masks),
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_params",
                    camera_params,
                    "--FeatureExtraction.use_gpu",
                    "1",
                    "--FeatureExtraction.gpu_index",
                    "0",
                    "--SiftExtraction.max_num_features",
                    "16384",
                    "--SiftExtraction.peak_threshold",
                    "0.002",
                    "--SiftExtraction.edge_threshold",
                    "20",
                    "--default_random_seed",
                    "0",
                ],
                environment,
            ),
            run_colmap_stage(
                "exhaustive_matcher",
                [
                    colmap,
                    "exhaustive_matcher",
                    "--database_path",
                    str(database),
                    "--FeatureMatching.use_gpu",
                    "1",
                    "--FeatureMatching.gpu_index",
                    "0",
                    "--FeatureMatching.guided_matching",
                    "1",
                    "--SiftMatching.max_ratio",
                    "0.9",
                    "--TwoViewGeometry.min_num_inliers",
                    "8",
                    "--default_random_seed",
                    "0",
                ],
                environment,
            ),
        ]
        triangulated_model.mkdir(parents=True)
        stages.append(
            run_colmap_stage(
                "point_triangulator",
                [
                    colmap,
                    "point_triangulator",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(source_scene / "image"),
                    "--input_path",
                    str(seed_model),
                    "--output_path",
                    str(triangulated_model),
                    "--clear_points",
                    "1",
                    "--refine_intrinsics",
                    "0",
                    "--Mapper.fix_existing_frames",
                    "1",
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_principal_point",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                    "--default_random_seed",
                    "0",
                ],
                environment,
            )
        )
        triangulated_text.mkdir(parents=True)
        stages.append(
            run_colmap_stage(
                "model_to_text",
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    str(triangulated_model),
                    "--output_path",
                    str(triangulated_text),
                    "--output_type",
                    "TXT",
                ],
                environment,
            )
        )

        points, reprojection_errors = parse_colmap_points(triangulated_text / "points3D.txt")
        if not len(points):
            raise RuntimeError("COLMAP did not triangulate any sparse points")
        if not np.isfinite(reprojection_errors).all():
            raise RuntimeError("COLMAP produced non-finite reprojection errors")
        bbox = robust_sparse_bbox(points)
        image_info = write_sugar_images(source_scene, temporary)
        sparse_output = temporary / "undistorted" / "sparse" / "0"
        sparse_output.mkdir(parents=True)
        shutil.copy2(triangulated_text / "cameras.txt", sparse_output / "cameras.txt")
        shutil.copy2(triangulated_text / "points3D.txt", sparse_output / "points3D.txt")
        changed = rewrite_colmap_image_extensions(
            triangulated_text / "images.txt", sparse_output / "images.txt"
        )
        if changed != VIEW_COUNT:
            raise RuntimeError(f"Expected to rewrite {VIEW_COUNT} COLMAP image names, changed {changed}")

        binary_model = workspace / "final_binary"
        binary_model.mkdir(parents=True)
        stages.append(
            run_colmap_stage(
                "model_to_binary",
                [
                    colmap,
                    "model_converter",
                    "--input_path",
                    str(sparse_output),
                    "--output_path",
                    str(binary_model),
                    "--output_type",
                    "BIN",
                ],
                environment,
            )
        )
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(binary_model / filename, sparse_output / filename)

        manifest = {
            "version": 1,
            "protocol": SUGAR_PROTOCOL,
            "scene": name,
            "source_scene": str(source_scene),
            "source_transforms_sha256": sha256_file(source_scene / "transforms.json"),
            "camera": {
                "source": "validated effective GShell c2w derived from Blender transforms.json",
                "colmap_convention": "world_to_camera_opencv_x_right_y_down_z_forward",
                "model": "PINHOLE",
                "width": RESOLUTION[0],
                "height": RESOLUTION[1],
                "focal_x": focal,
                "focal_y": focal,
                "principal_x": RESOLUTION[0] / 2.0,
                "principal_y": RESOLUTION[1] / 2.0,
                "view_count": VIEW_COUNT,
                "poses_are_fixed": True,
            },
            "images": image_info,
            "sparse_points": {
                "method": "masked_sift_exhaustive_matching_fixed_pose_triangulation",
                "count": int(len(points)),
                "mean_reprojection_error": float(np.mean(reprojection_errors)),
                "maximum_reprojection_error": float(np.max(reprojection_errors)),
            },
            "foreground_bbox": bbox,
            "training_contract": {
                "use_masks": True,
                "white_background": True,
                "uses_blender_camera_transforms": True,
                "uses_inverse_depth": False,
                "uses_ground_truth_mesh": False,
            },
            "colmap_stages": stages,
            "elapsed_seconds": time.time() - started,
        }
        (temporary / "masked_colmap_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        shutil.rmtree(workspace)
        result = validate_sugar_scene(temporary, source_scene)
        install_transactionally(temporary, target, args.overwrite)
        print(f"[ok-sugar] {name}: {result['sparse_point_count']} sparse points", flush=True)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        gpu_pool.put(gpu)


def run_prepare_sugar(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(prepare_sugar_record, args, record, gpu_pool): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[failed-sugar] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("SuGaR preparation failures:\n" + "\n".join(failures))


def run_validate_sugar(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    for record in records:
        name = str(record["name"])
        result = validate_sugar_scene(
            args.output_root.resolve() / name,
            args.input_root.resolve() / name,
        )
        print(json.dumps(result, sort_keys=True))


def prepare_neuraludf_record(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.resolve() / name
    target = args.output_root.resolve() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_neuraludf_scene(target, source_scene)
        print(f"[skip-neuraludf] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve()))
    try:
        print(f"[prepare-neuraludf] {name}", flush=True)
        write_neuraludf_scene(source_scene, temporary)
        result = validate_neuraludf_scene(temporary, source_scene)
        install_transactionally(temporary, target, args.overwrite)
        print(
            f"[ok-neuraludf] {name}: {result['train_count']} exact-camera training views",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_prepare_neuraludf(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    failures: list[str] = []
    for record in records:
        name = str(record["name"])
        try:
            prepare_neuraludf_record(args, record)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"[failed-neuraludf] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("NeuralUDF preparation failures:\n" + "\n".join(failures))


def run_validate_neuraludf(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    for record in records:
        name = str(record["name"])
        result = validate_neuraludf_scene(
            args.output_root.resolve() / name,
            args.input_root.resolve() / name,
        )
        print(json.dumps(result, sort_keys=True))


def blender_command(
    args: argparse.Namespace,
    action: str,
    record: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        str(args.blender.resolve()),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(SCRIPT_DIR / "blender_worker.py"),
        "--",
        action,
        "--manifest",
        str(args.manifest.resolve()),
        "--source-root",
        str(args.source_root.resolve()),
        "--shoe",
        str(record["name"]),
        "--output",
        str(output),
    ]


def parse_gpus(args: argparse.Namespace) -> list[int]:
    if getattr(args, "gpu", None) is not None:
        return [int(args.gpu)]
    raw = getattr(args, "gpus", None)
    if raw:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
        if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError(f"Invalid GPU list: {raw!r}")
        return values
    return [0]


def run_blender(command: list[str], gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, check=True, env=environment)


def build_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    gpu_pool: queue.Queue[int],
) -> dict[str, Any]:
    name = str(record["name"])
    target = args.output_root.resolve() / name
    if target.exists() and not args.overwrite:
        validation = validate_scene(target)
        print(f"[skip] {name}: existing scene is valid", flush=True)
        return validation
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve()))
    gpu = gpu_pool.get()
    try:
        print(f"[build] {name} on physical GPU {gpu}", flush=True)
        run_blender(blender_command(args, "build", record, temporary), gpu)
        validation = validate_scene(temporary)
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
        print(f"[ok] {name}: {VIEW_COUNT} exact Blender views", flush=True)
        return validation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        gpu_pool.put(gpu)


def run_build(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(build_record, args, record, gpu_pool): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[failed] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Build failures:\n" + "\n".join(failures))


def run_audit(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(
            args.manifest.resolve(),
            args.source_root.resolve(),
            require_reviewed=False,
        ),
        args.shoe,
        args.all,
    )
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def audit_record(record: dict[str, Any]) -> None:
        gpu = gpu_pool.get()
        try:
            target = args.output_dir.resolve() / str(record["name"])
            if target.exists():
                shutil.rmtree(target)
            run_blender(blender_command(args, "audit", record, target), gpu)
            print(f"[audit] {record['name']} -> {target}", flush=True)
        finally:
            gpu_pool.put(gpu)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(audit_record, record): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[audit-failed] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("Audit failures:\n" + "\n".join(failures))


def run_validate(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()), args.shoe, args.all
    )
    for record in records:
        result = validate_scene(args.output_root.resolve() / str(record["name"]))
        print(json.dumps(result, sort_keys=True))


def add_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shoe")
    selection.add_argument("--all", action="store_true")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)


def add_gpus(parser: argparse.ArgumentParser) -> None:
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int)
    devices.add_argument("--gpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Render temporary semantic audit views.")
    add_common(audit)
    add_selection(audit)
    add_gpus(audit)
    audit.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/golden_set_evaluation_blender_audit")
    )
    audit.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)

    build = commands.add_parser("build", help="Build transactional GShell-ready scenes.")
    add_common(build)
    add_selection(build)
    add_gpus(build)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    build.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate", help="Validate completed scenes.")
    add_common(validate)
    add_selection(validate)
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    prepare_sugar = commands.add_parser(
        "prepare-sugar", help="Create SuGaR data with exact cameras and triangulated sparse points."
    )
    add_common(prepare_sugar)
    add_selection(prepare_sugar)
    add_gpus(prepare_sugar)
    prepare_sugar.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_sugar.add_argument("--output-root", type=Path, default=DEFAULT_SUGAR_OUTPUT_ROOT)
    prepare_sugar.add_argument("--colmap-bin", type=Path, default=DEFAULT_COLMAP)
    prepare_sugar.add_argument("--overwrite", action="store_true")

    validate_sugar = commands.add_parser("validate-sugar", help="Validate SuGaR-ready scenes.")
    add_common(validate_sugar)
    add_selection(validate_sugar)
    validate_sugar.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_sugar.add_argument("--output-root", type=Path, default=DEFAULT_SUGAR_OUTPUT_ROOT)

    prepare_neuraludf = commands.add_parser(
        "prepare-neuraludf", help="Create NeuralUDF data from exact Blender cameras."
    )
    add_common(prepare_neuraludf)
    add_selection(prepare_neuraludf)
    prepare_neuraludf.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_neuraludf.add_argument(
        "--output-root", type=Path, default=DEFAULT_NEURALUDF_OUTPUT_ROOT
    )
    prepare_neuraludf.add_argument("--overwrite", action="store_true")

    validate_neuraludf = commands.add_parser(
        "validate-neuraludf", help="Validate NeuralUDF-ready scenes."
    )
    add_common(validate_neuraludf)
    add_selection(validate_neuraludf)
    validate_neuraludf.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_neuraludf.add_argument(
        "--output-root", type=Path, default=DEFAULT_NEURALUDF_OUTPUT_ROOT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "audit": run_audit,
        "build": run_build,
        "validate": run_validate,
        "prepare-sugar": run_prepare_sugar,
        "validate-sugar": run_validate_sugar,
        "prepare-neuraludf": run_prepare_neuraludf,
        "validate-neuraludf": run_validate_neuraludf,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
