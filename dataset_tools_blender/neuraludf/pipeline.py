"""Prepare and validate NeuralUDF-ready scenes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..core import (
    GSHELL_LOADER_LEFT_ROTATION,
    NEURALUDF_CAMERA_ATOL,
    NEURALUDF_GRID_RESOLUTION,
    NEURALUDF_SCALE_MARGIN,
    NEURALUDF_SEARCH_HALF_EXTENT,
    OPENGL_TO_OPENCV_CAMERA,
    RESOLUTION,
    TRAIN_INDICES,
    effective_to_colmap_w2c,
    install_transactionally,
    load_manifest,
    mask_array,
    read_json,
    selected_records,
    sugar_focal_length,
    validate_scene,
)


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
                    errors.append(
                        f"camera matrix changed for {output_name}/{matrix_name}: {error:.3g}"
                    )
            if all(
                f"{name}_{index}" in cameras
                for name in ("camera_mat", "world_mat", "scale_mat")
            ):
                recovered_pose = recover_neuraludf_pose(
                    cameras[f"camera_mat_{index}"],
                    cameras[f"world_mat_{index}"],
                    cameras[f"scale_mat_{index}"],
                )
                expected_pose = normalized_neuraludf_pose(effective_c2w, expected_scale)
                pose_error = float(np.max(np.abs(recovered_pose - expected_pose)))
                maximum_camera_error = max(maximum_camera_error, pose_error)
                if pose_error > NEURALUDF_CAMERA_ATOL:
                    errors.append(
                        f"normalized camera pose changed for {output_name}: {pose_error:.3g}"
                    )

                rotation = recovered_pose[:3, :3]
                rotation_error = float(
                    np.max(np.abs(rotation.T @ rotation - np.eye(3)))
                )
                determinant = float(np.linalg.det(rotation))
                maximum_rotation_error = max(maximum_rotation_error, rotation_error)
                minimum_rotation_determinant = min(
                    minimum_rotation_determinant, determinant
                )
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
                    errors.append(
                        f"camera ray does not bracket the unit sphere for {output_name}"
                    )

    if errors:
        raise RuntimeError(
            f"NeuralUDF validation failed for {scene}:\n" + "\n".join(errors[:100])
        )
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


def prepare_neuraludf_record(
    args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.resolve() / name
    target = args.output_root.resolve() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_neuraludf_scene(target, source_scene)
        print(f"[skip-neuraludf] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve())
    )
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
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
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
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        name = str(record["name"])
        result = validate_neuraludf_scene(
            args.output_root.resolve() / name,
            args.input_root.resolve() / name,
        )
        print(json.dumps(result, sort_keys=True))
