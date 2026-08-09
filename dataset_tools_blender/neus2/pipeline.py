"""Prepare and validate NeuS2-ready scenes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core import (
    FOV_X_DEG,
    GSHELL_LOADER_LEFT_ROTATION,
    NEURALUDF_SCALE_MARGIN,
    NEUS2_CAMERA_ATOL,
    NEUS2_PROTOCOL,
    OPENGL_TO_OPENCV_CAMERA,
    RESOLUTION,
    TEST_INDICES,
    TEST_STRIDE,
    TRAIN_INDICES,
    TURNTABLE_INDICES,
    TURNTABLE_TEST_INDICES,
    TURNTABLE_TRAIN_INDICES,
    VIEW_COUNT,
    install_transactionally,
    load_manifest,
    mask_array,
    numbered_names,
    read_json,
    selected_records,
    source_manifest_fields,
    validate_scene,
    validate_source_manifest,
)
from ..neuraludf.pipeline import neuraludf_intrinsic, neuraludf_scale_matrix


DEFAULT_NEUS2_TURNTABLE_OUTPUT_ROOT = Path(
    "/storage/Abhinay/home_ab5298/dataset/datasets/processed/"
    "neus2/golden_set_evaluation_turntable"
)
NEUS2_TURNTABLE_PROTOCOL = (
    "exact_blender_cameras_turntable_36_visual_hull_normalization_v1"
)
def effective_neus2_frames(
    source_scene: Path,
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Return exact effective and OpenCV camera poses for all Blender views."""
    payload = read_json(source_scene / "transforms.json")
    frames = payload.get("frames", [])
    if len(frames) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} source poses, found {len(frames)}")

    result: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for index, frame in enumerate(frames):
        source_name = f"img{index + 1:03d}.jpg"
        if frame.get("file_path") != f"image/{source_name}":
            raise ValueError(f"Unexpected NeuS2 source frame: {frame.get('file_path')!r}")
        saved_c2w = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if saved_c2w.shape != (4, 4) or not np.isfinite(saved_c2w).all():
            raise ValueError(f"Invalid saved pose for {source_name}")
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        opencv_c2w = effective_c2w @ OPENGL_TO_OPENCV_CAMERA
        result.append((index, source_name, effective_c2w, opencv_c2w))
    return result


def neus2_scale_offset(scale_matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Map a conservative world-space bounding sphere into NeuS2's unit cube."""
    scale_matrix = np.asarray(scale_matrix, dtype=np.float64)
    axis_scales = np.linalg.norm(scale_matrix[:3, :3], axis=0)
    radius = float(axis_scales.mean())
    if radius <= 0.0 or not np.allclose(
        axis_scales, radius, rtol=1e-7, atol=1e-9
    ):
        raise ValueError(
            f"NeuS2 normalization sphere must be uniform, found {axis_scales}"
        )
    center = scale_matrix[:3, 3]
    scale = 0.5 / radius
    offset = np.full(3, 0.5, dtype=np.float64) - scale * center
    if not math.isfinite(scale) or not np.isfinite(offset).all():
        raise ValueError("NeuS2 normalization contains non-finite values")
    return scale, offset


def neus2_normalization(
    source_scene: Path,
    frames: list[tuple[int, str, np.ndarray, np.ndarray]],
    indices: tuple[int, ...] = TRAIN_INDICES,
) -> tuple[np.ndarray, float, np.ndarray]:
    training_frames = [
        (f"img{index + 1:03d}.png", source_name, effective_c2w)
        for index, source_name, effective_c2w, _ in frames
        if index in indices
    ]
    if len(training_frames) != len(indices):
        raise ValueError("NeuS2 training split is incomplete")
    sphere = neuraludf_scale_matrix(source_scene, training_frames)
    scale, offset = neus2_scale_offset(sphere)
    return sphere, scale, offset


def neus2_intrinsic() -> np.ndarray:
    return neuraludf_intrinsic()


def neus2_transform_payload(
    frames: list[tuple[int, str, np.ndarray, np.ndarray]],
    indices: tuple[int, ...],
    scale: float,
    offset: np.ndarray,
) -> dict[str, Any]:
    by_index = {
        index: (source_name, opencv_c2w)
        for index, source_name, _, opencv_c2w in frames
    }
    intrinsic = neus2_intrinsic()
    output_frames = []
    for index in indices:
        source_name, opencv_c2w = by_index[index]
        output_frames.append(
            {
                "file_path": f"images/{Path(source_name).stem}.png",
                "transform_matrix": opencv_c2w.tolist(),
                "intrinsic_matrix": intrinsic.tolist(),
                "source_view_index": index,
            }
        )
    return {
        "w": RESOLUTION[0],
        "h": RESOLUTION[1],
        "camera_angle_x": math.radians(FOV_X_DEG),
        "aabb_scale": 1.0,
        "scale": float(scale),
        "offset": np.asarray(offset, dtype=np.float64).tolist(),
        "from_na": True,
        "frames": output_frames,
    }


def write_neus2_scene(source_scene: Path, destination: Path) -> None:
    frames = effective_neus2_frames(source_scene)
    sphere, scale, offset = neus2_normalization(source_scene, frames)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    foreground_fractions: list[float] = []
    for index, source_name, _, _ in frames:
        basename = Path(source_name).stem
        with Image.open(source_scene / "image" / source_name) as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(
            source_scene / "mask" / f"{basename}.png"
        ) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape or rgb.shape[1::-1] != RESOLUTION:
            raise ValueError(
                f"RGB/mask dimensions do not match the camera for {basename}"
            )
        if not np.any(mask):
            raise ValueError(f"Mask is empty for {basename}")
        Image.fromarray(np.dstack((rgb, mask)), mode="RGBA").save(
            image_dir / f"{basename}.png",
            compress_level=6,
        )
        foreground_fractions.append(float((mask > 0).mean()))

    train_payload = neus2_transform_payload(
        frames, TRAIN_INDICES, scale, offset
    )
    test_payload = neus2_transform_payload(frames, TEST_INDICES, scale, offset)
    (destination / "transform_train.json").write_text(
        json.dumps(train_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "transform_test.json").write_text(
        json.dumps(test_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **source_manifest_fields(source_scene),
        "protocol": NEUS2_PROTOCOL,
        "camera": {
            "source_pose": "Rx(+90deg) @ legacy_gshell_saved_c2w",
            "saved_pose": "effective_opengl_c2w @ diag(1,-1,-1,1)",
            "saved_pose_convention": "opencv_c2w_x_right_y_down_z_forward",
            "from_na": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "horizontal_fov_degrees": FOV_X_DEG,
            "focal_x": float(neus2_intrinsic()[0, 0]),
            "focal_y": float(neus2_intrinsic()[1, 1]),
            "principal_x": RESOLUTION[0] / 2.0,
            "principal_y": RESOLUTION[1] / 2.0,
        },
        "split": {
            "train_count": len(TRAIN_INDICES),
            "test_count": len(TEST_INDICES),
            "test_stride": TEST_STRIDE,
            "train_source_indices": list(TRAIN_INDICES),
            "test_source_indices": list(TEST_INDICES),
        },
        "normalization": {
            "source": "visual_hull_from_training_masks_and_exact_cameras",
            "sphere_center": sphere[:3, 3].tolist(),
            "sphere_radius": float(sphere[0, 0]),
            "sphere_margin": NEURALUDF_SCALE_MARGIN,
            "ngp_scale": float(scale),
            "ngp_offset": offset.tolist(),
            "mapped_center": (scale * sphere[:3, 3] + offset).tolist(),
            "mapped_radius": float(scale * sphere[0, 0]),
        },
        "images": {
            "count": VIEW_COUNT,
            "format": "rgba_png",
            "alpha_source": "binary_blender_mask",
            "foreground_fraction_min": float(np.min(foreground_fractions)),
            "foreground_fraction_mean": float(np.mean(foreground_fractions)),
            "foreground_fraction_max": float(np.max(foreground_fractions)),
        },
        "training_contract": {
            "uses_rgb": True,
            "uses_masks_as_alpha": True,
            "uses_exact_blender_cameras": True,
            "uses_inverse_depth": False,
            "uses_ground_truth_mesh": False,
        },
    }
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def write_neus2_turntable_scene(
    source_scene: Path, destination: Path
) -> None:
    """Write the 36-view level orbit used by the real turntable dataset."""
    frames = effective_neus2_frames(source_scene)
    sphere, scale, offset = neus2_normalization(
        source_scene,
        frames,
        TURNTABLE_TRAIN_INDICES,
    )
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    foreground_fractions: list[float] = []
    for index, source_name, _, _ in frames:
        if index not in TURNTABLE_INDICES:
            continue
        basename = Path(source_name).stem
        with Image.open(source_scene / "image" / source_name) as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        with Image.open(
            source_scene / "mask" / f"{basename}.png"
        ) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape or rgb.shape[1::-1] != RESOLUTION:
            raise ValueError(
                f"RGB/mask dimensions do not match the camera for {basename}"
            )
        if not np.any(mask):
            raise ValueError(f"Mask is empty for {basename}")
        Image.fromarray(np.dstack((rgb, mask)), mode="RGBA").save(
            image_dir / f"{basename}.png",
            compress_level=6,
        )
        foreground_fractions.append(float((mask > 0).mean()))

    payloads = {
        "transform.json": neus2_transform_payload(
            frames, TURNTABLE_INDICES, scale, offset
        ),
        "transform_train.json": neus2_transform_payload(
            frames, TURNTABLE_TRAIN_INDICES, scale, offset
        ),
        "transform_test.json": neus2_transform_payload(
            frames, TURNTABLE_TEST_INDICES, scale, offset
        ),
    }
    for filename, payload in payloads.items():
        (destination / filename).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        **source_manifest_fields(source_scene),
        "protocol": NEUS2_TURNTABLE_PROTOCOL,
        "camera": {
            "source_pose": "Rx(+90deg) @ legacy_gshell_saved_c2w",
            "saved_pose": "effective_opengl_c2w @ diag(1,-1,-1,1)",
            "saved_pose_convention": "opencv_c2w_x_right_y_down_z_forward",
            "from_na": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "horizontal_fov_degrees": FOV_X_DEG,
            "elevation_degrees": 0.0,
            "azimuth_step_degrees": 10.0,
        },
        "split": {
            "all_count": len(TURNTABLE_INDICES),
            "train_count": len(TURNTABLE_TRAIN_INDICES),
            "test_count": len(TURNTABLE_TEST_INDICES),
            "test_stride": TEST_STRIDE,
            "all_source_indices": list(TURNTABLE_INDICES),
            "train_source_indices": list(TURNTABLE_TRAIN_INDICES),
            "test_source_indices": list(TURNTABLE_TEST_INDICES),
        },
        "normalization": {
            "source": "visual_hull_from_turntable_training_masks_and_exact_cameras",
            "sphere_center": sphere[:3, 3].tolist(),
            "sphere_radius": float(sphere[0, 0]),
            "sphere_margin": NEURALUDF_SCALE_MARGIN,
            "ngp_scale": float(scale),
            "ngp_offset": offset.tolist(),
        },
        "images": {
            "count": len(TURNTABLE_INDICES),
            "format": "rgba_png",
            "alpha_source": "binary_blender_mask",
            "foreground_fraction_min": float(np.min(foreground_fractions)),
            "foreground_fraction_mean": float(np.mean(foreground_fractions)),
            "foreground_fraction_max": float(np.max(foreground_fractions)),
        },
        "training_contract": {
            "uses_rgb": True,
            "uses_masks_as_alpha": True,
            "uses_exact_blender_cameras": True,
            "uses_only_level_turntable_views": True,
            "uses_inverse_depth": False,
            "uses_ground_truth_mesh": False,
        },
    }
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_neus2_scene(
    scene: Path, source_scene: Path
) -> dict[str, Any]:
    errors: list[str] = []
    expected_images = numbered_names("images", "png")
    image_dir = scene / "images"
    actual_images = {path.name for path in image_dir.glob("*.png")}
    if actual_images != expected_images:
        errors.append(
            "NeuS2 image set is incomplete: "
            f"missing={sorted(expected_images - actual_images)[:5]}, "
            f"unexpected={sorted(actual_images - expected_images)[:5]}"
        )
    forbidden = [
        path
        for path in (
            scene / "reference_mesh.ply",
            scene / "invdepth",
            scene / "cameras_sphere.npz",
        )
        if path.exists()
    ]
    if forbidden:
        errors.append(f"NeuS2 output contains forbidden assets: {forbidden}")

    manifest_path = scene / "conversion_manifest.json"
    train_path = scene / "transform_train.json"
    test_path = scene / "transform_test.json"
    for path in (manifest_path, train_path, test_path):
        if not path.is_file():
            errors.append(f"missing {path.name}")
    if errors:
        raise RuntimeError(
            f"NeuS2 validation failed for {scene}:\n" + "\n".join(errors)
        )

    manifest = read_json(manifest_path)
    if manifest.get("protocol") != NEUS2_PROTOCOL:
        errors.append("incorrect NeuS2 conversion protocol")
    errors.extend(validate_source_manifest(manifest, source_scene))

    frames = effective_neus2_frames(source_scene)
    sphere, expected_scale, expected_offset = neus2_normalization(
        source_scene, frames
    )
    normalization = manifest.get("normalization", {})
    if not math.isclose(
        float(normalization.get("sphere_radius", -1.0)),
        float(sphere[0, 0]),
        rel_tol=0.0,
        abs_tol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 visual-hull radius changed")
    if not np.allclose(
        normalization.get("sphere_center", []),
        sphere[:3, 3],
        atol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 visual-hull center changed")
    if not math.isclose(
        float(normalization.get("ngp_scale", -1.0)),
        expected_scale,
        rel_tol=0.0,
        abs_tol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 scale changed")
    if not np.allclose(
        normalization.get("ngp_offset", []),
        expected_offset,
        atol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 offset changed")
    mapped_center = expected_scale * sphere[:3, 3] + expected_offset
    mapped_radius = expected_scale * float(sphere[0, 0])
    if not np.allclose(mapped_center, 0.5, atol=NEUS2_CAMERA_ATOL):
        errors.append(
            "NeuS2 visual-hull center does not map to the unit-cube center"
        )
    if not math.isclose(
        mapped_radius, 0.5, abs_tol=NEUS2_CAMERA_ATOL
    ):
        errors.append("NeuS2 visual-hull sphere does not fit the unit cube")

    expected_by_index = {
        index: (source_name, opencv_c2w)
        for index, source_name, _, opencv_c2w in frames
    }
    seen_indices: set[int] = set()
    maximum_pose_error = 0.0
    minimum_foreground_fraction = 1.0
    expected_intrinsic = neus2_intrinsic()
    for json_path, expected_indices in (
        (train_path, TRAIN_INDICES),
        (test_path, TEST_INDICES),
    ):
        payload = read_json(json_path)
        if (
            payload.get("w") != RESOLUTION[0]
            or payload.get("h") != RESOLUTION[1]
        ):
            errors.append(f"{json_path.name}: incorrect resolution")
        if (
            payload.get("from_na") is not True
            or float(payload.get("aabb_scale", -1.0)) != 1.0
        ):
            errors.append(f"{json_path.name}: incorrect NeuS2 camera flags")
        if not math.isclose(
            float(payload.get("camera_angle_x", -1.0)),
            math.radians(FOV_X_DEG),
            abs_tol=1e-10,
        ):
            errors.append(f"{json_path.name}: incorrect horizontal FOV")
        if not math.isclose(
            float(payload.get("scale", -1.0)),
            expected_scale,
            abs_tol=NEUS2_CAMERA_ATOL,
        ) or not np.allclose(
            payload.get("offset", []),
            expected_offset,
            atol=NEUS2_CAMERA_ATOL,
        ):
            errors.append(f"{json_path.name}: incorrect normalization")

        payload_frames = payload.get("frames", [])
        if len(payload_frames) != len(expected_indices):
            errors.append(
                f"{json_path.name}: expected {len(expected_indices)} frames, "
                f"found {len(payload_frames)}"
            )
            continue
        for frame, index in zip(payload_frames, expected_indices):
            seen_indices.add(index)
            source_name, expected_pose = expected_by_index[index]
            expected_png = f"images/{Path(source_name).stem}.png"
            if frame.get("file_path") != expected_png:
                errors.append(
                    f"{json_path.name}: view {index} has the wrong image path"
                )
            if frame.get("source_view_index") != index:
                errors.append(
                    f"{json_path.name}: view {index} has the wrong source index"
                )
            intrinsic = np.asarray(
                frame.get("intrinsic_matrix"), dtype=np.float64
            )
            if intrinsic.shape != (4, 4) or not np.allclose(
                intrinsic, expected_intrinsic, atol=1e-7
            ):
                errors.append(
                    f"{json_path.name}: view {index} has incorrect intrinsics"
                )
            pose = np.asarray(
                frame.get("transform_matrix"), dtype=np.float64
            )
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                errors.append(
                    f"{json_path.name}: view {index} has an invalid pose"
                )
                continue
            pose_error = float(np.max(np.abs(pose - expected_pose)))
            maximum_pose_error = max(maximum_pose_error, pose_error)
            if pose_error > NEUS2_CAMERA_ATOL:
                errors.append(
                    f"{json_path.name}: view {index} camera changed"
                )
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=NEUS2_CAMERA_ATOL,
            ):
                errors.append(
                    f"{json_path.name}: view {index} rotation is not rigid"
                )
            if not math.isclose(
                float(np.linalg.det(rotation)),
                1.0,
                abs_tol=NEUS2_CAMERA_ATOL,
            ):
                errors.append(
                    f"{json_path.name}: view {index} rotation determinant is not one"
                )
            internal_pose = pose.copy()
            internal_pose[:3, 3] = (
                expected_scale * pose[:3, 3] + expected_offset
            )
            if not np.isfinite(internal_pose).all():
                errors.append(
                    f"{json_path.name}: view {index} normalized pose is not finite"
                )
            center_ray = rotation[:, 2]
            if not math.isclose(
                float(np.linalg.norm(center_ray)),
                1.0,
                abs_tol=NEUS2_CAMERA_ATOL,
            ):
                errors.append(
                    f"{json_path.name}: view {index} has a non-unit center ray"
                )

    if seen_indices != set(range(VIEW_COUNT)):
        errors.append("NeuS2 train/test split is not complete and disjoint")

    if not errors:
        for index in range(VIEW_COUNT):
            basename = f"img{index + 1:03d}"
            with Image.open(image_dir / f"{basename}.png") as image_handle:
                if (
                    image_handle.mode != "RGBA"
                    or image_handle.size != RESOLUTION
                ):
                    errors.append(
                        f"{basename}: output is not a full-resolution RGBA PNG"
                    )
                    continue
                rgba = np.asarray(image_handle, dtype=np.uint8)
            with Image.open(
                source_scene / "image" / f"{basename}.jpg"
            ) as source_handle:
                source_rgb = np.asarray(
                    source_handle.convert("RGB"), dtype=np.uint8
                )
            source_mask = mask_array(
                source_scene / "mask" / f"{basename}.png"
            )
            if not np.array_equal(rgba[..., :3], source_rgb):
                errors.append(f"{basename}: RGB values changed")
            if not np.array_equal(rgba[..., 3] > 127, source_mask):
                errors.append(f"{basename}: alpha does not equal the source mask")
            minimum_foreground_fraction = min(
                minimum_foreground_fraction,
                float((rgba[..., 3] > 127).mean()),
            )

    if errors:
        raise RuntimeError(
            f"NeuS2 validation failed for {scene}:\n" + "\n".join(errors[:100])
        )
    return {
        "scene": scene.name,
        "view_count": VIEW_COUNT,
        "train_count": len(TRAIN_INDICES),
        "test_count": len(TEST_INDICES),
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "normalization_center": sphere[:3, 3].tolist(),
        "normalization_radius": float(sphere[0, 0]),
        "ngp_scale": expected_scale,
        "ngp_offset": expected_offset.tolist(),
        "minimum_foreground_fraction": minimum_foreground_fraction,
        "maximum_camera_error": maximum_pose_error,
    }


def validate_neus2_turntable_scene(
    scene: Path, source_scene: Path
) -> dict[str, Any]:
    """Validate the 36-view, 30/6 NeuS2 turntable contract."""
    errors: list[str] = []
    expected_images = {
        f"img{index + 1:03d}.png" for index in TURNTABLE_INDICES
    }
    image_dir = scene / "images"
    actual_images = (
        {path.name for path in image_dir.glob("*.png")}
        if image_dir.is_dir()
        else set()
    )
    if actual_images != expected_images:
        errors.append(
            "NeuS2 turntable image set is incomplete: "
            f"missing={sorted(expected_images - actual_images)[:5]}, "
            f"unexpected={sorted(actual_images - expected_images)[:5]}"
        )

    forbidden = [
        path
        for path in (
            scene / "reference_mesh.ply",
            scene / "invdepth",
            scene / "cameras_sphere.npz",
        )
        if path.exists()
    ]
    if forbidden:
        errors.append(
            f"NeuS2 turntable output contains forbidden assets: {forbidden}"
        )

    required = {
        "transform.json": TURNTABLE_INDICES,
        "transform_train.json": TURNTABLE_TRAIN_INDICES,
        "transform_test.json": TURNTABLE_TEST_INDICES,
    }
    manifest_path = scene / "conversion_manifest.json"
    for path in (manifest_path, *(scene / name for name in required)):
        if not path.is_file():
            errors.append(f"missing {path.name}")
    if errors:
        raise RuntimeError(
            f"NeuS2 turntable validation failed for {scene}:\n"
            + "\n".join(errors)
        )

    manifest = read_json(manifest_path)
    if manifest.get("protocol") != NEUS2_TURNTABLE_PROTOCOL:
        errors.append("incorrect NeuS2 turntable conversion protocol")
    errors.extend(validate_source_manifest(manifest, source_scene))

    frames = effective_neus2_frames(source_scene)
    sphere, expected_scale, expected_offset = neus2_normalization(
        source_scene,
        frames,
        TURNTABLE_TRAIN_INDICES,
    )
    normalization = manifest.get("normalization", {})
    if not math.isclose(
        float(normalization.get("sphere_radius", -1.0)),
        float(sphere[0, 0]),
        abs_tol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 turntable visual-hull radius changed")
    if not np.allclose(
        normalization.get("sphere_center", []),
        sphere[:3, 3],
        atol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 turntable visual-hull center changed")
    if not math.isclose(
        float(normalization.get("ngp_scale", -1.0)),
        expected_scale,
        abs_tol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 turntable scale changed")
    if not np.allclose(
        normalization.get("ngp_offset", []),
        expected_offset,
        atol=NEUS2_CAMERA_ATOL,
    ):
        errors.append("NeuS2 turntable offset changed")

    expected_by_index = {
        index: opencv_c2w
        for index, _, _, opencv_c2w in frames
        if index in TURNTABLE_INDICES
    }
    expected_intrinsic = neus2_intrinsic()
    seen_by_file: dict[str, set[int]] = {}
    maximum_pose_error = 0.0
    for filename, expected_indices in required.items():
        payload = read_json(scene / filename)
        if (
            payload.get("w") != RESOLUTION[0]
            or payload.get("h") != RESOLUTION[1]
        ):
            errors.append(f"{filename}: incorrect resolution")
        if (
            payload.get("from_na") is not True
            or float(payload.get("aabb_scale", -1.0)) != 1.0
        ):
            errors.append(f"{filename}: incorrect NeuS2 camera flags")
        if not math.isclose(
            float(payload.get("camera_angle_x", -1.0)),
            math.radians(FOV_X_DEG),
            abs_tol=1e-10,
        ):
            errors.append(f"{filename}: incorrect horizontal FOV")
        if not math.isclose(
            float(payload.get("scale", -1.0)),
            expected_scale,
            abs_tol=NEUS2_CAMERA_ATOL,
        ) or not np.allclose(
            payload.get("offset", []),
            expected_offset,
            atol=NEUS2_CAMERA_ATOL,
        ):
            errors.append(f"{filename}: incorrect normalization")

        payload_frames = payload.get("frames", [])
        if len(payload_frames) != len(expected_indices):
            errors.append(
                f"{filename}: expected {len(expected_indices)} frames, "
                f"found {len(payload_frames)}"
            )
            continue

        seen: set[int] = set()
        for frame, index in zip(payload_frames, expected_indices):
            seen.add(index)
            expected_path = f"images/img{index + 1:03d}.png"
            if frame.get("file_path") != expected_path:
                errors.append(
                    f"{filename}: view {index} has the wrong image path"
                )
            if frame.get("source_view_index") != index:
                errors.append(
                    f"{filename}: view {index} has the wrong source index"
                )
            intrinsic = np.asarray(
                frame.get("intrinsic_matrix"), dtype=np.float64
            )
            if intrinsic.shape != (4, 4) or not np.allclose(
                intrinsic, expected_intrinsic, atol=1e-7
            ):
                errors.append(
                    f"{filename}: view {index} has incorrect intrinsics"
                )
            pose = np.asarray(
                frame.get("transform_matrix"), dtype=np.float64
            )
            expected_pose = expected_by_index[index]
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                errors.append(f"{filename}: view {index} has an invalid pose")
                continue
            pose_error = float(np.max(np.abs(pose - expected_pose)))
            maximum_pose_error = max(maximum_pose_error, pose_error)
            if pose_error > NEUS2_CAMERA_ATOL:
                errors.append(f"{filename}: view {index} camera changed")
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=NEUS2_CAMERA_ATOL,
            ) or not math.isclose(
                float(np.linalg.det(rotation)),
                1.0,
                abs_tol=NEUS2_CAMERA_ATOL,
            ):
                errors.append(
                    f"{filename}: view {index} rotation is not rigid"
                )
            if not math.isclose(
                float(np.linalg.norm(rotation[:, 2])),
                1.0,
                abs_tol=NEUS2_CAMERA_ATOL,
            ):
                errors.append(
                    f"{filename}: view {index} has a non-unit center ray"
                )
        seen_by_file[filename] = seen

    all_seen = seen_by_file.get("transform.json", set())
    train_seen = seen_by_file.get("transform_train.json", set())
    test_seen = seen_by_file.get("transform_test.json", set())
    if all_seen != set(TURNTABLE_INDICES):
        errors.append("transform.json does not contain the complete level ring")
    if train_seen & test_seen:
        errors.append("NeuS2 turntable train/test split overlaps")
    if train_seen | test_seen != set(TURNTABLE_INDICES):
        errors.append("NeuS2 turntable train/test split is incomplete")

    split = manifest.get("split", {})
    if split.get("train_source_indices") != list(TURNTABLE_TRAIN_INDICES):
        errors.append("manifest training membership is incorrect")
    if split.get("test_source_indices") != list(TURNTABLE_TEST_INDICES):
        errors.append("manifest test membership is incorrect")

    minimum_foreground_fraction = 1.0
    if not errors:
        for index in TURNTABLE_INDICES:
            basename = f"img{index + 1:03d}"
            with Image.open(image_dir / f"{basename}.png") as image_handle:
                if (
                    image_handle.mode != "RGBA"
                    or image_handle.size != RESOLUTION
                ):
                    errors.append(
                        f"{basename}: output is not a full-resolution RGBA PNG"
                    )
                    continue
                rgba = np.asarray(image_handle, dtype=np.uint8)
            with Image.open(
                source_scene / "image" / f"{basename}.jpg"
            ) as source_handle:
                source_rgb = np.asarray(
                    source_handle.convert("RGB"), dtype=np.uint8
                )
            source_mask = mask_array(
                source_scene / "mask" / f"{basename}.png"
            )
            if not np.array_equal(rgba[..., :3], source_rgb):
                errors.append(f"{basename}: RGB values changed")
            if not np.array_equal(rgba[..., 3] > 127, source_mask):
                errors.append(
                    f"{basename}: alpha does not equal the source mask"
                )
            minimum_foreground_fraction = min(
                minimum_foreground_fraction,
                float((rgba[..., 3] > 127).mean()),
            )

    if errors:
        raise RuntimeError(
            f"NeuS2 turntable validation failed for {scene}:\n"
            + "\n".join(errors[:100])
        )
    return {
        "scene": scene.name,
        "view_count": len(TURNTABLE_INDICES),
        "train_count": len(TURNTABLE_TRAIN_INDICES),
        "test_count": len(TURNTABLE_TEST_INDICES),
        "test_source_indices": list(TURNTABLE_TEST_INDICES),
        "normalization_center": sphere[:3, 3].tolist(),
        "normalization_radius": float(sphere[0, 0]),
        "minimum_foreground_fraction": minimum_foreground_fraction,
        "maximum_camera_error": maximum_pose_error,
    }


def prepare_neus2_record(
    args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.resolve() / name
    target = args.output_root.resolve() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_neus2_scene(target, source_scene)
        print(f"[skip-neus2] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve())
    )
    try:
        print(f"[prepare-neus2] {name}", flush=True)
        write_neus2_scene(source_scene, temporary)
        result = validate_neus2_scene(temporary, source_scene)
        install_transactionally(temporary, target, args.overwrite)
        print(
            f"[ok-neus2] {name}: {result['train_count']} train + "
            f"{result['test_count']} held-out views",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_prepare_neus2(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    failures: list[str] = []
    for record in records:
        name = str(record["name"])
        try:
            prepare_neus2_record(args, record)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"[failed-neus2] {failures[-1]}", flush=True)
    if failures:
        raise RuntimeError("NeuS2 preparation failures:\n" + "\n".join(failures))


def run_validate_neus2(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        name = str(record["name"])
        result = validate_neus2_scene(
            args.output_root.resolve() / name,
            args.input_root.resolve() / name,
        )
        print(json.dumps(result, sort_keys=True))


def prepare_neus2_turntable_record(
    args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.resolve() / name
    target = args.output_root.resolve() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_neus2_turntable_scene(target, source_scene)
        print(
            f"[skip-neus2-turntable] {name}: existing scene is valid",
            flush=True,
        )
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.resolve())
    )
    try:
        print(f"[prepare-neus2-turntable] {name}", flush=True)
        write_neus2_turntable_scene(source_scene, temporary)
        result = validate_neus2_turntable_scene(temporary, source_scene)
        install_transactionally(temporary, target, args.overwrite)
        print(
            f"[ok-neus2-turntable] {name}: "
            f"{result['train_count']} train + "
            f"{result['test_count']} held-out views",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_prepare_neus2_turntable(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    failures: list[str] = []
    for record in records:
        name = str(record["name"])
        try:
            prepare_neus2_turntable_record(args, record)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(
                f"[failed-neus2-turntable] {failures[-1]}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            "NeuS2 turntable preparation failures:\n" + "\n".join(failures)
        )


def run_validate_neus2_turntable(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        name = str(record["name"])
        result = validate_neus2_turntable_scene(
            args.output_root.resolve() / name,
            args.input_root.resolve() / name,
        )
        print(json.dumps(result, sort_keys=True))
