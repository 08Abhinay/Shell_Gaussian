"""Prepare and validate MILo-ready fixed-camera Blender scenes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core import (
    CAMERA_RADIUS,
    DEFAULT_COLMAP,
    DEFAULT_GSHELL_TURNTABLE_OUTPUT_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    FOV_X_DEG,
    GSHELL_LOADER_LEFT_ROTATION,
    RESOLUTION,
    TEST_INDICES,
    TRAIN_INDICES,
    TURNTABLE_INDICES,
    TURNTABLE_TEST_INDICES,
    TURNTABLE_TRAIN_INDICES,
    VIEW_COUNT,
    effective_to_colmap_w2c,
    install_transactionally,
    load_manifest,
    parse_gpus,
    read_json,
    selected_records,
    sha256_file,
    sugar_focal_length,
    validate_scene,
)
from ..gshell import pipeline as gshell_pipeline
from ..sugar.pipeline import (
    colmap_database_image_ids,
    parse_colmap_camera,
    parse_colmap_image_ids,
    parse_colmap_images,
    point_track_image_ids,
    run_colmap_stage,
    write_seed_colmap_model,
)


DEFAULT_MILO_OUTPUT_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/milo/golden_set_evaluation"
)
DEFAULT_MILO_TURNTABLE_OUTPUT_ROOT = Path(
    "/home/ab5298/dataset/datasets/processed/milo/"
    "golden_set_evaluation_turntable"
)
MILO_MANIFEST_VERSION = 1
MILO_CAMERA_ATOL = 1e-6


@dataclass(frozen=True)
class MiloVariant:
    name: str
    protocol: str
    source_dataset: str
    indices: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    default_input_root: Path
    default_output_root: Path


FULL_VARIANT = MiloVariant(
    name="full",
    protocol="exact_blender_cameras_milo_180_train_only_triangulation_v1",
    source_dataset="gshell/golden_set_evaluation",
    indices=tuple(range(VIEW_COUNT)),
    train_indices=TRAIN_INDICES,
    test_indices=TEST_INDICES,
    default_input_root=DEFAULT_OUTPUT_ROOT,
    default_output_root=DEFAULT_MILO_OUTPUT_ROOT,
)
TURNTABLE_VARIANT = MiloVariant(
    name="turntable",
    protocol="exact_blender_cameras_milo_36_train_only_triangulation_v1",
    source_dataset="gshell/golden_set_evaluation_turntable",
    indices=TURNTABLE_INDICES,
    train_indices=TURNTABLE_TRAIN_INDICES,
    test_indices=TURNTABLE_TEST_INDICES,
    default_input_root=DEFAULT_GSHELL_TURNTABLE_OUTPUT_ROOT,
    default_output_root=DEFAULT_MILO_TURNTABLE_OUTPUT_ROOT,
)


def source_hashes(source_scene: Path) -> dict[str, str]:
    names = (
        "transforms.json",
        "transforms_train.json",
        "transforms_test.json",
    )
    return {name: sha256_file(source_scene / name) for name in names}


def effective_milo_frames(
    source_scene: Path, variant: MiloVariant
) -> dict[int, tuple[str, np.ndarray]]:
    payload = read_json(source_scene / "transforms.json")
    frames = payload.get("frames", [])
    if len(frames) != len(variant.indices):
        raise ValueError(
            f"Expected {len(variant.indices)} {variant.name} source poses, "
            f"found {len(frames)}"
        )

    result: dict[int, tuple[str, np.ndarray]] = {}
    for position, source_index in enumerate(variant.indices):
        frame = frames[position]
        source_name = Path(str(frame.get("file_path", ""))).name
        expected_name = f"img{source_index + 1:03d}.jpg"
        if source_name != expected_name:
            raise ValueError(
                f"Unexpected source image order: {source_name!r} != "
                f"{expected_name!r}"
            )
        saved_c2w = np.asarray(
            frame.get("transform_matrix"), dtype=np.float64
        )
        if saved_c2w.shape != (4, 4) or not np.isfinite(saved_c2w).all():
            raise ValueError(f"Invalid source camera matrix for {source_name}")
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        rotation = effective_c2w[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"Non-rigid source camera for {source_name}")
        if not math.isclose(
            float(np.linalg.norm(effective_c2w[:3, 3])),
            CAMERA_RADIUS,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Incorrect camera radius for {source_name}")
        result[source_index] = (source_name, effective_c2w)
    return result


def milo_transform_payload(
    frames: dict[int, tuple[str, np.ndarray]],
    indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "camera_angle_x": math.radians(FOV_X_DEG),
        "frames": [
            {
                "file_path": f"images/img{index + 1:03d}",
                "transform_matrix": frames[index][1].tolist(),
            }
            for index in indices
        ],
    }


def write_milo_images(
    source_scene: Path,
    destination: Path,
    variant: MiloVariant,
) -> dict[str, float | int]:
    output_images = destination / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    foreground_fractions: list[float] = []
    for index in variant.indices:
        basename = f"img{index + 1:03d}"
        with Image.open(source_scene / "image" / f"{basename}.jpg") as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(source_scene / "mask" / f"{basename}.png") as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape or rgb.shape[1::-1] != RESOLUTION:
            raise ValueError(f"RGB/mask dimensions differ for {basename}")
        if not np.any(mask):
            raise ValueError(f"Mask is empty for {basename}")
        Image.fromarray(np.dstack((rgb, mask)), mode="RGBA").save(
            output_images / f"{basename}.png", compress_level=6
        )
        foreground_fractions.append(float((mask > 0).mean()))
    return {
        "count": len(variant.indices),
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "foreground_fraction_min": float(np.min(foreground_fractions)),
        "foreground_fraction_mean": float(np.mean(foreground_fractions)),
        "foreground_fraction_max": float(np.max(foreground_fractions)),
    }


def write_milo_transforms(
    destination: Path,
    frames: dict[int, tuple[str, np.ndarray]],
    variant: MiloVariant,
) -> None:
    for split, indices in (
        ("train", variant.train_indices),
        ("test", variant.test_indices),
    ):
        payload = milo_transform_payload(frames, indices)
        (destination / f"transforms_{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def parse_colmap_points_detailed(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    points: list[list[float]] = []
    colors: list[list[int]] = []
    errors: list[float] = []
    track_image_ids: set[int] = set()
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 8 or (len(fields) - 8) % 2:
                raise ValueError(f"Malformed COLMAP point record in {path}")
            points.append([float(value) for value in fields[1:4]])
            colors.append([int(value) for value in fields[4:7]])
            errors.append(float(fields[7]))
            track_image_ids.update(
                int(fields[index]) for index in range(8, len(fields), 2)
            )
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(errors, dtype=np.float64),
        track_image_ids,
    )


MILO_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def write_milo_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("MILo point cloud must contain Nx3 points")
    if colors.shape != points.shape:
        raise ValueError("MILo point colors must match point positions")
    if not np.isfinite(points).all():
        raise ValueError("MILo point positions are non-finite")
    vertices = np.zeros(len(points), dtype=MILO_PLY_DTYPE)
    for axis in ("x", "y", "z"):
        vertices[axis] = points[:, "xyz".index(axis)]
    for channel, column in (("red", 0), ("green", 1), ("blue", 2)):
        vertices[channel] = colors[:, column]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def read_milo_ply(path: Path) -> np.ndarray:
    properties: list[str] = []
    vertex_count: int | None = None
    with path.open("rb") as handle:
        if handle.readline() != b"ply\n":
            raise ValueError(f"Not a PLY file: {path}")
        if handle.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError(f"MILo PLY must be binary little-endian: {path}")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Truncated PLY header: {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line.startswith("property "):
                properties.append(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"Missing or empty PLY vertex element: {path}")
        expected = list(MILO_PLY_DTYPE.names or ())
        if properties != expected:
            raise ValueError(
                f"Unexpected MILo PLY properties: {properties} != {expected}"
            )
        vertices = np.fromfile(handle, dtype=MILO_PLY_DTYPE, count=vertex_count)
        if len(vertices) != vertex_count or handle.read(1):
            raise ValueError(f"MILo PLY payload size is incorrect: {path}")
    coordinates = np.column_stack(
        (vertices["x"], vertices["y"], vertices["z"])
    )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"MILo PLY contains non-finite points: {path}")
    normals = np.column_stack(
        (vertices["nx"], vertices["ny"], vertices["nz"])
    )
    if not np.isfinite(normals).all() or np.any(normals != 0.0):
        raise ValueError(f"MILo PLY normals must be finite zeros: {path}")
    return vertices


def triangulate_training_points(
    source_scene: Path,
    destination: Path,
    workspace: Path,
    frames: dict[int, tuple[str, np.ndarray]],
    variant: MiloVariant,
    colmap_bin: Path,
    gpu: int,
) -> dict[str, Any]:
    train_images = workspace / "train_images"
    colmap_masks = workspace / "colmap_masks"
    seed_model = workspace / "seed_model"
    triangulated_model = workspace / "triangulated_model"
    triangulated_text = workspace / "triangulated_text"
    database = workspace / "database.db"
    train_images.mkdir(parents=True)
    colmap_masks.mkdir(parents=True)
    train_frames = [frames[index] for index in variant.train_indices]
    for image_name, _ in train_frames:
        shutil.copy2(
            source_scene / "image" / image_name,
            train_images / image_name,
        )
        shutil.copy2(
            source_scene / "mask" / f"{Path(image_name).stem}.png",
            colmap_masks / f"{image_name}.png",
        )

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    colmap = str(colmap_bin.resolve())
    focal = sugar_focal_length()
    camera_params = (
        f"{focal:.17g},{focal:.17g},"
        f"{RESOLUTION[0] / 2.0:.17g},{RESOLUTION[1] / 2.0:.17g}"
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
                str(train_images),
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
    database_ids = colmap_database_image_ids(database)
    expected_names = {name for name, _ in train_frames}
    if set(database_ids) != expected_names:
        raise RuntimeError(
            "COLMAP training membership changed: "
            f"missing={sorted(expected_names - set(database_ids))}, "
            f"unexpected={sorted(set(database_ids) - expected_names)}"
        )
    write_seed_colmap_model(seed_model, train_frames, database_ids)
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
                str(train_images),
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

    provenance = destination / "initialization" / "colmap"
    provenance.mkdir(parents=True)
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(triangulated_text / filename, provenance / filename)
    points, colors, reprojection_errors, track_ids = (
        parse_colmap_points_detailed(provenance / "points3D.txt")
    )
    if not len(points):
        raise RuntimeError("COLMAP did not triangulate any training-only points")
    if not np.isfinite(reprojection_errors).all():
        raise RuntimeError("COLMAP produced non-finite reprojection errors")
    if not track_ids.issubset(set(database_ids.values())):
        raise RuntimeError("Sparse-point tracks contain non-training cameras")
    write_milo_ply(destination / "points3d.ply", points, colors)
    return {
        "method": "masked_sift_exhaustive_matching_fixed_pose_train_only",
        "point_count": int(len(points)),
        "mean_reprojection_error": float(np.mean(reprojection_errors)),
        "maximum_reprojection_error": float(np.max(reprojection_errors)),
        "training_camera_count": len(train_frames),
        "track_camera_count": len(track_ids),
        "stages": stages,
    }


def _validate_source(
    source_scene: Path,
    full_source_scene: Path,
    variant: MiloVariant,
) -> None:
    if variant is FULL_VARIANT:
        validate_scene(source_scene)
    else:
        gshell_pipeline.validate_gshell_turntable_scene(
            source_scene, full_source_scene
        )


def validate_milo_scene(
    scene: Path,
    source_scene: Path,
    full_source_scene: Path,
    variant: MiloVariant,
) -> dict[str, Any]:
    _validate_source(source_scene, full_source_scene, variant)
    errors: list[str] = []
    manifest_path = scene / "milo_dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("version") != MILO_MANIFEST_VERSION:
        errors.append("incorrect MILo manifest version")
    if manifest.get("protocol") != variant.protocol:
        errors.append("incorrect MILo protocol")
    if manifest.get("variant") != variant.name:
        errors.append("incorrect MILo variant")
    if manifest.get("scene") != source_scene.name:
        errors.append("incorrect MILo scene name")
    if manifest.get("source_dataset") != variant.source_dataset:
        errors.append("incorrect MILo source dataset")
    if manifest.get("source_files_sha256") != source_hashes(source_scene):
        errors.append("MILo source transforms changed after preparation")

    expected_pngs = {
        f"img{index + 1:03d}.png" for index in variant.indices
    }
    image_dir = scene / "images"
    actual_pngs = (
        {path.name for path in image_dir.glob("*.png")}
        if image_dir.is_dir()
        else set()
    )
    if actual_pngs != expected_pngs:
        errors.append("MILo RGBA image set is incomplete")
    elif not errors:
        for index in variant.indices:
            basename = f"img{index + 1:03d}"
            with Image.open(image_dir / f"{basename}.png") as rgba_handle:
                if rgba_handle.mode != "RGBA" or rgba_handle.size != RESOLUTION:
                    errors.append(f"invalid MILo RGBA image {basename}.png")
                    continue
                rgba = np.asarray(rgba_handle, dtype=np.uint8)
            with Image.open(
                source_scene / "image" / f"{basename}.jpg"
            ) as rgb_handle:
                source_rgb = np.asarray(rgb_handle.convert("RGB"), dtype=np.uint8)
            with Image.open(
                source_scene / "mask" / f"{basename}.png"
            ) as mask_handle:
                source_mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
            source_mask = np.where(source_mask >= 128, 255, 0).astype(np.uint8)
            if not np.array_equal(rgba[:, :, :3], source_rgb):
                errors.append(f"MILo RGB changed for {basename}.png")
            if not np.array_equal(rgba[:, :, 3], source_mask):
                errors.append(f"MILo alpha changed for {basename}.png")

    expected_frames = effective_milo_frames(source_scene, variant)
    observed_indices: set[int] = set()
    maximum_camera_error = 0.0
    for split, indices in (
        ("train", variant.train_indices),
        ("test", variant.test_indices),
    ):
        path = scene / f"transforms_{split}.json"
        if not path.is_file():
            errors.append(f"missing transforms_{split}.json")
            continue
        payload = read_json(path)
        if not math.isclose(
            float(payload.get("camera_angle_x", -1.0)),
            math.radians(FOV_X_DEG),
            abs_tol=1e-12,
        ):
            errors.append(f"incorrect {split} horizontal FOV")
        actual_frames = payload.get("frames", [])
        if len(actual_frames) != len(indices):
            errors.append(
                f"{split} frame count {len(actual_frames)} != {len(indices)}"
            )
            continue
        for frame, index in zip(actual_frames, indices):
            observed_indices.add(index)
            expected_path = f"images/img{index + 1:03d}"
            if frame.get("file_path") != expected_path:
                errors.append(f"incorrect MILo frame path for index {index}")
            matrix = np.asarray(
                frame.get("transform_matrix"), dtype=np.float64
            )
            expected = expected_frames[index][1]
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                errors.append(f"invalid MILo camera matrix for index {index}")
                continue
            camera_error = float(np.max(np.abs(matrix - expected)))
            maximum_camera_error = max(maximum_camera_error, camera_error)
            if camera_error > MILO_CAMERA_ATOL:
                errors.append(
                    f"MILo camera changed for index {index}: {camera_error:.3g}"
                )
            milo_opencv_c2w = matrix.copy()
            milo_opencv_c2w[:3, 1:3] *= -1
            milo_w2c = np.linalg.inv(milo_opencv_c2w)
            expected_w2c = effective_to_colmap_w2c(expected)
            if not np.allclose(milo_w2c, expected_w2c, atol=1e-9):
                errors.append(f"MILo loader conversion changed index {index}")
    if observed_indices != set(variant.indices):
        errors.append("MILo train/test splits do not cover every view exactly once")

    ply_path = scene / "points3d.ply"
    if not ply_path.is_file():
        errors.append("missing points3d.ply")
        vertices = np.empty(0, dtype=MILO_PLY_DTYPE)
    else:
        try:
            vertices = read_milo_ply(ply_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            vertices = np.empty(0, dtype=MILO_PLY_DTYPE)

    provenance = scene / "initialization" / "colmap"
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (provenance / filename).is_file():
            errors.append(f"missing initialization/colmap/{filename}")
    if not errors:
        camera = parse_colmap_camera(provenance / "cameras.txt")
        focal = sugar_focal_length()
        expected_params = np.asarray(
            [focal, focal, RESOLUTION[0] / 2.0, RESOLUTION[1] / 2.0]
        )
        if (
            camera["model"] != "PINHOLE"
            or (camera["width"], camera["height"]) != RESOLUTION
            or not np.allclose(camera["params"], expected_params, atol=1e-7)
        ):
            errors.append("MILo triangulation intrinsics changed")
        colmap_poses = parse_colmap_images(provenance / "images.txt")
        expected_train_names = {
            expected_frames[index][0] for index in variant.train_indices
        }
        if set(colmap_poses) != expected_train_names:
            errors.append("MILo sparse model contains non-training cameras")
        for index in variant.train_indices:
            name, expected = expected_frames[index]
            actual = colmap_poses.get(name)
            if actual is None or not np.allclose(
                actual, expected, atol=MILO_CAMERA_ATOL
            ):
                errors.append(f"MILo sparse camera changed for {name}")
        image_ids = parse_colmap_image_ids(provenance / "images.txt")
        track_ids = point_track_image_ids(provenance / "points3D.txt")
        if not track_ids.issubset(set(image_ids.values())):
            errors.append("MILo sparse tracks contain non-training camera IDs")
        points, colors, reprojection_errors, detailed_track_ids = (
            parse_colmap_points_detailed(provenance / "points3D.txt")
        )
        if not len(points) or not np.isfinite(reprojection_errors).all():
            errors.append("MILo sparse points or errors are invalid")
        if detailed_track_ids != track_ids:
            errors.append("MILo sparse track parsing is inconsistent")
        if len(points) != len(vertices):
            errors.append("MILo PLY and COLMAP point counts differ")
        elif not np.allclose(
            points.astype(np.float32),
            np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
            atol=1e-7,
        ):
            errors.append("MILo PLY point positions differ from COLMAP")
        if len(points) == len(vertices) and not np.array_equal(
            colors,
            np.column_stack(
                (vertices["red"], vertices["green"], vertices["blue"])
            ),
        ):
            errors.append("MILo PLY colors differ from COLMAP")
        initialization = manifest.get("initialization", {})
        if initialization.get("point_count") != len(points):
            errors.append("MILo manifest point count changed")
        stages = initialization.get("stages", [])
        stage_names = [stage.get("name") for stage in stages]
        if stage_names != [
            "feature_extractor",
            "exhaustive_matcher",
            "point_triangulator",
            "model_to_text",
        ]:
            errors.append("MILo COLMAP stage sequence changed")
        if any(
            len(stage.get("command", [])) > 1
            and stage["command"][1] == "mapper"
            for stage in stages
        ):
            errors.append("MILo preparation ran COLMAP mapper")

    for forbidden in ("sparse", "invdepth", "reference_mesh.ply"):
        if (scene / forbidden).exists():
            errors.append(f"MILo dataset contains forbidden {forbidden}")
    if any(path.is_symlink() for path in scene.rglob("*")):
        errors.append("MILo dataset contains symbolic links")
    if errors:
        raise RuntimeError(
            f"MILo validation failed for {scene}:\n" + "\n".join(errors[:100])
        )
    return {
        "scene": scene.name,
        "variant": variant.name,
        "view_count": len(variant.indices),
        "train_count": len(variant.train_indices),
        "test_count": len(variant.test_indices),
        "sparse_point_count": len(vertices),
        "maximum_camera_matrix_error": maximum_camera_error,
    }


def prepare_milo_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    gpu_pool: queue.Queue[int],
    variant: MiloVariant,
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.absolute() / name
    full_source_scene = args.full_input_root.absolute() / name
    target = args.output_root.absolute() / name
    _validate_source(source_scene, full_source_scene, variant)
    if target.exists() and not args.overwrite:
        result = validate_milo_scene(
            target, source_scene, full_source_scene, variant
        )
        print(
            f"[skip-milo-{variant.name}] {name}: existing scene is valid",
            flush=True,
        )
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.absolute())
    )
    workspace = temporary / ".workspace"
    workspace.mkdir()
    gpu = gpu_pool.get()
    started = time.time()
    try:
        print(
            f"[prepare-milo-{variant.name}] {name} on physical GPU {gpu}",
            flush=True,
        )
        frames = effective_milo_frames(source_scene, variant)
        image_info = write_milo_images(source_scene, temporary, variant)
        write_milo_transforms(temporary, frames, variant)
        initialization = triangulate_training_points(
            source_scene,
            temporary,
            workspace,
            frames,
            variant,
            args.colmap_bin,
            gpu,
        )
        manifest = {
            "version": MILO_MANIFEST_VERSION,
            "protocol": variant.protocol,
            "variant": variant.name,
            "scene": name,
            "source_dataset": variant.source_dataset,
            "source_files_sha256": source_hashes(source_scene),
            "camera": {
                "source": "validated effective GShell cameras from Blender",
                "format": "milo_blender_opengl_camera_to_world",
                "horizontal_fov_degrees": FOV_X_DEG,
                "radius": CAMERA_RADIUS,
                "width": RESOLUTION[0],
                "height": RESOLUTION[1],
                "view_count": len(variant.indices),
            },
            "images": image_info,
            "training_contract": {
                "white_background": True,
                "require_eval_flag": True,
                "train_count": len(variant.train_indices),
                "test_count": len(variant.test_indices),
                "test_stride": 6,
                "sparse_points_use_training_images_only": True,
                "uses_inverse_depth": False,
                "uses_ground_truth_mesh": False,
            },
            "initialization": initialization,
            "elapsed_seconds": time.time() - started,
        }
        (temporary / "milo_dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        shutil.rmtree(workspace)
        result = validate_milo_scene(
            temporary, source_scene, full_source_scene, variant
        )
        install_transactionally(temporary, target, args.overwrite)
        print(
            f"[ok-milo-{variant.name}] {name}: "
            f"{result['sparse_point_count']} training-only sparse points",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        gpu_pool.put(gpu)


def _selected(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_manifest(
        args.manifest.resolve(),
        args.source_root.resolve(),
        verify_hashes=False,
    )
    return selected_records(records, args.shoe, args.all)


def _run_prepare(args: argparse.Namespace, variant: MiloVariant) -> None:
    records = _selected(args)
    gpus = parse_gpus(args)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(gpus)
    ) as executor:
        futures = {
            executor.submit(
                prepare_milo_record, args, record, gpu_pool, variant
            ): str(record["name"])
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                message = f"{name}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[failed-milo-{variant.name}] {message}", flush=True)
    if failures:
        raise RuntimeError(
            f"MILo {variant.name} preparation failures:\n"
            + "\n".join(failures)
        )


def _run_validate(args: argparse.Namespace, variant: MiloVariant) -> None:
    for record in _selected(args):
        name = str(record["name"])
        result = validate_milo_scene(
            args.output_root.absolute() / name,
            args.input_root.absolute() / name,
            args.full_input_root.absolute() / name,
            variant,
        )
        print(json.dumps(result, sort_keys=True))


def run_prepare_milo(args: argparse.Namespace) -> None:
    _run_prepare(args, FULL_VARIANT)


def run_validate_milo(args: argparse.Namespace) -> None:
    _run_validate(args, FULL_VARIANT)


def run_prepare_milo_turntable(args: argparse.Namespace) -> None:
    _run_prepare(args, TURNTABLE_VARIANT)


def run_validate_milo_turntable(args: argparse.Namespace) -> None:
    _run_validate(args, TURNTABLE_VARIANT)


__all__ = [
    "DEFAULT_MILO_OUTPUT_ROOT",
    "DEFAULT_MILO_TURNTABLE_OUTPUT_ROOT",
    "FULL_VARIANT",
    "TURNTABLE_VARIANT",
    "effective_milo_frames",
    "milo_transform_payload",
    "parse_colmap_points_detailed",
    "read_milo_ply",
    "run_prepare_milo",
    "run_prepare_milo_turntable",
    "run_validate_milo",
    "run_validate_milo_turntable",
    "validate_milo_scene",
    "write_milo_ply",
]
