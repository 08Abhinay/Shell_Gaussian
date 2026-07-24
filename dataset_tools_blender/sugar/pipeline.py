"""Prepare and validate SuGaR-ready scenes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core import (
    CAMERA_RADIUS,
    GSHELL_LOADER_LEFT_ROTATION,
    RESOLUTION,
    SUGAR_BBOX_HIGH_QUANTILE,
    SUGAR_BBOX_LOW_QUANTILE,
    SUGAR_BBOX_MARGIN,
    SUGAR_CAMERA_ATOL,
    SUGAR_PROTOCOL,
    TEST_INDICES,
    TRAIN_INDICES,
    VIEW_COUNT,
    colmap_w2c_to_effective,
    effective_to_colmap_w2c,
    install_transactionally,
    load_manifest,
    numbered_names,
    parse_gpus,
    read_json,
    selected_records,
    sha256_file,
    sugar_focal_length,
    validate_scene,
)


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


def effective_sugar_frames(
    source_scene: Path,
) -> list[tuple[str, np.ndarray]]:
    payload = read_json(source_scene / "transforms.json")
    frames = payload.get("frames", [])
    if len(frames) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} source poses, found {len(frames)}")
    result: list[tuple[str, np.ndarray]] = []
    for index, frame in enumerate(frames):
        source_name = Path(str(frame.get("file_path", ""))).name
        expected_name = f"img{index + 1:03d}.jpg"
        if source_name != expected_name:
            raise ValueError(
                f"Unexpected source image order: {source_name!r} != {expected_name!r}"
            )
        saved_c2w = np.asarray(
            frame.get("transform_matrix"), dtype=np.float64
        )
        effective_c2w = GSHELL_LOADER_LEFT_ROTATION @ saved_c2w
        result.append((source_name, effective_c2w))
    return result


def write_seed_colmap_model(
    model_dir: Path, frames: list[tuple[str, np.ndarray]]
) -> None:
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
            f"{image_id} "
            + " ".join(f"{value:.17g}" for value in values)
            + f" 1 {name}\n\n"
        )
    (model_dir / "images.txt").write_text(
        "".join(image_lines), encoding="utf-8"
    )
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
            if (
                len(fields) != 10
                or not fields[0].isdigit()
                or not fields[8].isdigit()
            ):
                continue
            quaternion = np.asarray(
                [float(value) for value in fields[1:5]], dtype=np.float64
            )
            translation = np.asarray(
                [float(value) for value in fields[5:8]], dtype=np.float64
            )
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
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def robust_sparse_bbox(points: np.ndarray) -> dict[str, Any]:
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not len(points)
    ):
        raise ValueError(
            "Cannot compute a bounding box without finite sparse points"
        )
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


def rewrite_colmap_image_extensions(
    source: Path, destination: Path
) -> int:
    changed = 0
    output_lines: list[str] = []
    with source.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if (
                len(fields) == 10
                and fields[0].isdigit()
                and fields[8].isdigit()
            ):
                if not fields[9].lower().endswith((".jpg", ".jpeg")):
                    raise ValueError(
                        f"Unexpected COLMAP image extension: {fields[9]}"
                    )
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
            f"COLMAP stage {name} failed with exit code "
            f"{completed.returncode}:\n{completed.stdout}"
        )
    return {
        "name": name,
        "command": command,
        "elapsed_seconds": time.time() - started,
        "last_output_lines": completed.stdout.splitlines()[-20:],
    }


def write_sugar_images(
    source_scene: Path, destination: Path
) -> dict[str, float | int]:
    output_images = destination / "undistorted" / "images"
    output_masks = destination / "undistorted" / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    foreground_fractions: list[float] = []
    for index in range(1, VIEW_COUNT + 1):
        basename = f"img{index:03d}"
        with Image.open(
            source_scene / "image" / f"{basename}.jpg"
        ) as image_handle:
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
        rgba = np.dstack((rgb, mask))
        Image.fromarray(rgba, mode="RGBA").save(
            output_images / f"{basename}.png", compress_level=6
        )
        Image.fromarray(mask, mode="L").save(
            output_masks / f"{basename}.png", compress_level=6
        )
        foreground_fractions.append(float((mask > 0).mean()))
    return {
        "count": VIEW_COUNT,
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "foreground_fraction_min": float(np.min(foreground_fractions)),
        "foreground_fraction_mean": float(np.mean(foreground_fractions)),
        "foreground_fraction_max": float(np.max(foreground_fractions)),
    }


def split_image_names(indices: tuple[int, ...]) -> set[str]:
    return {f"img{index + 1:03d}.png" for index in indices}


def write_sugar_splits(
    source_scene: Path, destination: Path
) -> dict[str, int]:
    output_root = destination / "undistorted"
    output_root.mkdir(parents=True, exist_ok=True)
    expected = {
        "train": split_image_names(TRAIN_INDICES),
        "test": split_image_names(TEST_INDICES),
    }
    for split, indices in (
        ("train", TRAIN_INDICES),
        ("test", TEST_INDICES),
    ):
        source_payload = read_json(
            source_scene / f"transforms_{split}.json"
        )
        source_names = {
            f"{Path(str(frame.get('file_path', ''))).stem}.png"
            for frame in source_payload.get("frames", [])
        }
        if source_names != expected[split]:
            raise ValueError(
                f"Source {split} membership changed: "
                f"missing={sorted(expected[split] - source_names)}, "
                f"unexpected={sorted(source_names - expected[split])}"
            )
        payload = {
            "schema_version": 1,
            "purpose": "colmap_camera_split_membership",
            "frames": [
                {"file_path": f"images/img{index + 1:03d}.png"}
                for index in indices
            ],
        }
        (output_root / f"transforms_{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "train_count": len(TRAIN_INDICES),
        "test_count": len(TEST_INDICES),
    }


def validate_sugar_splits(scene: Path) -> list[str]:
    errors: list[str] = []
    expected = {
        "train": split_image_names(TRAIN_INDICES),
        "test": split_image_names(TEST_INDICES),
    }
    actual: dict[str, set[str]] = {}
    for split in ("train", "test"):
        path = scene / "undistorted" / f"transforms_{split}.json"
        if not path.is_file():
            errors.append(f"missing transforms_{split}.json")
            actual[split] = set()
            continue
        payload = read_json(path)
        frames = payload.get("frames", [])
        actual[split] = {
            Path(str(frame.get("file_path", ""))).name
            for frame in frames
        }
        if actual[split] != expected[split]:
            errors.append(
                f"incorrect {split} membership: "
                f"missing={sorted(expected[split] - actual[split])}, "
                f"unexpected={sorted(actual[split] - expected[split])}"
            )
        if any(
            not str(frame.get("file_path", "")).startswith("images/")
            for frame in frames
        ):
            errors.append(
                f"transforms_{split}.json must reference images/*.png"
            )
    overlap = actual.get("train", set()) & actual.get("test", set())
    if overlap:
        errors.append(f"train/test membership overlaps: {sorted(overlap)}")
    if actual.get("train", set()) | actual.get("test", set()) != (
        expected["train"] | expected["test"]
    ):
        errors.append("train/test membership does not cover all 180 cameras")
    return errors


def validate_sugar_scene(
    scene: Path, source_scene: Path
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = scene / "masked_colmap_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != SUGAR_PROTOCOL:
        errors.append("incorrect SuGaR protocol")
    if (
        Path(str(manifest.get("source_scene", ""))).resolve()
        != source_scene.resolve()
    ):
        errors.append("source scene path does not match")
    source_transforms = source_scene / "transforms.json"
    if not source_transforms.is_file():
        errors.append("source transforms.json is missing")
    elif manifest.get("source_transforms_sha256") != sha256_file(
        source_transforms
    ):
        errors.append(
            "source transforms.json changed after SuGaR preparation"
        )

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
    errors.extend(validate_sugar_splits(scene))
    for name in (
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ):
        path = sparse_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty sparse/0/{name}")

    expected_frames = effective_sugar_frames(source_scene)
    if not errors:
        camera = parse_colmap_camera(sparse_dir / "cameras.txt")
        focal = sugar_focal_length()
        expected_params = np.asarray(
            [
                focal,
                focal,
                RESOLUTION[0] / 2.0,
                RESOLUTION[1] / 2.0,
            ],
            dtype=np.float64,
        )
        if (
            camera["model"] != "PINHOLE"
            or (camera["width"], camera["height"]) != RESOLUTION
        ):
            errors.append("COLMAP camera model or resolution changed")
        elif not np.allclose(
            camera["params"], expected_params, atol=1e-7
        ):
            errors.append("COLMAP intrinsics changed")

        colmap_poses = parse_colmap_images(sparse_dir / "images.txt")
        if len(colmap_poses) != VIEW_COUNT:
            errors.append(
                f"registered camera count {len(colmap_poses)} != {VIEW_COUNT}"
            )
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
                errors.append(
                    f"COLMAP camera changed for {png_name}: {pose_error:.3g}"
                )
            if not math.isclose(
                float(np.linalg.norm(actual_pose[:3, 3])),
                CAMERA_RADIUS,
                abs_tol=1e-6,
            ):
                errors.append(f"COLMAP camera radius changed for {png_name}")

        points, reprojection_errors = parse_colmap_points(
            sparse_dir / "points3D.txt"
        )
        if not len(points):
            errors.append("COLMAP did not triangulate any sparse points")
        elif (
            not np.isfinite(points).all()
            or not np.isfinite(reprojection_errors).all()
        ):
            errors.append(
                "sparse points or reprojection errors are non-finite"
            )
        else:
            computed_box = robust_sparse_bbox(points)
            recorded_box = manifest.get("foreground_bbox", {})
            if not np.allclose(
                recorded_box.get("min", []),
                computed_box["min"],
                atol=1e-9,
            ):
                errors.append(
                    "recorded foreground bounding-box minimum changed"
                )
            if not np.allclose(
                recorded_box.get("max", []),
                computed_box["max"],
                atol=1e-9,
            ):
                errors.append(
                    "recorded foreground bounding-box maximum changed"
                )
            if computed_box["points_outside_fraction"] > 0.05 + 1e-12:
                errors.append(
                    "foreground bounding box contains fewer than 95% "
                    "of sparse points"
                )

        for index in range(1, VIEW_COUNT + 1):
            basename = f"img{index:03d}.png"
            with (
                Image.open(image_dir / basename) as rgba_handle,
                Image.open(mask_dir / basename) as mask_handle,
            ):
                if (
                    rgba_handle.mode != "RGBA"
                    or rgba_handle.size != RESOLUTION
                ):
                    errors.append(f"invalid RGBA image {basename}")
                    continue
                alpha = np.asarray(
                    rgba_handle.getchannel("A"), dtype=np.uint8
                )
                mask = np.asarray(
                    mask_handle.convert("L"), dtype=np.uint8
                )
                if not np.array_equal(alpha, mask):
                    errors.append(
                        f"RGBA alpha and mask differ for {basename}"
                    )
    else:
        maximum_pose_error = float("inf")
        points = np.empty((0, 3), dtype=np.float64)
        reprojection_errors = np.empty((0,), dtype=np.float64)

    if (scene / "reference_mesh.ply").exists() or (
        scene / "invdepth"
    ).exists():
        errors.append(
            "SuGaR output contains forbidden ground-truth geometry "
            "or inverse depth"
        )
    if errors:
        raise RuntimeError(
            f"SuGaR validation failed for {scene}:\n"
            + "\n".join(errors[:100])
        )
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
    source_scene = args.input_root.absolute() / name
    target = args.output_root.absolute() / name
    validate_scene(source_scene)
    if target.exists() and not args.overwrite:
        result = validate_sugar_scene(target, source_scene)
        print(f"[skip-sugar] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.absolute())
    )
    workspace = temporary / ".workspace"
    seed_model = workspace / "seed_model"
    triangulated_model = workspace / "triangulated_model"
    triangulated_text = workspace / "triangulated_text"
    colmap_masks = workspace / "colmap_masks"
    database = workspace / "database.db"
    gpu = gpu_pool.get()
    started = time.time()
    try:
        print(
            f"[prepare-sugar] {name} on physical GPU {gpu}", flush=True
        )
        frames = effective_sugar_frames(source_scene)
        write_seed_colmap_model(seed_model, frames)
        colmap_masks.mkdir(parents=True)
        for image_name, _ in frames:
            source_mask = (
                source_scene / "mask" / f"{Path(image_name).stem}.png"
            )
            shutil.copy2(
                source_mask, colmap_masks / f"{image_name}.png"
            )

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        colmap = str(args.colmap_bin.resolve())
        focal = sugar_focal_length()
        camera_params = (
            f"{focal:.17g},{focal:.17g},"
            f"{RESOLUTION[0] / 2.0:.17g},"
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

        points, reprojection_errors = parse_colmap_points(
            triangulated_text / "points3D.txt"
        )
        if not len(points):
            raise RuntimeError(
                "COLMAP did not triangulate any sparse points"
            )
        if not np.isfinite(reprojection_errors).all():
            raise RuntimeError(
                "COLMAP produced non-finite reprojection errors"
            )
        bbox = robust_sparse_bbox(points)
        image_info = write_sugar_images(source_scene, temporary)
        split_info = write_sugar_splits(source_scene, temporary)
        sparse_output = temporary / "undistorted" / "sparse" / "0"
        sparse_output.mkdir(parents=True)
        shutil.copy2(
            triangulated_text / "cameras.txt",
            sparse_output / "cameras.txt",
        )
        shutil.copy2(
            triangulated_text / "points3D.txt",
            sparse_output / "points3D.txt",
        )
        changed = rewrite_colmap_image_extensions(
            triangulated_text / "images.txt",
            sparse_output / "images.txt",
        )
        if changed != VIEW_COUNT:
            raise RuntimeError(
                f"Expected to rewrite {VIEW_COUNT} COLMAP image names, "
                f"changed {changed}"
            )

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
        for filename in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
        ):
            shutil.copy2(
                binary_model / filename, sparse_output / filename
            )

        manifest = {
            "version": 1,
            "protocol": SUGAR_PROTOCOL,
            "scene": name,
            "source_scene": str(source_scene),
            "source_transforms_sha256": sha256_file(
                source_scene / "transforms.json"
            ),
            "camera": {
                "source": (
                    "validated effective GShell c2w derived from "
                    "Blender transforms.json"
                ),
                "colmap_convention": (
                    "world_to_camera_opencv_x_right_y_down_z_forward"
                ),
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
                "method": (
                    "masked_sift_exhaustive_matching_fixed_pose_triangulation"
                ),
                "count": int(len(points)),
                "mean_reprojection_error": float(
                    np.mean(reprojection_errors)
                ),
                "maximum_reprojection_error": float(
                    np.max(reprojection_errors)
                ),
            },
            "foreground_bbox": bbox,
            "training_contract": {
                "use_masks": True,
                "white_background": True,
                "split_source": "explicit_source_membership",
                **split_info,
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
        print(
            f"[ok-sugar] {name}: {result['sparse_point_count']} "
            "sparse points",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        gpu_pool.put(gpu)


def run_prepare_sugar(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
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
                prepare_sugar_record, args, record, gpu_pool
            ): str(record["name"])
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
        raise RuntimeError(
            "SuGaR preparation failures:\n" + "\n".join(failures)
        )


def run_validate_sugar(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        name = str(record["name"])
        result = validate_sugar_scene(
            args.output_root.absolute() / name,
            args.input_root.absolute() / name,
        )
        print(json.dumps(result, sort_keys=True))
