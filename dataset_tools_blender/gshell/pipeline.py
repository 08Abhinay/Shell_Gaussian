"""Prepare and validate 36-view G-Shell turntable scenes."""

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
    CAMERA_RADIUS,
    FOV_X_DEG,
    MIN_INVDEPTH_MASK_IOU,
    RESOLUTION,
    TURNTABLE_INDICES,
    TURNTABLE_TEST_INDICES,
    TURNTABLE_TRAIN_INDICES,
    copy_sparse_npy,
    install_transactionally,
    inverse_depth_mask_iou,
    load_manifest,
    mask_array,
    read_json,
    selected_records,
    source_manifest_fields,
    validate_frame_payload,
    validate_scene,
    validate_source_manifest,
)


GSHELL_TURNTABLE_PROTOCOL = "exact_blender_cameras_turntable_36_gshell_v1"


def _source_frames(source_scene: Path) -> list[dict[str, Any]]:
    frames = read_json(source_scene / "transforms.json").get("frames", [])
    if len(frames) < len(TURNTABLE_INDICES):
        raise ValueError(
            f"Expected at least {len(TURNTABLE_INDICES)} source frames, "
            f"found {len(frames)}"
        )
    return frames


def _transform_payload(
    source_payload: dict[str, Any],
    frames: list[dict[str, Any]],
    indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        key: value for key, value in source_payload.items() if key != "frames"
    } | {"frames": [frames[index] for index in indices]}


def write_gshell_turntable_scene(
    source_scene: Path, destination: Path
) -> None:
    source_payload = read_json(source_scene / "transforms.json")
    frames = _source_frames(source_scene)
    for folder in ("image", "mask", "invdepth"):
        (destination / folder).mkdir(parents=True, exist_ok=True)

    for index in TURNTABLE_INDICES:
        basename = f"img{index + 1:03d}"
        shutil.copy2(
            source_scene / "image" / f"{basename}.jpg",
            destination / "image" / f"{basename}.jpg",
        )
        shutil.copy2(
            source_scene / "mask" / f"{basename}.png",
            destination / "mask" / f"{basename}.png",
        )
        copy_sparse_npy(
            source_scene / "invdepth" / f"{basename}.npy",
            destination / "invdepth" / f"{basename}.npy",
        )

    payloads = {
        "transforms.json": TURNTABLE_INDICES,
        "transforms_train.json": TURNTABLE_TRAIN_INDICES,
        "transforms_test.json": TURNTABLE_TEST_INDICES,
    }
    for filename, indices in payloads.items():
        payload = _transform_payload(source_payload, frames, indices)
        (destination / filename).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        **source_manifest_fields(source_scene),
        "protocol": GSHELL_TURNTABLE_PROTOCOL,
        "camera": {
            "view_count": len(TURNTABLE_INDICES),
            "radius": CAMERA_RADIUS,
            "horizontal_fov_degrees": FOV_X_DEG,
            "elevation_degrees": 0.0,
            "azimuth_step_degrees": 10.0,
            "uses_exact_blender_cameras": True,
        },
        "split": {
            "all_source_indices": list(TURNTABLE_INDICES),
            "train_source_indices": list(TURNTABLE_TRAIN_INDICES),
            "test_source_indices": list(TURNTABLE_TEST_INDICES),
            "train_count": len(TURNTABLE_TRAIN_INDICES),
            "test_count": len(TURNTABLE_TEST_INDICES),
        },
        "training_contract": {
            "uses_rgb": True,
            "uses_masks": True,
            "retains_first_surface_inverse_depth": True,
            "uses_ground_truth_mesh": False,
        },
    }
    (destination / "turntable_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def validate_gshell_turntable_scene(
    scene: Path, source_scene: Path
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = scene / "turntable_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing turntable_manifest.json")
    else:
        manifest = read_json(manifest_path)
        if manifest.get("protocol") != GSHELL_TURNTABLE_PROTOCOL:
            errors.append("incorrect G-Shell turntable protocol")
        errors.extend(validate_source_manifest(manifest, source_scene))

    expected_names = {
        "image": {f"img{index + 1:03d}.jpg" for index in TURNTABLE_INDICES},
        "mask": {f"img{index + 1:03d}.png" for index in TURNTABLE_INDICES},
        "invdepth": {f"img{index + 1:03d}.npy" for index in TURNTABLE_INDICES},
    }
    for folder, expected in expected_names.items():
        directory = scene / folder
        actual = (
            {path.name for path in directory.iterdir()}
            if directory.is_dir()
            else set()
        )
        if actual != expected:
            errors.append(f"incorrect {folder} membership")
        if any(path.is_symlink() for path in directory.glob("*") if directory.is_dir()):
            errors.append(f"{folder} contains symbolic links")

    payloads = {
        "transforms.json": TURNTABLE_INDICES,
        "transforms_train.json": TURNTABLE_TRAIN_INDICES,
        "transforms_test.json": TURNTABLE_TEST_INDICES,
    }
    memberships: dict[str, set[str]] = {}
    for filename, indices in payloads.items():
        path = scene / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        payload = read_json(path)
        frame_errors = validate_frame_payload(
            payload.get("frames", []), indices, scene
        )
        errors.extend(f"{filename}: {error}" for error in frame_errors)
        memberships[filename] = {
            str(frame.get("file_path")) for frame in payload.get("frames", [])
        }
    train = memberships.get("transforms_train.json", set())
    test = memberships.get("transforms_test.json", set())
    all_views = memberships.get("transforms.json", set())
    if train & test:
        errors.append("train/test membership overlaps")
    if train | test != all_views:
        errors.append("train/test membership does not cover all views")

    minimum_iou = 1.0
    for index in TURNTABLE_INDICES:
        basename = f"img{index + 1:03d}"
        image_path = scene / "image" / f"{basename}.jpg"
        mask_path = scene / "mask" / f"{basename}.png"
        depth_path = scene / "invdepth" / f"{basename}.npy"
        if not (image_path.is_file() and mask_path.is_file() and depth_path.is_file()):
            continue
        with Image.open(image_path) as image:
            if image.size != RESOLUTION:
                errors.append(f"{basename}: incorrect image resolution")
        mask = mask_array(mask_path)
        depth = np.load(depth_path, mmap_mode="r")
        if mask.shape != (RESOLUTION[1], RESOLUTION[0]) or not mask.any():
            errors.append(f"{basename}: invalid mask")
            continue
        if depth.shape != mask.shape or depth.dtype != np.float32:
            errors.append(f"{basename}: invalid inverse depth")
            continue
        iou = inverse_depth_mask_iou(mask, depth)
        minimum_iou = min(minimum_iou, iou)
        if iou < MIN_INVDEPTH_MASK_IOU:
            errors.append(f"{basename}: inverse-depth/mask IoU is {iou:.6f}")

    if (scene / "reference_mesh.ply").exists():
        errors.append("G-Shell turntable output contains a ground-truth mesh")
    if errors:
        raise RuntimeError(
            f"G-Shell turntable validation failed for {scene}:\n"
            + "\n".join(errors[:100])
        )
    return {
        "scene": scene.name,
        "view_count": len(TURNTABLE_INDICES),
        "train_count": len(TURNTABLE_TRAIN_INDICES),
        "test_count": len(TURNTABLE_TEST_INDICES),
        "minimum_invdepth_mask_iou": minimum_iou,
    }


def prepare_gshell_turntable_record(
    args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    name = str(record["name"])
    source_scene = args.input_root.absolute() / name
    target = args.output_root.absolute() / name
    validate_scene(source_scene, validate_pixels=False)
    if target.exists() and not args.overwrite:
        result = validate_gshell_turntable_scene(target, source_scene)
        print(f"[skip-gshell-turntable] {name}: existing scene is valid", flush=True)
        return result

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=args.output_root.absolute())
    )
    try:
        print(f"[prepare-gshell-turntable] {name}", flush=True)
        write_gshell_turntable_scene(source_scene, temporary)
        result = validate_gshell_turntable_scene(temporary, source_scene)
        install_transactionally(temporary, target, args.overwrite)
        print(
            f"[ok-gshell-turntable] {name}: {result['train_count']} train + "
            f"{result['test_count']} held-out views",
            flush=True,
        )
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_prepare_gshell_turntable(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    failures: list[str] = []
    for record in records:
        try:
            prepare_gshell_turntable_record(args, record)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"{record['name']}: {type(exc).__name__}: {exc}"
            )
    if failures:
        raise RuntimeError(
            "G-Shell turntable preparation failures:\n" + "\n".join(failures)
        )


def run_validate_gshell_turntable(args: argparse.Namespace) -> None:
    records = selected_records(
        load_manifest(args.manifest.resolve(), args.source_root.resolve()),
        args.shoe,
        args.all,
    )
    for record in records:
        name = str(record["name"])
        result = validate_gshell_turntable_scene(
            args.output_root.absolute() / name,
            args.input_root.absolute() / name,
        )
        print(json.dumps(result, sort_keys=True))
