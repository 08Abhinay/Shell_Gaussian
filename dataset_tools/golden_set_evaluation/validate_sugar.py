#!/usr/bin/env python3
"""Validate a SuGaR adapter scene built from the common COLMAP dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_IMAGE_COUNT = 180
MAX_BBOX_OUTSIDE_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_records(path: Path) -> list[list[str]]:
    records: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 10 and fields[0].isdigit() and fields[8].isdigit():
                records.append(fields)
    return records


def main() -> int:
    args = parse_args()
    scene = (args.dataset_root / args.scene).resolve()
    manifest_path = scene / "masked_colmap_manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_protocols = {"compact_fresh_colmap_with_rgba_masks"}
    if manifest.get("protocol") not in valid_protocols:
        errors.append("unexpected dataset protocol")

    source_scene = Path(manifest.get("source_rgb_only_scene", ""))
    source_manifest = source_scene / "conversion_manifest.json"
    if not source_manifest.is_file():
        errors.append("source RGB-only manifest is missing")
    elif sha256(source_manifest) != manifest.get("source_manifest_sha256"):
        errors.append("source RGB-only manifest hash changed")

    contract = manifest.get("training_contract", {})
    expected_contract = {
        "use_masks": True,
        "white_background": True,
        "center_bbox": False,
        "foreground_only": True,
        "constrain_points_to_bbox": True,
        "filter_gaussians_by_bbox": True,
        "uses_blender_camera_transforms": False,
        "uses_inverse_depth": False,
        "uses_ground_truth_mesh": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) is not expected:
            errors.append(f"training contract {key} is not {expected}")

    image_info = manifest.get("images", {})
    width = int(image_info.get("width", 0))
    height = int(image_info.get("height", 0))
    records = image_info.get("files", [])
    if len(records) != EXPECTED_IMAGE_COUNT:
        errors.append(f"manifest contains {len(records)} image records, expected {EXPECTED_IMAGE_COUNT}")

    rgba_paths = sorted((scene / "undistorted" / "images").glob("*.png"))
    mask_paths = sorted((scene / "undistorted" / "masks").glob("*.png"))
    if len(rgba_paths) != EXPECTED_IMAGE_COUNT:
        errors.append(f"found {len(rgba_paths)} RGBA images, expected {EXPECTED_IMAGE_COUNT}")
    if len(mask_paths) != EXPECTED_IMAGE_COUNT:
        errors.append(f"found {len(mask_paths)} masks, expected {EXPECTED_IMAGE_COUNT}")

    coverage: list[float] = []
    for record in records:
        rgba_path = scene / "undistorted" / "images" / record["name"]
        mask_path = scene / "undistorted" / "masks" / record["name"]
        if not rgba_path.is_file() or not mask_path.is_file():
            errors.append(f"missing RGBA image or mask: {record['name']}")
            continue
        if sha256(rgba_path) != record.get("rgba_sha256"):
            errors.append(f"RGBA hash mismatch: {record['name']}")
        if sha256(mask_path) != record.get("mask_sha256"):
            errors.append(f"mask hash mismatch: {record['name']}")
        with Image.open(rgba_path) as rgba_handle, Image.open(mask_path) as mask_handle:
            if rgba_handle.mode != "RGBA":
                errors.append(f"image is not RGBA: {record['name']}")
                continue
            if rgba_handle.size != (width, height) or mask_handle.size != (width, height):
                errors.append(f"dimension mismatch: {record['name']}")
                continue
            alpha = np.asarray(rgba_handle.getchannel("A"), dtype=np.uint8)
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
            if not np.array_equal(alpha, mask):
                errors.append(f"RGBA alpha does not equal stored mask: {record['name']}")
            unique = set(np.unique(mask).tolist())
            if not unique.issubset({0, 255}):
                errors.append(f"mask is not binary: {record['name']}")
            fraction = float((mask > 0).mean())
            coverage.append(fraction)
            if abs(fraction - float(record.get("foreground_fraction", -1))) > 1e-12:
                errors.append(f"foreground coverage mismatch: {record['name']}")

    source_records = image_records(source_scene / "undistorted" / "sparse" / "0" / "images.txt")
    masked_records = image_records(scene / "undistorted" / "sparse" / "0" / "images.txt")
    if len(source_records) != EXPECTED_IMAGE_COUNT or len(masked_records) != EXPECTED_IMAGE_COUNT:
        errors.append("source or masked COLMAP model does not contain 180 image records")
    elif len(source_records) == len(masked_records):
        for source, masked in zip(source_records, masked_records):
            if source[:9] != masked[:9]:
                errors.append(f"COLMAP camera pose changed for image id {source[0]}")
                break
            if Path(source[9]).stem != Path(masked[9]).stem or Path(masked[9]).suffix != ".png":
                errors.append(f"COLMAP image-name conversion is invalid for image id {source[0]}")
                break

    source_camera_text = source_scene / "undistorted" / "sparse" / "0" / "cameras.txt"
    masked_camera_text = scene / "undistorted" / "sparse" / "0" / "cameras.txt"
    source_points_text = source_scene / "undistorted" / "sparse" / "0" / "points3D.txt"
    masked_points_text = scene / "undistorted" / "sparse" / "0" / "points3D.txt"
    if sha256(source_camera_text) != sha256(masked_camera_text):
        errors.append("COLMAP camera intrinsics changed")
    if sha256(source_points_text) != sha256(masked_points_text):
        errors.append("COLMAP sparse points changed")

    for name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"):
        if not (scene / "undistorted" / "sparse" / "0" / name).is_file():
            errors.append(f"missing undistorted/sparse/0/{name}")
    for redundant in (scene / "images", scene / "masks", scene / "colmap"):
        if redundant.exists():
            errors.append(f"redundant common-dataset copy exists: {redundant.name}")
    montage = scene / manifest.get("validation_montage", "")
    if not montage.is_file():
        errors.append("mask validation montage is missing")

    box = manifest.get("foreground_bbox", {})
    bbox_min = np.asarray(box.get("min", []), dtype=float)
    bbox_max = np.asarray(box.get("max", []), dtype=float)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,) or not np.all(bbox_max > bbox_min):
        errors.append("foreground bounding box is invalid")
    outside_fraction = float(box.get("points_outside_fraction", 1.0))
    if outside_fraction > MAX_BBOX_OUTSIDE_FRACTION:
        errors.append(
            f"{outside_fraction:.3%} of COLMAP sparse points lie outside the robust "
            f"bounding box; maximum is {MAX_BBOX_OUTSIDE_FRACTION:.1%}"
        )

    report = {
        "scene": args.scene,
        "ok": not errors,
        "errors": errors,
        "rgba_images": len(rgba_paths),
        "masks": len(mask_paths),
        "registered_images": len(masked_records),
        "mean_foreground_fraction": float(np.mean(coverage)) if coverage else None,
        "bbox_min": bbox_min.tolist() if bbox_min.shape == (3,) else None,
        "bbox_max": bbox_max.tolist() if bbox_max.shape == (3,) else None,
        "max_gaussian_scale_ratio": contract.get("max_gaussian_scale_ratio"),
    }
    report_path = scene / "masked_colmap_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = "OK" if not errors else "FAILED"
    print(f"{args.scene}: {status}")
    print(f"  RGBA/masks/cameras: {len(rgba_paths)}/{len(mask_paths)}/{len(masked_records)}")
    if coverage:
        print(f"  Mean foreground coverage: {np.mean(coverage) * 100:.3f}%")
    print(f"  Bbox min: {report['bbox_min']}")
    print(f"  Bbox max: {report['bbox_max']}")
    print(f"  Scale ratio: {report['max_gaussian_scale_ratio']}")
    for error in errors:
        print(f"  - {error}")
    print(f"  Report: {report_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
