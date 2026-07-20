#!/usr/bin/env python3
"""Validate compact Golden evaluation COLMAP scenes and their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_IMAGE_COUNT = 180
SPARSE_FILES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "cameras.txt",
    "images.txt",
    "points3D.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", action="append")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--require-all-registered", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_names(images_txt: Path) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(
        r"^\d+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+"
        r"[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+\d+\s+(\S+)$"
    )
    with images_txt.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                names.add(match.group(1))
    return names


def text_point_count(points_txt: Path) -> int:
    with points_txt.open(encoding="utf-8", errors="strict") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def validate_source_records(
    records: list[dict[str, str]], directory: Path, label: str, errors: list[str]
) -> set[str]:
    names: set[str] = set()
    if len(records) != EXPECTED_IMAGE_COUNT:
        errors.append(f"manifest {label} count is {len(records)}, expected {EXPECTED_IMAGE_COUNT}")
    for record in records:
        name = record.get("name", "")
        path = directory / name
        names.add(name)
        if not path.is_file():
            errors.append(f"missing source {label[:-1]} {name}")
        elif sha256(path) != record.get("sha256"):
            errors.append(f"source {label[:-1]} hash mismatch {name}")
    return names


def validate_scene(scene: Path, require_all_registered: bool) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = scene / "conversion_manifest.json"
    if not manifest_path.is_file():
        return {"scene": scene.name, "ok": False, "errors": ["missing conversion_manifest.json"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    prohibited = manifest.get("prohibited_training_inputs", {})
    for key in (
        "uses_blender_camera_transforms",
        "uses_inverse_depth",
        "uses_masks_in_colmap",
        "uses_ground_truth_mesh",
        "uses_canonical_bbox",
    ):
        if prohibited.get(key) is not False:
            errors.append(f"manifest does not explicitly set {key}=false")

    storage = manifest.get("storage_contract", {})
    expected_storage = {
        "raw_rgb_and_masks_are_external": True,
        "duplicates_raw_inputs": False,
        "raw_model_retained": False,
        "compaction_complete": True,
    }
    for key, expected in expected_storage.items():
        if storage.get(key) is not expected:
            errors.append(f"storage contract {key} is not {expected}")

    source_scene_value = manifest.get("source_scene")
    source_scene = Path(source_scene_value) if source_scene_value else Path()
    if not source_scene_value or not source_scene.is_dir():
        errors.append(f"raw source scene is missing: {source_scene_value!r}")
    inputs = manifest.get("inputs", {})
    source_names = validate_source_records(
        inputs.get("images", []), source_scene / "images", "images", errors
    )
    validate_source_records(inputs.get("masks", []), source_scene / "masks", "masks", errors)

    for redundant_name in ("images", "masks", "colmap"):
        if (scene / redundant_name).exists():
            errors.append(f"redundant processed input copy exists: {redundant_name}")
    if (scene / "undistorted" / "stereo").exists():
        errors.append("unused undistorted/stereo directory exists")

    sparse = scene / "undistorted" / "sparse" / "0"
    for name in SPARSE_FILES:
        if not (sparse / name).is_file():
            errors.append(f"missing undistorted/sparse/0/{name}")
    ply_files = list(scene.rglob("*.ply"))
    if ply_files:
        errors.append(f"redundant PLY files present: {ply_files}")

    images_txt = sparse / "images.txt"
    names = registered_names(images_txt) if images_txt.is_file() else set()
    if not names.issubset(source_names):
        errors.append("COLMAP model contains unexpected image names")
    if require_all_registered and names != source_names:
        errors.append(f"registered {len(names)}/{len(source_names)} images")

    points_path = sparse / "points3D.txt"
    point_count = text_point_count(points_path) if points_path.is_file() else 0
    if point_count <= 0:
        errors.append("COLMAP sparse point cloud is empty")

    image_count = len(list((scene / "undistorted" / "images").glob("*.jpg")))
    mask_count = len(list((scene / "undistorted" / "masks").glob("*.png")))
    if image_count != len(names):
        errors.append(f"undistorted image count {image_count} != registered images {len(names)}")
    if mask_count != len(names):
        errors.append(f"undistorted mask count {mask_count} != registered images {len(names)}")
    if not (scene / "logs").is_dir():
        errors.append("missing COLMAP logs directory")

    forbidden = list(scene.glob("transforms*.json")) + list(scene.rglob("invdepth"))
    if forbidden:
        errors.append(f"forbidden Blender/depth outputs present: {forbidden}")

    return {
        "scene": scene.name,
        "ok": not errors,
        "errors": errors,
        "registered_images": len(names),
        "source_images": len(source_names),
        "undistorted_images": image_count,
        "undistorted_masks": mask_count,
        "points": point_count,
    }


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    scenes = (
        sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        if args.all
        else [root / name for name in args.scene]
    )
    reports = []
    for scene in scenes:
        report = validate_scene(scene, args.require_all_registered)
        reports.append(report)
        status = "OK" if report["ok"] else "FAILED"
        print(
            f"{scene.name}: {status} registered={report.get('registered_images', 0)}/"
            f"{report.get('source_images', 0)} points={report.get('points', 0)}"
        )
        for error in report["errors"]:
            print(f"  - {error}")
    output = args.report_path.resolve() if args.report_path else root / "colmap_validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scenes": reports}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
