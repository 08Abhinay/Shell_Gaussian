#!/usr/bin/env python3
"""Canonicalize GShell shoe scenes by deterministic turntable phase.

The processed GShell shoe dataset uses camera poses exported from each shoe's
COLMAP reconstruction. COLMAP poses are self-consistent per scene, but their
global yaw/turntable phase can differ across scenes. For turntable shoe data,
the image filename order is a stronger dataset-wide signal: ``img01.jpg`` should
represent the same camera slot for every shoe.

This script writes a new processed dataset root with the same images, masks, and
intrinsics, but yaw-rotates every camera-to-world transform in each scene so the
reference frame, by default ``img01.jpg``, lands at a fixed raw XY orbit angle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


KEY_FRAME_NAMES = ("img01.jpg", "img10.jpg", "img19.jpg", "img28.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes"),
        help="Input processed GShell dataset root, or one scene folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes_turntable_canonical"),
        help="Output processed dataset root to create.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Process only this scene name. May be repeated.",
    )
    parser.add_argument(
        "--reference-frame",
        default="img01.jpg",
        help="Frame basename whose camera orbit angle should become --target-angle-deg.",
    )
    parser.add_argument(
        "--target-angle-deg",
        type=float,
        default=90.0,
        help="Target raw XY orbit angle for the reference frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute summaries without writing output scenes.",
    )
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


def resolve_scene_dirs(input_root: Path, scene_names: list[str] | None) -> list[Path]:
    if (input_root / "transforms.json").is_file():
        return [input_root]

    if scene_names:
        scene_dirs = [input_root / name for name in scene_names]
    else:
        scene_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())

    missing = [path for path in scene_dirs if not (path / "transforms.json").is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing transforms.json for: " + ", ".join(str(path) for path in missing)
        )
    return scene_dirs


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_delta_deg(target: float, current: float) -> float:
    return normalize_angle_deg(target - current)


def raw_xy_orbit_angle_deg(matrix: np.ndarray) -> float:
    center = matrix[:3, 3]
    return normalize_angle_deg(math.degrees(math.atan2(float(center[1]), float(center[0]))))


def raw_z_rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rot


def frame_basename(frame: dict[str, Any]) -> str:
    return Path(str(frame["file_path"])).name


def load_payload(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "transforms.json").open("r") as f:
        payload = json.load(f)
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise ValueError(f"{scene_dir}: transforms.json must contain a frames list")
    return payload


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


def key_angle_map(angles: dict[str, float]) -> dict[str, float | None]:
    return {name: angles.get(name) for name in KEY_FRAME_NAMES}


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


def relink_dir(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output path exists; pass --overwrite: {target}")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    os.symlink(source.resolve(), target, target_is_directory=True)


def write_output_scene(
    scene_dir: Path,
    output_scene: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    if output_scene.resolve() == scene_dir.resolve():
        raise ValueError("Output scene cannot be the same as input scene")
    if output_scene.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene exists; pass --overwrite: {output_scene}")
        shutil.rmtree(output_scene)

    output_scene.mkdir(parents=True, exist_ok=True)
    relink_dir(scene_dir / "image", output_scene / "image", overwrite=True)
    relink_dir(scene_dir / "mask", output_scene / "mask", overwrite=True)

    with (output_scene / "transforms.json").open("w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
        f.write("\n")

    with (output_scene / "turntable_canonicalization.json").open("w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
        f.write("\n")


def validate_output_scene(output_scene: Path, payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload["frames"]
    image_ok = True
    mask_ok = True
    paths_ok = True
    for frame in frames:
        image_path = output_scene / frame["file_path"]
        mask_path = output_scene / str(frame["file_path"]).replace("image/", "mask/").replace(
            ".jpg", ".png"
        )
        paths_ok = paths_ok and image_path.exists() and mask_path.exists()
        image_ok = image_ok and (output_scene / "image").exists()
        mask_ok = mask_ok and (output_scene / "mask").exists()

    return {
        "frame_count": len(frames),
        "frame_count_is_36": len(frames) == 36,
        "image_symlink_valid": image_ok,
        "mask_symlink_valid": mask_ok,
        "frame_image_mask_paths_valid": paths_ok,
    }


def canonicalize_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_payload(scene_dir)
    frames = payload["frames"]
    reference_frame = find_reference_frame(frames, args.reference_frame)
    reference_matrix = matrix_for_frame(reference_frame, scene_dir.name)

    before_angles = angle_map(frames, scene_dir.name)
    before_reference_angle = raw_xy_orbit_angle_deg(reference_matrix)
    delta = angle_delta_deg(float(args.target_angle_deg), before_reference_angle)
    yaw = raw_z_rotation_matrix(delta)

    output_payload = json.loads(json.dumps(payload))
    for out_frame in output_payload["frames"]:
        matrix = matrix_for_frame(out_frame, scene_dir.name)
        out_frame["transform_matrix"] = (yaw @ matrix).tolist()

    after_frames = output_payload["frames"]
    after_angles = angle_map(after_frames, scene_dir.name)
    after_reference_frame = find_reference_frame(after_frames, args.reference_frame)
    after_reference_angle = raw_xy_orbit_angle_deg(
        matrix_for_frame(after_reference_frame, scene_dir.name)
    )

    pre_stats = rotation_stats(frames, scene_dir.name)
    post_stats = rotation_stats(after_frames, scene_dir.name)
    target_error = abs(angle_delta_deg(float(args.target_angle_deg), after_reference_angle))

    output_scene = args.output_root / scene_dir.name
    validation = {
        "target_reference_angle_error_deg": target_error,
        "reference_angle_passed": target_error < 1e-6,
        "before_rotation_stats": pre_stats,
        "after_rotation_stats": post_stats,
        "postwrite": None,
    }

    metadata = {
        "scene": scene_dir.name,
        "source_scene": str(scene_dir),
        "output_scene": str(output_scene),
        "method": "raw_xy_turntable_phase_alignment",
        "settings": {
            "reference_frame": args.reference_frame,
            "target_angle_deg": float(args.target_angle_deg),
            "orbit_plane": "raw_xy",
            "rotation_axis": "raw_z",
            "changed_fields": ["frames[*].transform_matrix"],
            "unchanged_fields": ["frames[*].camera_angle_x", "frames[*].file_path"],
        },
        "phase_correction": {
            "before_reference_angle_deg": before_reference_angle,
            "target_reference_angle_deg": float(args.target_angle_deg),
            "delta_yaw_deg": delta,
            "after_reference_angle_deg": after_reference_angle,
            "raw_z_rotation_matrix": yaw,
        },
        "before": {
            "key_frame_angles_deg": key_angle_map(before_angles),
            "median_frame_step_deg": median_step_deg(before_angles, frames),
        },
        "after": {
            "key_frame_angles_deg": key_angle_map(after_angles),
            "median_frame_step_deg": median_step_deg(after_angles, after_frames),
        },
        "validation": validation,
    }

    if not args.dry_run:
        write_output_scene(scene_dir, output_scene, output_payload, metadata, args.overwrite)
        postwrite = validate_output_scene(output_scene, output_payload)
        metadata["validation"]["postwrite"] = postwrite
        with (output_scene / "turntable_canonicalization.json").open("w") as f:
            json.dump(to_jsonable(metadata), f, indent=2)
            f.write("\n")
    else:
        postwrite = {
            "frame_count": len(after_frames),
            "frame_count_is_36": len(after_frames) == 36,
            "image_symlink_valid": None,
            "mask_symlink_valid": None,
            "frame_image_mask_paths_valid": None,
        }

    status = "ok"
    if not post_stats["rotations_passed"] or target_error >= 1e-6:
        status = "failed"
    if not args.dry_run and not all(
        bool(postwrite[key])
        for key in [
            "frame_count_is_36",
            "image_symlink_valid",
            "mask_symlink_valid",
            "frame_image_mask_paths_valid",
        ]
    ):
        status = "failed"

    row = {
        "scene": scene_dir.name,
        "status": status,
        "frames": len(frames),
        "source_scene": str(scene_dir),
        "output_scene": str(output_scene),
        "transforms_json": str(output_scene / "transforms.json"),
        "turntable_canonicalization_json": str(output_scene / "turntable_canonicalization.json"),
        "reference_frame": args.reference_frame,
        "target_angle_deg": float(args.target_angle_deg),
        "before_reference_angle_deg": before_reference_angle,
        "after_reference_angle_deg": after_reference_angle,
        "delta_yaw_deg": delta,
        "before_median_step_deg": metadata["before"]["median_frame_step_deg"],
        "after_median_step_deg": metadata["after"]["median_frame_step_deg"],
        "target_error_deg": target_error,
        "rotations_passed": post_stats["rotations_passed"],
        "rotation_det_min": post_stats["rotation_det_min"],
        "rotation_det_max": post_stats["rotation_det_max"],
        "rotation_orthonormal_error_max": post_stats["rotation_orthonormal_error_max"],
        "frame_count_is_36": postwrite["frame_count_is_36"],
        "image_symlink_valid": postwrite["image_symlink_valid"],
        "mask_symlink_valid": postwrite["mask_symlink_valid"],
        "frame_image_mask_paths_valid": postwrite["frame_image_mask_paths_valid"],
    }

    for name in KEY_FRAME_NAMES:
        row[f"before_{Path(name).stem}_angle_deg"] = before_angles.get(name)
        row[f"after_{Path(name).stem}_angle_deg"] = after_angles.get(name)

    return row


def aggregate_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows if row.get(key) not in (None, "")])

    def circular_mean_std_deg(angles_deg: np.ndarray) -> tuple[float | None, float | None]:
        if not angles_deg.size:
            return None, None
        angles_rad = np.radians(angles_deg)
        sin_mean = float(np.mean(np.sin(angles_rad)))
        cos_mean = float(np.mean(np.cos(angles_rad)))
        mean = normalize_angle_deg(math.degrees(math.atan2(sin_mean, cos_mean)))
        resultant = min(max(math.hypot(sin_mean, cos_mean), 1e-12), 1.0)
        std = math.degrees(math.sqrt(max(-2.0 * math.log(resultant), 0.0)))
        return mean, std

    key_stats: dict[str, Any] = {}
    for name in KEY_FRAME_NAMES:
        stem = Path(name).stem
        before = values(f"before_{stem}_angle_deg")
        after = values(f"after_{stem}_angle_deg")
        before_circ_mean, before_circ_std = circular_mean_std_deg(before)
        after_circ_mean, after_circ_std = circular_mean_std_deg(after)
        key_stats[name] = {
            "before_mean_deg": float(np.mean(before)) if before.size else None,
            "before_std_deg": float(np.std(before)) if before.size else None,
            "before_circular_mean_deg": before_circ_mean,
            "before_circular_std_deg": before_circ_std,
            "before_min_deg": float(np.min(before)) if before.size else None,
            "before_max_deg": float(np.max(before)) if before.size else None,
            "after_mean_deg": float(np.mean(after)) if after.size else None,
            "after_std_deg": float(np.std(after)) if after.size else None,
            "after_circular_mean_deg": after_circ_mean,
            "after_circular_std_deg": after_circ_std,
            "after_min_deg": float(np.min(after)) if after.size else None,
            "after_max_deg": float(np.max(after)) if after.size else None,
        }

    return {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "scene_count": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({row["status"] for row in rows})
        },
        "settings": {
            "reference_frame": args.reference_frame,
            "target_angle_deg": float(args.target_angle_deg),
            "orbit_plane": "raw_xy",
            "rotation_axis": "raw_z",
        },
        "key_frame_angle_stats": key_stats,
        "max_target_error_deg": float(max(float(row["target_error_deg"]) for row in rows))
        if rows
        else None,
        "all_rotations_passed": all(bool(row["rotations_passed"]) for row in rows),
        "all_frame_counts_36": all(bool(row["frame_count_is_36"]) for row in rows),
        "original_dataset_modified": False,
        "rows": to_jsonable(rows),
    }


def write_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_root / "summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate_summary(rows, args)
    with (args.output_root / "summary.json").open("w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.output_root.resolve() == args.input_root.resolve():
        raise ValueError("Output root must not be the same as input root")

    scene_dirs = resolve_scene_dirs(args.input_root, args.scene)
    rows: list[dict[str, Any]] = []
    for idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{idx}/{len(scene_dirs)}] {scene_dir.name}")
        row = canonicalize_scene(scene_dir, args)
        rows.append(row)
        print(
            "  "
            f"{row['status']} "
            f"img01 {float(row['before_reference_angle_deg']):.2f}"
            f" -> {float(row['after_reference_angle_deg']):.2f} deg "
            f"(yaw {float(row['delta_yaw_deg']):+.2f})"
        )

    if not args.dry_run:
        write_summary(rows, args)
        print(f"\nWrote summary: {args.output_root / 'summary.csv'}")
        print(f"Wrote summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
