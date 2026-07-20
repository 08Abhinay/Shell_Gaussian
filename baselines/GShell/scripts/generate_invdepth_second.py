#!/usr/bin/env python3
"""Generate second visible inverse-depth targets from a canonical mesh.

The input dataset is the existing GShell-style multi_elevation_360 folder with
Blender camera transforms. The output is:

    all/invdepth_second/img001.npy ...
    train/invdepth_second/img001.npy ...
    val/invdepth_second/img001.npy ...

Each pixel stores 1 / camera-distance for the second surface hit along that
camera ray, or 0 when there is no second hit.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument(
        "--pose-convention",
        choices=("blender-raw", "gshell-legacy-saved"),
        default="blender-raw",
        help=(
            "Convention used by transform_matrix. GShell-compatible Blender datasets "
            "store Rx(-90deg) @ c2w and require conversion back to raw Blender c2w "
            "before ray casting."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-auto-flip", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def make_scene(mesh_npz: Path) -> o3d.t.geometry.RaycastingScene:
    payload = np.load(mesh_npz)
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["faces"].astype(np.uint32)
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    mesh.triangle["indices"] = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def rotate_x_matrix(angle: float) -> np.ndarray:
    s, c = math.sin(angle), math.cos(angle)
    matrix = np.eye(4, dtype=np.float32)
    matrix[1, 1] = c
    matrix[1, 2] = s
    matrix[2, 1] = -s
    matrix[2, 2] = c
    return matrix


GSHELL_DATASET_ROTATION = rotate_x_matrix(-math.pi / 2.0)


def camera_rays(
    frame: dict[str, Any],
    height: int,
    width: int,
    pose_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(frame["transform_matrix"], dtype=np.float32)
    if pose_convention == "gshell-legacy-saved":
        camera = np.linalg.inv(GSHELL_DATASET_ROTATION) @ camera
    origin = camera[:3, 3].astype(np.float32)
    rotation = camera[:3, :3].astype(np.float32)
    fov_x = float(frame["camera_angle_x"])
    tan_x = math.tan(fov_x * 0.5)
    tan_y = tan_x * height / width

    xs = (2.0 * ((np.arange(width, dtype=np.float32) + 0.5) / width) - 1.0) * tan_x
    ys = (1.0 - 2.0 * ((np.arange(height, dtype=np.float32) + 0.5) / height)) * tan_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs_camera = np.stack(
        [grid_x, grid_y, -np.ones_like(grid_x, dtype=np.float32)],
        axis=-1,
    ).reshape(-1, 3)
    dirs_camera /= np.linalg.norm(dirs_camera, axis=1, keepdims=True)
    dirs_world = dirs_camera @ rotation.T
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
    return origin, dirs_world.astype(np.float32)


def cast_rays(
    scene: o3d.t.geometry.RaycastingScene,
    origins: np.ndarray,
    dirs: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    hits = np.empty((dirs.shape[0],), dtype=np.float32)
    for start in range(0, dirs.shape[0], chunk_size):
        end = min(start + chunk_size, dirs.shape[0])
        rays = np.concatenate([origins[start:end], dirs[start:end]], axis=1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        hits[start:end] = result["t_hit"].numpy()
    return hits


def render_invdepth_pair(
    scene: o3d.t.geometry.RaycastingScene,
    frame: dict[str, Any],
    height: int,
    width: int,
    eps: float,
    chunk_size: int,
    pose_convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    origin, dirs = camera_rays(frame, height, width, pose_convention)
    origins = np.repeat(origin[None, :], dirs.shape[0], axis=0).astype(np.float32)

    first_t = cast_rays(scene, origins, dirs, chunk_size)
    first_valid = np.isfinite(first_t)
    first = np.zeros((dirs.shape[0],), dtype=np.float32)
    first[first_valid] = 1.0 / np.maximum(first_t[first_valid], 1e-6)

    second = np.zeros((dirs.shape[0],), dtype=np.float32)
    valid_indices = np.flatnonzero(first_valid)
    if valid_indices.size:
        first_dirs = dirs[valid_indices]
        first_t_valid = first_t[valid_indices]
        second_origins = origin[None, :] + first_dirs * (first_t_valid[:, None] + eps)
        second_t = cast_rays(scene, second_origins.astype(np.float32), first_dirs, chunk_size)
        second_valid = np.isfinite(second_t)
        second_indices = valid_indices[second_valid]
        total_distance = first_t_valid[second_valid] + eps + second_t[second_valid]
        second[second_indices] = 1.0 / np.maximum(total_distance, 1e-6)

    return first.reshape(height, width), second.reshape(height, width)


def resize_for_compare(invdepth: np.ndarray, height: int, width: int) -> np.ndarray | None:
    if invdepth.shape == (height, width):
        return invdepth.astype(np.float32)
    old_h, old_w = invdepth.shape[:2]
    if old_h % height == 0 and old_w % width == 0:
        sy = old_h // height
        sx = old_w // width
        return invdepth.reshape(height, sy, width, sx).mean(axis=(1, 3)).astype(np.float32)
    return None


def resolve_invdepth_path(split_dir: Path, frame: dict[str, Any], key: str = "invdepth_path", folder: str = "invdepth") -> Path:
    if key in frame:
        return split_dir / frame[key]
    rel = Path(frame["file_path"])
    parts = list(rel.parts)
    if "image" in parts:
        parts[parts.index("image")] = folder
    else:
        parts.insert(0, folder)
    parts[-1] = Path(parts[-1]).with_suffix(".npy").name
    return split_dir.joinpath(*parts)


def detect_flip(
    scene: o3d.t.geometry.RaycastingScene,
    all_dir: Path,
    frame: dict[str, Any],
    height: int,
    width: int,
    eps: float,
    chunk_size: int,
    pose_convention: str,
) -> tuple[bool, dict[str, Any]]:
    target_path = resolve_invdepth_path(all_dir, frame)
    if not target_path.exists():
        return False, {"status": "skipped", "reason": f"missing first-layer target {target_path}"}

    first_pred, _ = render_invdepth_pair(
        scene,
        frame,
        height,
        width,
        eps,
        chunk_size,
        pose_convention,
    )
    target = np.load(target_path).astype(np.float32)
    target = target[..., 0] if target.ndim == 3 else target
    target_small = resize_for_compare(target, height, width)
    if target_small is None:
        return False, {
            "status": "skipped",
            "reason": f"cannot compare target shape {list(target.shape)} with {(height, width)}",
        }

    mask = target_small > 0
    if not np.any(mask):
        return False, {"status": "skipped", "reason": "first-layer target has no positive pixels"}

    mae = float(np.abs(first_pred[mask] - target_small[mask]).mean())
    mae_flip = float(np.abs(np.flipud(first_pred)[mask] - target_small[mask]).mean())
    flip = mae_flip < 0.85 * mae
    return flip, {
        "status": "ok",
        "target_path": str(target_path),
        "mae_no_flip": mae,
        "mae_flip": mae_flip,
        "flip_vertical": flip,
    }


def generate_all(args: argparse.Namespace, scene: o3d.t.geometry.RaycastingScene, flip_vertical: bool) -> dict[str, Any]:
    all_dir = args.dataset_root / "all"
    transforms_path = all_dir / "transforms.json"
    payload = load_json(transforms_path)
    frames = payload["frames"]
    out_dir = all_dir / "invdepth_second"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for idx, frame in enumerate(frames, start=1):
        basename = Path(frame["file_path"]).with_suffix(".npy").name
        output_path = out_dir / basename
        if output_path.exists() and not args.overwrite:
            second = np.load(output_path)
        else:
            _, second = render_invdepth_pair(
                scene,
                frame,
                args.height,
                args.width,
                args.eps,
                args.chunk_size,
                args.pose_convention,
            )
            if flip_vertical:
                second = np.flipud(second)
            np.save(output_path, second.astype(np.float32))
        frame["invdepth_second_path"] = f"invdepth_second/{basename}"
        valid = second > 0
        stats.append(
            {
                "frame": basename,
                "valid_pixels": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "mean_positive_invdepth": float(second[valid].mean()) if np.any(valid) else 0.0,
            }
        )
        if idx % 10 == 0 or idx == len(frames):
            print(f"Generated second-layer invdepth {idx}/{len(frames)}")

    write_json(transforms_path, payload)
    valid_counts = [item["valid_pixels"] for item in stats]
    return {
        "count": len(stats),
        "height": args.height,
        "width": args.width,
        "mean_valid_pixels": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "min_valid_pixels": int(np.min(valid_counts)) if valid_counts else 0,
        "max_valid_pixels": int(np.max(valid_counts)) if valid_counts else 0,
    }


def copy_split_invdepth_second(args: argparse.Namespace, split: str) -> dict[str, Any]:
    all_dir = args.dataset_root / "all"
    split_dir = args.dataset_root / split
    transforms_path = split_dir / "transforms.json"
    if not transforms_path.exists():
        return {"split": split, "status": "missing"}

    payload = load_json(transforms_path)
    out_dir = split_dir / "invdepth_second"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for frame in payload["frames"]:
        all_file_path = frame.get("all_file_path", frame["file_path"])
        all_basename = Path(all_file_path).with_suffix(".npy").name
        split_basename = Path(frame["file_path"]).with_suffix(".npy").name
        src = all_dir / "invdepth_second" / all_basename
        dst = out_dir / split_basename
        if not src.exists():
            raise FileNotFoundError(f"Missing all split second-layer invdepth: {src}")
        if args.overwrite or not dst.exists():
            shutil.copy2(src, dst)
        frame["invdepth_second_path"] = f"invdepth_second/{split_basename}"
        frame["all_invdepth_second_path"] = f"invdepth_second/{all_basename}"
        copied += 1

    write_json(transforms_path, payload)
    return {"split": split, "status": "ok", "count": copied}


def main() -> None:
    args = parse_args()
    all_transforms = args.dataset_root / "all" / "transforms.json"
    if not all_transforms.exists():
        raise FileNotFoundError(f"Expected all split transforms at {all_transforms}")

    scene = make_scene(args.mesh_npz)
    first_frame = load_json(all_transforms)["frames"][0]
    flip_vertical = False
    flip_summary: dict[str, Any] = {"status": "disabled"}
    if not args.no_auto_flip:
        flip_vertical, flip_summary = detect_flip(
            scene,
            args.dataset_root / "all",
            first_frame,
            args.height,
            args.width,
            args.eps,
            args.chunk_size,
            args.pose_convention,
        )
        print(f"First-depth orientation check: {flip_summary}")

    all_summary = generate_all(args, scene, flip_vertical)
    split_summaries = [copy_split_invdepth_second(args, split) for split in ("train", "val")]
    summary = {
        "dataset_root": str(args.dataset_root),
        "mesh_npz": str(args.mesh_npz),
        "height": args.height,
        "width": args.width,
        "eps": args.eps,
        "pose_convention": args.pose_convention,
        "flip_vertical": flip_vertical,
        "orientation_check": flip_summary,
        "all": all_summary,
        "splits": split_summaries,
    }
    write_json(args.dataset_root / "invdepth_second_summary.json", summary)
    print(f"Wrote summary to {args.dataset_root / 'invdepth_second_summary.json'}")


if __name__ == "__main__":
    main()
