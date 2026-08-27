#!/usr/bin/env python3
"""Canonicalize processed GShell shoe scenes.

The processed shoe datasets store camera-to-world matrices in the raw
``transforms.json`` frame. GShell training then applies the same convention as
``DatasetNERF``:

    X_train = Rx(+90deg) @ X_raw

This script estimates each shoe's semantic frame from images, masks, and camera
poses only, then writes a new dataset root with symlinked image/mask folders and
updated camera poses. Camera rotations are rotated into the canonical frame;
camera translations are also uniformly scaled so every shoe has the same robust
length in training coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRAIN_FROM_RAW = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
RAW_FROM_TRAIN = TRAIN_FROM_RAW.T


@dataclass(frozen=True)
class SceneData:
    name: str
    path: Path
    payload: dict[str, Any]
    frames: list[dict[str, Any]]
    width: int
    height: int
    cx: float
    cy: float
    c2ws_raw: list[np.ndarray]
    cam_centers_raw: np.ndarray
    fxs: np.ndarray
    fys: np.ndarray
    masks: list[np.ndarray]
    mask_centroids: np.ndarray
    mask_area_frac: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes"),
        help="Input processed dataset root, or a single scene folder with transforms.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes_canonical"),
        help="Output dataset root to create.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Process only this scene name. May be repeated.",
    )
    parser.add_argument("--target-length", type=float, default=0.30)
    parser.add_argument("--mesh-scale", type=float, default=3.6)
    parser.add_argument("--coarse-res", type=int, default=96)
    parser.add_argument("--refine-res", type=int, default=192)
    parser.add_argument("--extent-percentile", type=float, default=1.0)
    parser.add_argument("--end-fraction", type=float, default=0.20)
    parser.add_argument("--heel-confidence-thresh", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute canonicalization and validation summaries without writing the output dataset.",
    )
    return parser.parse_args()


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


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1] if mask.shape[-1] == 4 else mask[..., 0]
    return mask > 127


def load_scene(scene_dir: Path) -> SceneData:
    with (scene_dir / "transforms.json").open("r") as f:
        payload = json.load(f)
    frames = payload["frames"]

    first_image_path = scene_dir / frames[0]["file_path"]
    with Image.open(first_image_path) as first_image:
        width, height = first_image.size

    c2ws_raw: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    fxs: list[float] = []
    fys: list[float] = []
    masks: list[np.ndarray] = []
    centroids: list[list[float]] = []
    area_fracs: list[float] = []

    for frame in frames:
        c2w_raw = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w_raw.shape != (4, 4):
            raise ValueError(f"{scene_dir.name}: transform_matrix must have shape 4x4")
        c2ws_raw.append(c2w_raw)
        centers.append(c2w_raw[:3, 3])

        angle_x = float(frame["camera_angle_x"])
        fx = 0.5 * width / math.tan(angle_x / 2.0)
        fxs.append(fx)
        fys.append(fx)

        image_path = scene_dir / frame["file_path"]
        mask_path = scene_dir / frame["file_path"].replace("image/", "mask/").replace(".jpg", ".png")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        mask = load_mask(mask_path)
        masks.append(mask)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise ValueError(f"{scene_dir.name}: empty mask for {mask_path}")
        centroids.append([float(xs.mean()), float(ys.mean())])
        area_fracs.append(float(mask.mean()))

    return SceneData(
        name=scene_dir.name,
        path=scene_dir,
        payload=payload,
        frames=frames,
        width=width,
        height=height,
        cx=width / 2.0,
        cy=height / 2.0,
        c2ws_raw=c2ws_raw,
        cam_centers_raw=np.stack(centers),
        fxs=np.asarray(fxs, dtype=np.float64),
        fys=np.asarray(fys, dtype=np.float64),
        masks=masks,
        mask_centroids=np.asarray(centroids, dtype=np.float64),
        mask_area_frac=np.asarray(area_fracs, dtype=np.float64),
    )


def estimate_line_intersection(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    a_mat = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, dirs):
        direction = direction / np.linalg.norm(direction)
        proj = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
        a_mat += proj
        b_vec += proj @ origin
    return np.linalg.solve(a_mat, b_vec)


def choose_optical_axis_sign(scene: SceneData) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for z_sign in (+1.0, -1.0):
        centroid_rays = []
        for c2w_raw, fx, fy, (u, v) in zip(
            scene.c2ws_raw, scene.fxs, scene.fys, scene.mask_centroids
        ):
            cam_dir = np.array(
                [(u - scene.cx) / fx, -(v - scene.cy) / fy, z_sign],
                dtype=np.float64,
            )
            cam_dir /= np.linalg.norm(cam_dir)
            world_dir = c2w_raw[:3, :3] @ cam_dir
            world_dir /= np.linalg.norm(world_dir)
            centroid_rays.append(world_dir)

        centroid_rays_arr = np.stack(centroid_rays)
        centroid_center_raw = estimate_line_intersection(scene.cam_centers_raw, centroid_rays_arr)

        reproj_errs = []
        in_front = 0
        for c2w_raw, fx, fy, (u_gt, v_gt) in zip(
            scene.c2ws_raw, scene.fxs, scene.fys, scene.mask_centroids
        ):
            pc = (centroid_center_raw[None, :] - c2w_raw[:3, 3]) @ c2w_raw[:3, :3]
            depth = z_sign * pc[0, 2]
            if depth > 1e-8:
                in_front += 1
                u = fx * (pc[0, 0] / depth) + scene.cx
                v = scene.cy - fy * (pc[0, 1] / depth)
                reproj_errs.append(math.hypot(u - u_gt, v - v_gt))
            else:
                reproj_errs.append(1e9)

        candidate = {
            "z_sign": int(z_sign),
            "centroid_center_raw": centroid_center_raw,
            "centroid_rays_raw": centroid_rays_arr,
            "centroid_reproj_err_px_mean": float(np.mean(reproj_errs)),
            "centroid_reproj_in_front": int(in_front),
        }
        if best is None:
            best = candidate
        elif candidate["centroid_reproj_in_front"] > best["centroid_reproj_in_front"]:
            best = candidate
        elif (
            candidate["centroid_reproj_in_front"] == best["centroid_reproj_in_front"]
            and candidate["centroid_reproj_err_px_mean"] < best["centroid_reproj_err_px_mean"]
        ):
            best = candidate

    assert best is not None
    return best


def project_points(
    points: np.ndarray,
    c2w_raw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc = (points - c2w_raw[:3, 3]) @ c2w_raw[:3, :3]
    depth = z_sign * pc[:, 2]
    front = depth > 1e-8
    u = np.full(depth.shape, np.nan, dtype=np.float64)
    v = np.full(depth.shape, np.nan, dtype=np.float64)
    if np.any(front):
        u[front] = fx * (pc[front, 0] / depth[front]) + cx
        v[front] = cy - fy * (pc[front, 1] / depth[front])
    return front, u, v


def carve_visual_hull(
    scene: SceneData,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    resolution: int,
    z_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    cell = (bounds_max - bounds_min) / float(resolution)

    xs = np.linspace(bounds_min[0], bounds_max[0], resolution, endpoint=False) + cell[0] / 2.0
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution, endpoint=False) + cell[1] / 2.0
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution, endpoint=False) + cell[2] / 2.0
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    occupied = np.ones(points.shape[0], dtype=bool)
    for c2w_raw, fx, fy, mask in zip(scene.c2ws_raw, scene.fxs, scene.fys, scene.masks):
        idx = np.flatnonzero(occupied)
        if idx.size == 0:
            break

        keep = np.zeros(idx.size, dtype=bool)
        chunk_size = 250_000
        for start in range(0, idx.size, chunk_size):
            sel = idx[start : start + chunk_size]
            front, u, v = project_points(points[sel], c2w_raw, fx, fy, scene.cx, scene.cy, z_sign)

            chunk_keep = np.zeros(sel.size, dtype=bool)
            finite_front = front & np.isfinite(u) & np.isfinite(v)
            if np.any(finite_front):
                front_idx = np.flatnonzero(finite_front)
                u_round = np.rint(u[finite_front])
                v_round = np.rint(v[finite_front])
                inside = (
                    (u_round >= 0)
                    & (u_round < scene.width)
                    & (v_round >= 0)
                    & (v_round < scene.height)
                )
                if np.any(inside):
                    u_int = u_round[inside].astype(np.int32)
                    v_int = v_round[inside].astype(np.int32)
                    front_keep = np.zeros(np.count_nonzero(finite_front), dtype=bool)
                    front_keep[inside] = mask[v_int, u_int]
                    chunk_keep[front_idx] = front_keep

            keep[start : start + chunk_size] = chunk_keep
        occupied[idx] = keep

    return points[occupied].astype(np.float64), cell.astype(np.float64)


def estimate_visual_hull(scene: SceneData, args: argparse.Namespace) -> dict[str, Any]:
    sign_info = choose_optical_axis_sign(scene)
    z_sign = sign_info["z_sign"]
    domain_half = 0.5 * args.mesh_scale
    init_half = np.array([0.95, 0.95, 0.95], dtype=np.float64)

    coarse_bounds_min = np.maximum(sign_info["centroid_center_raw"] - init_half, -domain_half)
    coarse_bounds_max = np.minimum(sign_info["centroid_center_raw"] + init_half, domain_half)
    coarse_occ_raw, coarse_cell = carve_visual_hull(
        scene, coarse_bounds_min, coarse_bounds_max, args.coarse_res, z_sign
    )

    if coarse_occ_raw.size == 0:
        coarse_bounds_min = np.full(3, -domain_half, dtype=np.float64)
        coarse_bounds_max = np.full(3, domain_half, dtype=np.float64)
        coarse_occ_raw, coarse_cell = carve_visual_hull(
            scene, coarse_bounds_min, coarse_bounds_max, args.coarse_res, z_sign
        )
    if coarse_occ_raw.size == 0:
        raise RuntimeError(f"{scene.name}: coarse visual hull is empty")

    refine_bounds_min = np.maximum(coarse_occ_raw.min(axis=0) - 3.0 * coarse_cell, -domain_half)
    refine_bounds_max = np.minimum(coarse_occ_raw.max(axis=0) + 3.0 * coarse_cell, domain_half)
    refine_occ_raw, refine_cell = carve_visual_hull(
        scene, refine_bounds_min, refine_bounds_max, args.refine_res, z_sign
    )
    if refine_occ_raw.size == 0:
        raise RuntimeError(f"{scene.name}: refined visual hull is empty")

    half_cell = refine_cell / 2.0
    offsets = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    expanded_raw = (
        refine_occ_raw[:, None, :] + offsets[None, :, :] * half_cell[None, None, :]
    ).reshape(-1, 3)
    expanded_train = expanded_raw @ TRAIN_FROM_RAW.T

    return {
        "points_raw": expanded_raw,
        "points_train": expanded_train,
        "refine_cell": refine_cell,
        "coarse_bounds_min": coarse_bounds_min,
        "coarse_bounds_max": coarse_bounds_max,
        "refine_bounds_min": refine_bounds_min,
        "refine_bounds_max": refine_bounds_max,
        "optical_axis_sign": z_sign,
        "centroid_reproj_err_px_mean": sign_info["centroid_reproj_err_px_mean"],
        "centroid_reproj_in_front": sign_info["centroid_reproj_in_front"],
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def robust_min_max(values: np.ndarray, low_percentile: float) -> tuple[float, float]:
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, 100.0 - low_percentile))
    return low, high


def end_asymmetry_score(
    points_train: np.ndarray,
    length_scores: np.ndarray,
    end_mask: np.ndarray,
) -> dict[str, float]:
    y_values = points_train[:, 1]
    end_y = y_values[end_mask]
    if end_y.size == 0:
        return {
            "upper_fraction": 0.0,
            "vertical_extent_norm": 0.0,
            "top_reach": 0.0,
            "score": 0.0,
            "count": 0,
        }

    global_y05 = float(np.percentile(y_values, 5.0))
    global_y35 = float(np.percentile(y_values, 35.0))
    global_y95 = float(np.percentile(y_values, 95.0))
    global_extent = max(global_y95 - global_y05, 1e-8)

    end_y05 = float(np.percentile(end_y, 5.0))
    end_y95 = float(np.percentile(end_y, 95.0))
    upper_fraction = float(np.mean(end_y <= global_y35))
    vertical_extent_norm = float((end_y95 - end_y05) / global_extent)
    top_reach = float((global_y95 - end_y05) / global_extent)
    score = upper_fraction + 0.50 * vertical_extent_norm + 0.35 * top_reach
    return {
        "upper_fraction": upper_fraction,
        "vertical_extent_norm": vertical_extent_norm,
        "top_reach": top_reach,
        "score": float(score),
        "count": int(end_y.size),
    }


def estimate_canonical_frame(points_train: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    horizontal = points_train[:, [0, 2]]
    horizontal_centered = horizontal - horizontal.mean(axis=0, keepdims=True)
    covariance = np.cov(horizontal_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_2d = eigenvectors[:, int(np.argmax(eigenvalues))]
    principal_2d = normalize(principal_2d)

    raw_length_dir = np.array([principal_2d[0], 0.0, principal_2d[1]], dtype=np.float64)
    raw_length_scores = points_train @ raw_length_dir
    low_cut = float(np.percentile(raw_length_scores, args.end_fraction * 100.0))
    high_cut = float(np.percentile(raw_length_scores, 100.0 - args.end_fraction * 100.0))

    negative_score = end_asymmetry_score(
        points_train,
        raw_length_scores,
        raw_length_scores <= low_cut,
    )
    positive_score = end_asymmetry_score(
        points_train,
        raw_length_scores,
        raw_length_scores >= high_cut,
    )

    if positive_score["score"] > negative_score["score"]:
        heel_end = "positive"
        canonical_x_dir = -raw_length_dir
    else:
        heel_end = "negative"
        canonical_x_dir = raw_length_dir

    canonical_x_dir = normalize(canonical_x_dir)
    canonical_y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    canonical_z_dir = normalize(np.cross(canonical_x_dir, canonical_y_dir))
    basis_train_to_canonical = np.stack(
        [canonical_x_dir, canonical_y_dir, canonical_z_dir],
        axis=0,
    )

    unscaled_coords = points_train @ basis_train_to_canonical.T
    low_p = float(args.extent_percentile)
    q_min = np.array([robust_min_max(unscaled_coords[:, i], low_p)[0] for i in range(3)])
    q_max = np.array([robust_min_max(unscaled_coords[:, i], low_p)[1] for i in range(3)])
    q_center = (q_min + q_max) * 0.5
    robust_extent = q_max - q_min
    robust_length = float(robust_extent[0])
    if robust_length <= 1e-8:
        raise ValueError("Robust shoe length is zero")

    scale = float(args.target_length / robust_length)
    center_train = basis_train_to_canonical.T @ q_center
    canonical_points = scale * ((points_train - center_train[None, :]) @ basis_train_to_canonical.T)
    can_low = np.array([robust_min_max(canonical_points[:, i], low_p)[0] for i in range(3)])
    can_high = np.array([robust_min_max(canonical_points[:, i], low_p)[1] for i in range(3)])
    canonical_extent = can_high - can_low

    score_sum = abs(positive_score["score"]) + abs(negative_score["score"])
    heel_confidence = (
        abs(float(positive_score["score"] - negative_score["score"])) / max(score_sum, 1e-8)
    )
    horizontal_ratio = robust_extent[0] / max(robust_extent[2], 1e-8)
    needs_review = bool(
        heel_confidence < args.heel_confidence_thresh
        or horizontal_ratio < 1.15
        or canonical_extent[0] <= canonical_extent[2]
    )

    canonical_transform = np.eye(4, dtype=np.float64)
    canonical_transform[:3, :3] = scale * basis_train_to_canonical
    canonical_transform[:3, 3] = -scale * basis_train_to_canonical @ center_train

    return {
        "basis_train_to_canonical": basis_train_to_canonical,
        "center_train": center_train,
        "scale": scale,
        "target_length": float(args.target_length),
        "pre_scale_robust_extent": robust_extent,
        "post_scale_robust_extent": canonical_extent,
        "robust_bounds_before_scale_min": q_min,
        "robust_bounds_before_scale_max": q_max,
        "canonical_bounds_min": can_low,
        "canonical_bounds_max": can_high,
        "canonical_transform_train": canonical_transform,
        "heel_end_before_sign": heel_end,
        "heel_confidence": float(heel_confidence),
        "negative_end_score": negative_score,
        "positive_end_score": positive_score,
        "horizontal_length_width_ratio": float(horizontal_ratio),
        "needs_review": needs_review,
    }


def transform_camera_pose_raw(
    c2w_raw: np.ndarray,
    basis_train_to_canonical: np.ndarray,
    center_train: np.ndarray,
    scale: float,
) -> np.ndarray:
    r_train = TRAIN_FROM_RAW @ c2w_raw[:3, :3]
    t_train = TRAIN_FROM_RAW @ c2w_raw[:3, 3]
    r_canonical = basis_train_to_canonical @ r_train
    t_canonical = scale * (basis_train_to_canonical @ (t_train - center_train))

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = RAW_FROM_TRAIN @ r_canonical
    out[:3, 3] = RAW_FROM_TRAIN @ t_canonical
    return out


def rotation_validation(payload: dict[str, Any]) -> dict[str, Any]:
    dets = []
    ortho_errors = []
    for frame in payload["frames"]:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rotation = matrix[:3, :3]
        dets.append(float(np.linalg.det(rotation)))
        ortho_errors.append(float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")))

    return {
        "rotation_det_min": float(np.min(dets)),
        "rotation_det_max": float(np.max(dets)),
        "rotation_orthonormal_error_max": float(np.max(ortho_errors)),
        "rotations_passed": bool(
            min(dets) > 0.999
            and max(dets) < 1.001
            and max(ortho_errors) < 1e-5
        ),
    }


def robust_axis_extent(points_train: np.ndarray, low_percentile: float) -> np.ndarray:
    bounds_min = np.array(
        [robust_min_max(points_train[:, i], low_percentile)[0] for i in range(3)],
        dtype=np.float64,
    )
    bounds_max = np.array(
        [robust_min_max(points_train[:, i], low_percentile)[1] for i in range(3)],
        dtype=np.float64,
    )
    return bounds_max - bounds_min


def validate_output_poses(
    payload: dict[str, Any],
    target_length: float,
    canonical_extent: np.ndarray,
) -> dict[str, Any]:
    validation = rotation_validation(payload)
    length_error_fraction = abs(float(canonical_extent[0]) - target_length) / max(target_length, 1e-8)
    horizontal_length_is_x = bool(canonical_extent[0] > canonical_extent[2])
    validation.update(
        {
            "canonical_length_error_fraction": float(length_error_fraction),
            "horizontal_length_is_x": horizontal_length_is_x,
            "passed": bool(
                validation["rotations_passed"]
                and length_error_fraction <= 0.01
                and horizontal_length_is_x
            ),
        }
    )
    return validation


def validate_written_scene(output_scene: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene(output_scene)
    structure_validation = {
        "frame_count": len(scene.frames),
        "frame_count_is_36": len(scene.frames) == 36,
        "image_symlink_valid": (output_scene / "image").is_symlink()
        and (output_scene / "image").exists(),
        "mask_symlink_valid": (output_scene / "mask").is_symlink()
        and (output_scene / "mask").exists(),
    }
    frame_paths_valid = True
    for frame in scene.frames:
        image_path = output_scene / frame["file_path"]
        mask_path = output_scene / frame["file_path"].replace("image/", "mask/").replace(".jpg", ".png")
        frame_paths_valid = frame_paths_valid and image_path.exists() and mask_path.exists()
    structure_validation["frame_image_mask_paths_valid"] = frame_paths_valid

    hull = estimate_visual_hull(scene, args)
    recomputed_extent = robust_axis_extent(
        hull["points_train"],
        low_percentile=float(args.extent_percentile),
    )
    pose_validation = validate_output_poses(
        scene.payload,
        target_length=float(args.target_length),
        canonical_extent=recomputed_extent,
    )

    passed = bool(
        structure_validation["frame_count_is_36"]
        and structure_validation["image_symlink_valid"]
        and structure_validation["mask_symlink_valid"]
        and structure_validation["frame_image_mask_paths_valid"]
        and pose_validation["passed"]
    )
    return {
        **structure_validation,
        **pose_validation,
        "recomputed_axis_extent": recomputed_extent,
        "recomputed_canonical_length": float(recomputed_extent[0]),
        "passed": passed,
    }


def write_scene_output(
    scene: SceneData,
    output_scene: Path,
    output_payload: dict[str, Any],
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    if output_scene.resolve() == scene.path.resolve():
        raise ValueError("Output scene path must not be the same as the input scene path")
    if output_scene.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene exists; pass --overwrite: {output_scene}")
        shutil.rmtree(output_scene)

    output_scene.mkdir(parents=True, exist_ok=True)
    for dirname in ["image", "mask"]:
        source = scene.path / dirname
        target = output_scene / dirname
        os.symlink(source, target, target_is_directory=True)

    with (output_scene / "transforms.json").open("w") as f:
        json.dump(to_jsonable(output_payload), f, indent=2)
        f.write("\n")
    with (output_scene / "canonicalization.json").open("w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
        f.write("\n")


def canonicalize_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene(scene_dir)
    hull = estimate_visual_hull(scene, args)
    frame = estimate_canonical_frame(hull["points_train"], args)

    output_payload = json.loads(json.dumps(scene.payload))
    for out_frame, c2w_raw in zip(output_payload["frames"], scene.c2ws_raw):
        c2w_out = transform_camera_pose_raw(
            c2w_raw,
            frame["basis_train_to_canonical"],
            frame["center_train"],
            frame["scale"],
        )
        out_frame["transform_matrix"] = c2w_out.tolist()

    prewrite_validation = validate_output_poses(
        output_payload,
        target_length=float(args.target_length),
        canonical_extent=frame["post_scale_robust_extent"],
    )

    output_scene = args.output_root / scene.name
    metadata = {
        "scene": scene.name,
        "source_scene": str(scene.path),
        "output_scene": str(output_scene),
        "target_convention_training_world": {
            "+X": "heel_to_toe_length",
            "+Y": "sole_base_bottom_side",
            "-Y": "opening_up_side",
            "+Z": "width",
        },
        "settings": {
            "target_length": float(args.target_length),
            "mesh_scale": float(args.mesh_scale),
            "coarse_res": int(args.coarse_res),
            "refine_res": int(args.refine_res),
            "extent_percentile": float(args.extent_percentile),
            "end_fraction": float(args.end_fraction),
            "heel_confidence_thresh": float(args.heel_confidence_thresh),
        },
        "visual_hull": {
            "num_points_expanded": int(hull["points_train"].shape[0]),
            "optical_axis_sign": int(hull["optical_axis_sign"]),
            "centroid_reproj_err_px_mean": hull["centroid_reproj_err_px_mean"],
            "centroid_reproj_in_front": hull["centroid_reproj_in_front"],
            "raw_coarse_bounds_min": hull["coarse_bounds_min"],
            "raw_coarse_bounds_max": hull["coarse_bounds_max"],
            "raw_refine_bounds_min": hull["refine_bounds_min"],
            "raw_refine_bounds_max": hull["refine_bounds_max"],
        },
        "canonicalization": frame,
        "validation": {
            "prewrite_transformed_hull": prewrite_validation,
            "postwrite_recomputed_hull": None,
        },
    }

    validation = prewrite_validation
    summary_extent = frame["post_scale_robust_extent"]
    if not args.dry_run:
        write_scene_output(scene, output_scene, output_payload, metadata, args.overwrite)
        validation = validate_written_scene(output_scene, args)
        summary_extent = np.asarray(validation["recomputed_axis_extent"], dtype=np.float64)
        metadata["validation"]["postwrite_recomputed_hull"] = validation
        with (output_scene / "canonicalization.json").open("w") as f:
            json.dump(to_jsonable(metadata), f, indent=2)
            f.write("\n")

    validation_passed = bool(validation["passed"])
    status = "ok"
    if not validation_passed:
        status = "failed"
    elif frame["needs_review"]:
        status = "needs_review"

    return {
        "scene": scene.name,
        "status": status,
        "frames": len(scene.frames),
        "output_scene": str(output_scene),
        "scale": frame["scale"],
        "target_length": float(args.target_length),
        "canonical_length": float(summary_extent[0]),
        "canonical_width": float(summary_extent[2]),
        "canonical_height": float(summary_extent[1]),
        "length_width_ratio": frame["horizontal_length_width_ratio"],
        "heel_end": frame["heel_end_before_sign"],
        "heel_confidence": frame["heel_confidence"],
        "needs_review": frame["needs_review"],
        "validation_passed": validation_passed,
        "frame_count_is_36": validation.get("frame_count_is_36"),
        "image_symlink_valid": validation.get("image_symlink_valid"),
        "mask_symlink_valid": validation.get("mask_symlink_valid"),
        "frame_image_mask_paths_valid": validation.get("frame_image_mask_paths_valid"),
        "rotations_passed": validation.get("rotations_passed"),
        "rotation_det_min": validation["rotation_det_min"],
        "rotation_det_max": validation["rotation_det_max"],
        "rotation_orthonormal_error_max": validation["rotation_orthonormal_error_max"],
        "canonical_length_error_fraction": validation["canonical_length_error_fraction"],
        "horizontal_length_is_x": validation["horizontal_length_is_x"],
        "canonicalization_json": str(output_scene / "canonicalization.json"),
        "transforms_json": str(output_scene / "transforms.json"),
        "error": "" if validation_passed else "validation failed",
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    fieldnames = [
        "scene",
        "status",
        "validation_passed",
        "needs_review",
        "scale",
        "target_length",
        "canonical_length",
        "canonical_width",
        "canonical_height",
        "length_width_ratio",
        "heel_end",
        "heel_confidence",
        "canonical_length_error_fraction",
        "frame_count_is_36",
        "image_symlink_valid",
        "mask_symlink_valid",
        "frame_image_mask_paths_valid",
        "rotations_passed",
        "rotation_det_min",
        "rotation_det_max",
        "rotation_orthonormal_error_max",
        "horizontal_length_is_x",
        "frames",
        "output_scene",
        "transforms_json",
        "canonicalization_json",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "target_length": float(args.target_length),
        "scene_count": len(rows),
        "ok_count": int(sum(row.get("status") == "ok" for row in rows)),
        "needs_review_count": int(sum(row.get("needs_review", False) for row in rows)),
        "failed_count": int(sum(row.get("status") == "failed" for row in rows)),
        "validation_passed_count": int(sum(row.get("validation_passed", False) for row in rows)),
        "rows": rows,
    }
    with (output_root / "summary.json").open("w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
        f.write("\n")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    return obj


def compact_row(row: dict[str, Any]) -> str:
    if row.get("status") == "failed":
        return f"{row['scene']:<58} failed {row.get('error', '')}"
    return (
        f"{row['scene']:<58} "
        f"{row['status']:<12} "
        f"len={row['canonical_length']:.4f} "
        f"width={row['canonical_width']:.4f} "
        f"height={row['canonical_height']:.4f} "
        f"scale={row['scale']:.3f} "
        f"heel_conf={row['heel_confidence']:.3f} "
        f"valid={row['validation_passed']}"
    )


def main() -> None:
    args = parse_args()
    if args.target_length <= 0.0:
        raise ValueError("--target-length must be positive")
    if args.output_root.resolve() == args.input_root.resolve():
        raise ValueError("--output-root must differ from --input-root")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if args.output_root.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output root exists; pass --overwrite: {args.output_root}")

    scene_dirs = resolve_scene_dirs(args.input_root, args.scene)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{index}/{len(scene_dirs)}] {scene_dir.name}")
        try:
            row = canonicalize_scene(scene_dir, args)
        except Exception as exc:
            row = {
                "scene": scene_dir.name,
                "status": "failed",
                "needs_review": True,
                "validation_passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "output_scene": str(args.output_root / scene_dir.name),
            }
        rows.append(row)
        print("  " + compact_row(row))

    if not args.dry_run:
        write_summary(args.output_root, rows, args)
        print(f"\nWrote canonicalized dataset: {args.output_root}")
        print(f"Wrote summary: {args.output_root / 'summary.csv'}")
    else:
        print("\nDry run complete; no files were written.")


if __name__ == "__main__":
    main()
