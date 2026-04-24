#!/usr/bin/env python3
"""Batch-check whether shoe scenes are well normalized for GShell.

This script mirrors the effective camera convention in `dataset_nerf.py`:
`X_train = Rx(+90deg) @ X_raw`.

When no object geometry is present in the dataset, it estimates object shape from
camera poses plus binary masks using a silhouette visual hull.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRAIN_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class SceneData:
    name: str
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


def configure_cache(cache_root: str | None) -> None:
    if not cache_root:
        return

    os.environ["PIP_CACHE_DIR"] = os.path.join(cache_root, ".cache", "pip")
    os.environ["TORCH_HOME"] = os.path.join(cache_root, ".cache", "torch")
    os.environ["HF_HOME"] = os.path.join(cache_root, ".cache", "huggingface")
    os.environ["XDG_CACHE_HOME"] = os.path.join(cache_root, ".cache")
    os.environ["TMPDIR"] = os.path.join(cache_root, "tmp")

    for key in ["PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "TMPDIR"]:
        os.makedirs(os.environ[key], exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing scene folders, or a single scene folder with transforms.json.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Scene name(s) to analyze. If omitted, analyze every scene under --dataset-root.",
    )
    parser.add_argument("--mesh-scale", type=float, default=3.6)
    parser.add_argument("--sphere-init-norm", type=float, default=0.5)
    parser.add_argument("--coarse-res", type=int, default=96)
    parser.add_argument("--refine-res", type=int, default=192)
    parser.add_argument("--cache-root", type=str, default="/data/abelde")
    parser.add_argument("--center-thresh", type=float, default=0.05)
    parser.add_argument("--orbit-offset-thresh", type=float, default=0.05)
    parser.add_argument("--sphere-ratio-min", type=float, default=0.75)
    parser.add_argument("--sphere-ratio-max", type=float, default=1.5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the full JSON report.",
    )
    return parser.parse_args()


def resolve_scene_dirs(dataset_root: Path, scene_names: list[str] | None) -> list[Path]:
    if (dataset_root / "transforms.json").is_file():
        return [dataset_root]

    if scene_names:
        scene_dirs = [dataset_root / name for name in scene_names]
    else:
        scene_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    missing = [path for path in scene_dirs if not (path / "transforms.json").is_file()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing transforms.json for: {missing_str}")
    return scene_dirs


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1] if mask.shape[-1] == 4 else mask[..., 0]
    return mask > 127


def load_scene(scene_dir: Path) -> SceneData:
    cfg = json.load(open(scene_dir / "transforms.json"))
    frames = cfg["frames"]

    first_image = Image.open(scene_dir / frames[0]["file_path"])
    width, height = first_image.size

    c2ws_raw: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    fxs: list[float] = []
    fys: list[float] = []
    masks: list[np.ndarray] = []
    centroids: list[list[float]] = []
    area_fracs: list[float] = []

    for frame in frames:
        c2w_raw = np.array(frame["transform_matrix"], dtype=np.float64)
        c2ws_raw.append(c2w_raw)
        centers.append(c2w_raw[:3, 3])

        angle_x = float(frame["camera_angle_x"])
        fx = 0.5 * width / math.tan(angle_x / 2.0)
        fxs.append(fx)
        fys.append(fx)

        mask_path = scene_dir / frame["file_path"].replace("image/", "mask/").replace(".jpg", ".png")
        mask = load_mask(mask_path)
        masks.append(mask)

        ys, xs = np.nonzero(mask)
        centroids.append([float(xs.mean()), float(ys.mean())])
        area_fracs.append(float(mask.mean()))

    return SceneData(
        name=scene_dir.name,
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
            continue

        if candidate["centroid_reproj_in_front"] > best["centroid_reproj_in_front"]:
            best = candidate
            continue

        if (
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
    res: int,
    z_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = bounds_min.astype(np.float32)
    bounds_max = bounds_max.astype(np.float32)
    cell = (bounds_max - bounds_min) / float(res)

    xs = np.linspace(bounds_min[0], bounds_max[0], res, endpoint=False, dtype=np.float32) + cell[0] / 2.0
    ys = np.linspace(bounds_min[1], bounds_max[1], res, endpoint=False, dtype=np.float32) + cell[1] / 2.0
    zs = np.linspace(bounds_min[2], bounds_max[2], res, endpoint=False, dtype=np.float32) + cell[2] / 2.0
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    occupied = np.ones(len(points), dtype=bool)
    for c2w_raw, fx, fy, mask in zip(scene.c2ws_raw, scene.fxs, scene.fys, scene.masks):
        idx = np.flatnonzero(occupied)
        if len(idx) == 0:
            break

        keep = np.zeros(len(idx), dtype=bool)
        chunk_size = 250_000
        for start in range(0, len(idx), chunk_size):
            sel = idx[start : start + chunk_size]
            front, u, v = project_points(points[sel], c2w_raw, fx, fy, scene.cx, scene.cy, z_sign)

            chunk_keep = np.zeros(len(sel), dtype=bool)
            if np.any(front):
                front_idx = np.flatnonzero(front)
                u_int = np.rint(u[front]).astype(np.int32)
                v_int = np.rint(v[front]).astype(np.int32)
                inside = (
                    (u_int >= 0)
                    & (u_int < scene.width)
                    & (v_int >= 0)
                    & (v_int < scene.height)
                )
                if np.any(inside):
                    front_keep = np.zeros(np.count_nonzero(front), dtype=bool)
                    front_keep[inside] = mask[v_int[inside], u_int[inside]]
                    chunk_keep[front_idx] = front_keep

            keep[start : start + chunk_size] = chunk_keep

        occupied[idx] = keep

    return points[occupied].astype(np.float64), cell.astype(np.float64)


def to_training_world(points_raw: np.ndarray) -> np.ndarray:
    return points_raw @ TRAIN_ROT.T


def summarize_training_world(
    points_train: np.ndarray,
    cam_centers_train: np.ndarray,
    cam_forwards_train: np.ndarray,
    centroid_center_train: np.ndarray,
    mesh_scale: float,
    sphere_init_norm: float,
) -> dict[str, Any]:
    domain_half = 0.5 * mesh_scale

    bbox_min = points_train.min(axis=0)
    bbox_max = points_train.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_extent = bbox_max - bbox_min

    occ_center = points_train.mean(axis=0)
    radial = np.linalg.norm(points_train - bbox_center[None, :], axis=1)
    occ_radius_max = float(radial.max())
    occ_radius_mean = float(radial.mean())
    center_norm = float(np.linalg.norm(bbox_center))

    orbit_center = estimate_line_intersection(cam_centers_train, cam_forwards_train)
    orbit_radii = np.linalg.norm(cam_centers_train - orbit_center[None, :], axis=1)
    cam_radius_obj = np.linalg.norm(cam_centers_train - bbox_center[None, :], axis=1)

    look_angles = []
    for cam_center, cam_forward in zip(cam_centers_train, cam_forwards_train):
        to_obj = bbox_center - cam_center
        to_obj /= np.linalg.norm(to_obj)
        cosang = np.clip(np.dot(cam_forward, to_obj), -1.0, 1.0)
        look_angles.append(np.degrees(np.arccos(cosang)))

    bbox_max_abs = float(np.max(np.abs(np.concatenate([bbox_min, bbox_max]))))
    domain_margin = float(domain_half - bbox_max_abs)
    sphere_ratio = float(sphere_init_norm / occ_radius_max)

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_center": bbox_center,
        "bbox_extent": bbox_extent,
        "bbox_center_norm": center_norm,
        "occ_center_mean": occ_center,
        "occ_radius_max": occ_radius_max,
        "occ_radius_mean": occ_radius_mean,
        "orbit_center_from_view_axes": orbit_center,
        "orbit_center_to_obj_center": float(np.linalg.norm(orbit_center - bbox_center)),
        "centroid_center_to_obj_center": float(np.linalg.norm(centroid_center_train - bbox_center)),
        "cam_radius_about_orbit_center": orbit_radii,
        "cam_radius_about_obj_center": cam_radius_obj,
        "look_angle_deg_to_obj_center": np.asarray(look_angles, dtype=np.float64),
        "bbox_max_abs_coord": bbox_max_abs,
        "domain_margin_min": domain_margin,
        "domain_half_extent": domain_half,
        "sphere_init_norm": sphere_init_norm,
        "sphere_to_obj_radius_ratio": sphere_ratio,
        "object_smaller_than_camera": bool(occ_radius_max < cam_radius_obj.mean()),
        "obj_to_cam_radius_ratio_mean": float(occ_radius_max / cam_radius_obj.mean()),
        "recommended_uniform_scene_scale_for_sphere": sphere_ratio,
        "recommended_sphere_init_norm": occ_radius_max,
    }


def classify_scene(stats: dict[str, Any], args: argparse.Namespace) -> dict[str, bool | str]:
    centered_near_origin = stats["bbox_center_norm"] <= args.center_thresh
    orbit_centered = stats["orbit_center_to_obj_center"] <= args.orbit_offset_thresh
    fits_domain = stats["bbox_max_abs_coord"] <= stats["domain_half_extent"]
    sphere_sensible = args.sphere_ratio_min <= stats["sphere_to_obj_radius_ratio"] <= args.sphere_ratio_max
    under_scaled_for_sphere = stats["sphere_to_obj_radius_ratio"] > args.sphere_ratio_max

    if centered_near_origin and orbit_centered and fits_domain and sphere_sensible:
        verdict = "good"
    elif centered_near_origin and orbit_centered and fits_domain and under_scaled_for_sphere:
        verdict = "centered_but_under_scaled"
    elif fits_domain and under_scaled_for_sphere:
        verdict = "under_scaled_and_off_center"
    elif not fits_domain:
        verdict = "out_of_domain"
    else:
        verdict = "mixed"

    return {
        "centered_near_origin": centered_near_origin,
        "orbit_centered_on_object": orbit_centered,
        "fits_inside_domain": fits_domain,
        "sphere_init_sensible": sphere_sensible,
        "under_scaled_for_sphere": under_scaled_for_sphere,
        "verdict": verdict,
    }


def analyze_scene(scene_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene(scene_dir)
    sign_info = choose_optical_axis_sign(scene)
    z_sign = sign_info["z_sign"]

    domain_half = 0.5 * args.mesh_scale
    init_half = np.array([0.95, 0.95, 0.95], dtype=np.float64)

    coarse_bounds_min = np.maximum(sign_info["centroid_center_raw"] - init_half, -domain_half)
    coarse_bounds_max = np.minimum(sign_info["centroid_center_raw"] + init_half, domain_half)

    coarse_occ_raw, coarse_cell = carve_visual_hull(
        scene, coarse_bounds_min, coarse_bounds_max, args.coarse_res, z_sign
    )
    if len(coarse_occ_raw) == 0:
        coarse_bounds_min = np.array([-domain_half, -domain_half, -domain_half], dtype=np.float64)
        coarse_bounds_max = np.array([domain_half, domain_half, domain_half], dtype=np.float64)
        coarse_occ_raw, coarse_cell = carve_visual_hull(
            scene, coarse_bounds_min, coarse_bounds_max, args.coarse_res, z_sign
        )
    if len(coarse_occ_raw) == 0:
        raise RuntimeError(f"Visual hull is empty for {scene.name}")

    coarse_min = coarse_occ_raw.min(axis=0)
    coarse_max = coarse_occ_raw.max(axis=0)
    refine_bounds_min = np.maximum(coarse_min - 3.0 * coarse_cell, -domain_half)
    refine_bounds_max = np.minimum(coarse_max + 3.0 * coarse_cell, domain_half)

    refine_occ_raw, refine_cell = carve_visual_hull(
        scene, refine_bounds_min, refine_bounds_max, args.refine_res, z_sign
    )
    if len(refine_occ_raw) == 0:
        raise RuntimeError(f"Refined visual hull is empty for {scene.name}")

    half_cell = refine_cell / 2.0
    offsets = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    expanded_points_raw = (
        refine_occ_raw[:, None, :] + offsets[None, :, :] * half_cell[None, None, :]
    ).reshape(-1, 3)

    points_train = to_training_world(expanded_points_raw)
    cam_centers_train = to_training_world(scene.cam_centers_raw)
    cam_forwards_raw = np.stack(
        [z_sign * c2w_raw[:3, 2] / np.linalg.norm(c2w_raw[:3, 2]) for c2w_raw in scene.c2ws_raw]
    )
    cam_forwards_train = to_training_world(cam_forwards_raw)
    centroid_center_train = to_training_world(sign_info["centroid_center_raw"][None, :])[0]

    stats = summarize_training_world(
        points_train,
        cam_centers_train,
        cam_forwards_train,
        centroid_center_train,
        mesh_scale=args.mesh_scale,
        sphere_init_norm=args.sphere_init_norm,
    )
    flags = classify_scene(stats, args)

    result: dict[str, Any] = {
        "scene": scene.name,
        "n_cams": len(scene.frames),
        "image_size_hw": [scene.height, scene.width],
        "camera_angle_x_deg_unique": np.unique(
            np.round(
                np.degrees(np.array([float(frame["camera_angle_x"]) for frame in scene.frames])),
                10,
            )
        ),
        "optical_axis_sign_along_raw_camera_z": z_sign,
        "mask_area_frac": scene.mask_area_frac,
        "centroid_reproj_err_px_mean": sign_info["centroid_reproj_err_px_mean"],
        "centroid_reproj_in_front": sign_info["centroid_reproj_in_front"],
        "raw_coarse_bounds_min": coarse_bounds_min,
        "raw_coarse_bounds_max": coarse_bounds_max,
        "raw_refine_bounds_min": refine_bounds_min,
        "raw_refine_bounds_max": refine_bounds_max,
    }
    result.update(stats)
    result.update(flags)
    return result


def stats_dict(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    return obj


def compact_row(result: dict[str, Any]) -> str:
    return (
        f"{result['scene']:<58} "
        f"center={result['bbox_center_norm']:.4f} "
        f"radius={result['occ_radius_max']:.4f} "
        f"orbit_off={result['orbit_center_to_obj_center']:.4f} "
        f"sphere_ratio={result['sphere_to_obj_radius_ratio']:.2f} "
        f"margin={result['domain_margin_min']:.4f} "
        f"{result['verdict']}"
    )


def build_summary(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    center_norms = np.array([result["bbox_center_norm"] for result in results], dtype=np.float64)
    radii = np.array([result["occ_radius_max"] for result in results], dtype=np.float64)
    sphere_ratios = np.array(
        [result["sphere_to_obj_radius_ratio"] for result in results], dtype=np.float64
    )
    orbit_offsets = np.array(
        [result["orbit_center_to_obj_center"] for result in results], dtype=np.float64
    )

    return {
        "dataset_root": str(args.dataset_root),
        "mesh_scale": args.mesh_scale,
        "sphere_init_norm": args.sphere_init_norm,
        "domain_half_extent": 0.5 * args.mesh_scale,
        "scene_count": len(results),
        "counts": {
            "centered_near_origin": int(sum(result["centered_near_origin"] for result in results)),
            "orbit_centered_on_object": int(
                sum(result["orbit_centered_on_object"] for result in results)
            ),
            "fits_inside_domain": int(sum(result["fits_inside_domain"] for result in results)),
            "sphere_init_sensible": int(sum(result["sphere_init_sensible"] for result in results)),
            "under_scaled_for_sphere": int(sum(result["under_scaled_for_sphere"] for result in results)),
        },
        "aggregate_stats": {
            "bbox_center_norm": stats_dict(center_norms),
            "occ_radius_max": stats_dict(radii),
            "sphere_to_obj_radius_ratio": stats_dict(sphere_ratios),
            "orbit_center_to_obj_center": stats_dict(orbit_offsets),
        },
        "best_centered_scenes": [
            result["scene"]
            for result in sorted(results, key=lambda item: item["bbox_center_norm"])[:5]
        ],
        "worst_centered_scenes": [
            result["scene"]
            for result in sorted(results, key=lambda item: item["bbox_center_norm"], reverse=True)[:5]
        ],
        "largest_scale_up_needed_for_current_sphere": [
            {
                "scene": result["scene"],
                "scale_factor": float(result["recommended_uniform_scene_scale_for_sphere"]),
            }
            for result in sorted(
                results,
                key=lambda item: item["recommended_uniform_scene_scale_for_sphere"],
                reverse=True,
            )[:5]
        ],
    }


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_root)

    scene_dirs = resolve_scene_dirs(args.dataset_root, args.scene)
    results = []
    for scene_dir in scene_dirs:
        result = analyze_scene(scene_dir, args)
        results.append(result)
        print(compact_row(result))

    summary = build_summary(results, args)
    report = {
        "summary": summary,
        "training_world_note": (
            "All geometry and camera stats are reported in the GShell training world "
            "used by DatasetNERF: X_train = Rx(+90deg) @ X_raw."
        ),
        "scenes": [to_jsonable(result) for result in results],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {args.output_json}")

    print("\nSummary:")
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
