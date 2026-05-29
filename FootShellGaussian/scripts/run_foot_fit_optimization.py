#!/usr/bin/env python3
"""Run Section 1.4 optimization-based foot fitting for trained shoe meshes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback
from typing import Iterable, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOOTSHELL_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FOOTSHELL_ROOT) not in sys.path:
    sys.path.insert(0, str(FOOTSHELL_ROOT))

from FootShellGaussian.foot_prior import (  # noqa: E402
    FootAlignment,
    FootFitOptimizerConfig,
    MeshData,
    evaluate_fit_numpy,
    load_pseudo_cavity_from_support_json,
    load_triangle_mesh,
    optimize_foot_fit,
    sole_block_masks,
    write_obj_mesh,
)


DEFAULT_MESH_ROOT = PROJECT_ROOT / "baselines/GShell/output/turntable-512-768"
DEFAULT_SUPPORT_ROOT = PROJECT_ROOT / "baselines/GShell/output/support_footbed_analysis_v4"
DEFAULT_LEGACY_SUPPORT_ROOT = PROJECT_ROOT / "baselines/GShell/output/support_footbed_analysis"
DEFAULT_BASELINE_ALIGNMENT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_turntable-512-768"
DEFAULT_WARM_START_ALIGNMENT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_optimized_turntable-512-768"
DEFAULT_V4_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_optimized_v4_hybrid_turntable-512-768"
DEFAULT_V5_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_v5_turntable-512-768"
DEFAULT_SIMPLE_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_simple_turntable-512-768"
DEFAULT_LEGACY_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/foot_alignment_legacy_turntable-512-768"
DEFAULT_FINAL_1D_SUPPORT_ROOT = PROJECT_ROOT / "baselines/GShell/output/final/support_1d"
DEFAULT_FINAL_1D_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/final/foot_alignment_1d_heightmap"
DEFAULT_FOOT_OBJ = PROJECT_ROOT / "baselines/SUPR/output/debug_playground/supr_male_right_foot_neutral.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT_ROOT)
    parser.add_argument("--baseline-alignment-root", type=Path, default=DEFAULT_BASELINE_ALIGNMENT_ROOT)
    parser.add_argument("--warm-start-alignment-root", type=Path, default=DEFAULT_WARM_START_ALIGNMENT_ROOT)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--fit-version", choices=["v4", "v5", "simple", "legacy", "final_1d"], default="v4")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--foot-obj", type=Path, default=DEFAULT_FOOT_OBJ)
    parser.add_argument("--scene", action="append", help="Scene directory name, for example Foo_turntable. Repeatable.")
    parser.add_argument("--shoe-name", action="append", help="Shoe name without _turntable. Repeatable.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--style-mode", choices=["auto", "normal", "boot"], default="auto")
    parser.add_argument("--adam-steps", type=int, default=160)
    parser.add_argument("--lbfgs-steps", type=int, default=25)
    parser.add_argument("--adam-lr", type=float, default=0.035)
    args = parser.parse_args()
    if args.output_root is None:
        if args.fit_version == "v5":
            args.output_root = DEFAULT_V5_OUTPUT_ROOT
        elif args.fit_version == "simple":
            args.output_root = DEFAULT_SIMPLE_OUTPUT_ROOT
        elif args.fit_version == "legacy":
            args.output_root = DEFAULT_LEGACY_OUTPUT_ROOT
        elif args.fit_version == "final_1d":
            args.output_root = DEFAULT_FINAL_1D_OUTPUT_ROOT
        else:
            args.output_root = DEFAULT_V4_OUTPUT_ROOT
    if args.fit_version == "legacy" and args.support_root == DEFAULT_SUPPORT_ROOT:
        args.support_root = DEFAULT_LEGACY_SUPPORT_ROOT
    if args.fit_version == "final_1d" and args.support_root == DEFAULT_SUPPORT_ROOT:
        args.support_root = DEFAULT_FINAL_1D_SUPPORT_ROOT
    if args.fit_version in {"v5", "simple", "legacy", "final_1d"}:
        args.no_warm_start = True
    return args


def scene_to_shoe_name(scene_name: str) -> str:
    return scene_name[:-10] if scene_name.endswith("_turntable") else scene_name


def shoe_to_scene_name(shoe_name: str) -> str:
    return shoe_name if shoe_name.endswith("_turntable") else f"{shoe_name}_turntable"


def discover_scenes(mesh_root: Path, scenes: Optional[Iterable[str]], shoes: Optional[Iterable[str]]) -> list[str]:
    requested: list[str] = []
    if scenes:
        requested.extend(str(scene) for scene in scenes)
    if shoes:
        requested.extend(shoe_to_scene_name(str(shoe)) for shoe in shoes)
    if requested:
        return sorted(dict.fromkeys(requested))

    found = []
    for scene_dir in sorted(mesh_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if (scene_dir / "mesh/mesh.obj").exists() and (scene_dir / "mesh_watertight/mesh.obj").exists():
            found.append(scene_dir.name)
    return found


def process_scene(args: argparse.Namespace, scene_name: str) -> dict[str, object]:
    shoe_name = scene_to_shoe_name(scene_name)
    scene_dir = args.mesh_root / scene_name
    support_json = args.support_root / scene_name / "support_footprint.json"
    support_npz = None if args.fit_version == "legacy" else args.support_root / scene_name / "pseudo_footbed_heightmap.npz"
    baseline_alignment_json = args.baseline_alignment_root / shoe_name / "alignment.json"
    warm_start_alignment_json = args.warm_start_alignment_root / shoe_name / "alignment_optimized.json"
    input_alignment_json = (
        baseline_alignment_json
        if args.no_warm_start or not warm_start_alignment_json.exists()
        else warm_start_alignment_json
    )
    output_dir = args.output_root / shoe_name

    row: dict[str, object] = {
        "scene": scene_name,
        "shoe_name": shoe_name,
        "status": "pending",
        "scene_dir": str(scene_dir),
        "support_json": str(support_json),
        "support_npz": "" if support_npz is None else str(support_npz),
        "baseline_alignment_json": str(baseline_alignment_json),
        "warm_start_alignment_json": str(warm_start_alignment_json),
        "input_alignment_json": str(input_alignment_json),
        "warm_start_used": bool(input_alignment_json == warm_start_alignment_json),
        "fit_version": str(args.fit_version),
        "output_dir": str(output_dir),
    }

    if output_dir.exists() and (output_dir / "fit_metrics.json").exists() and not args.overwrite:
        row.update(status="skipped_existing")
        return row

    mesh_path = scene_dir / "mesh_watertight" / "mesh.obj"
    open_mesh_path = scene_dir / "mesh" / "mesh.obj"
    required_paths = [
        ("watertight mesh", mesh_path),
        ("open mesh", open_mesh_path),
        ("support JSON", support_json),
        ("input alignment", input_alignment_json),
        ("foot OBJ", args.foot_obj),
    ]
    if support_npz is not None:
        required_paths.append(("support footbed NPZ", support_npz))
    for label, path in required_paths:
        if not path.exists():
            row.update(status="missing_input", error=f"Missing {label}: {path}")
            return row

    output_dir.mkdir(parents=True, exist_ok=True)
    stale_error = output_dir / "error.txt"
    if stale_error.exists():
        stale_error.unlink()
    shoe_mesh = load_triangle_mesh(mesh_path)
    open_mesh = load_triangle_mesh(open_mesh_path)
    foot_mesh = load_triangle_mesh(args.foot_obj)
    baseline_alignment = FootAlignment.from_json(input_alignment_json)
    cavity = load_pseudo_cavity_from_support_json(
        support_json,
        shoe_mesh,
        footbed_npz_path=support_npz,
        use_sidecar_npz=support_npz is not None,
    )
    optimizer_config = FootFitOptimizerConfig(
        device=args.device,
        style_mode=args.style_mode,
        fit_version=args.fit_version,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
    )

    result = optimize_foot_fit(
        foot_mesh=foot_mesh,
        shoe_mesh=shoe_mesh,
        baseline_alignment=baseline_alignment,
        cavity=cavity,
        config=optimizer_config,
    )

    optimized_mesh = MeshData(vertices=result.aligned_vertices, faces=foot_mesh.faces)
    baseline_mesh = MeshData(vertices=result.baseline_vertices, faces=foot_mesh.faces)
    result.alignment.save_json(output_dir / "alignment_optimized.json")
    write_obj_mesh(
        output_dir / "foot_aligned_optimized.obj",
        optimized_mesh,
        comments=["Optimized SUPR-Foot mesh in GShell shoe coordinates."],
    )
    _write_colored_overlay(
        output_dir / "foot_inside_shoe_optimized.ply",
        shoe_mesh,
        baseline_mesh,
        optimized_mesh,
    )
    result.cavity.to_npz(output_dir / "pseudo_cavity.npz", metrics_json=result.metrics)

    baseline_offset = float(result.metrics["offset_bounds"]["init"])
    baseline_cavity_metrics = evaluate_fit_numpy(result.baseline_vertices, cavity, baseline_offset, optimizer_config)
    optimized_cavity_metrics = evaluate_fit_numpy(
        result.aligned_vertices,
        cavity,
        float(result.metrics["optimized_params"]["footbed_offset"]),
        optimizer_config,
    )

    sole_masks = sole_block_masks(
        shoe_mesh,
        result.aligned_vertices,
        result.alignment,
        footprint_margin=0.012,
        plantar_band=result.alignment.config.plantar_band,
    )
    result.metrics.update(
        {
            "scene": scene_name,
            "shoe_name": shoe_name,
            "inputs": {
                "mesh_path": str(mesh_path),
                "open_mesh_path": str(open_mesh_path),
                "support_json": str(support_json),
                "support_npz": "" if support_npz is None else str(support_npz),
                "baseline_alignment_json": str(baseline_alignment_json),
                "warm_start_alignment_json": str(warm_start_alignment_json),
                "input_alignment_json": str(input_alignment_json),
                "warm_start_used": bool(input_alignment_json == warm_start_alignment_json),
                "fit_version": str(args.fit_version),
            },
            "extra_cavity_metrics": {
                "baseline": baseline_cavity_metrics,
                "optimized": optimized_cavity_metrics,
            },
            "sole_block": {
                "vertex_count": int(np.asarray(sole_masks["vertex_sole_block"], dtype=bool).sum()),
                "vertex_fraction": float(np.asarray(sole_masks["vertex_sole_block"], dtype=bool).mean()),
                "face_count": int(np.asarray(sole_masks["face_sole_block"], dtype=bool).sum()),
                "face_fraction": float(np.asarray(sole_masks["face_sole_block"], dtype=bool).mean()),
            },
        }
    )

    with (output_dir / "fit_metrics.json").open("w") as f:
        json.dump(result.metrics, f, indent=2)
        f.write("\n")

    _write_plots(output_dir, shoe_name, shoe_mesh, open_mesh, baseline_mesh, optimized_mesh, result)
    _make_local_contact_sheet(output_dir)

    row.update(
        status="ok",
        warm_start_used=bool(input_alignment_json == warm_start_alignment_json),
        fit_version=str(args.fit_version),
        loss_before=float(result.metrics["baseline_loss"]["total"]),
        loss_after=float(result.metrics["optimized_loss"]["total"]),
        loss_improved=bool(result.metrics["loss_improved"]),
        scale=float(result.alignment.scale),
        scale_delta=float(result.metrics["optimized_params"]["scale_delta"]),
        yaw_delta_degrees=float(result.metrics["optimized_params"]["yaw_degrees"]),
        pitch_delta_degrees=float(result.metrics["optimized_params"]["pitch_degrees"]),
        roll_delta_degrees=float(result.metrics["optimized_params"]["roll_degrees"]),
        tx=float(result.metrics["optimized_params"]["tx"]),
        ty=float(result.metrics["optimized_params"]["ty"]),
        tz=float(result.metrics["optimized_params"]["tz"]),
        footbed_offset=float(result.metrics["optimized_params"]["footbed_offset"]),
        style_mode=str(result.metrics["shoe_style"]["mode"]),
        style_height_ratio=float(result.metrics["shoe_style"]["height_ratio"]),
        multistart_count=int(result.metrics["shoe_style"]["multistart_count"]),
        cavity_violation_fraction=float(optimized_cavity_metrics["cavity_violation_fraction"]),
        footbed_mask_violation_fraction=float(result.metrics["optimized_loss"]["footbed_mask_violation_fraction"]),
        plantar_clearance_mean=float(result.metrics["optimized_loss"]["plantar_clearance_mean"]),
        side_gap_abs_mean=float(result.metrics["optimized_loss"]["side_gap_abs_mean"]),
        side_total_clearance_mean=float(result.metrics["optimized_loss"]["side_total_clearance_mean"]),
        length_balance=float(result.metrics["optimized_loss"]["length_balance"]),
        fit_before_after_png=str(output_dir / "fit_before_after.png"),
        fit_contact_sheet_png=str(output_dir / "fit_contact_sheet.png"),
        fit_metrics_json=str(output_dir / "fit_metrics.json"),
        alignment_optimized_json=str(output_dir / "alignment_optimized.json"),
    )
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    fieldnames = [
        "scene",
        "shoe_name",
        "status",
        "warm_start_used",
        "fit_version",
        "loss_before",
        "loss_after",
        "loss_improved",
        "scale",
        "scale_delta",
        "yaw_delta_degrees",
        "pitch_delta_degrees",
        "roll_delta_degrees",
        "tx",
        "ty",
        "tz",
        "footbed_offset",
        "style_mode",
        "style_height_ratio",
        "multistart_count",
        "cavity_violation_fraction",
        "footbed_mask_violation_fraction",
        "plantar_clearance_mean",
        "side_gap_abs_mean",
        "side_total_clearance_mean",
        "length_balance",
        "fit_before_after_png",
        "fit_contact_sheet_png",
        "fit_metrics_json",
        "alignment_optimized_json",
        "error",
    ]
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_colored_overlay(path: Path, shoe_mesh: MeshData, baseline_mesh: MeshData, optimized_mesh: MeshData) -> None:
    items = [
        (shoe_mesh, np.asarray([175, 175, 175], dtype=np.uint8)),
        (baseline_mesh, np.asarray([253, 184, 99], dtype=np.uint8)),
        (optimized_mesh, np.asarray([44, 123, 182], dtype=np.uint8)),
    ]
    vertices_all = []
    faces_all = []
    colors_all = []
    offset = 0
    for mesh, color in items:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        colors_all.append(np.tile(color[None, :], (vertices.shape[0], 1)))
        offset += vertices.shape[0]

    vertices = np.concatenate(vertices_all, axis=0)
    faces = np.concatenate(faces_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _write_plots(
    output_dir: Path,
    shoe_name: str,
    shoe_mesh: MeshData,
    open_mesh: MeshData,
    baseline_mesh: MeshData,
    optimized_mesh: MeshData,
    result,
) -> None:
    del open_mesh
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cavity = result.cavity
    offset = float(result.metrics["optimized_params"]["footbed_offset"])
    center_footbed_y = cavity.sample_footbed_numpy(cavity.centerline_x, cavity.centerline_z) + np.sign(
        cavity.config.shoe_up_sign
    ) * offset

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{shoe_name} - baseline vs optimized foot fit", fontsize=13)
    ax = axes[0]
    ax.fill_betweenx(cavity.centerline_x, cavity.left_boundary_z, cavity.right_boundary_z, color="#f0f0f0", label="pseudo-cavity footprint")
    ax.plot(cavity.centerline_z, cavity.centerline_x, color="#d7191c", linewidth=2.0, label="shoe centerline")
    ax.scatter(baseline_mesh.vertices[:, 2], baseline_mesh.vertices[:, 0], s=1.5, alpha=0.22, color="#fdae61", label="baseline foot")
    ax.scatter(optimized_mesh.vertices[:, 2], optimized_mesh.vertices[:, 0], s=1.5, alpha=0.22, color="#2c7bb6", label="optimized foot")
    ax.set_xlabel("Z width")
    ax.set_ylabel("X heel to toe")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(cavity.centerline_x, cavity.floor_y, color="#969696", linewidth=1.5, label="smooth outer floor")
    ax.plot(cavity.centerline_x, center_footbed_y, color="#1a9641", linewidth=2.0, label="V4 smooth footbed")
    ax.scatter(baseline_mesh.vertices[:, 0], baseline_mesh.vertices[:, 1], s=1.4, alpha=0.18, color="#fdae61", label="baseline foot")
    ax.scatter(optimized_mesh.vertices[:, 0], optimized_mesh.vertices[:, 1], s=1.4, alpha=0.18, color="#2c7bb6", label="optimized foot")
    ax.set_xlabel("X heel to toe")
    ax.set_ylabel("Y coordinate (+Y bottom, -Y opening)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fit_before_after.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    slice_fracs = [0.18, 0.62, 0.90]
    for ax, frac in zip(axes, slice_fracs):
        x = float(cavity.centerline_x.min() + frac * (cavity.centerline_x.max() - cavity.centerline_x.min()))
        left = np.interp(x, cavity.centerline_x, cavity.left_boundary_z)
        right = np.interp(x, cavity.centerline_x, cavity.right_boundary_z)
        floor = np.interp(x, cavity.centerline_x, cavity.floor_y)
        z_line = np.linspace(left, right, 120, dtype=np.float32)
        x_line = np.full_like(z_line, x, dtype=np.float32)
        bed_line = cavity.sample_footbed_numpy(x_line, z_line) + np.sign(cavity.config.shoe_up_sign) * offset
        band = max(cavity.support_length * 0.025, 0.004)
        mask = np.abs(optimized_mesh.vertices[:, 0] - x) <= band
        ax.axvspan(left, right, color="#f0f0f0")
        ax.axhline(floor, color="#969696", linewidth=1.5, label="floor")
        ax.plot(z_line, bed_line, color="#1a9641", linewidth=1.8, label="V4 footbed")
        ax.scatter(optimized_mesh.vertices[mask, 2], optimized_mesh.vertices[mask, 1], s=5, alpha=0.45, color="#2c7bb6")
        ax.set_title(f"X slice {frac:.0%}")
        ax.set_xlabel("Z width")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cavity_slices.png", dpi=170)
    plt.close(fig)

    x = optimized_mesh.vertices[:, 0]
    y = optimized_mesh.vertices[:, 1]
    z = optimized_mesh.vertices[:, 2]
    bed = cavity.sample_footbed_numpy(x, z) + np.sign(cavity.config.shoe_up_sign) * offset
    clearance = bed - y
    plantar_mask = np.abs(clearance) <= 0.030
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    sc = ax.scatter(z[plantar_mask], x[plantar_mask], c=clearance[plantar_mask], s=7, cmap="coolwarm", vmin=-0.015, vmax=0.025)
    ax.fill_betweenx(cavity.centerline_x, cavity.left_boundary_z, cavity.right_boundary_z, color="#f0f0f0", alpha=0.45)
    if cavity.footbed_x_centers is not None and cavity.footbed_z_centers is not None and cavity.footbed_mask is not None:
        ax.contour(
            cavity.footbed_z_centers,
            cavity.footbed_x_centers,
            cavity.footbed_mask.astype(float),
            levels=[0.5],
            colors="#1a9641",
            linewidths=1.0,
        )
    ax.plot(cavity.left_boundary_z, cavity.centerline_x, color="#636363", linewidth=1.0)
    ax.plot(cavity.right_boundary_z, cavity.centerline_x, color="#636363", linewidth=1.0)
    ax.set_title("Plantar clearance: footbed Y minus foot Y")
    ax.set_xlabel("Z width")
    ax.set_ylabel("X heel to toe")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="clearance")
    fig.tight_layout()
    fig.savefig(output_dir / "plantar_clearance.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"{shoe_name} - optimized 3D overlay", fontsize=13)
    views = [("side", 10, -70), ("heel/toe", 8, 0), ("top", 65, -90), ("bottom", -55, -90)]
    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        _add_mesh(ax, shoe_mesh, "#bdbdbd", 0.15, Poly3DCollection, max_faces=4500)
        _add_mesh(ax, baseline_mesh, "#fdae61", 0.22, Poly3DCollection, max_faces=1200)
        _add_mesh(ax, optimized_mesh, "#2c7bb6", 0.35, Poly3DCollection, max_faces=1200)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
        _set_axes_equal(ax, [shoe_mesh.vertices, baseline_mesh.vertices, optimized_mesh.vertices])
    fig.tight_layout()
    fig.savefig(output_dir / "fit_3d_overlay.png", dpi=160)
    plt.close(fig)


def _add_mesh(ax, mesh: MeshData, color: str, alpha: float, collection_cls, max_faces: int = 4000) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    collection = collection_cls(vertices[faces], facecolors=color, edgecolors="#222222", linewidths=0.03, alpha=alpha)
    ax.add_collection3d(collection)


def _set_axes_equal(ax, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate([np.asarray(points, dtype=np.float32) for points in point_sets], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _make_local_contact_sheet(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    names = ["fit_before_after.png", "cavity_slices.png", "plantar_clearance.png", "fit_3d_overlay.png"]
    images = [output_dir / name for name in names if (output_dir / name).exists()]
    if not images:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, images):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "fit_contact_sheet.png", dpi=150)
    plt.close(fig)


def make_dataset_contact_sheet(output_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row.get("status") == "ok" and Path(str(row.get("fit_contact_sheet_png", ""))).exists()]
    if not ok_rows:
        return
    cols = 4
    row_count = int(np.ceil(len(ok_rows) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(cols * 4.4, row_count * 4.1))
    axes = np.atleast_1d(axes).reshape(row_count, cols)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, row in zip(axes.reshape(-1), ok_rows):
        image = mpimg.imread(str(row["fit_contact_sheet_png"]))
        ax.imshow(image)
        ax.set_title(str(row["shoe_name"]), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_root / "all_shoes_optimized_contact_sheet.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for label, path in [
        ("mesh root", args.mesh_root),
        ("support root", args.support_root),
        ("baseline alignment root", args.baseline_alignment_root),
        ("foot OBJ", args.foot_obj),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.fit_version != "v5" and not args.no_warm_start and not args.warm_start_alignment_root.exists():
        print(f"Warm-start root missing, falling back to baseline alignments: {args.warm_start_alignment_root}")

    scenes = discover_scenes(args.mesh_root, args.scene, args.shoe_name)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, scene_name in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene_name}")
        try:
            row = process_scene(args, scene_name)
            print(
                f"  status={row.get('status')} "
                f"loss={row.get('loss_before', '')}->{row.get('loss_after', '')} "
                f"offset={row.get('footbed_offset', '')}"
            )
        except Exception as exc:
            shoe_name = scene_to_shoe_name(scene_name)
            error_dir = args.output_root / shoe_name
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.txt").write_text(traceback.format_exc())
            row = {
                "scene": scene_name,
                "shoe_name": shoe_name,
                "status": "failed",
                "output_dir": str(error_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {row['error']}")
        rows.append(row)
        write_summary(rows, args.output_root)

    make_dataset_contact_sheet(args.output_root, rows)
    write_summary(rows, args.output_root)
    print(f"Wrote {args.output_root / 'summary.csv'}")
    if (args.output_root / "all_shoes_optimized_contact_sheet.png").exists():
        print(f"Wrote {args.output_root / 'all_shoes_optimized_contact_sheet.png'}")


if __name__ == "__main__":
    main()
