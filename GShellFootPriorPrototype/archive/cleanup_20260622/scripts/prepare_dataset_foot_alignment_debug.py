#!/usr/bin/env python
"""Prepare full-dataset SUPR-foot alignment debug assets.

This script is diagnostic only. It does not train and it does not change the
GShell cutting code. For each shoe, it prefers the already-exported GShell
``mesh/mesh.obj`` and ``mesh_watertight/mesh.obj`` files, falls back to
checkpoint export only when needed, aligns the neutral SUPR foot, and writes
visual assets that make the foot placement and sole-block region easy to
inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch


def _repo_paths() -> tuple[Path, Path]:
    footshell_root = Path(__file__).resolve().parents[1]
    project_root = footshell_root.parent
    return footshell_root, project_root


def _add_footshell_to_path() -> tuple[Path, Path]:
    footshell_root, project_root = _repo_paths()
    if str(footshell_root) not in sys.path:
        sys.path.insert(0, str(footshell_root))
    return footshell_root, project_root


def _jsonify_bounds(vertices: np.ndarray) -> dict[str, list[float]]:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    return {
        "min": [float(v) for v in bounds_min],
        "max": [float(v) for v in bounds_max],
        "size": [float(v) for v in (bounds_max - bounds_min)],
        "center": [float(v) for v in ((bounds_min + bounds_max) * 0.5)],
    }


def _base_export_flags(config_path: Optional[Path], sdf_pretrain_steps: int) -> SimpleNamespace:
    """Match the training-time geometry flags needed to reload model.pt."""

    flags = SimpleNamespace()
    flags.gshell_grid = 64
    flags.mesh_scale = 3.6
    flags.boxscale = [1, 1, 1]
    flags.use_foot_prior = False
    flags.use_sdf_mlp = True
    flags.use_msdf_mlp = False
    flags.use_tanh_deform = False
    flags.sphere_init = False
    flags.sphere_init_norm = 0.5
    flags.n_hidden = 6
    flags.d_hidden = 256
    flags.n_freq = 6
    flags.skip_in = [3]
    flags.use_float16 = False
    flags.visualize_watertight = True
    flags.sdf_mlp_pretrain_steps = max(1, int(sdf_pretrain_steps))

    if config_path is not None and config_path.exists():
        with config_path.open("r") as f:
            payload = json.load(f)
        for key, value in payload.items():
            setattr(flags, key, value)

    flags.visualize_watertight = True
    flags.use_foot_prior = False
    flags.sdf_mlp_pretrain_steps = max(1, int(sdf_pretrain_steps))
    return flags


def _mesh_from_render_mesh(render_mesh) -> "MeshData":
    from foot_prior import MeshData

    return MeshData(
        vertices=render_mesh.v_pos.detach().cpu().numpy().astype(np.float32),
        faces=render_mesh.t_pos_idx.detach().cpu().numpy().astype(np.int64),
    )


def export_gshell_meshes_from_checkpoint(
    checkpoint_path: Path,
    config_path: Optional[Path],
    footshell_root: Path,
    sdf_pretrain_steps: int,
) -> tuple["MeshData", "MeshData"]:
    """Reload a trained GShell geometry checkpoint and export open/watertight meshes."""

    from geometry.gshell_tets_geometry import GShellTetsGeometry

    flags = _base_export_flags(config_path, sdf_pretrain_steps)
    previous_cwd = Path.cwd()
    os.chdir(footshell_root)
    try:
        geometry = GShellTetsGeometry(flags.gshell_grid, flags.mesh_scale, flags)
        state = torch.load(checkpoint_path, map_location="cuda")
        geometry.load_state_dict(state, strict=True)
        geometry.eval()
        with torch.no_grad():
            mesh_result = geometry.getMesh({})
            open_mesh = _mesh_from_render_mesh(mesh_result["imesh"])
            if "imesh_watertight" not in mesh_result:
                raise RuntimeError("geometry.getMesh did not return imesh_watertight")
            watertight_mesh = _mesh_from_render_mesh(mesh_result["imesh_watertight"])
        return open_mesh, watertight_mesh
    finally:
        os.chdir(previous_cwd)
        torch.cuda.empty_cache()


def stage_mesh_reference(source_path: Path, staged_path: Path, overwrite: bool) -> Path:
    """Expose a training-exported mesh inside the debug folder without copying it."""

    source_resolved = source_path.resolve()
    staged_path.parent.mkdir(parents=True, exist_ok=True)

    if staged_path.exists() or staged_path.is_symlink():
        if staged_path.is_symlink():
            try:
                if staged_path.resolve() == source_resolved:
                    return staged_path
            except FileNotFoundError:
                pass
        if not overwrite:
            return source_path
        if staged_path.is_dir() and not staged_path.is_symlink():
            raise IsADirectoryError(f"Cannot replace directory with mesh symlink: {staged_path}")
        staged_path.unlink()

    os.symlink(source_resolved, staged_path)
    return staged_path


def write_colored_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != (vertices.shape[0], 3):
        raise ValueError("colors must have shape [num_vertices, 3]")

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


def combine_colored_meshes(mesh_items: Iterable[tuple["MeshData", np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_all = []
    faces_all = []
    colors_all = []
    offset = 0
    for mesh, color in mesh_items:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        colors = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (vertices.shape[0], 1))
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        colors_all.append(colors)
        offset += vertices.shape[0]
    return (
        np.concatenate(vertices_all, axis=0),
        np.concatenate(faces_all, axis=0),
        np.concatenate(colors_all, axis=0),
    )


def compact_face_subset(mesh: "MeshData", face_mask: np.ndarray) -> "MeshData":
    from foot_prior import MeshData

    selected_faces = np.asarray(mesh.faces, dtype=np.int64)[np.asarray(face_mask, dtype=bool)]
    if selected_faces.shape[0] == 0:
        return MeshData(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
        )
    used_vertices = np.unique(selected_faces.reshape(-1))
    remap = np.full((mesh.vertices.shape[0],), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(used_vertices.shape[0])
    return MeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float32)[used_vertices],
        faces=remap[selected_faces],
    )


def display_coords(points: np.ndarray, config: "FootAlignmentConfig") -> np.ndarray:
    """Map raw shoe coordinates to a human-readable display frame."""

    points = np.asarray(points, dtype=np.float32)
    return np.stack(
        [
            points[:, config.shoe_length_axis] * np.sign(config.shoe_length_sign),
            points[:, config.shoe_width_axis] * np.sign(config.shoe_width_sign),
            points[:, config.shoe_up_axis] * np.sign(config.shoe_up_sign),
        ],
        axis=1,
    )


def display_vectors(vectors: np.ndarray, config: "FootAlignmentConfig") -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return np.stack(
        [
            vectors[:, config.shoe_length_axis] * np.sign(config.shoe_length_sign),
            vectors[:, config.shoe_width_axis] * np.sign(config.shoe_width_sign),
            vectors[:, config.shoe_up_axis] * np.sign(config.shoe_up_sign),
        ],
        axis=1,
    )


def set_axes_equal(ax, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate([p for p in point_sets if p.shape[0] > 0], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def add_mesh(ax, mesh: "MeshData", config: "FootAlignmentConfig", color: str, alpha: float, max_faces: int = 4000, linewidth: float = 0.03) -> None:
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return
    vertices = display_coords(mesh.vertices, config)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    collection = Poly3DCollection(vertices[faces], facecolors=color, edgecolors="#222222", linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


def add_axis_arrows(ax, mesh: "MeshData", config: "FootAlignmentConfig") -> None:
    points = display_coords(mesh.vertices, config)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    length = float((maxs - mins).max()) * 0.32
    arrows = [
        ((length, 0.0, 0.0), "#d7191c", "+X length"),
        ((0.0, length, 0.0), "#2c7bb6", "+Z width"),
        ((0.0, 0.0, length), "#1a9641", "-Y opening/up"),
        ((0.0, 0.0, -length), "#1b9e77", "+Y base/bottom"),
    ]
    for delta, color, label in arrows:
        delta = np.asarray(delta, dtype=np.float32)
        ax.quiver(center[0], center[1], center[2], delta[0], delta[1], delta[2], color=color, linewidth=2.0)
        end = center + delta
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=8)


def add_foot_anatomy_arrows(ax, foot_mesh: "MeshData", alignment: "FootAlignment", config: "FootAlignmentConfig") -> None:
    """Show how SUPR anatomy directions ended up in shoe/display space."""

    center_raw = np.asarray(foot_mesh.vertices, dtype=np.float32).mean(axis=0)
    center = display_coords(center_raw[None, :], config)[0]
    length = max(float(np.ptp(display_coords(foot_mesh.vertices, config), axis=0).max()) * 0.42, 1e-4)
    linear = np.asarray(alignment.foot_to_shoe[:3, :3], dtype=np.float32)
    anatomy = [
        (linear @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32), "#f46d43", "SUPR toes"),
        (linear @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32), "#7b3294", "SUPR ankle cut"),
        (linear @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32), "#3288bd", "SUPR sole"),
    ]
    for raw_vector, color, label in anatomy:
        norm = max(float(np.linalg.norm(raw_vector)), 1e-8)
        vector = display_vectors((raw_vector / norm)[None, :], config)[0] * length
        ax.quiver(center[0], center[1], center[2], vector[0], vector[1], vector[2], color=color, linewidth=2.0)
        end = center + vector
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=8)


def render_scene_panel(
    ax,
    watertight_mesh: "MeshData",
    foot_mesh: "MeshData",
    config: "FootAlignmentConfig",
    title: str,
    elev: int,
    azim: int,
    selected_mesh: Optional["MeshData"] = None,
    open_mesh: Optional["MeshData"] = None,
    alignment: Optional["FootAlignment"] = None,
    show_anatomy_arrows: bool = False,
) -> None:
    add_mesh(ax, watertight_mesh, config, color="#bdbdbd", alpha=0.12, max_faces=4500)
    if open_mesh is not None:
        add_mesh(ax, open_mesh, config, color="#9ecae1", alpha=0.08, max_faces=4500)
    if selected_mesh is not None:
        add_mesh(ax, selected_mesh, config, color="#d7191c", alpha=0.88, max_faces=8000, linewidth=0.05)
    add_mesh(ax, foot_mesh, config, color="#fdae6b", alpha=0.34, max_faces=1200, linewidth=0.08)
    add_axis_arrows(ax, watertight_mesh, config)
    if show_anatomy_arrows and alignment is not None:
        add_foot_anatomy_arrows(ax, foot_mesh, alignment, config)
    bounds = [display_coords(watertight_mesh.vertices, config), display_coords(foot_mesh.vertices, config)]
    if selected_mesh is not None and selected_mesh.vertices.shape[0] > 0:
        bounds.append(display_coords(selected_mesh.vertices, config))
    set_axes_equal(ax, bounds)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def render_debug_pngs(
    out_dir: Path,
    shoe_name: str,
    open_mesh: "MeshData",
    watertight_mesh: "MeshData",
    foot_mesh: "MeshData",
    sole_block_face_mesh: "MeshData",
    alignment: "FootAlignment",
    config: "FootAlignmentConfig",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection

    globals()["Poly3DCollection"] = _Poly3DCollection

    contact_views = [
        ("side: foot inside shoe", 12, -70, None, True),
        ("toe/heel direction", 10, 0, None, True),
        ("top/opening direction", 60, -90, None, False),
        ("bottom/base direction", -60, -90, None, False),
    ]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"{shoe_name} - axes + aligned SUPR foot", fontsize=13)
    for index, (title, elev, azim, selected, anatomy_arrows) in enumerate(contact_views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_scene_panel(
            ax,
            watertight_mesh,
            foot_mesh,
            config,
            title,
            elev,
            azim,
            selected_mesh=selected,
            open_mesh=open_mesh,
            alignment=alignment,
            show_anatomy_arrows=anatomy_arrows,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=160)
    plt.close(fig)

    sole_views = [
        ("sole block: side", 12, -70),
        ("sole block: toe/heel", 10, 0),
        ("sole block: top/opening", 60, -90),
        ("sole block: bottom/base", -60, -90),
    ]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"{shoe_name} - red surface is sole block under foot", fontsize=13)
    for index, (title, elev, azim) in enumerate(sole_views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_scene_panel(
            ax,
            watertight_mesh,
            foot_mesh,
            config,
            title,
            elev,
            azim,
            selected_mesh=sole_block_face_mesh,
            open_mesh=None,
            alignment=alignment,
            show_anatomy_arrows=False,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "sole_block_views.png", dpi=160)
    fig.savefig(out_dir / "overview.png", dpi=120)
    plt.close(fig)

    axis_views = [
        ("raw shoe axes: side", 12, -70),
        ("raw shoe axes: toe/heel", 10, 0),
        ("aligned SUPR anatomy arrows", 18, -55),
        ("opening/up vs base/bottom", 60, -90),
    ]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"{shoe_name} - coordinate sanity check", fontsize=13)
    for index, (title, elev, azim) in enumerate(axis_views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        render_scene_panel(
            ax,
            watertight_mesh,
            foot_mesh,
            config,
            title,
            elev,
            azim,
            selected_mesh=None,
            open_mesh=open_mesh,
            alignment=alignment,
            show_anatomy_arrows=index in (3, 4),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "axis_alignment_views.png", dpi=160)
    plt.close(fig)


def make_dataset_contact_sheet(out_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row.get("status") == "ok" and Path(str(row.get("overview_png", ""))).exists()]
    if not ok_rows:
        return
    cols = 4
    rows_count = int(np.ceil(len(ok_rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 4.2, rows_count * 4.2))
    axes = np.atleast_1d(axes).reshape(rows_count, cols)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, row in zip(axes.reshape(-1), ok_rows):
        image = mpimg.imread(str(row["overview_png"]))
        ax.imshow(image)
        ax.set_title(str(row["shoe_name"]), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_root / "all_shoes_contact_sheet.png", dpi=150)
    plt.close(fig)


def process_one_shoe(args, shoe_name: str, footshell_root: Path, project_root: Path) -> dict[str, object]:
    from foot_prior import (
        FootAlignmentConfig,
        FootSDFGrid,
        MeshData,
        build_alignment_from_meshes,
        classify_shoe_points,
        colors_from_regions,
        detect_shoe_opening_boundary,
        get_single_boundary_loop,
        load_triangle_mesh,
        query_foot_sdf_in_shoe_space,
        region_summary,
        sole_block_masks,
        write_colored_ply,
        write_obj_mesh,
    )

    dataset_scene = args.dataset_root / shoe_name
    trained_dir = args.baseline_output_root / args.baseline_subdir / f"{shoe_name}{args.baseline_suffix}"
    checkpoint_path = trained_dir / "mesh" / "model.pt"
    trained_open_mesh_path = trained_dir / "mesh" / "mesh.obj"
    trained_watertight_mesh_path = trained_dir / "mesh_watertight" / "mesh.obj"
    out_dir = args.out_root / shoe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, object] = {
        "shoe_name": shoe_name,
        "status": "pending",
        "dataset_scene": str(dataset_scene),
        "trained_dir": str(trained_dir),
        "checkpoint": str(checkpoint_path),
        "trained_open_mesh": str(trained_open_mesh_path),
        "trained_watertight_mesh": str(trained_watertight_mesh_path),
        "out_dir": str(out_dir),
    }

    if not dataset_scene.exists():
        row.update(status="missing_dataset_scene", error=f"Missing dataset scene: {dataset_scene}")
        return row

    open_mesh_path = out_dir / "shoe_open_mesh.obj"
    watertight_mesh_path = out_dir / "shoe_watertight_mesh.obj"

    use_training_export = (
        not args.force_reexport_from_checkpoint
        and trained_open_mesh_path.exists()
        and trained_watertight_mesh_path.exists()
    )
    if use_training_export:
        mesh_source = "training_export"
        open_mesh = load_triangle_mesh(trained_open_mesh_path)
        watertight_mesh = load_triangle_mesh(trained_watertight_mesh_path)
        open_mesh_output_path = stage_mesh_reference(trained_open_mesh_path, open_mesh_path, args.overwrite)
        watertight_mesh_output_path = stage_mesh_reference(
            trained_watertight_mesh_path,
            watertight_mesh_path,
            args.overwrite,
        )
    else:
        missing_exported = [
            path
            for path in [trained_open_mesh_path, trained_watertight_mesh_path]
            if not path.exists()
        ]
        if not checkpoint_path.exists():
            missing_text = ", ".join(str(path) for path in missing_exported) or "none"
            row.update(
                status="missing_meshes",
                error=(
                    f"Missing exported meshes: {missing_text}; "
                    f"missing checkpoint fallback: {checkpoint_path}"
                ),
            )
            return row

        mesh_source = (
            "checkpoint_forced"
            if args.force_reexport_from_checkpoint
            else "checkpoint_fallback"
        )
        open_mesh_output_path = open_mesh_path
        watertight_mesh_output_path = watertight_mesh_path
        if args.overwrite or not (open_mesh_path.exists() and watertight_mesh_path.exists()):
            open_mesh, watertight_mesh = export_gshell_meshes_from_checkpoint(
                checkpoint_path,
                args.gshell_config,
                footshell_root,
                args.sdf_pretrain_steps_for_export,
            )
            write_obj_mesh(open_mesh_path, open_mesh, comments=["Open mesh after learned mSDF cutting."])
            write_obj_mesh(watertight_mesh_path, watertight_mesh, comments=["Watertight SDF surface before learned mSDF cutting."])
        else:
            open_mesh = load_triangle_mesh(open_mesh_path)
            watertight_mesh = load_triangle_mesh(watertight_mesh_path)

    foot_mesh = load_triangle_mesh(args.foot_obj)
    foot_sdf = FootSDFGrid.from_npz(str(args.foot_sdf), device=torch.device(args.device))
    config = FootAlignmentConfig(
        length_ratio=args.length_ratio,
        scale_multiplier=args.scale_multiplier,
        plantar_clearance=args.plantar_clearance,
        plantar_band=args.plantar_band,
        surface_band=args.surface_band,
        clearance=args.clearance,
        ankle_radius=args.ankle_radius,
        align_ankle_to_opening=not args.no_align_ankle_to_opening,
        auto_yaw=not args.no_auto_yaw,
        yaw_degrees=args.yaw_degrees,
        pitch_degrees=args.pitch_degrees,
        roll_degrees=args.roll_degrees,
        translation_offset=(args.tx, args.ty, args.tz),
    )

    alignment = build_alignment_from_meshes(
        foot_mesh=foot_mesh,
        shoe_mesh=watertight_mesh,
        opening_mesh=open_mesh,
        config=config,
    )
    foot_vertices_aligned = alignment.transform_foot_to_shoe(foot_mesh.vertices)
    aligned_foot_mesh = MeshData(vertices=foot_vertices_aligned, faces=foot_mesh.faces)
    ankle_loop_shoe = alignment.transform_foot_to_shoe(foot_mesh.vertices[get_single_boundary_loop(foot_mesh)])

    open_regions = classify_shoe_points(open_mesh.vertices, foot_sdf, alignment, ankle_loop_shoe)
    watertight_regions = classify_shoe_points(watertight_mesh.vertices, foot_sdf, alignment, ankle_loop_shoe)
    sole_masks = sole_block_masks(
        watertight_mesh,
        foot_vertices_aligned,
        alignment,
        footprint_margin=args.footprint_margin,
        plantar_band=args.plantar_band,
    )
    watertight_sdf = query_foot_sdf_in_shoe_space(watertight_mesh.vertices, foot_sdf, alignment)

    sole_vertex_colors = np.full((watertight_mesh.vertices.shape[0], 3), [180, 180, 180], dtype=np.uint8)
    sole_vertex_colors[sole_masks["vertex_under_footprint"]] = [253, 174, 97]
    sole_vertex_colors[sole_masks["vertex_sole_block"]] = [215, 48, 39]
    write_colored_mesh_ply(out_dir / "sole_block_vertices.ply", watertight_mesh.vertices, watertight_mesh.faces, sole_vertex_colors)

    sole_block_face_mesh = compact_face_subset(watertight_mesh, sole_masks["face_sole_block"])
    write_colored_mesh_ply(
        out_dir / "sole_block_faces.ply",
        sole_block_face_mesh.vertices,
        sole_block_face_mesh.faces,
        np.tile(np.asarray([[215, 48, 39]], dtype=np.uint8), (sole_block_face_mesh.vertices.shape[0], 1)),
    )

    overlay_vertices, overlay_faces, overlay_colors = combine_colored_meshes(
        [
            (watertight_mesh, np.asarray([185, 185, 185], dtype=np.uint8)),
            (aligned_foot_mesh, np.asarray([253, 174, 97], dtype=np.uint8)),
        ]
    )
    write_colored_mesh_ply(out_dir / "foot_inside_shoe_overlay.ply", overlay_vertices, overlay_faces, overlay_colors)

    opening_colors = np.full((open_mesh.vertices.shape[0], 3), [180, 180, 180], dtype=np.uint8)
    opening = detect_shoe_opening_boundary(open_mesh, config)
    if opening is not None:
        opening_colors[np.asarray(opening["vertices"], dtype=np.int64)] = [117, 112, 179]
    write_colored_mesh_ply(out_dir / "opening_boundary_debug.ply", open_mesh.vertices, open_mesh.faces, opening_colors)

    write_obj_mesh(
        out_dir / "foot_aligned.obj",
        aligned_foot_mesh,
        comments=["Aligned neutral SUPR-Foot mesh in GShell shoe coordinates."],
    )
    alignment.save_json(out_dir / "alignment.json")
    write_colored_ply(out_dir / "shoe_open_regions.ply", open_mesh, colors_from_regions(open_regions))
    write_colored_ply(out_dir / "shoe_watertight_regions.ply", watertight_mesh, colors_from_regions(watertight_regions))
    render_debug_pngs(
        out_dir,
        shoe_name,
        open_mesh,
        watertight_mesh,
        aligned_foot_mesh,
        sole_block_face_mesh,
        alignment,
        config,
    )

    foot_bounds = _jsonify_bounds(foot_vertices_aligned)
    watertight_bounds = _jsonify_bounds(watertight_mesh.vertices)
    sole_vertex_mask = np.asarray(sole_masks["vertex_sole_block"], dtype=bool)
    sole_face_mask = np.asarray(sole_masks["face_sole_block"], dtype=bool)
    selected_sdf = watertight_sdf[sole_vertex_mask] if sole_vertex_mask.any() else np.asarray([], dtype=np.float32)
    summary = {
        "shoe_name": shoe_name,
        "status": "ok",
        "dataset_scene": str(dataset_scene),
        "trained_dir": str(trained_dir),
        "checkpoint": str(checkpoint_path),
        "mesh_source": mesh_source,
        "source_meshes": {
            "training_open_mesh_obj": str(trained_open_mesh_path),
            "training_watertight_mesh_obj": str(trained_watertight_mesh_path),
            "debug_open_mesh_obj": str(open_mesh_output_path),
            "debug_watertight_mesh_obj": str(watertight_mesh_output_path),
        },
        "alignment": alignment.to_dict(),
        "bounds": {
            "open_mesh": _jsonify_bounds(open_mesh.vertices),
            "watertight_mesh": watertight_bounds,
            "foot_aligned": foot_bounds,
        },
        "regions": {
            "open_vertices": region_summary(open_regions),
            "watertight_vertices": region_summary(watertight_regions),
        },
        "sole_block": {
            "vertex_count": int(sole_vertex_mask.sum()),
            "vertex_fraction": float(sole_vertex_mask.mean()),
            "face_count": int(sole_face_mask.sum()),
            "face_fraction": float(sole_face_mask.mean()),
            "footprint_hull_points": np.asarray(sole_masks["footprint_hull"], dtype=float).tolist(),
            "selected_foot_sdf_min": None if selected_sdf.size == 0 else float(selected_sdf.min()),
            "selected_foot_sdf_mean": None if selected_sdf.size == 0 else float(selected_sdf.mean()),
            "selected_foot_sdf_max": None if selected_sdf.size == 0 else float(selected_sdf.max()),
        },
        "opening": {
            "detected": opening is not None,
            "component_index": None if opening is None else int(opening["index"]),
            "component_vertices": 0 if opening is None else int(np.asarray(opening["vertices"]).shape[0]),
        },
        "outputs": {
            "alignment_json": str(out_dir / "alignment.json"),
            "summary_json": str(out_dir / "summary.json"),
            "foot_aligned_obj": str(out_dir / "foot_aligned.obj"),
            "shoe_open_mesh_obj": str(open_mesh_output_path),
            "shoe_watertight_mesh_obj": str(watertight_mesh_output_path),
            "sole_block_vertices_ply": str(out_dir / "sole_block_vertices.ply"),
            "sole_block_faces_ply": str(out_dir / "sole_block_faces.ply"),
            "foot_inside_shoe_overlay_ply": str(out_dir / "foot_inside_shoe_overlay.ply"),
            "opening_boundary_debug_ply": str(out_dir / "opening_boundary_debug.ply"),
            "contact_sheet_png": str(out_dir / "contact_sheet.png"),
            "sole_block_views_png": str(out_dir / "sole_block_views.png"),
            "axis_alignment_views_png": str(out_dir / "axis_alignment_views.png"),
            "overview_png": str(out_dir / "overview.png"),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    row.update(
        status="ok",
        mesh_source=mesh_source,
        opening_detected=opening is not None,
        foot_scale=alignment.scale,
        plantar_z=alignment.plantar_z,
        sole_vertex_count=int(sole_vertex_mask.sum()),
        sole_vertex_fraction=float(sole_vertex_mask.mean()),
        sole_face_count=int(sole_face_mask.sum()),
        sole_face_fraction=float(sole_face_mask.mean()),
        selected_foot_sdf_mean="" if selected_sdf.size == 0 else float(selected_sdf.mean()),
        contact_sheet_png=str(out_dir / "contact_sheet.png"),
        sole_block_views_png=str(out_dir / "sole_block_views.png"),
        axis_alignment_views_png=str(out_dir / "axis_alignment_views.png"),
        overview_png=str(out_dir / "overview.png"),
        summary_json=str(out_dir / "summary.json"),
    )
    return row


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shoe_name",
        "status",
        "mesh_source",
        "opening_detected",
        "foot_scale",
        "plantar_z",
        "sole_vertex_count",
        "sole_vertex_fraction",
        "sole_face_count",
        "sole_face_fraction",
        "selected_foot_sdf_mean",
        "dataset_scene",
        "trained_dir",
        "checkpoint",
        "trained_open_mesh",
        "trained_watertight_mesh",
        "out_dir",
        "summary_json",
        "contact_sheet_png",
        "sole_block_views_png",
        "axis_alignment_views_png",
        "overview_png",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_summary_rows(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    if not path.exists():
        return [], {}
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    order: list[str] = []
    by_name: dict[str, dict[str, object]] = {}
    for row in rows:
        shoe_name = str(row.get("shoe_name", ""))
        if not shoe_name:
            continue
        order.append(shoe_name)
        by_name[shoe_name] = row
    return order, by_name


def main() -> None:
    footshell_root, project_root = _add_footshell_to_path()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/abelde/datasets/processed/gshell_shoes"))
    parser.add_argument("--baseline-output-root", type=Path, default=project_root / "baselines" / "GShell" / "output")
    parser.add_argument("--baseline-subdir", type=str, default="res-512-768")
    parser.add_argument("--baseline-suffix", type=str, default="_normfix")
    parser.add_argument(
        "--gshell-config",
        type=Path,
        default=project_root / "baselines" / "GShell" / "configs" / "shoes_mc_normfix.json",
    )
    parser.add_argument(
        "--foot-obj",
        type=Path,
        default=project_root / "baselines" / "SUPR" / "output" / "debug_playground" / "supr_male_right_foot_neutral.obj",
    )
    parser.add_argument(
        "--foot-sdf",
        type=Path,
        default=footshell_root / "data" / "foot_prior" / "supr_male_right_foot_sdf.npz",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=footshell_root / "output" / "foot_alignment_dataset_debug",
    )
    parser.add_argument("--shoe-name", action="append", default=None, help="Process only this shoe; may be repeated.")
    parser.add_argument("--max-shoes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-reexport-from-checkpoint",
        action="store_true",
        help=(
            "Ignore existing mesh/mesh.obj and mesh_watertight/mesh.obj files "
            "and regenerate shoe meshes from mesh/model.pt."
        ),
    )
    parser.add_argument("--sdf-pretrain-steps-for-export", type=int, default=1)
    parser.add_argument("--length-ratio", type=float, default=0.78)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--plantar-clearance", type=float, default=0.032)
    parser.add_argument("--plantar-band", type=float, default=0.012)
    parser.add_argument("--surface-band", type=float, default=0.005)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--ankle-radius", type=float, default=0.025)
    parser.add_argument("--footprint-margin", type=float, default=0.012)
    parser.add_argument("--no-align-ankle-to-opening", action="store_true")
    parser.add_argument("--no-auto-yaw", action="store_true")
    parser.add_argument("--yaw-degrees", type=float, default=0.0)
    parser.add_argument("--pitch-degrees", type=float, default=0.0)
    parser.add_argument("--roll-degrees", type=float, default=0.0)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)
    args = parser.parse_args()

    for label, path in [("dataset root", args.dataset_root), ("foot OBJ", args.foot_obj), ("foot SDF", args.foot_sdf)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.shoe_name:
        shoe_names = args.shoe_name
    else:
        shoe_names = sorted(path.name for path in args.dataset_root.iterdir() if path.is_dir())
    if args.max_shoes is not None:
        shoe_names = shoe_names[: args.max_shoes]

    merge_existing_summary = bool(args.shoe_name and (args.out_root / "summary.csv").exists())
    summary_order, rows_by_name = (
        read_existing_summary_rows(args.out_root / "summary.csv")
        if merge_existing_summary
        else ([], {})
    )
    for shoe_name in shoe_names:
        if shoe_name not in rows_by_name:
            summary_order.append(shoe_name)

    rows: list[dict[str, object]] = [rows_by_name[name] for name in summary_order if name in rows_by_name]
    for index, shoe_name in enumerate(shoe_names, start=1):
        print(f"[{index}/{len(shoe_names)}] {shoe_name}")
        try:
            row = process_one_shoe(args, shoe_name, footshell_root, project_root)
            print(f"  status={row.get('status')} sole_faces={row.get('sole_face_count', '')}")
        except Exception as exc:
            row = {
                "shoe_name": shoe_name,
                "status": "failed",
                "dataset_scene": str(args.dataset_root / shoe_name),
                "checkpoint": str(args.baseline_output_root / args.baseline_subdir / f"{shoe_name}{args.baseline_suffix}" / "mesh" / "model.pt"),
                "out_dir": str(args.out_root / shoe_name),
                "error": f"{type(exc).__name__}: {exc}",
            }
            error_path = args.out_root / shoe_name / "error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc())
            print(f"  failed: {row['error']}")
        rows_by_name[shoe_name] = row
        rows = [rows_by_name[name] for name in summary_order if name in rows_by_name]
        write_summary_csv(args.out_root / "summary.csv", rows)

    make_dataset_contact_sheet(args.out_root, rows)
    write_summary_csv(args.out_root / "summary.csv", rows)
    print(f"\nWrote summary: {args.out_root / 'summary.csv'}")
    if (args.out_root / "all_shoes_contact_sheet.png").exists():
        print(f"Wrote contact sheet: {args.out_root / 'all_shoes_contact_sheet.png'}")


if __name__ == "__main__":
    main()
