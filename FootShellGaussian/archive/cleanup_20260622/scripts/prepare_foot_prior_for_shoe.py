#!/usr/bin/env python
"""Prepare neutral SUPR-Foot alignment assets for one shoe.

This is the training-facing version of the notebook/diagnostic workflow. It
uses an existing baseline GShell reconstruction to place the neutral foot, then
saves the alignment and debug assets that FootShellGaussian training consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def main() -> None:
    footshell_root, project_root = _add_footshell_to_path()

    from foot_prior import (
        FootAlignmentConfig,
        FootSDFGrid,
        MeshData,
        build_alignment_from_meshes,
        classify_shoe_points,
        colors_from_regions,
        get_single_boundary_loop,
        load_triangle_mesh,
        make_hybrid_mesh,
        region_summary,
        select_faces_below_signed_axis,
        write_colored_ply,
        write_obj_mesh,
    )

    default_shoe = "Adidas-Yeezy-Boost-350-V2-Static-Non-Reflective-Kids"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe-name", type=str, default=default_shoe)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/abelde/datasets/processed/gshell_shoes"),
    )
    parser.add_argument(
        "--baseline-output-root",
        type=Path,
        default=project_root / "baselines" / "GShell" / "output",
    )
    parser.add_argument("--baseline-suffix", type=str, default="_sole_debug")
    parser.add_argument("--shell-mesh", type=Path, default=None)
    parser.add_argument("--watertight-mesh", type=Path, default=None)
    parser.add_argument(
        "--foot-obj",
        type=Path,
        default=project_root
        / "baselines"
        / "SUPR"
        / "output"
        / "debug_playground"
        / "supr_male_right_foot_neutral.obj",
    )
    parser.add_argument(
        "--foot-sdf",
        type=Path,
        default=footshell_root / "data" / "foot_prior" / "supr_male_right_foot_sdf.npz",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=footshell_root / "output" / "foot_prior_debug",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--length-ratio", type=float, default=0.78)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--plantar-clearance", type=float, default=0.032)
    parser.add_argument("--plantar-band", type=float, default=0.012)
    parser.add_argument("--surface-band", type=float, default=0.005)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--ankle-radius", type=float, default=0.025)
    parser.add_argument("--no-align-ankle-to-opening", action="store_true")
    parser.add_argument("--no-auto-yaw", action="store_true")
    parser.add_argument("--yaw-degrees", type=float, default=0.0)
    parser.add_argument("--pitch-degrees", type=float, default=0.0)
    parser.add_argument("--roll-degrees", type=float, default=0.0)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)
    args = parser.parse_args()

    dataset_scene = args.dataset_root / args.shoe_name
    baseline_dir = args.baseline_output_root / f"{args.shoe_name}{args.baseline_suffix}"
    shell_mesh_path = args.shell_mesh or baseline_dir / "mesh" / "mesh.obj"
    watertight_mesh_path = args.watertight_mesh or baseline_dir / "mesh_watertight" / "mesh.obj"
    out_dir = args.out_dir or args.out_root / args.shoe_name

    for label, path in [
        ("dataset scene", dataset_scene),
        ("shell mesh", shell_mesh_path),
        ("watertight mesh", watertight_mesh_path),
        ("foot OBJ", args.foot_obj),
        ("foot SDF", args.foot_sdf),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    shell_mesh = load_triangle_mesh(shell_mesh_path)
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
        opening_mesh=shell_mesh,
        config=config,
    )

    foot_vertices_aligned = alignment.transform_foot_to_shoe(foot_mesh.vertices)
    aligned_foot_mesh = MeshData(vertices=foot_vertices_aligned, faces=foot_mesh.faces)
    ankle_loop_shoe = alignment.transform_foot_to_shoe(
        foot_mesh.vertices[get_single_boundary_loop(foot_mesh)]
    )

    shell_regions = classify_shoe_points(
        shell_mesh.vertices,
        foot_sdf=foot_sdf,
        alignment=alignment,
        ankle_loop_shoe=ankle_loop_shoe,
    )
    watertight_regions = classify_shoe_points(
        watertight_mesh.vertices,
        foot_sdf=foot_sdf,
        alignment=alignment,
        ankle_loop_shoe=ankle_loop_shoe,
    )
    retain_mask = select_faces_below_signed_axis(
        watertight_mesh,
        axis=config.shoe_up_axis,
        axis_direction=config.shoe_up_sign,
        plantar_value=alignment.plantar_z,
        band=config.plantar_band,
    )
    hybrid_mesh = make_hybrid_mesh(shell_mesh, watertight_mesh, retain_mask)

    alignment.save_json(out_dir / "alignment.json")
    write_obj_mesh(
        out_dir / "foot_aligned.obj",
        aligned_foot_mesh,
        comments=[
            "Aligned neutral SUPR-Foot mesh in GShell shoe coordinates.",
            f"source foot: {args.foot_obj}",
        ],
    )
    write_colored_ply(out_dir / "shoe_regions.ply", shell_mesh, colors_from_regions(shell_regions))
    write_colored_ply(
        out_dir / "watertight_regions.ply",
        watertight_mesh,
        colors_from_regions(watertight_regions),
    )
    write_obj_mesh(
        out_dir / "hybrid_shell_foot_retained.obj",
        hybrid_mesh,
        comments=[
            "Diagnostic mesh: shell plus watertight triangles below the aligned plantar band.",
            f"selected watertight faces: {int(retain_mask.sum())} / {int(retain_mask.shape[0])}",
        ],
    )

    summary = {
        "shoe_name": args.shoe_name,
        "dataset_scene": str(dataset_scene),
        "shell_mesh": str(shell_mesh_path),
        "watertight_mesh": str(watertight_mesh_path),
        "foot_obj": str(args.foot_obj),
        "foot_sdf": str(args.foot_sdf),
        "alignment": alignment.to_dict(),
        "bounds": {
            "shell": _jsonify_bounds(shell_mesh.vertices),
            "watertight": _jsonify_bounds(watertight_mesh.vertices),
            "foot_raw": _jsonify_bounds(foot_mesh.vertices),
            "foot_aligned": _jsonify_bounds(foot_vertices_aligned),
        },
        "regions": {
            "shell_vertices": region_summary(shell_regions),
            "watertight_vertices": region_summary(watertight_regions),
        },
        "hybrid": {
            "retained_watertight_faces": int(retain_mask.sum()),
            "total_watertight_faces": int(retain_mask.shape[0]),
            "retained_watertight_face_fraction": float(retain_mask.mean()),
            "hybrid_vertices": int(hybrid_mesh.vertices.shape[0]),
            "hybrid_faces": int(hybrid_mesh.faces.shape[0]),
        },
        "outputs": {
            "alignment": str(out_dir / "alignment.json"),
            "foot_aligned": str(out_dir / "foot_aligned.obj"),
            "shoe_regions": str(out_dir / "shoe_regions.ply"),
            "watertight_regions": str(out_dir / "watertight_regions.ply"),
            "hybrid_shell_foot_retained": str(out_dir / "hybrid_shell_foot_retained.obj"),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Saved foot-prior preparation outputs to: {out_dir}")
    print(f"  alignment: {out_dir / 'alignment.json'}")
    print(f"  scale: {alignment.scale:.6f}")
    print(f"  plantar coordinate: {alignment.plantar_z:.6f}")
    print(
        "  shell clearance violations: "
        f"{summary['regions']['shell_vertices']['clearance_violation_count']} / "
        f"{summary['regions']['shell_vertices']['num_points']}"
    )
    print(
        "  retained watertight faces: "
        f"{summary['hybrid']['retained_watertight_faces']} / "
        f"{summary['hybrid']['total_watertight_faces']}"
    )


if __name__ == "__main__":
    main()
