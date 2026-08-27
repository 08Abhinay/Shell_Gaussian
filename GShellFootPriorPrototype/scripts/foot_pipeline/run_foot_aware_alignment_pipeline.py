#!/usr/bin/env python3
"""Run the final foot-aware alignment pipeline for trained GShell shoe meshes.

Per shoe, this script runs:

    trained mesh -> support footprint/footbed -> initial SUPR alignment -> optimized fitting

It is the final user-facing path for foot placement artifacts. Debug helper
scripts still exist, but this runner keeps the reproducible flow in one place.
"""

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
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, FOOTSHELL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from FootShellGaussian.foot_prior import (  # noqa: E402
    FootAlignmentConfig,
    FootFitOptimizerConfig,
    MeshData,
    SupportFootprintConfig,
    build_alignment_from_meshes,
    evaluate_fit_numpy,
    extract_support_footprint,
    load_pseudo_cavity_from_support_json,
    load_triangle_mesh,
    optimize_foot_fit,
    save_support_footprint_artifacts,
    sole_block_masks,
    write_obj_mesh,
)
from run_foot_fit_optimization import _make_local_contact_sheet, _write_colored_overlay, _write_plots, make_dataset_contact_sheet  # noqa: E402


DEFAULT_MESH_ROOT = PROJECT_ROOT / "baselines/GShell/output/turntable-512-768"
DEFAULT_OUTPUT_ROOT = FOOTSHELL_ROOT / "output/foot_aware_alignment"
DEFAULT_FOOT_OBJ = PROJECT_ROOT / "baselines/SUPR/output/debug_playground/supr_male_right_foot_neutral.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--foot-obj", type=Path, default=DEFAULT_FOOT_OBJ)
    parser.add_argument("--scene", action="append", help="Scene directory name, for example Foo_turntable. Repeatable.")
    parser.add_argument("--shoe-name", action="append", help="Shoe name without _turntable. Repeatable.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--style-mode", choices=["auto", "normal", "boot"], default="auto")
    parser.add_argument("--diagnostics", choices=["minimal", "full"], default="minimal")

    parser.add_argument("--length-ratio", type=float, default=0.85)
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

    parser.add_argument("--adam-steps", type=int, default=160)
    parser.add_argument("--lbfgs-steps", type=int, default=25)
    parser.add_argument("--adam-lr", type=float, default=0.035)

    parser.add_argument("--grid-resolution", type=int, default=192)
    parser.add_argument("--footbed-offset", type=float, default=0.015)
    parser.add_argument("--heightmap-min-samples-per-cell", type=int, default=2)
    parser.add_argument("--heightmap-smooth-sigma", type=float, default=1.25)
    parser.add_argument("--heightmap-profile-clip", type=float, default=0.025)
    parser.add_argument("--footbed-inner-margin-cells", type=int, default=7)
    parser.add_argument("--smooth-footbed-window-fraction", type=float, default=0.18)
    parser.add_argument("--footbed-height-fraction", type=float, default=0.22)
    parser.add_argument("--open-boundary-footbed-offset", type=float, default=None)
    parser.add_argument("--open-boundary-footbed-offset-ratio", type=float, default=0.055)
    parser.add_argument("--open-boundary-footbed-offset-min", type=float, default=0.008)
    parser.add_argument("--open-boundary-footbed-offset-max", type=float, default=0.022)
    return parser.parse_args()


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


def make_support_config(args: argparse.Namespace) -> SupportFootprintConfig:
    return SupportFootprintConfig(
        grid_resolution=args.grid_resolution,
        footbed_offset=args.footbed_offset,
        heightmap_min_samples_per_cell=args.heightmap_min_samples_per_cell,
        heightmap_smooth_sigma=args.heightmap_smooth_sigma,
        heightmap_profile_clip=args.heightmap_profile_clip,
        footbed_inner_margin_cells=args.footbed_inner_margin_cells,
        smooth_footbed_window_fraction=args.smooth_footbed_window_fraction,
        smooth_footbed_height_fraction_from_bottom=args.footbed_height_fraction,
        open_boundary_footbed_offset=args.open_boundary_footbed_offset,
        open_boundary_footbed_offset_ratio=args.open_boundary_footbed_offset_ratio,
        open_boundary_footbed_offset_min=args.open_boundary_footbed_offset_min,
        open_boundary_footbed_offset_max=args.open_boundary_footbed_offset_max,
    )


def make_alignment_config(args: argparse.Namespace) -> FootAlignmentConfig:
    return FootAlignmentConfig(
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


def make_optimizer_config(args: argparse.Namespace) -> FootFitOptimizerConfig:
    return FootFitOptimizerConfig(
        device=args.device,
        style_mode=args.style_mode,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
    )


def process_scene(args: argparse.Namespace, scene_name: str) -> dict[str, object]:
    shoe_name = scene_to_shoe_name(scene_name)
    scene_dir = args.mesh_root / scene_name
    mesh_path = scene_dir / "mesh_watertight" / "mesh.obj"
    open_mesh_path = scene_dir / "mesh" / "mesh.obj"
    output_dir = args.output_root / shoe_name
    support_dir = output_dir / "support"
    support_json = support_dir / "support_footprint.json"
    support_npz = support_dir / "pseudo_footbed_heightmap.npz"
    alignment_initial_json = output_dir / "alignment_initial.json"

    row: dict[str, object] = {
        "scene": scene_name,
        "shoe_name": shoe_name,
        "status": "pending",
        "scene_dir": str(scene_dir),
        "mesh_path": str(mesh_path),
        "open_mesh_path": str(open_mesh_path),
        "support_dir": str(support_dir),
        "support_json": str(support_json),
        "support_npz": str(support_npz),
        "alignment_initial_json": str(alignment_initial_json),
        "output_dir": str(output_dir),
    }

    if output_dir.exists() and (output_dir / "fit_metrics.json").exists() and not args.overwrite:
        row.update(status="skipped_existing")
        return row

    required_paths = [
        ("watertight mesh", mesh_path),
        ("open mesh", open_mesh_path),
        ("foot OBJ", args.foot_obj),
    ]
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

    support_config = make_support_config(args)
    footprint = extract_support_footprint(shoe_mesh, support_config, open_mesh=open_mesh)
    support_artifacts = save_support_footprint_artifacts(
        shoe_mesh,
        footprint,
        support_dir,
        diagnostics=args.diagnostics,
    )

    alignment_config = make_alignment_config(args)
    initial_alignment = build_alignment_from_meshes(
        foot_mesh,
        shoe_mesh=shoe_mesh,
        opening_mesh=open_mesh,
        config=alignment_config,
    )
    initial_vertices = initial_alignment.transform_foot_to_shoe(foot_mesh.vertices)
    initial_mesh = MeshData(vertices=initial_vertices.astype(np.float32), faces=foot_mesh.faces)
    initial_alignment.save_json(alignment_initial_json)
    write_obj_mesh(
        output_dir / "foot_aligned_initial.obj",
        initial_mesh,
        comments=["Initial SUPR-Foot mesh in GShell shoe coordinates."],
    )

    cavity = load_pseudo_cavity_from_support_json(
        support_json,
        shoe_mesh,
        footbed_npz_path=support_npz,
        use_sidecar_npz=True,
    )
    optimizer_config = make_optimizer_config(args)
    result = optimize_foot_fit(
        foot_mesh=foot_mesh,
        shoe_mesh=shoe_mesh,
        baseline_alignment=initial_alignment,
        cavity=cavity,
        config=optimizer_config,
    )

    optimized_mesh = MeshData(vertices=result.aligned_vertices, faces=foot_mesh.faces)
    result.alignment.save_json(output_dir / "alignment_optimized.json")
    write_obj_mesh(
        output_dir / "foot_aligned_optimized.obj",
        optimized_mesh,
        comments=["Optimized SUPR-Foot mesh in GShell shoe coordinates."],
    )
    _write_colored_overlay(
        output_dir / "foot_inside_shoe_optimized.ply",
        shoe_mesh,
        initial_mesh,
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
                "foot_obj": str(args.foot_obj),
                "support_dir": str(support_dir),
                "support_json": str(support_json),
                "support_npz": str(support_npz),
                "alignment_initial_json": str(alignment_initial_json),
                "length_ratio": float(args.length_ratio),
                "diagnostics": str(args.diagnostics),
            },
            "support_artifacts": support_artifacts,
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

    _write_plots(output_dir, shoe_name, shoe_mesh, open_mesh, initial_mesh, optimized_mesh, result, diagnostics=args.diagnostics)
    if args.diagnostics == "full":
        _make_local_contact_sheet(output_dir)

    row.update(
        status="ok",
        loss_before=float(result.metrics["baseline_loss"]["total"]),
        loss_after=float(result.metrics["optimized_loss"]["total"]),
        loss_improved=bool(result.metrics["loss_improved"]),
        initial_scale=float(initial_alignment.scale),
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
        fit_before_after_png=str(output_dir / "fit_before_after.png") if (output_dir / "fit_before_after.png").exists() else "",
        fit_3d_overlay_png=str(output_dir / "fit_3d_overlay.png"),
        plantar_clearance_png=str(output_dir / "plantar_clearance.png"),
        fit_contact_sheet_png=str(output_dir / "fit_contact_sheet.png") if (output_dir / "fit_contact_sheet.png").exists() else "",
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
        "loss_before",
        "loss_after",
        "loss_improved",
        "initial_scale",
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
        "support_json",
        "alignment_initial_json",
        "alignment_optimized_json",
        "fit_before_after_png",
        "fit_3d_overlay_png",
        "plantar_clearance_png",
        "fit_contact_sheet_png",
        "fit_metrics_json",
        "error",
    ]
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    for label, path in [
        ("mesh root", args.mesh_root),
        ("foot OBJ", args.foot_obj),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

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
                f"style={row.get('style_mode', '')}"
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

    if args.diagnostics == "full":
        make_dataset_contact_sheet(args.output_root, rows)
    write_summary(rows, args.output_root)
    print(f"Wrote {args.output_root / 'summary.csv'}")
    if (args.output_root / "all_shoes_optimized_contact_sheet.png").exists():
        print(f"Wrote {args.output_root / 'all_shoes_optimized_contact_sheet.png'}")


if __name__ == "__main__":
    main()
