#!/usr/bin/env python3
"""Run support-footprint/footbed analysis for trained GShell shoe meshes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FootShellGaussian.foot_prior import load_triangle_mesh, mesh_bounds  # noqa: E402
from FootShellGaussian.foot_prior.support_footprint import (  # noqa: E402
    SupportFootprintConfig,
    extract_support_footprint,
    save_support_footprint_artifacts,
)


DEFAULT_MESH_ROOT = PROJECT_ROOT / "baselines/GShell/output/turntable-512-768"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "baselines/GShell/output/support_footbed_analysis_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", help="Scene directory name to process. Repeatable.")
    parser.add_argument("--shoe-name", action="append", help="Shoe name without _turntable. Repeatable.")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def shoe_to_scene_name(shoe_name: str) -> str:
    return shoe_name if shoe_name.endswith("_turntable") else f"{shoe_name}_turntable"


def discover_scenes(
    mesh_root: Path,
    requested: Iterable[str] | None = None,
    requested_shoes: Iterable[str] | None = None,
) -> list[str]:
    explicit = []
    if requested:
        explicit.extend(str(scene) for scene in requested)
    if requested_shoes:
        explicit.extend(shoe_to_scene_name(str(shoe)) for shoe in requested_shoes)
    if explicit:
        return sorted(dict.fromkeys(explicit))
    scenes = []
    for scene_dir in sorted(mesh_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if (scene_dir / "mesh_watertight" / "mesh.obj").exists() and (scene_dir / "mesh" / "mesh.obj").exists():
            scenes.append(scene_dir.name)
    return scenes


def run_scene(scene_name: str, mesh_root: Path, output_root: Path, config: SupportFootprintConfig) -> dict[str, object]:
    scene_dir = mesh_root / scene_name
    mesh_path = scene_dir / "mesh_watertight" / "mesh.obj"
    open_mesh_path = scene_dir / "mesh" / "mesh.obj"
    mesh = load_triangle_mesh(mesh_path)
    open_mesh = load_triangle_mesh(open_mesh_path)
    footprint = extract_support_footprint(mesh, config, open_mesh=open_mesh)

    scene_out = output_root / scene_name
    artifacts = save_support_footprint_artifacts(mesh, footprint, scene_out)
    _, _, shoe_size, bbox_center = mesh_bounds(mesh.vertices)

    row = {
        "scene": scene_name,
        "mesh_path": str(mesh_path),
        "open_mesh_path": str(open_mesh_path),
        "shoe_length": float(shoe_size[config.shoe_length_axis]),
        "shoe_width": float(shoe_size[config.shoe_width_axis]),
        "support_length": float(footprint.length_extent),
        "support_median_width": float(footprint.median_width),
        "bbox_width_center": float(bbox_center[config.shoe_width_axis]),
        "support_width_anchor": float(footprint.suggested_width_anchor),
        "support_minus_bbox_width": float(footprint.suggested_width_anchor - bbox_center[config.shoe_width_axis]),
        "support_face_count": int(footprint.support_face_indices.size),
        "profile_point_count": int(footprint.centerline_x.size),
        "floor_sample_count": int(footprint.floor_sample_points.shape[0]),
        "floor_valid_slice_count": int(sum(int(count) >= config.floor_min_samples_per_slice for count in footprint.floor_sample_count_profile)),
        "heightmap_shape": f"{footprint.floor_heightmap.shape[0]}x{footprint.floor_heightmap.shape[1]}",
        "heightmap_valid_cell_count": int(np.isfinite(footprint.raw_floor_heightmap[footprint.footprint_mask]).sum()),
        "heightmap_inside_cell_count": int(footprint.footprint_mask.sum()),
        "footbed_mask_cell_count": int(footprint.footbed_mask.sum()),
        "footbed_mask_area_fraction": float(footprint.footbed_mask.sum() / max(int(footprint.footprint_mask.sum()), 1)),
        "heightmap_valid_cell_fraction": float(
            np.isfinite(footprint.raw_floor_heightmap[footprint.footprint_mask]).sum()
            / max(int(footprint.footprint_mask.sum()), 1)
        ),
        "heightmap_floor_min": float(np.nanmin(footprint.floor_heightmap)),
        "heightmap_floor_max": float(np.nanmax(footprint.floor_heightmap)),
        "heightmap_footbed_min": float(np.nanmin(footprint.footbed_heightmap)),
        "heightmap_footbed_max": float(np.nanmax(footprint.footbed_heightmap)),
        "smooth_footbed_min": float(np.nanmin(footprint.smooth_footbed_heightmap)),
        "smooth_footbed_max": float(np.nanmax(footprint.smooth_footbed_heightmap)),
        "smooth_footbed_offset": float(footprint.smooth_footbed_offset),
        "smooth_footbed_source": footprint.smooth_footbed_source,
        "smooth_footbed_height_fraction_from_bottom": float(config.smooth_footbed_height_fraction_from_bottom),
        "floor_min_samples_per_slice": int(footprint.floor_sample_count_profile[footprint.floor_sample_count_profile > 0].min())
        if (footprint.floor_sample_count_profile > 0).any()
        else 0,
        "floor_median_samples_per_slice": float(np.median(footprint.floor_sample_count_profile[footprint.floor_sample_count_profile > 0]))
        if (footprint.floor_sample_count_profile > 0).any()
        else 0.0,
        "footprint_source": str(footprint.confidence.get("footprint_mask_source")),
        "axis_profile_source": str(footprint.confidence.get("axis_profile_source")),
        "filter_source": str(footprint.confidence.get("open_boundary_filter_selected_source")),
        "lower_boundary": None if footprint.lower_boundary is None else footprint.lower_boundary.to_summary_dict(),
        "artifacts": artifacts,
    }
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "support_footbed_summary.json"
    with summary_json.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    summary_csv = output_root / "support_footbed_summary.csv"
    fieldnames = [
        "scene",
        "shoe_length",
        "shoe_width",
        "support_length",
        "support_median_width",
        "bbox_width_center",
        "support_width_anchor",
        "support_minus_bbox_width",
        "support_face_count",
        "profile_point_count",
        "floor_sample_count",
        "floor_valid_slice_count",
        "heightmap_shape",
        "heightmap_valid_cell_count",
        "heightmap_inside_cell_count",
        "footbed_mask_cell_count",
        "footbed_mask_area_fraction",
        "heightmap_valid_cell_fraction",
        "heightmap_floor_min",
        "heightmap_floor_max",
        "heightmap_footbed_min",
        "heightmap_footbed_max",
        "smooth_footbed_min",
        "smooth_footbed_max",
        "smooth_footbed_offset",
        "smooth_footbed_source",
        "smooth_footbed_height_fraction_from_bottom",
        "floor_min_samples_per_slice",
        "floor_median_samples_per_slice",
        "footprint_source",
        "axis_profile_source",
        "filter_source",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    config = SupportFootprintConfig(
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
    scenes = discover_scenes(args.mesh_root, args.scene, args.shoe_name)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene_name in scenes:
        scene_out = args.output_root / scene_name
        if scene_out.exists() and not args.overwrite:
            print(f"SKIP {scene_name}: output exists")
            continue
        try:
            row = run_scene(scene_name, args.mesh_root, args.output_root, config)
            rows.append(row)
            print(
                f"{scene_name:72s} "
                f"len={row['support_length']:.4f} "
                f"width={row['support_median_width']:.4f} "
                f"faces={row['support_face_count']:5d} "
                f"source={row['footprint_source']}"
            )
        except Exception as exc:
            rows.append({"scene": scene_name, "error": str(exc)})
            print(f"FAILED {scene_name}: {exc}")

    write_summary(rows, args.output_root)
    print(f"Wrote {args.output_root / 'support_footbed_summary.json'}")


if __name__ == "__main__":
    main()
