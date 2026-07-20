#!/usr/bin/env python3
"""Run only coarse SuGaR optimization and coarse-mesh extraction.

This intentionally stops before refinement so the first mesh can be inspected
before a bad coarse result is propagated into later stages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from sugar_trainers.coarse_density_and_dn_consistency import (
    coarse_training_with_density_regularization_and_dn_consistency,
)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bbox-min", nargs=3, type=float, required=True)
    parser.add_argument("--bbox-max", nargs=3, type=float, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--max-gaussian-scale-ratio", type=float, default=0.05)
    parser.add_argument("--surface-level", type=float, default=0.3)
    parser.add_argument("--vertices", type=int, default=1_000_000)
    parser.add_argument("--poisson-depth", type=int, default=10)
    parser.add_argument("--vertices-density-quantile", type=float, default=0.1)
    parser.add_argument("--surface-point-budget", type=int, default=10_000_000)
    return parser.parse_args()


def tuple_text(values: list[float]) -> str:
    return ",".join(f"{value:.17g}" for value in values)


def main() -> int:
    args = parse_args()
    scene_path = args.scene_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    # SuGaR's legacy GaussianSplattingWrapper concatenates filenames onto this
    # string (for example, checkpoint_path + "cameras.json").
    checkpoint_argument = str(checkpoint_path) + "/"
    output_root = args.output_root.resolve()
    coarse_output = output_root / "coarse" / scene_path.name
    mesh_output = output_root / "coarse_mesh" / scene_path.name
    output_root.mkdir(parents=True, exist_ok=True)

    bbox_min = tuple_text(args.bbox_min)
    bbox_max = tuple_text(args.bbox_max)
    bbox_diagonal = sum(
        (maximum - minimum) ** 2
        for minimum, maximum in zip(args.bbox_min, args.bbox_max)
    ) ** 0.5
    maximum_scale = bbox_diagonal * args.max_gaussian_scale_ratio

    run_config = {
        "protocol": "coarse_only_masked_colmap_bounded_scale_capped",
        "scene_path": str(scene_path),
        "vanilla_3dgs_checkpoint_path": str(checkpoint_path),
        "vanilla_3dgs_iteration": args.iteration,
        "output_root": str(output_root),
        "gpu": args.gpu,
        "use_masks": True,
        "white_background": True,
        "bbox_min": args.bbox_min,
        "bbox_max": args.bbox_max,
        "bbox_diagonal": bbox_diagonal,
        "center_bbox": False,
        "foreground_only": True,
        "constrain_points_to_bbox": True,
        "filter_gaussians_by_bbox": True,
        "max_gaussian_scale_ratio": args.max_gaussian_scale_ratio,
        "maximum_gaussian_scale": maximum_scale,
        "regularization": "dn_consistency",
        "poisson_depth": args.poisson_depth,
        "vertices_density_quantile": args.vertices_density_quantile,
        "surface_point_budget": args.surface_point_budget,
        "stops_after": "coarse_mesh",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_path = output_root / "coarse_pilot_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    print("=== Masked, bounded, scale-capped coarse SuGaR pilot ===", flush=True)
    print(f"Scene: {scene_path}", flush=True)
    print(f"Existing vanilla 3DGS: {checkpoint_path}", flush=True)
    print("Masks: enabled from RGBA alpha", flush=True)
    print(f"Foreground bbox: {args.bbox_min} -> {args.bbox_max}", flush=True)
    print(f"BBox diagonal: {bbox_diagonal:.9f}", flush=True)
    print(
        f"Maximum Gaussian scale: {args.max_gaussian_scale_ratio:.4f} x diagonal = {maximum_scale:.9f}",
        flush=True,
    )
    print("Pipeline will stop after coarse mesh extraction.", flush=True)
    print(f"Run config: {config_path}", flush=True)

    coarse_args = AttrDict(
        checkpoint_path=checkpoint_argument,
        scene_path=str(scene_path),
        iteration_to_load=args.iteration,
        output_dir=str(coarse_output),
        eval=True,
        estimation_factor=0.2,
        normal_factor=0.2,
        gpu=args.gpu,
        white_background=True,
        bboxmin=bbox_min,
        bboxmax=bbox_max,
        use_masks=True,
        entropy_regularization=True,
        prune_at_start=False,
        constrain_points_to_bbox=True,
        max_gaussian_scale_ratio=args.max_gaussian_scale_ratio,
    )
    coarse_model_path = coarse_training_with_density_regularization_and_dn_consistency(
        coarse_args
    )

    mesh_args = AttrDict(
        scene_path=str(scene_path),
        checkpoint_path=checkpoint_argument,
        iteration_to_load=args.iteration,
        coarse_model_path=coarse_model_path,
        surface_level=args.surface_level,
        decimation_target=args.vertices,
        project_mesh_on_surface_points=True,
        mesh_output_dir=str(mesh_output),
        bboxmin=bbox_min,
        bboxmax=bbox_max,
        center_bbox=False,
        foreground_only=True,
        use_masks=True,
        filter_gaussians_by_bbox=True,
        white_background=True,
        poisson_depth=args.poisson_depth,
        vertices_density_quantile=args.vertices_density_quantile,
        surface_point_budget=args.surface_point_budget,
        gpu=args.gpu,
        eval=True,
        use_centers_to_extract_mesh=False,
        use_marching_cubes=False,
        use_vanilla_3dgs=False,
    )
    mesh_paths = extract_mesh_from_coarse_sugar(mesh_args)
    mesh_path = mesh_paths[0]
    result = {
        **run_config,
        "coarse_model_path": str(coarse_model_path),
        "coarse_mesh_path": str(mesh_path),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",
    }
    result_path = output_root / "coarse_pilot_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("=== Coarse pilot complete; refinement was not run ===", flush=True)
    print(f"Coarse checkpoint: {coarse_model_path}", flush=True)
    print(f"Coarse mesh: {mesh_path}", flush=True)
    print(f"Result manifest: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
