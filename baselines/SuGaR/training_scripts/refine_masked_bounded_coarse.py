#!/usr/bin/env python3
"""Resume a completed masked/bounded coarse pilot at SuGaR refinement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar
from sugar_trainers.refine import refined_training


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-result", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--refinement-iterations", type=int, default=15_000)
    parser.add_argument("--gaussians-per-triangle", type=int, default=1)
    parser.add_argument("--vertices", type=int, default=1_000_000)
    parser.add_argument("--square-size", type=int, default=8)
    parser.add_argument("--normal-consistency-factor", type=float, default=0.1)
    return parser.parse_args()


def tuple_text(values: list[float]) -> str:
    return ",".join(f"{value:.17g}" for value in values)


def main() -> int:
    args = parse_args()
    coarse_result_path = args.coarse_result.absolute()
    coarse_result = json.loads(coarse_result_path.read_text(encoding="utf-8"))
    if coarse_result.get("protocol") != "coarse_only_masked_colmap_bounded_scale_capped":
        raise ValueError("Refusing to refine an unexpected coarse-result protocol")
    if coarse_result.get("status") != "complete":
        raise ValueError("Coarse pilot is not marked complete")

    scene_path = Path(coarse_result["scene_path"]).absolute()
    checkpoint_path = Path(
        coarse_result["vanilla_3dgs_checkpoint_path"]
    ).absolute()
    checkpoint_argument = str(checkpoint_path) + "/"
    coarse_mesh_path = Path(coarse_result["coarse_mesh_path"]).absolute()
    coarse_model_path = Path(coarse_result["coarse_model_path"]).absolute()
    output_root = Path(coarse_result["output_root"]).absolute()
    bbox_min = coarse_result["bbox_min"]
    bbox_max = coarse_result["bbox_max"]
    iteration = int(coarse_result["vanilla_3dgs_iteration"])
    refined_output = output_root / "refined" / scene_path.name
    refined_mesh_output = output_root / "refined_mesh" / scene_path.name

    for required in (
        scene_path / "images",
        checkpoint_path / "cameras.json",
        checkpoint_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        coarse_mesh_path,
        coarse_model_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    config = {
        "protocol": "resume_refinement_from_masked_bounded_coarse",
        "coarse_result_manifest": str(coarse_result_path),
        "scene_path": str(scene_path),
        "vanilla_3dgs_checkpoint_path": str(checkpoint_path),
        "coarse_model_path": str(coarse_model_path),
        "accepted_coarse_mesh_path": str(coarse_mesh_path),
        "output_root": str(output_root),
        "gpu": args.gpu,
        "use_masks": True,
        "white_background": True,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "refinement_iterations": args.refinement_iterations,
        "gaussians_per_triangle": args.gaussians_per_triangle,
        "n_vertices_in_foreground": args.vertices,
        "normal_consistency_factor": args.normal_consistency_factor,
        "export_ply": True,
        "export_uv_textured_mesh": True,
        "texture_square_size": args.square_size,
        "postprocess_mesh": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_path = output_root / "refinement_run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("=== Resuming accepted coarse mesh at SuGaR refinement ===", flush=True)
    print(f"Accepted coarse mesh: {coarse_mesh_path}", flush=True)
    print(f"Coarse optimization will NOT be repeated: {coarse_model_path}", flush=True)
    print("Poisson reconstruction will NOT be repeated.", flush=True)
    print(f"Refinement iterations: {args.refinement_iterations}", flush=True)
    print("Masks: enabled from aligned RGBA alpha", flush=True)
    print(f"GPU: {args.gpu}", flush=True)
    print(f"Run config: {config_path}", flush=True)

    refined_args = AttrDict(
        scene_path=str(scene_path),
        checkpoint_path=checkpoint_argument,
        mesh_path=str(coarse_mesh_path),
        output_dir=str(refined_output),
        iteration_to_load=iteration,
        normal_consistency_factor=args.normal_consistency_factor,
        gaussians_per_triangle=args.gaussians_per_triangle,
        n_vertices_in_fg=args.vertices,
        refinement_iterations=args.refinement_iterations,
        bboxmin=tuple_text(bbox_min),
        bboxmax=tuple_text(bbox_max),
        export_ply=True,
        eval=True,
        gpu=args.gpu,
        white_background=True,
        use_masks=True,
    )
    refined_model_path = refined_training(refined_args)

    refined_mesh_args = AttrDict(
        scene_path=str(scene_path),
        iteration_to_load=iteration,
        checkpoint_path=checkpoint_argument,
        refined_model_path=refined_model_path,
        coarse_mesh_path=str(coarse_mesh_path),
        mesh_output_dir=str(refined_mesh_output),
        n_gaussians_per_surface_triangle=args.gaussians_per_triangle,
        square_size=args.square_size,
        eval=True,
        gpu=args.gpu,
        postprocess_mesh=False,
        postprocess_density_threshold=0.1,
        postprocess_iterations=5,
        white_background=True,
    )
    refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)

    result = {
        **config,
        "refined_model_path": str(refined_model_path),
        "refined_mesh_path": str(refined_mesh_path),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "complete",
    }
    result_path = output_root / "refinement_run_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("=== Refinement and final mesh export complete ===", flush=True)
    print(f"Refined checkpoint: {refined_model_path}", flush=True)
    print(f"Final mesh: {refined_mesh_path}", flush=True)
    print(f"Result manifest: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
