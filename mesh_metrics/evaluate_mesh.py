"""Run alignment, geometry metrics, and optional held-out view metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .alignment import AlignmentConfig, align_meshes
from .geometry_metrics import GeometryMetricConfig, compute_geometry_metrics
from .mesh_io import load_mesh, mesh_summary, transform_mesh
from .render_metrics import (
    RenderMetricConfig,
    compute_heldout_metrics,
    validate_reference_render,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align and evaluate a reconstructed shoe mesh."
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Blender evaluation scene containing reference_mesh.ply and test assets.",
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-aligned", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--training-view-set",
        choices=("train", "all", "unknown"),
        default="unknown",
        help="Declare whether the method trained on 150 train views or all views.",
    )

    parser.add_argument("--alignment-samples", type=int, default=50_000)
    parser.add_argument("--coarse-samples", type=int, default=5_000)
    parser.add_argument("--alignment-candidates", type=int, default=4)
    parser.add_argument("--inlier-fraction", type=float, default=0.8)
    parser.add_argument("--alignment-iterations", type=int, default=100)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-7)
    parser.add_argument("--alignment-seed", type=int, default=0)

    parser.add_argument("--metric-samples", type=int, default=200_000)
    parser.add_argument("--metric-seed", type=int, default=10_000)
    parser.add_argument("--query-chunk-size", type=int, default=250_000)
    parser.add_argument("--boundary-tolerance-px", type=float, default=2.0)
    parser.add_argument(
        "--minimum-reference-mask-iou",
        type=float,
        default=0.98,
        help=(
            "Minimum per-view IoU required when validating the exported "
            "ground-truth mesh against the Blender masks."
        ),
    )
    return parser


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    prediction_path = args.prediction.expanduser().resolve()
    scene_root = args.scene.expanduser().resolve()
    ground_truth_path = (
        args.ground_truth.expanduser().resolve()
        if args.ground_truth
        else scene_root / "reference_mesh.ply"
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    prediction = load_mesh(prediction_path)
    ground_truth = load_mesh(ground_truth_path)
    alignment_config = AlignmentConfig(
        sample_count=args.alignment_samples,
        coarse_sample_count=args.coarse_samples,
        candidate_count=args.alignment_candidates,
        inlier_fraction=args.inlier_fraction,
        max_iterations=args.alignment_iterations,
        tolerance=args.alignment_tolerance,
        seed=args.alignment_seed,
    )
    alignment = align_meshes(prediction, ground_truth, alignment_config)
    aligned_prediction = transform_mesh(prediction, alignment.transform)
    alignment_payload: dict[str, object] = {
        "schema_version": 1,
        "prediction": str(prediction_path),
        "ground_truth": str(ground_truth_path),
        "prediction_mesh": mesh_summary(prediction),
        "ground_truth_mesh": mesh_summary(ground_truth),
        "configuration": {
            "sample_count": alignment_config.sample_count,
            "coarse_sample_count": alignment_config.coarse_sample_count,
            "candidate_count": alignment_config.candidate_count,
            "inlier_fraction": alignment_config.inlier_fraction,
            "max_iterations": alignment_config.max_iterations,
            "tolerance": alignment_config.tolerance,
            "seed": alignment_config.seed,
            "allow_reflection": False,
            "allow_nonuniform_scale": False,
        },
        "alignment": alignment.to_dict(),
    }
    _atomic_json(output / "alignment.json", alignment_payload)
    if args.save_aligned:
        aligned_prediction.export(output / "aligned_prediction.ply")

    geometry_config = GeometryMetricConfig(
        sample_count=args.metric_samples,
        seed=args.metric_seed,
        query_chunk_size=args.query_chunk_size,
    )
    geometry = compute_geometry_metrics(
        aligned_prediction,
        ground_truth,
        geometry_config,
    )
    geometry["prediction"] = str(prediction_path)
    geometry["ground_truth"] = str(ground_truth_path)
    _atomic_json(output / "geometry_metrics.json", geometry)

    if not args.geometry_only:
        render_config = RenderMetricConfig(
            boundary_tolerance_px=args.boundary_tolerance_px,
            ray_chunk_size=args.query_chunk_size,
            minimum_reference_mask_iou=args.minimum_reference_mask_iou,
        )
        diagonal = float(geometry["ground_truth_bbox_diagonal"])
        reference_validation = validate_reference_render(
            ground_truth,
            scene_root,
            diagonal,
            render_config,
        )
        _atomic_json(
            output / "reference_render_validation.json",
            reference_validation,
        )
        view_metrics = compute_heldout_metrics(
            aligned_prediction,
            scene_root,
            diagonal,
            render_config,
        )
        view_metrics["prediction"] = str(prediction_path)
        view_metrics["scene"] = str(scene_root)
        view_metrics["training_view_set"] = args.training_view_set
        view_metrics["heldout_eligible"] = args.training_view_set == "train"
        _atomic_json(output / "view_metrics.json", view_metrics)

    print(f"Evaluation written to {output}")
    print(json.dumps(geometry["headline"], indent=2))
    if not args.geometry_only:
        print(json.dumps(view_metrics["headline"], indent=2))
        if args.training_view_set != "train":
            print("View metrics are not labelled held out because training-view-set is not train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
