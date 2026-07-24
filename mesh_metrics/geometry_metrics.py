"""Area-weighted point-to-triangle geometry metrics for aligned meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .mesh_io import sample_surface
from .surface_queries import TriangleSurface


@dataclass(frozen=True)
class GeometryMetricConfig:
    sample_count: int = 200_000
    seed: int = 10_000
    fscore_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02)
    query_chunk_size: int = 250_000

    def validate(self) -> None:
        if self.sample_count < 100:
            raise ValueError("sample_count must be at least 100")
        if not self.fscore_thresholds:
            raise ValueError("At least one F-score threshold is required")
        if any(threshold <= 0.0 for threshold in self.fscore_thresholds):
            raise ValueError("F-score thresholds must be positive")
        if not any(np.isclose(threshold, 0.01) for threshold in self.fscore_thresholds):
            raise ValueError("fscore_thresholds must include the headline 1% threshold")
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")


def _normal_consistency(source_normals: np.ndarray, target_normals: np.ndarray) -> float:
    source_norms = np.linalg.norm(source_normals, axis=1)
    target_norms = np.linalg.norm(target_normals, axis=1)
    valid = (source_norms > 1e-12) & (target_norms > 1e-12)
    if not np.any(valid):
        return 0.0
    source = source_normals[valid] / source_norms[valid, None]
    target = target_normals[valid] / target_norms[valid, None]
    return float(np.mean(np.abs(np.sum(source * target, axis=1))))


def _edge_diagnostics(mesh: trimesh.Trimesh, diagonal: float) -> dict[str, object]:
    usage = np.bincount(
        mesh.edges_unique_inverse,
        minlength=len(mesh.edges_unique),
    )
    boundary = usage == 1
    nonmanifold = usage > 2
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
    )
    boundary_length = float(np.sum(mesh.edges_unique_length[boundary]))
    return {
        "connected_components": int(len(components)),
        "boundary_edge_count": int(np.count_nonzero(boundary)),
        "boundary_length": boundary_length,
        "boundary_length_over_gt_diagonal": float(boundary_length / diagonal),
        "nonmanifold_edge_count": int(np.count_nonzero(nonmanifold)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "is_watertight": bool(mesh.is_watertight),
        "postload_invalid_vertex_count": int(
            np.count_nonzero(~np.all(np.isfinite(mesh.vertices), axis=1))
        ),
        "postload_degenerate_face_count": int(
            np.count_nonzero(~mesh.nondegenerate_faces())
        ),
    }


def compute_geometry_metrics(
    aligned_prediction: trimesh.Trimesh,
    ground_truth: trimesh.Trimesh,
    config: GeometryMetricConfig | None = None,
) -> dict[str, object]:
    """Compute symmetric surface metrics after global similarity alignment."""
    settings = config or GeometryMetricConfig()
    settings.validate()
    diagonal = float(np.linalg.norm(np.asarray(ground_truth.extents, dtype=np.float64)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Ground-truth bounding-box diagonal is invalid")

    prediction_points, prediction_normals = sample_surface(
        aligned_prediction,
        settings.sample_count,
        settings.seed,
    )
    ground_truth_points, ground_truth_normals = sample_surface(
        ground_truth,
        settings.sample_count,
        settings.seed + 1,
    )

    prediction_surface = TriangleSurface(aligned_prediction)
    ground_truth_surface = TriangleSurface(ground_truth)
    prediction_to_gt = ground_truth_surface.closest_points(
        prediction_points,
        settings.query_chunk_size,
    )
    gt_to_prediction = prediction_surface.closest_points(
        ground_truth_points,
        settings.query_chunk_size,
    )

    accuracy = float(np.mean(prediction_to_gt.distances))
    completeness = float(np.mean(gt_to_prediction.distances))
    accuracy_p95 = float(np.percentile(prediction_to_gt.distances, 95.0))
    completeness_p95 = float(np.percentile(gt_to_prediction.distances, 95.0))
    prediction_normal_consistency = _normal_consistency(
        prediction_normals,
        prediction_to_gt.normals,
    )
    gt_normal_consistency = _normal_consistency(
        ground_truth_normals,
        gt_to_prediction.normals,
    )

    f_scores: dict[str, object] = {}
    for threshold_fraction in settings.fscore_thresholds:
        threshold = threshold_fraction * diagonal
        precision = float(np.mean(prediction_to_gt.distances <= threshold))
        recall = float(np.mean(gt_to_prediction.distances <= threshold))
        denominator = precision + recall
        f_score = 2.0 * precision * recall / denominator if denominator > 0.0 else 0.0
        key = f"{100.0 * threshold_fraction:g}_percent"
        f_scores[key] = {
            "threshold_fraction": float(threshold_fraction),
            "threshold_distance": float(threshold),
            "precision": precision,
            "recall": recall,
            "f_score": float(f_score),
        }

    prediction_diagnostics = _edge_diagnostics(aligned_prediction, diagonal)
    ground_truth_diagnostics = _edge_diagnostics(ground_truth, diagonal)
    headline = {
        "accuracy_percent": 100.0 * accuracy / diagonal,
        "completeness_percent": 100.0 * completeness / diagonal,
        "chamfer_l1_percent": 100.0 * 0.5 * (accuracy + completeness) / diagonal,
        "f_score_1_percent": f_scores["1_percent"]["f_score"],
        "normal_consistency": 0.5
        * (prediction_normal_consistency + gt_normal_consistency),
        "p95_distance_percent": 100.0
        * max(accuracy_p95, completeness_p95)
        / diagonal,
    }
    return {
        "schema_version": 1,
        "normalization": "percentage_of_ground_truth_bbox_diagonal",
        "ground_truth_bbox_diagonal": diagonal,
        "sample_count_per_surface": settings.sample_count,
        "sample_seed_prediction": settings.seed,
        "sample_seed_ground_truth": settings.seed + 1,
        "headline": headline,
        "distances": {
            "accuracy": accuracy,
            "completeness": completeness,
            "chamfer_l1": 0.5 * (accuracy + completeness),
            "accuracy_p95": accuracy_p95,
            "completeness_p95": completeness_p95,
            "p95_max": max(accuracy_p95, completeness_p95),
        },
        "f_scores": f_scores,
        "normal_consistency": {
            "prediction_to_ground_truth": prediction_normal_consistency,
            "ground_truth_to_prediction": gt_normal_consistency,
            "symmetric_mean": headline["normal_consistency"],
            "orientation_invariant": True,
        },
        "diagnostics": {
            "prediction": prediction_diagnostics,
            "ground_truth": ground_truth_diagnostics,
            "prediction_to_ground_truth_surface_area_ratio": float(
                aligned_prediction.area / ground_truth.area
            ),
        },
    }
