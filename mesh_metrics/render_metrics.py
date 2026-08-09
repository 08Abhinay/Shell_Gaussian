"""Held-out silhouette and axial-depth metrics for aligned meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
import trimesh

from .camera_io import EvaluationCamera, load_test_cameras
from .surface_queries import TriangleSurface


@dataclass(frozen=True)
class RenderMetricConfig:
    boundary_tolerance_px: float = 2.0
    depth_border_erosion_px: int = 1
    ray_chunk_size: int = 250_000
    minimum_reference_mask_iou: float = 0.98
    maximum_reference_depth_relative_p95: float = 0.01

    def validate(self) -> None:
        if self.boundary_tolerance_px <= 0.0:
            raise ValueError("boundary_tolerance_px must be positive")
        if self.depth_border_erosion_px < 0:
            raise ValueError("depth_border_erosion_px cannot be negative")
        if self.ray_chunk_size < 1:
            raise ValueError("ray_chunk_size must be positive")
        if not 0.0 < self.minimum_reference_mask_iou <= 1.0:
            raise ValueError("minimum_reference_mask_iou must be in (0, 1]")
        if self.maximum_reference_depth_relative_p95 <= 0.0:
            raise ValueError("maximum_reference_depth_relative_p95 must be positive")


class HeldOutRenderer:
    """Render first-surface masks and camera-Z depth from exact test cameras."""

    def __init__(self, width: int, height: int, fov_x_rad: float) -> None:
        self.width = int(width)
        self.height = int(height)
        focal = self.width / (2.0 * np.tan(0.5 * float(fov_x_rad)))
        columns = np.arange(self.width, dtype=np.float32) + 0.5
        rows = np.arange(self.height, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(columns, rows)
        x = (grid_x - self.width / 2.0) / focal
        y = -(grid_y - self.height / 2.0) / focal
        local = np.stack((x, y, -np.ones_like(x)), axis=-1)
        local /= np.linalg.norm(local, axis=-1, keepdims=True)
        self.local_directions = np.ascontiguousarray(local.reshape(-1, 3), dtype=np.float32)
        self.axial_factors = np.ascontiguousarray(-local[..., 2], dtype=np.float64)

    def render(
        self,
        surface: TriangleSurface,
        camera: EvaluationCamera,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if camera.width != self.width or camera.height != self.height:
            raise ValueError("All held-out cameras must share one resolution")
        rotation = camera.effective_c2w[:3, :3]
        origin = camera.effective_c2w[:3, 3]
        hit_parts: list[np.ndarray] = []
        for start in range(0, len(self.local_directions), chunk_size):
            local = self.local_directions[start : start + chunk_size]
            directions = local.astype(np.float64) @ rotation.T
            origins = np.broadcast_to(origin, directions.shape)
            rays = np.concatenate((origins, directions), axis=1)
            hit_parts.append(surface.cast_rays(rays, chunk_size))
        ray_depth = np.concatenate(hit_parts).reshape(self.height, self.width)
        mask = np.isfinite(ray_depth)
        axial_depth = ray_depth * self.axial_factors
        axial_depth[~mask] = 0.0
        return mask, axial_depth


def load_ground_truth_view(camera: EvaluationCamera) -> tuple[np.ndarray, np.ndarray]:
    """Load the top-left mask and camera-Z depth for one held-out frame."""
    mask = np.asarray(Image.open(camera.mask_path).convert("L")) > 127
    inverse_depth = np.load(camera.invdepth_path, mmap_mode="r")
    if mask.shape != (camera.height, camera.width):
        raise ValueError(f"Unexpected mask shape for {camera.frame_name}: {mask.shape}")
    if inverse_depth.shape != mask.shape:
        raise ValueError(
            f"Inverse-depth shape mismatch for {camera.frame_name}: {inverse_depth.shape}"
        )
    depth = np.zeros(mask.shape, dtype=np.float64)
    valid = np.isfinite(inverse_depth) & (inverse_depth > 0.0)
    depth[valid] = 1.0 / inverse_depth[valid]
    return mask, depth


def silhouette_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    union = int(np.count_nonzero(prediction | ground_truth))
    return float(intersection / union) if union else 1.0


def boundary_f_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_px: float,
) -> dict[str, float]:
    prediction_boundary = prediction ^ binary_erosion(prediction)
    ground_truth_boundary = ground_truth ^ binary_erosion(ground_truth)
    prediction_count = int(np.count_nonzero(prediction_boundary))
    ground_truth_count = int(np.count_nonzero(ground_truth_boundary))
    if prediction_count == 0 and ground_truth_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f_score": 1.0}
    if prediction_count == 0 or ground_truth_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
    distance_to_gt = distance_transform_edt(~ground_truth_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)
    precision = float(np.mean(distance_to_gt[prediction_boundary] <= tolerance_px))
    recall = float(np.mean(distance_to_prediction[ground_truth_boundary] <= tolerance_px))
    denominator = precision + recall
    score = 2.0 * precision * recall / denominator if denominator > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f_score": float(score)}


def evaluate_rendered_view(
    predicted_mask: np.ndarray,
    predicted_depth: np.ndarray,
    ground_truth_mask: np.ndarray,
    ground_truth_depth: np.ndarray,
    camera: EvaluationCamera,
    diagonal: float,
    config: RenderMetricConfig,
) -> dict[str, object]:
    """Compare one rendered mask/depth pair against its held-out target."""
    depth_region = ground_truth_mask
    if config.depth_border_erosion_px:
        depth_region = binary_erosion(
            depth_region,
            iterations=config.depth_border_erosion_px,
        )
    ground_truth_depth_valid = (
        depth_region
        & np.isfinite(ground_truth_depth)
        & (ground_truth_depth > 0.0)
        & (ground_truth_depth < 2.0 * camera.radius)
    )
    overlap = ground_truth_depth_valid & predicted_mask & (predicted_depth > 0.0)
    ground_truth_count = int(np.count_nonzero(ground_truth_depth_valid))
    overlap_count = int(np.count_nonzero(overlap))
    coverage = float(overlap_count / ground_truth_count) if ground_truth_count else 0.0
    if overlap_count:
        depth_error = np.abs(predicted_depth[overlap] - ground_truth_depth[overlap])
        depth_mae = float(np.mean(depth_error))
        depth_relative = depth_error / ground_truth_depth[overlap]
        relative_median = float(np.median(depth_relative))
        relative_p95 = float(np.percentile(depth_relative, 95.0))
    else:
        depth_mae = float("inf")
        relative_median = float("inf")
        relative_p95 = float("inf")
    boundary = boundary_f_score(
        predicted_mask,
        ground_truth_mask,
        config.boundary_tolerance_px,
    )
    return {
        "frame": camera.frame_name,
        "ring_index": camera.ring_index,
        "elevation_deg": camera.elevation_deg,
        "azimuth_deg": camera.azimuth_deg,
        "silhouette_iou": silhouette_iou(predicted_mask, ground_truth_mask),
        "boundary_precision": boundary["precision"],
        "boundary_recall": boundary["recall"],
        "boundary_f_score": boundary["f_score"],
        "depth_overlap_coverage": coverage,
        "depth_mae": depth_mae,
        "depth_mae_percent": 100.0 * depth_mae / diagonal,
        "depth_relative_median": relative_median,
        "depth_relative_p95": relative_p95,
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.mean(values))


def _evaluation_protocol(rows: list[dict[str, object]]) -> str:
    elevations = np.asarray(
        [float(row["elevation_deg"]) for row in rows], dtype=np.float64
    )
    if len(rows) == 6 and np.all(np.isclose(elevations, 0.0)):
        return "turntable"

    full_elevations = (-25.0, 0.0, 20.0, 45.0, 65.0)
    if len(rows) == 30 and all(
        int(np.count_nonzero(np.isclose(elevations, expected))) == 6
        for expected in full_elevations
    ):
        return "full_view"

    raise ValueError(
        "Held-out cameras must follow either the 30-view full-view contract "
        "or the 6-view level turntable contract"
    )


def compute_heldout_metrics(
    mesh: trimesh.Trimesh,
    scene_root: str | Path,
    ground_truth_diagonal: float,
    config: RenderMetricConfig | None = None,
) -> dict[str, object]:
    """Render and score one aligned mesh using a supported held-out protocol."""
    settings = config or RenderMetricConfig()
    settings.validate()
    cameras = load_test_cameras(scene_root)
    renderer = HeldOutRenderer(cameras[0].width, cameras[0].height, cameras[0].fov_x_rad)
    surface = TriangleSurface(mesh)
    rows: list[dict[str, object]] = []
    for camera in cameras:
        predicted_mask, predicted_depth = renderer.render(
            surface,
            camera,
            settings.ray_chunk_size,
        )
        ground_truth_mask, ground_truth_depth = load_ground_truth_view(camera)
        rows.append(
            evaluate_rendered_view(
                predicted_mask,
                predicted_depth,
                ground_truth_mask,
                ground_truth_depth,
                camera,
                ground_truth_diagonal,
                settings,
            )
        )

    protocol = _evaluation_protocol(rows)
    headline = {
        "silhouette_iou": _mean(rows, "silhouette_iou"),
        "boundary_f_score": _mean(rows, "boundary_f_score"),
        "depth_mae_percent": _mean(rows, "depth_mae_percent"),
        "depth_overlap_coverage": _mean(rows, "depth_overlap_coverage"),
    }
    if protocol == "full_view":
        underside = [row for row in rows if np.isclose(row["elevation_deg"], -25.0)]
        top = [
            row
            for row in rows
            if np.isclose(row["elevation_deg"], 45.0)
            or np.isclose(row["elevation_deg"], 65.0)
        ]
        headline.update(
            {
                "underside_depth_mae_percent": _mean(underside, "depth_mae_percent"),
                "top_view_depth_mae_percent": _mean(top, "depth_mae_percent"),
            }
        )
    return {
        "schema_version": 1,
        "evaluation_protocol": protocol,
        "view_count": len(rows),
        "depth_definition": "camera_z_depth",
        "depth_normalization": "percentage_of_ground_truth_bbox_diagonal",
        "boundary_tolerance_px": settings.boundary_tolerance_px,
        "headline": headline,
        "per_view": rows,
    }


def validate_reference_render(
    reference_mesh: trimesh.Trimesh,
    scene_root: str | Path,
    ground_truth_diagonal: float,
    config: RenderMetricConfig | None = None,
) -> dict[str, object]:
    """Require the exported GT mesh to reproduce Blender masks and depth."""
    settings = config or RenderMetricConfig()
    result = compute_heldout_metrics(
        reference_mesh,
        scene_root,
        ground_truth_diagonal,
        settings,
    )
    minimum_iou = min(float(row["silhouette_iou"]) for row in result["per_view"])
    maximum_depth_relative_p95 = max(
        float(row["depth_relative_p95"]) for row in result["per_view"]
    )
    passed = (
        minimum_iou >= settings.minimum_reference_mask_iou
        and maximum_depth_relative_p95
        <= settings.maximum_reference_depth_relative_p95
    )
    validation = {
        "passed": bool(passed),
        "minimum_silhouette_iou": minimum_iou,
        "required_minimum_silhouette_iou": settings.minimum_reference_mask_iou,
        "maximum_depth_relative_p95": maximum_depth_relative_p95,
        "allowed_maximum_depth_relative_p95": settings.maximum_reference_depth_relative_p95,
        "summary": result["headline"],
    }
    if not passed:
        raise RuntimeError(f"Reference camera validation failed: {validation}")
    return validation
