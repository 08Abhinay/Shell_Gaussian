"""Support-footprint and simple pseudo-footbed extraction for shoe meshes.

This module implements the Section 1.3 geometry stage from the foot fitting
notes: identify lower sole/support geometry, project it into the X-Z footprint
plane, extract a centerline and width profile, and build a first footbed profile
from the support surface plus a conservative sole/insole offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .foot_alignment import MeshData, find_boundary_components, mesh_bounds

try:  # Optional, but available in the current research environment.
    from scipy import ndimage
except Exception:  # pragma: no cover - fallback for lean environments.
    ndimage = None


@dataclass(frozen=True)
class SupportFootprintConfig:
    """Knobs for extracting a shoe support footprint.

    The defaults match the current canonical GShell shoe convention:

        +X = heel-to-toe length
        +Y = sole/base/bottom side
        -Y = opening/up side
        +Z = width
    """

    shoe_length_axis: int = 0
    shoe_up_axis: int = 1
    shoe_width_axis: int = 2
    shoe_up_sign: float = -1.0
    bottom_quantile: float = 0.80
    fallback_bottom_quantile: float = 0.70
    normal_angle_degrees: float = 65.0
    fallback_normal_angle_degrees: float = 88.0
    min_support_faces: int = 40
    grid_resolution: int = 256
    grid_padding_fraction: float = 0.03
    morphology_iterations: int = 3
    component_keep_ratio: float = 0.04
    min_component_pixels: int = 8
    min_slice_pixels: int = 3
    centerline_smooth_window: int = 9
    support_axis_quantile: float = 0.90
    floor_axis_quantile: float = 0.90
    floor_smooth_window: int = 17
    floor_min_samples_per_slice: int = 8
    heightmap_min_samples_per_cell: int = 2
    heightmap_smooth_sigma: float = 1.25
    heightmap_profile_clip: float = 0.025
    footbed_inner_margin_cells: int = 7
    footbed_inner_min_area_ratio: float = 0.35
    smooth_footbed_window_fraction: float = 0.18
    smooth_footbed_quantile: float = 0.50
    smooth_footbed_height_fraction_from_bottom: float = 0.22
    open_boundary_footbed_offset: Optional[float] = None
    open_boundary_footbed_offset_ratio: float = 0.055
    open_boundary_footbed_offset_min: float = 0.008
    open_boundary_footbed_offset_max: float = 0.022
    footbed_offset: float = 0.015
    use_open_boundary: bool = True
    open_bottom_quantile: float = 0.58
    broad_support_quantile: float = 0.55
    open_bottom_morphology_iterations: int = 5
    open_bottom_dilation_iterations: int = 2
    open_bottom_min_pixels: int = 64
    boundary_sample_spacing_fraction: float = 0.45
    boundary_morphology_iterations: int = 4
    boundary_filter_margin: float = 0.004
    boundary_axis_quantile: float = 0.90
    boundary_min_vertices: int = 20
    boundary_min_length_ratio: float = 0.20
    boundary_min_width_ratio: float = 0.20
    heel_fraction: Tuple[float, float] = (0.00, 0.15)
    midfoot_fraction: Tuple[float, float] = (0.25, 0.55)
    ball_fraction: Tuple[float, float] = (0.60, 0.75)
    toe_fraction: Tuple[float, float] = (0.85, 1.00)

    def __post_init__(self) -> None:
        axes = [self.shoe_length_axis, self.shoe_up_axis, self.shoe_width_axis]
        if sorted(axes) != [0, 1, 2]:
            raise ValueError("length/up/width axes must be a permutation of 0,1,2")
        if not 0.0 < self.bottom_quantile < 1.0:
            raise ValueError("bottom_quantile must be in (0, 1)")
        if not 0.0 < self.fallback_bottom_quantile < 1.0:
            raise ValueError("fallback_bottom_quantile must be in (0, 1)")
        if self.grid_resolution < 32:
            raise ValueError("grid_resolution must be at least 32")
        if self.component_keep_ratio < 0.0:
            raise ValueError("component_keep_ratio must be non-negative")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be positive")
        if not 0.0 <= self.support_axis_quantile <= 1.0:
            raise ValueError("support_axis_quantile must be in [0, 1]")
        if not 0.0 <= self.floor_axis_quantile <= 1.0:
            raise ValueError("floor_axis_quantile must be in [0, 1]")
        if self.floor_smooth_window < 1:
            raise ValueError("floor_smooth_window must be positive")
        if self.floor_min_samples_per_slice < 1:
            raise ValueError("floor_min_samples_per_slice must be positive")
        if self.heightmap_min_samples_per_cell < 1:
            raise ValueError("heightmap_min_samples_per_cell must be positive")
        if self.heightmap_smooth_sigma < 0.0:
            raise ValueError("heightmap_smooth_sigma must be non-negative")
        if self.heightmap_profile_clip < 0.0:
            raise ValueError("heightmap_profile_clip must be non-negative")
        if self.footbed_inner_margin_cells < 0:
            raise ValueError("footbed_inner_margin_cells must be non-negative")
        if not 0.0 < self.footbed_inner_min_area_ratio <= 1.0:
            raise ValueError("footbed_inner_min_area_ratio must be in (0, 1]")
        if self.smooth_footbed_window_fraction < 0.0:
            raise ValueError("smooth_footbed_window_fraction must be non-negative")
        if not 0.0 <= self.smooth_footbed_quantile <= 1.0:
            raise ValueError("smooth_footbed_quantile must be in [0, 1]")
        if not 0.0 <= self.smooth_footbed_height_fraction_from_bottom <= 1.0:
            raise ValueError("smooth_footbed_height_fraction_from_bottom must be in [0, 1]")
        if self.open_boundary_footbed_offset is not None and self.open_boundary_footbed_offset < 0.0:
            raise ValueError("open_boundary_footbed_offset must be non-negative")
        if self.open_boundary_footbed_offset_ratio < 0.0:
            raise ValueError("open_boundary_footbed_offset_ratio must be non-negative")
        if self.open_boundary_footbed_offset_min < 0.0:
            raise ValueError("open_boundary_footbed_offset_min must be non-negative")
        if self.open_boundary_footbed_offset_max < self.open_boundary_footbed_offset_min:
            raise ValueError("open_boundary_footbed_offset_max must be >= open_boundary_footbed_offset_min")
        if not 0.0 <= self.boundary_axis_quantile <= 1.0:
            raise ValueError("boundary_axis_quantile must be in [0, 1]")
        if self.boundary_filter_margin < 0.0:
            raise ValueError("boundary_filter_margin must be non-negative")
        if self.boundary_sample_spacing_fraction <= 0.0:
            raise ValueError("boundary_sample_spacing_fraction must be positive")
        if self.boundary_min_vertices < 1:
            raise ValueError("boundary_min_vertices must be positive")
        if self.footbed_offset < 0.0:
            raise ValueError("footbed_offset must be non-negative")
        if not 0.0 < self.open_bottom_quantile < 1.0:
            raise ValueError("open_bottom_quantile must be in (0, 1)")
        if not 0.0 < self.broad_support_quantile < 1.0:
            raise ValueError("broad_support_quantile must be in (0, 1)")
        if self.open_bottom_morphology_iterations < 0:
            raise ValueError("open_bottom_morphology_iterations must be non-negative")
        if self.open_bottom_dilation_iterations < 0:
            raise ValueError("open_bottom_dilation_iterations must be non-negative")
        if self.open_bottom_min_pixels < 1:
            raise ValueError("open_bottom_min_pixels must be positive")


@dataclass(frozen=True)
class LowerOpenBoundary:
    """Trusted lower cut boundary detected from the open/mSDF shoe mesh."""

    component_index: int
    vertex_indices: np.ndarray
    edge_indices: np.ndarray
    points: np.ndarray
    sample_points: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    size: np.ndarray
    center: np.ndarray
    score: float
    confidence: Dict[str, object]
    source: str = "boundary_components"
    footprint_mask: Optional[np.ndarray] = None
    x_edges: Optional[np.ndarray] = None
    z_edges: Optional[np.ndarray] = None
    outline_points: Optional[np.ndarray] = None

    def to_summary_dict(self) -> Dict[str, object]:
        payload = {
            "source": self.source,
            "component_index": int(self.component_index),
            "vertex_count": int(self.vertex_indices.size),
            "edge_count": int(self.edge_indices.shape[0]),
            "sample_count": int(self.sample_points.shape[0]),
            "outline_sample_count": 0 if self.outline_points is None else int(self.outline_points.shape[0]),
            "bounds_min": self.bounds_min.astype(float).tolist(),
            "bounds_max": self.bounds_max.astype(float).tolist(),
            "size": self.size.astype(float).tolist(),
            "center": self.center.astype(float).tolist(),
            "score": float(self.score),
            "confidence": _jsonify(self.confidence),
        }
        if self.footprint_mask is not None:
            payload["footprint_pixel_count"] = int(np.asarray(self.footprint_mask, dtype=bool).sum())
            payload["footprint_fill_fraction"] = float(np.asarray(self.footprint_mask, dtype=bool).mean())
        return payload


@dataclass(frozen=True)
class SupportFootprint:
    """Measured support-footprint geometry for one shoe mesh."""

    support_face_indices: np.ndarray
    support_vertex_indices: np.ndarray
    support_points: np.ndarray
    footprint_mask: np.ndarray
    x_edges: np.ndarray
    z_edges: np.ndarray
    centerline_x: np.ndarray
    centerline_z: np.ndarray
    left_boundary_z: np.ndarray
    right_boundary_z: np.ndarray
    width_profile: np.ndarray
    raw_floor_axis_profile: np.ndarray
    floor_sample_count_profile: np.ndarray
    support_face_axis_profile: np.ndarray
    support_axis_profile: np.ndarray
    footbed_axis_profile: np.ndarray
    floor_sample_points: np.ndarray
    raw_floor_heightmap: np.ndarray
    floor_heightmap: np.ndarray
    footbed_mask: np.ndarray
    footbed_heightmap: np.ndarray
    smooth_footbed_axis_profile: np.ndarray
    smooth_footbed_heightmap: np.ndarray
    smooth_footbed_offset: float
    smooth_footbed_source: str
    heightmap_sample_count: np.ndarray
    heel_mask: np.ndarray
    midfoot_mask: np.ndarray
    ball_mask: np.ndarray
    toe_mask: np.ndarray
    bbox_center: np.ndarray
    footprint_center: np.ndarray
    suggested_width_anchor: float
    suggested_length_anchor: float
    lower_boundary: Optional[LowerOpenBoundary]
    confidence: Dict[str, object]
    config: SupportFootprintConfig

    @property
    def length_extent(self) -> float:
        if self.centerline_x.size == 0:
            return 0.0
        return float(self.centerline_x.max() - self.centerline_x.min())

    @property
    def median_width(self) -> float:
        if self.width_profile.size == 0:
            return 0.0
        return float(np.median(self.width_profile))

    def to_summary_dict(self, include_profile: bool = True) -> Dict[str, object]:
        """Return a JSON-serializable summary without the dense support points."""

        payload: Dict[str, object] = {
            "support_face_count": int(self.support_face_indices.size),
            "support_vertex_count": int(self.support_vertex_indices.size),
            "footprint_pixel_count": int(self.footprint_mask.sum()),
            "profile_point_count": int(self.centerline_x.size),
            "length_extent": self.length_extent,
            "median_width": self.median_width,
            "bbox_center": self.bbox_center.astype(float).tolist(),
            "footprint_center": self.footprint_center.astype(float).tolist(),
            "suggested_length_anchor": float(self.suggested_length_anchor),
            "suggested_width_anchor": float(self.suggested_width_anchor),
            "heightmap_shape": [int(v) for v in self.floor_heightmap.shape],
            "heightmap_inside_cell_count": int(self.footprint_mask.sum()),
            "heightmap_valid_cell_count": int(np.isfinite(self.raw_floor_heightmap[self.footprint_mask]).sum()),
            "footbed_mask_cell_count": int(self.footbed_mask.sum()),
            "footbed_mask_area_fraction": float(self.footbed_mask.sum() / max(int(self.footprint_mask.sum()), 1)),
            "heightmap_floor_y_range": _finite_range(self.floor_heightmap),
            "heightmap_footbed_y_range": _finite_range(self.footbed_heightmap),
            "smooth_footbed_y_range": _finite_range(self.smooth_footbed_heightmap),
            "smooth_footbed_offset": float(self.smooth_footbed_offset),
            "smooth_footbed_source": self.smooth_footbed_source,
            "smooth_footbed_height_fraction_from_bottom": float(self.config.smooth_footbed_height_fraction_from_bottom),
            "lower_boundary": None if self.lower_boundary is None else self.lower_boundary.to_summary_dict(),
            "confidence": _jsonify(self.confidence),
            "config": asdict(self.config),
        }
        if include_profile:
            payload.update(
                {
                    "centerline_x": self.centerline_x.astype(float).tolist(),
                    "centerline_z": self.centerline_z.astype(float).tolist(),
                    "left_boundary_z": self.left_boundary_z.astype(float).tolist(),
                    "right_boundary_z": self.right_boundary_z.astype(float).tolist(),
                    "width_profile": self.width_profile.astype(float).tolist(),
                    "raw_floor_axis_profile": self.raw_floor_axis_profile.astype(float).tolist(),
                    "floor_sample_count_profile": self.floor_sample_count_profile.astype(int).tolist(),
                    "support_face_axis_profile": self.support_face_axis_profile.astype(float).tolist(),
                    "support_axis_profile": self.support_axis_profile.astype(float).tolist(),
                    "footbed_axis_profile": self.footbed_axis_profile.astype(float).tolist(),
                    "heel_indices": np.flatnonzero(self.heel_mask).astype(int).tolist(),
                    "midfoot_indices": np.flatnonzero(self.midfoot_mask).astype(int).tolist(),
                    "ball_indices": np.flatnonzero(self.ball_mask).astype(int).tolist(),
                    "toe_indices": np.flatnonzero(self.toe_mask).astype(int).tolist(),
                }
            )
        return payload

    def save_json(self, path: str | Path, include_profile: bool = True) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.to_summary_dict(include_profile=include_profile), f, indent=2)
            f.write("\n")


def extract_support_footprint(
    mesh: MeshData,
    config: Optional[SupportFootprintConfig] = None,
    *,
    open_mesh: Optional[MeshData] = None,
) -> SupportFootprint:
    """Extract lower support footprint and a simple pseudo-footbed profile."""

    cfg = config or SupportFootprintConfig()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh.faces must have triangular shape [F, 3]")

    lower_boundary = None
    if cfg.use_open_boundary and open_mesh is not None:
        try:
            lower_boundary = detect_open_bottom_silhouette(open_mesh, cfg, reference_mesh=mesh)
        except Exception:
            lower_boundary = detect_lower_open_boundary(open_mesh, cfg, reference_mesh=mesh)

    centroids, normals, areas = _face_geometry(vertices, faces)
    support_indices, selection_info = _select_support_faces(centroids, normals, cfg)
    if lower_boundary is not None:
        support_indices, filter_info = _apply_open_bottom_filter(
            support_indices,
            centroids,
            vertices,
            cfg,
            lower_boundary,
        )
        selection_info.update(filter_info)
    else:
        selection_info["open_boundary_filter_used"] = False

    support_vertex_indices = np.unique(faces[support_indices].reshape(-1))
    support_points = _support_sample_points(vertices, faces[support_indices], centroids[support_indices])

    if lower_boundary is not None:
        if lower_boundary.footprint_mask is not None and lower_boundary.x_edges is not None and lower_boundary.z_edges is not None:
            footprint_mask = np.asarray(lower_boundary.footprint_mask, dtype=bool)
            x_edges = np.asarray(lower_boundary.x_edges, dtype=np.float32)
            z_edges = np.asarray(lower_boundary.z_edges, dtype=np.float32)
            footprint_mask_source = lower_boundary.source
        else:
            footprint_mask, x_edges, z_edges = _rasterize_boundary_footprint(
                vertices,
                lower_boundary.sample_points,
                cfg,
            )
            footprint_mask_source = lower_boundary.source
    else:
        footprint_mask, x_edges, z_edges = _rasterize_footprint(vertices, support_points, cfg)
        footprint_mask = _clean_footprint_mask(footprint_mask, cfg)
        footprint_mask_source = "support_faces"

    profile = _profile_from_mask(footprint_mask, x_edges, z_edges, cfg)
    if profile is None:
        raise ValueError("Could not extract a usable footprint profile from support geometry")

    (
        centerline_x,
        centerline_z,
        left_boundary_z,
        right_boundary_z,
        width_profile,
    ) = profile

    support_face_axis_profile = _axis_profile_from_points(
        support_points,
        centerline_x,
        x_edges,
        cfg,
        quantile=cfg.support_axis_quantile,
    )
    floor_estimate = _estimate_floor_profile_from_footprint(
        vertices,
        faces,
        centroids,
        centerline_x,
        x_edges,
        z_edges,
        footprint_mask,
        lower_boundary,
        cfg,
        fallback_profile=support_face_axis_profile,
    )
    support_axis_profile = floor_estimate["smooth_profile"]
    raw_floor_axis_profile = floor_estimate["raw_profile"]
    floor_sample_count_profile = floor_estimate["sample_counts"]
    floor_sample_points = floor_estimate["sample_points"]
    raw_floor_heightmap = floor_estimate["raw_heightmap"]
    floor_heightmap = floor_estimate["smooth_heightmap"]
    heightmap_sample_count = floor_estimate["heightmap_sample_counts"]
    axis_profile_source = str(floor_estimate["source"])

    footbed_axis_profile = support_axis_profile + _axis_sign(cfg.shoe_up_sign) * cfg.footbed_offset
    footbed_mask = _make_inner_footbed_mask(footprint_mask, cfg)
    footbed_heightmap = floor_heightmap + _axis_sign(cfg.shoe_up_sign) * cfg.footbed_offset
    footbed_heightmap = footbed_heightmap.astype(np.float32)
    footbed_heightmap[~footbed_mask] = np.nan
    smooth_footbed_axis_profile, smooth_footbed_offset, smooth_footbed_source = _smooth_footbed_profile_from_open_boundary(
        lower_boundary,
        vertices,
        centerline_x,
        x_edges,
        cfg,
    )
    smooth_footbed_heightmap = _profile_grid_from_arrays(
        footbed_mask,
        x_edges,
        centerline_x,
        smooth_footbed_axis_profile,
    )
    heel_mask = _fraction_mask(centerline_x, cfg.heel_fraction)
    midfoot_mask = _fraction_mask(centerline_x, cfg.midfoot_fraction)
    ball_mask = _fraction_mask(centerline_x, cfg.ball_fraction)
    toe_mask = _fraction_mask(centerline_x, cfg.toe_fraction)

    _, _, _, bbox_center = mesh_bounds(vertices)
    footprint_center = np.asarray(
        [
            float(np.mean(centerline_x)),
            float(np.median(support_axis_profile)),
            float(np.mean(centerline_z)),
        ],
        dtype=np.float32,
    )
    anchor_mask = ball_mask | midfoot_mask
    if not np.any(anchor_mask):
        anchor_mask = np.ones_like(centerline_x, dtype=bool)
    suggested_width_anchor = float(np.average(centerline_z[anchor_mask], weights=np.maximum(width_profile[anchor_mask], 1e-8)))
    suggested_length_anchor = float(np.average(centerline_x[anchor_mask], weights=np.maximum(width_profile[anchor_mask], 1e-8)))

    support_area = float(np.sum(areas[support_indices]))
    total_area = float(np.sum(areas))
    confidence: Dict[str, object] = {
        **selection_info,
        "support_area_fraction": support_area / max(total_area, 1e-12),
        "footprint_fill_fraction": float(footprint_mask.mean()),
        "profile_point_count": int(centerline_x.size),
        "median_width": float(np.median(width_profile)),
        "width_iqr": float(np.percentile(width_profile, 75) - np.percentile(width_profile, 25)),
        "used_scipy_morphology": bool(ndimage is not None),
        "footprint_mask_source": footprint_mask_source,
        "axis_profile_source": axis_profile_source,
        "floor_profile": floor_estimate["confidence"],
        "floor_heightmap": floor_estimate["heightmap_confidence"],
    }

    return SupportFootprint(
        support_face_indices=support_indices.astype(np.int64),
        support_vertex_indices=support_vertex_indices.astype(np.int64),
        support_points=support_points.astype(np.float32),
        footprint_mask=footprint_mask.astype(bool),
        x_edges=x_edges.astype(np.float32),
        z_edges=z_edges.astype(np.float32),
        centerline_x=centerline_x.astype(np.float32),
        centerline_z=centerline_z.astype(np.float32),
        left_boundary_z=left_boundary_z.astype(np.float32),
        right_boundary_z=right_boundary_z.astype(np.float32),
        width_profile=width_profile.astype(np.float32),
        raw_floor_axis_profile=raw_floor_axis_profile.astype(np.float32),
        floor_sample_count_profile=floor_sample_count_profile.astype(np.int32),
        support_face_axis_profile=support_face_axis_profile.astype(np.float32),
        support_axis_profile=support_axis_profile.astype(np.float32),
        footbed_axis_profile=footbed_axis_profile.astype(np.float32),
        floor_sample_points=floor_sample_points.astype(np.float32),
        raw_floor_heightmap=raw_floor_heightmap.astype(np.float32),
        floor_heightmap=floor_heightmap.astype(np.float32),
        footbed_mask=footbed_mask.astype(bool),
        footbed_heightmap=footbed_heightmap.astype(np.float32),
        smooth_footbed_axis_profile=smooth_footbed_axis_profile.astype(np.float32),
        smooth_footbed_heightmap=smooth_footbed_heightmap.astype(np.float32),
        smooth_footbed_offset=float(smooth_footbed_offset),
        smooth_footbed_source=smooth_footbed_source,
        heightmap_sample_count=heightmap_sample_count.astype(np.int32),
        heel_mask=heel_mask,
        midfoot_mask=midfoot_mask,
        ball_mask=ball_mask,
        toe_mask=toe_mask,
        bbox_center=bbox_center.astype(np.float32),
        footprint_center=footprint_center,
        suggested_width_anchor=suggested_width_anchor,
        suggested_length_anchor=suggested_length_anchor,
        lower_boundary=lower_boundary,
        confidence=confidence,
        config=cfg,
    )


def detect_open_bottom_silhouette(
    open_mesh: MeshData,
    config: Optional[SupportFootprintConfig] = None,
    *,
    reference_mesh: Optional[MeshData] = None,
) -> LowerOpenBoundary:
    """Build the trusted outer bottom outline from the open/mSDF mesh.

    This is intentionally different from reading raw mesh boundary loops. Crocs
    and sandals can have holes or broken inner cuts on the bottom. For foot
    fitting we want the outside bottom outline, so we project lower open-mesh
    surface samples into X-Z, fill holes, and keep the outer shape.
    """

    cfg = config or SupportFootprintConfig()
    vertices = np.asarray(open_mesh.vertices, dtype=np.float32)
    faces = np.asarray(open_mesh.faces, dtype=np.int64)
    reference_vertices = vertices if reference_mesh is None else np.asarray(reference_mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("open_mesh.vertices must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("open_mesh.faces must have triangular shape [F, 3]")

    centroids, _, _ = _face_geometry(vertices, faces)
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    bottom_scores = sole_sign * centroids[:, cfg.shoe_up_axis]
    quantile_attempts = (
        cfg.open_bottom_quantile,
        min(cfg.fallback_bottom_quantile, cfg.open_bottom_quantile),
        0.45,
    )

    best: Optional[Dict[str, object]] = None
    attempts = []
    for quantile in quantile_attempts:
        threshold = float(np.quantile(bottom_scores, quantile))
        face_indices = np.flatnonzero(bottom_scores >= threshold)
        if face_indices.size == 0:
            continue
        points = _support_sample_points(vertices, faces[face_indices], centroids[face_indices])
        mask, x_edges, z_edges = _rasterize_open_bottom_silhouette(reference_vertices, points, cfg)
        pixel_count = int(mask.sum())
        attempt = {
            "quantile": float(quantile),
            "threshold": threshold,
            "face_count": int(face_indices.size),
            "point_count": int(points.shape[0]),
            "pixel_count": pixel_count,
            "fill_fraction": float(mask.mean()),
        }
        attempts.append(attempt)
        if best is None or pixel_count > int(best["pixel_count"]):
            best = {
                **attempt,
                "face_indices": face_indices,
                "points": points,
                "mask": mask,
                "x_edges": x_edges,
                "z_edges": z_edges,
            }
        if pixel_count >= cfg.open_bottom_min_pixels:
            break

    if best is None:
        raise ValueError("Could not build an open bottom silhouette")

    points = np.asarray(best["points"], dtype=np.float32)
    mask = np.asarray(best["mask"], dtype=bool)
    x_edges = np.asarray(best["x_edges"], dtype=np.float32)
    z_edges = np.asarray(best["z_edges"], dtype=np.float32)
    outline_points = _outline_points_from_mask(mask, x_edges, z_edges, points, cfg)
    bounds_min, bounds_max, size, center = mesh_bounds(points)
    confidence = {
        "method": "lower_surface_projection_fill_holes",
        "attempts": attempts,
        "selected_quantile": best["quantile"],
        "selected_face_count": best["face_count"],
        "selected_point_count": best["point_count"],
        "footprint_pixel_count": int(mask.sum()),
        "footprint_fill_fraction": float(mask.mean()),
        "outline_sample_count": int(outline_points.shape[0]),
    }
    return LowerOpenBoundary(
        component_index=-1,
        vertex_indices=np.unique(faces[np.asarray(best["face_indices"], dtype=np.int64)].reshape(-1)).astype(np.int64),
        edge_indices=np.empty((0, 2), dtype=np.int64),
        points=points.astype(np.float32),
        sample_points=points.astype(np.float32),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        size=size.astype(np.float32),
        center=center.astype(np.float32),
        score=float(mask.sum()),
        confidence=confidence,
        source="open_bottom_silhouette",
        footprint_mask=mask.astype(bool),
        x_edges=x_edges.astype(np.float32),
        z_edges=z_edges.astype(np.float32),
        outline_points=outline_points.astype(np.float32),
    )


def detect_lower_open_boundary(
    open_mesh: MeshData,
    config: Optional[SupportFootprintConfig] = None,
    *,
    reference_mesh: Optional[MeshData] = None,
) -> LowerOpenBoundary:
    """Choose the lower cut boundary from an open/mSDF shoe mesh."""

    cfg = config or SupportFootprintConfig()
    components, edge_components = find_boundary_components(open_mesh)
    if not components:
        raise ValueError("Open mesh has no boundary components")

    vertices = np.asarray(open_mesh.vertices, dtype=np.float32)
    reference_vertices = vertices if reference_mesh is None else np.asarray(reference_mesh.vertices, dtype=np.float32)
    _, _, reference_size, _ = mesh_bounds(reference_vertices)
    length_extent = max(float(reference_size[cfg.shoe_length_axis]), 1e-8)
    width_extent = max(float(reference_size[cfg.shoe_width_axis]), 1e-8)
    up_extent = max(float(reference_size[cfg.shoe_up_axis]), 1e-8)

    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    signed_sole_all = sole_sign * vertices[:, cfg.shoe_up_axis]
    signed_sole_min = float(signed_sole_all.min())
    signed_sole_extent = max(float(signed_sole_all.max() - signed_sole_min), 1e-8)
    max_component_vertices = max(int(component.shape[0]) for component in components)
    best: Optional[Dict[str, object]] = None
    candidates: list[Dict[str, object]] = []

    for index, component in enumerate(components):
        points = vertices[component]
        bounds_min, bounds_max, size, center = mesh_bounds(points)
        length_ratio = float(size[cfg.shoe_length_axis] / length_extent)
        width_ratio = float(size[cfg.shoe_width_axis] / width_extent)
        up_ratio = float(size[cfg.shoe_up_axis] / up_extent)
        signed_center_sole = float(sole_sign * center[cfg.shoe_up_axis])
        sole_position = float((signed_center_sole - signed_sole_min) / signed_sole_extent)
        vertex_score = math.log1p(float(component.shape[0])) / max(math.log1p(float(max_component_vertices)), 1e-8)

        score = 1.25 * sole_position + 0.90 * length_ratio + 0.65 * width_ratio + 0.10 * vertex_score - 0.20 * up_ratio
        if component.shape[0] < cfg.boundary_min_vertices:
            score -= 1.0
        if length_ratio < cfg.boundary_min_length_ratio and width_ratio < cfg.boundary_min_width_ratio:
            score -= 1.0

        candidate = {
            "component_index": index,
            "vertex_indices": component,
            "edge_indices": edge_components[index],
            "vertex_count": int(component.shape[0]),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "size": size,
            "center": center,
            "score": float(score),
            "length_ratio": length_ratio,
            "width_ratio": width_ratio,
            "up_ratio": up_ratio,
            "sole_position": sole_position,
            "vertex_score": float(vertex_score),
        }
        candidates.append(candidate)
        if best is None or score > float(best["score"]):
            best = candidate

    if best is None:
        raise ValueError("Could not choose a lower open boundary")

    best_sole_position = float(best["sole_position"])
    selected_candidates = [
        candidate
        for candidate in candidates
        if int(candidate["vertex_count"]) >= cfg.boundary_min_vertices
        and float(candidate["sole_position"]) >= max(0.70, best_sole_position - 0.12)
        and (
            float(candidate["length_ratio"]) >= cfg.boundary_min_length_ratio
            or float(candidate["width_ratio"]) >= cfg.boundary_min_width_ratio
        )
    ]
    if not selected_candidates:
        selected_candidates = [best]
    elif int(best["component_index"]) not in {int(candidate["component_index"]) for candidate in selected_candidates}:
        selected_candidates.append(best)

    component = np.unique(
        np.concatenate([np.asarray(candidate["vertex_indices"], dtype=np.int64) for candidate in selected_candidates])
    )
    edge_arrays = [np.asarray(candidate["edge_indices"], dtype=np.int64) for candidate in selected_candidates]
    edge_indices = np.concatenate(edge_arrays, axis=0) if edge_arrays else np.empty((0, 2), dtype=np.int64)
    points = vertices[component]
    sample_points = _sample_boundary_edges(vertices, edge_indices, reference_vertices, cfg)
    if sample_points.size == 0:
        sample_points = points

    confidence = {
        "num_boundary_components": int(len(components)),
        "length_ratio": best["length_ratio"],
        "width_ratio": best["width_ratio"],
        "up_ratio": best["up_ratio"],
        "sole_position": best["sole_position"],
        "vertex_score": best["vertex_score"],
        "selected_component_indices": [int(candidate["component_index"]) for candidate in selected_candidates],
        "selected_component_count": int(len(selected_candidates)),
    }
    return LowerOpenBoundary(
        component_index=int(best["component_index"]),
        vertex_indices=component.astype(np.int64),
        edge_indices=edge_indices.astype(np.int64),
        points=points.astype(np.float32),
        sample_points=sample_points.astype(np.float32),
        bounds_min=np.asarray(best["bounds_min"], dtype=np.float32),
        bounds_max=np.asarray(best["bounds_max"], dtype=np.float32),
        size=np.asarray(best["size"], dtype=np.float32),
        center=np.asarray(best["center"], dtype=np.float32),
        score=float(best["score"]),
        confidence=confidence,
    )


def save_support_footprint_artifacts(
    mesh: MeshData,
    footprint: SupportFootprint,
    output_dir: str | Path,
    *,
    prefix: str = "",
    diagnostics: str = "full",
) -> Dict[str, str]:
    """Write summary JSON and standard visualizations for one footprint."""

    if diagnostics not in {"minimal", "full"}:
        raise ValueError("diagnostics must be 'minimal' or 'full'")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    paths = {
        "summary": output_path / f"{stem}support_footprint.json",
        "footprint": output_path / f"{stem}footprint_centerline.png",
        "width_profile": output_path / f"{stem}width_profile.png",
        "footbed_profile": output_path / f"{stem}footbed_profile.png",
        "floor_profile_v2": output_path / f"{stem}floor_profile_v2.png",
        "floor_samples_overlay": output_path / f"{stem}floor_samples_overlay.png",
        "floor_surface_v2": output_path / f"{stem}floor_surface_v2.png",
        "floor_diagnostic_v2": output_path / f"{stem}floor_diagnostic_v2.png",
        "pseudo_footbed_heightmap_npz": output_path / f"{stem}pseudo_footbed_heightmap.npz",
        "outer_floor_surface_obj": output_path / f"{stem}outer_floor_surface.obj",
        "pseudo_footbed_surface_obj": output_path / f"{stem}pseudo_footbed_surface.obj",
        "pseudo_footbed_smooth_surface_obj": output_path / f"{stem}pseudo_footbed_smooth_surface.obj",
        "pseudo_footbed_heightmap": output_path / f"{stem}pseudo_footbed_heightmap.png",
        "pseudo_footbed_cross_sections": output_path / f"{stem}pseudo_footbed_cross_sections.png",
        "pseudo_footbed_surface_preview": output_path / f"{stem}pseudo_footbed_surface_preview.png",
        "pseudo_footbed_smooth_surface_preview": output_path / f"{stem}pseudo_footbed_smooth_surface_preview.png",
        "support_overlay": output_path / f"{stem}support_faces_overlay.png",
    }
    footprint.save_json(paths["summary"])
    save_pseudo_footbed_heightmap_npz(footprint, paths["pseudo_footbed_heightmap_npz"])
    write_heightmap_surface_obj(footprint, paths["outer_floor_surface_obj"], surface="floor")
    write_heightmap_surface_obj(footprint, paths["pseudo_footbed_surface_obj"], surface="footbed")
    write_heightmap_surface_obj(footprint, paths["pseudo_footbed_smooth_surface_obj"], surface="smooth_footbed")
    plot_pseudo_footbed_surface_preview(footprint, paths["pseudo_footbed_surface_preview"])
    plot_pseudo_footbed_surface_preview(
        footprint,
        paths["pseudo_footbed_smooth_surface_preview"],
        surface="smooth_footbed",
    )
    written = {
        "summary": paths["summary"],
        "pseudo_footbed_heightmap_npz": paths["pseudo_footbed_heightmap_npz"],
        "outer_floor_surface_obj": paths["outer_floor_surface_obj"],
        "pseudo_footbed_surface_obj": paths["pseudo_footbed_surface_obj"],
        "pseudo_footbed_smooth_surface_obj": paths["pseudo_footbed_smooth_surface_obj"],
        "pseudo_footbed_surface_preview": paths["pseudo_footbed_surface_preview"],
        "pseudo_footbed_smooth_surface_preview": paths["pseudo_footbed_smooth_surface_preview"],
    }
    if diagnostics == "full":
        plot_footprint(footprint, paths["footprint"])
        plot_width_profile(footprint, paths["width_profile"])
        plot_footbed_profile(footprint, paths["footbed_profile"])
        plot_floor_profile_v2(footprint, paths["floor_profile_v2"])
        plot_floor_samples_overlay(footprint, paths["floor_samples_overlay"])
        plot_floor_surface_v2(footprint, paths["floor_surface_v2"])
        plot_floor_diagnostic_v2(footprint, paths["floor_diagnostic_v2"])
        plot_pseudo_footbed_heightmap(footprint, paths["pseudo_footbed_heightmap"])
        plot_pseudo_footbed_cross_sections(footprint, paths["pseudo_footbed_cross_sections"])
        plot_support_faces(mesh, footprint, paths["support_overlay"])
        written.update(paths)
    return {key: str(value) for key, value in written.items()}


def save_pseudo_footbed_heightmap_npz(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Save dense floor/footbed maps used by later fitting and mSDF stages."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_centers = 0.5 * (footprint.x_edges[:-1] + footprint.x_edges[1:])
    z_centers = 0.5 * (footprint.z_edges[:-1] + footprint.z_edges[1:])
    np.savez_compressed(
        output,
        x_edges=footprint.x_edges.astype(np.float32),
        z_edges=footprint.z_edges.astype(np.float32),
        x_centers=x_centers.astype(np.float32),
        z_centers=z_centers.astype(np.float32),
        footprint_mask=footprint.footprint_mask.astype(bool),
        footbed_mask=footprint.footbed_mask.astype(bool),
        raw_floor_heightmap=footprint.raw_floor_heightmap.astype(np.float32),
        floor_heightmap=footprint.floor_heightmap.astype(np.float32),
        footbed_heightmap=footprint.footbed_heightmap.astype(np.float32),
        smooth_footbed_heightmap=footprint.smooth_footbed_heightmap.astype(np.float32),
        heightmap_sample_count=footprint.heightmap_sample_count.astype(np.int32),
        centerline_x=footprint.centerline_x.astype(np.float32),
        centerline_z=footprint.centerline_z.astype(np.float32),
        left_boundary_z=footprint.left_boundary_z.astype(np.float32),
        right_boundary_z=footprint.right_boundary_z.astype(np.float32),
        support_axis_profile=footprint.support_axis_profile.astype(np.float32),
        footbed_axis_profile=footprint.footbed_axis_profile.astype(np.float32),
        smooth_footbed_axis_profile=footprint.smooth_footbed_axis_profile.astype(np.float32),
        footbed_offset=np.asarray([footprint.config.footbed_offset], dtype=np.float32),
        smooth_footbed_offset=np.asarray([footprint.smooth_footbed_offset], dtype=np.float32),
        smooth_footbed_source=np.asarray(footprint.smooth_footbed_source),
        smooth_footbed_height_fraction_from_bottom=np.asarray(
            [footprint.config.smooth_footbed_height_fraction_from_bottom],
            dtype=np.float32,
        ),
        config_json=np.asarray(json.dumps(asdict(footprint.config))),
    )


def write_heightmap_surface_obj(
    footprint: SupportFootprint,
    output_path: str | Path,
    *,
    surface: str,
) -> None:
    """Write a triangulated OBJ surface from either the floor or footbed map."""

    if surface == "floor":
        heightmap = footprint.floor_heightmap
    elif surface == "footbed":
        heightmap = footprint.footbed_heightmap
    elif surface == "smooth_footbed":
        heightmap = footprint.smooth_footbed_heightmap
    else:
        raise ValueError("surface must be 'floor', 'footbed', or 'smooth_footbed'")

    mask = footprint.footprint_mask & np.isfinite(heightmap)
    x_centers = 0.5 * (footprint.x_edges[:-1] + footprint.x_edges[1:])
    z_centers = 0.5 * (footprint.z_edges[:-1] + footprint.z_edges[1:])
    index_map = -np.ones(mask.shape, dtype=np.int64)
    rows, cols = np.nonzero(mask)
    vertices = np.zeros((rows.size, 3), dtype=np.float32)
    for vertex_index, (row, col) in enumerate(zip(rows, cols)):
        point = np.zeros((3,), dtype=np.float32)
        point[footprint.config.shoe_length_axis] = x_centers[row]
        point[footprint.config.shoe_up_axis] = heightmap[row, col]
        point[footprint.config.shoe_width_axis] = z_centers[col]
        vertices[vertex_index] = point
        index_map[row, col] = vertex_index

    faces = []
    for row in range(mask.shape[0] - 1):
        for col in range(mask.shape[1] - 1):
            a = index_map[row, col]
            b = index_map[row + 1, col]
            c = index_map[row + 1, col + 1]
            d = index_map[row, col + 1]
            if min(a, b, c, d) < 0:
                continue
            faces.append((a, b, c))
            faces.append((a, c, d))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write(f"# {surface} heightmap surface\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def plot_footprint(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot the 2D X-Z support footprint, centerline, and region spans."""

    import matplotlib.pyplot as plt

    x_edges = footprint.x_edges
    z_edges = footprint.z_edges
    extent = [float(z_edges[0]), float(z_edges[-1]), float(x_edges[0]), float(x_edges[-1])]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.imshow(
        footprint.footprint_mask,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Greys",
        alpha=0.88,
    )
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    try:
        ax.contour(
            z_centers,
            x_centers,
            footprint.footprint_mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.8,
        )
    except Exception:
        pass
    ax.plot(footprint.centerline_z, footprint.centerline_x, color="#d7191c", linewidth=2.0, label="centerline")
    ax.plot(footprint.left_boundary_z, footprint.centerline_x, color="#2c7bb6", linewidth=1.0, label="left/right boundary")
    ax.plot(footprint.right_boundary_z, footprint.centerline_x, color="#2c7bb6", linewidth=1.0)
    if footprint.lower_boundary is not None and footprint.lower_boundary.outline_points is not None:
        ax.scatter(
            footprint.lower_boundary.outline_points[:, footprint.config.shoe_width_axis],
            footprint.lower_boundary.outline_points[:, footprint.config.shoe_length_axis],
            s=2,
            color="#31a354",
            alpha=0.35,
            linewidth=0.0,
            label="open bottom outline samples",
        )
    _shade_regions(ax, footprint)
    ax.scatter(
        [footprint.bbox_center[footprint.config.shoe_width_axis]],
        [footprint.bbox_center[footprint.config.shoe_length_axis]],
        s=42,
        color="#fdae61",
        edgecolor="black",
        linewidth=0.5,
        label="bbox center",
        zorder=4,
    )
    ax.scatter(
        [footprint.suggested_width_anchor],
        [footprint.suggested_length_anchor],
        s=48,
        color="#1a9641",
        edgecolor="black",
        linewidth=0.5,
        label="support anchor",
        zorder=5,
    )
    ax.set_xlabel("width axis Z")
    ax.set_ylabel("length axis X")
    ax.set_title("Support Footprint and Centerline")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_width_profile(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot support width as a function of shoe length."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(footprint.centerline_x, footprint.width_profile, color="#2c7bb6", linewidth=2.0)
    _shade_regions(ax, footprint, vertical=False)
    ax.set_xlabel("length axis X")
    ax.set_ylabel("support width along Z")
    ax.set_title("Support Width Profile")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_footbed_profile(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot the robust floor curve and offset pseudo-footbed profile."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(
        footprint.centerline_x,
        footprint.raw_floor_axis_profile,
        color="#8c8c8c",
        linewidth=1.2,
        linestyle="--",
        alpha=0.80,
        label="raw floor samples",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.support_axis_profile,
        color="#2c7bb6",
        linewidth=2.0,
        label="smooth floor",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.footbed_axis_profile,
        color="#1a9641",
        linewidth=2.0,
        label=f"pseudo-footbed offset={footprint.config.footbed_offset:.4f}",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.support_face_axis_profile,
        color="#7b3294",
        linewidth=1.0,
        alpha=0.45,
        label="old red-face profile",
    )
    _shade_regions(ax, footprint, vertical=False)
    ax.set_xlabel("length axis X")
    ax.set_ylabel(_up_axis_label(footprint.config))
    ax.set_title("Robust Floor and Pseudo-Footbed Profile")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_floor_profile_v2(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot the V2 floor curve and sample counts in separate panels."""

    import matplotlib.pyplot as plt

    fig, (ax, ax_counts) = plt.subplots(
        2,
        1,
        figsize=(9.0, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        constrained_layout=True,
    )
    ax.plot(
        footprint.centerline_x,
        footprint.raw_floor_axis_profile,
        color="#8c8c8c",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label="raw robust floor",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.support_axis_profile,
        color="#2c7bb6",
        linewidth=2.2,
        label="smooth floor",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.footbed_axis_profile,
        color="#1a9641",
        linewidth=2.0,
        label="pseudo-footbed",
    )
    ax.plot(
        footprint.centerline_x,
        footprint.support_face_axis_profile,
        color="#7b3294",
        linewidth=0.9,
        alpha=0.35,
        label="old red-face profile",
    )
    _shade_regions(ax, footprint, vertical=False, include_labels=False)
    ax.set_ylabel(_up_axis_label(footprint.config))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, fontsize=8, frameon=True)
    ax.set_title("Floor Profile V2")

    ax_counts.bar(
        footprint.centerline_x,
        footprint.floor_sample_count_profile,
        width=_median_step(footprint.centerline_x),
        color="#fdae61",
        alpha=0.35,
        edgecolor="none",
        label="sample count",
    )
    _shade_regions(ax_counts, footprint, vertical=False, include_labels=False)
    ax_counts.set_xlabel("length axis X")
    ax_counts.set_ylabel("samples per X slice")
    ax_counts.grid(alpha=0.20)
    _save_figure(fig, output_path)


def plot_floor_samples_overlay(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot floor-sample density inside the footprint without overplotting points."""

    import matplotlib.pyplot as plt

    x_edges = footprint.x_edges
    z_edges = footprint.z_edges
    extent = [float(z_edges[0]), float(z_edges[-1]), float(x_edges[0]), float(x_edges[-1])]
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    ax.imshow(
        footprint.footprint_mask,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Greys",
        alpha=0.18,
    )

    samples = footprint.floor_sample_points
    if samples.size:
        hist, _, _ = np.histogram2d(
            samples[:, footprint.config.shoe_length_axis],
            samples[:, footprint.config.shoe_width_axis],
            bins=(x_edges, z_edges),
        )
        density = np.ma.masked_where(~footprint.footprint_mask | (hist <= 0), np.log1p(hist))
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        image = ax.imshow(
            density,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            alpha=0.82,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="log(1 + floor samples)")

    _draw_footprint_guides(ax, footprint, show_width=True)
    _shade_regions(ax, footprint, alpha=0.08, include_labels=False)
    ax.set_xlabel("width axis Z")
    ax.set_ylabel("length axis X")
    ax.set_title("Floor Sample Density Inside Footprint")
    _save_figure(fig, output_path)


def plot_floor_surface_v2(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot the estimated smooth floor surface inside the black footprint."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    extent = _footprint_extent(footprint)
    floor_grid = footprint.floor_heightmap
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    image = ax.imshow(
        np.ma.masked_invalid(floor_grid),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=_up_axis_label(footprint.config))
    _draw_footprint_guides(ax, footprint, show_width=True)
    _shade_regions(ax, footprint, alpha=0.08, include_labels=False)
    ax.set_xlabel("width axis Z")
    ax.set_ylabel("length axis X")
    ax.set_title("Smooth Floor Height Map V2")
    _save_figure(fig, output_path)


def plot_floor_diagnostic_v2(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Write a compact four-panel diagnostic with minimal overlap."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    ax_footprint, ax_surface, ax_curve, ax_counts = axes.reshape(-1)

    extent = _footprint_extent(footprint)
    ax_footprint.imshow(
        footprint.footprint_mask,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Greys",
        alpha=0.70,
    )
    _draw_footprint_guides(ax_footprint, footprint, show_width=True)
    _shade_regions(ax_footprint, footprint, alpha=0.08, include_labels=False)
    ax_footprint.set_title("1. Filled Bottom Footprint")
    ax_footprint.set_xlabel("width axis Z")
    ax_footprint.set_ylabel("length axis X")

    floor_grid = footprint.floor_heightmap
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    image = ax_surface.imshow(
        np.ma.masked_invalid(floor_grid),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )
    _draw_footprint_guides(ax_surface, footprint, show_width=False)
    ax_surface.set_title("2. Smooth Floor Height Map")
    ax_surface.set_xlabel("width axis Z")
    ax_surface.set_ylabel("length axis X")
    fig.colorbar(image, ax=ax_surface, fraction=0.046, pad=0.04, label=_up_axis_label(footprint.config))

    ax_curve.plot(
        footprint.centerline_x,
        footprint.raw_floor_axis_profile,
        color="#8c8c8c",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label="raw floor",
    )
    ax_curve.plot(footprint.centerline_x, footprint.support_axis_profile, color="#2c7bb6", linewidth=2.2, label="smooth floor")
    ax_curve.plot(footprint.centerline_x, footprint.footbed_axis_profile, color="#1a9641", linewidth=2.0, label="pseudo-footbed")
    _shade_regions(ax_curve, footprint, vertical=False, alpha=0.08, include_labels=False)
    ax_curve.grid(alpha=0.25)
    ax_curve.set_title("3. Floor Curve")
    ax_curve.set_xlabel("length axis X")
    ax_curve.set_ylabel(_up_axis_label(footprint.config))
    ax_curve.legend(loc="best", fontsize=8)

    ax_counts.bar(
        footprint.centerline_x,
        footprint.floor_sample_count_profile,
        width=_median_step(footprint.centerline_x),
        color="#fdae61",
        alpha=0.55,
        edgecolor="none",
    )
    _shade_regions(ax_counts, footprint, vertical=False, alpha=0.08, include_labels=False)
    ax_counts.grid(alpha=0.20)
    ax_counts.set_title("4. Samples Per X Slice")
    ax_counts.set_xlabel("length axis X")
    ax_counts.set_ylabel("sample count")

    _save_figure(fig, output_path)


def plot_pseudo_footbed_heightmap(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Plot the real X-Z pseudo-footbed map."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True)
    extent = _footprint_extent(footprint)
    maps = [
        ("Outer floor height map", footprint.floor_heightmap, "viridis"),
        ("Detailed pseudo-footbed", footprint.footbed_heightmap, "YlGnBu"),
        ("Smooth pseudo-footbed", footprint.smooth_footbed_heightmap, "YlGnBu"),
    ]
    for ax, (title, values, cmap_name) in zip(axes, maps):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        image = ax.imshow(
            np.ma.masked_invalid(values),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        _draw_footprint_guides(ax, footprint, show_width=True)
        _shade_regions(ax, footprint, alpha=0.08, include_labels=False)
        ax.set_xlabel("width axis Z")
        ax.set_ylabel("length axis X")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=_up_axis_label(footprint.config))
    _save_figure(fig, output_path)


def plot_pseudo_footbed_cross_sections(footprint: SupportFootprint, output_path: str | Path) -> None:
    """Show X and Z slices through the estimated pseudo-footbed surface."""

    import matplotlib.pyplot as plt

    x_centers = 0.5 * (footprint.x_edges[:-1] + footprint.x_edges[1:])
    z_centers = 0.5 * (footprint.z_edges[:-1] + footprint.z_edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    ax_long, ax_counts, ax_z1, ax_z2 = axes.reshape(-1)

    center_z = np.interp(x_centers, footprint.centerline_x, footprint.centerline_z)
    center_cols = np.searchsorted(footprint.z_edges, center_z, side="right") - 1
    center_cols = np.clip(center_cols, 0, footprint.footprint_mask.shape[1] - 1)
    floor_center = footprint.floor_heightmap[np.arange(x_centers.size), center_cols]
    footbed_center = footprint.footbed_heightmap[np.arange(x_centers.size), center_cols]
    smooth_center = footprint.smooth_footbed_heightmap[np.arange(x_centers.size), center_cols]

    ax_long.plot(x_centers, floor_center, color="#2c7bb6", linewidth=2.0, label="floor at centerline")
    ax_long.plot(x_centers, footbed_center, color="#1a9641", linewidth=1.0, alpha=0.35, label="detailed footbed")
    ax_long.plot(x_centers, smooth_center, color="#006d2c", linewidth=2.3, label="smooth footbed")
    ax_long.plot(footprint.centerline_x, footprint.support_axis_profile, color="#8c8c8c", linewidth=1.0, linestyle="--", label="old F(x)")
    ax_long.plot(footprint.centerline_x, footprint.footbed_axis_profile, color="#74c476", linewidth=1.0, linestyle="--", label="old B(x)")
    _shade_regions(ax_long, footprint, vertical=False, alpha=0.08, include_labels=False)
    ax_long.set_xlabel("length axis X")
    ax_long.set_ylabel(_up_axis_label(footprint.config))
    ax_long.set_title("Centerline height section")
    ax_long.grid(alpha=0.25)
    ax_long.legend(fontsize=8, loc="best")

    valid_counts = np.where(footprint.footprint_mask, footprint.heightmap_sample_count, 0)
    ax_counts.plot(x_centers, valid_counts.sum(axis=1), color="#fdae61", linewidth=1.8)
    _shade_regions(ax_counts, footprint, vertical=False, alpha=0.08, include_labels=False)
    ax_counts.set_xlabel("length axis X")
    ax_counts.set_ylabel("height samples")
    ax_counts.set_title("Samples per X row")
    ax_counts.grid(alpha=0.25)

    cross_fracs = [0.30, 0.70]
    for ax, frac in zip([ax_z1, ax_z2], cross_fracs):
        x_value = float(footprint.centerline_x.min() + frac * (footprint.centerline_x.max() - footprint.centerline_x.min()))
        row = int(np.clip(np.searchsorted(footprint.x_edges, x_value, side="right") - 1, 0, footprint.footprint_mask.shape[0] - 1))
        row_mask = footprint.footprint_mask[row] & np.isfinite(footprint.floor_heightmap[row])
        ax.plot(z_centers[row_mask], footprint.floor_heightmap[row, row_mask], color="#2c7bb6", linewidth=2.0, label="floor")
        footbed_row_mask = footprint.footbed_mask[row] & np.isfinite(footprint.footbed_heightmap[row])
        ax.plot(z_centers[footbed_row_mask], footprint.footbed_heightmap[row, footbed_row_mask], color="#1a9641", linewidth=1.0, alpha=0.35, label="detailed footbed")
        ax.plot(z_centers[footbed_row_mask], footprint.smooth_footbed_heightmap[row, footbed_row_mask], color="#006d2c", linewidth=2.2, label="smooth footbed")
        left = np.interp(x_value, footprint.centerline_x, footprint.left_boundary_z)
        right = np.interp(x_value, footprint.centerline_x, footprint.right_boundary_z)
        ax.axvspan(left, right, color="#f0f0f0", alpha=0.35)
        ax.set_xlabel("width axis Z")
        ax.set_ylabel(_up_axis_label(footprint.config))
        ax.set_title(f"Width section at X fraction {frac:.0%}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    _save_figure(fig, output_path)


def plot_pseudo_footbed_surface_preview(
    footprint: SupportFootprint,
    output_path: str | Path,
    *,
    surface: str = "footbed",
) -> None:
    """Write a lightweight 3D preview of the pseudo-footbed surface."""

    import matplotlib.pyplot as plt

    if surface == "smooth_footbed":
        heightmap = footprint.smooth_footbed_heightmap
        title = "Smooth Pseudo-Footbed Surface Preview"
    elif surface == "footbed":
        heightmap = footprint.footbed_heightmap
        title = "Pseudo-Footbed Surface Preview"
    else:
        raise ValueError("surface must be 'footbed' or 'smooth_footbed'")

    x_centers = 0.5 * (footprint.x_edges[:-1] + footprint.x_edges[1:])
    z_centers = 0.5 * (footprint.z_edges[:-1] + footprint.z_edges[1:])
    stride = max(1, int(math.ceil(max(heightmap.shape) / 90)))
    xs = x_centers[::stride]
    zs = z_centers[::stride]
    zz, xx = np.meshgrid(zs, xs)
    yy = heightmap[::stride, ::stride]
    yy = np.ma.masked_invalid(yy)

    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="YlGnBu", linewidth=0.0, antialiased=True, alpha=0.92)
    ax.set_xlabel("X length")
    ax.set_ylabel(_up_axis_label(footprint.config))
    ax.set_zlabel("Z width")
    ax.set_title(title)
    try:
        ax.set_box_aspect((1.6, 0.45, 1.0))
    except Exception:
        pass
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_support_faces(
    mesh: MeshData,
    footprint: SupportFootprint,
    output_path: str | Path,
    *,
    max_faces: int = 8000,
) -> None:
    """Plot a 3D mesh view with selected support faces highlighted."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_indices = np.arange(faces.shape[0])
    if faces.shape[0] > max_faces:
        stride = int(math.ceil(faces.shape[0] / max_faces))
        face_indices = face_indices[::stride]

    support_set = set(map(int, footprint.support_face_indices.tolist()))
    support_indices = np.asarray([idx for idx in face_indices if int(idx) in support_set], dtype=np.int64)
    background_indices = np.asarray([idx for idx in face_indices if int(idx) not in support_set], dtype=np.int64)

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    if background_indices.size:
        background = Poly3DCollection(
            vertices[faces[background_indices]],
            facecolors="#bdbdbd",
            edgecolors="none",
            alpha=0.20,
        )
        ax.add_collection3d(background)
    if support_indices.size:
        support = Poly3DCollection(
            vertices[faces[support_indices]],
            facecolors="#d7191c",
            edgecolors="#7f0000",
            linewidths=0.02,
            alpha=0.92,
        )
        ax.add_collection3d(support)
    if footprint.lower_boundary is not None:
        boundary = (
            footprint.lower_boundary.outline_points
            if footprint.lower_boundary.outline_points is not None
            else footprint.lower_boundary.sample_points
        )
        ax.scatter(
            boundary[:, 0],
            boundary[:, 1],
            boundary[:, 2],
            s=4,
            color="black",
            alpha=0.75,
            depthshade=False,
        )

    _set_axes_equal_3d(ax, vertices)
    ax.set_xlabel("X length")
    ax.set_ylabel("Y up/bottom")
    ax.set_zlabel("Z width")
    ax.set_title("Detected Support Faces")
    fig.tight_layout()
    _save_figure(fig, output_path)


def _axis_sign(sign: float) -> float:
    return 1.0 if sign >= 0.0 else -1.0


def _up_axis_label(cfg: SupportFootprintConfig) -> str:
    if cfg.shoe_up_axis == 1 and cfg.shoe_up_sign < 0.0:
        return "Y coordinate (+Y bottom, -Y opening)"
    return f"axis {cfg.shoe_up_axis} coordinate"


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    norm = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross, dtype=np.float32)
    valid = norm > 1e-12
    normals[valid] = cross[valid] / norm[valid, None]
    areas = 0.5 * norm
    return centroids.astype(np.float32), normals.astype(np.float32), areas.astype(np.float32)


def _select_support_faces(
    centroids: np.ndarray,
    normals: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    bottom_scores = sole_sign * centroids[:, cfg.shoe_up_axis]
    normal_scores = sole_sign * normals[:, cfg.shoe_up_axis]
    attempts = [
        (cfg.bottom_quantile, cfg.normal_angle_degrees, True),
        (cfg.fallback_bottom_quantile, cfg.fallback_normal_angle_degrees, True),
        (cfg.fallback_bottom_quantile, 180.0, False),
    ]

    best_indices: Optional[np.ndarray] = None
    best_info: Dict[str, object] = {}
    for quantile, angle_degrees, use_normals in attempts:
        threshold = float(np.quantile(bottom_scores, quantile))
        near_bottom = bottom_scores >= threshold
        if use_normals:
            normal_threshold = math.cos(math.radians(angle_degrees))
            normal_ok = normal_scores >= normal_threshold
        else:
            normal_threshold = -1.0
            normal_ok = np.ones_like(near_bottom, dtype=bool)
        indices = np.flatnonzero(near_bottom & normal_ok)
        best_indices = indices
        best_info = {
            "bottom_quantile_used": float(quantile),
            "bottom_score_threshold": threshold,
            "normal_angle_degrees_used": float(angle_degrees),
            "normal_threshold_used": float(normal_threshold),
            "normal_filter_used": bool(use_normals),
            "support_face_count": int(indices.size),
        }
        if indices.size >= cfg.min_support_faces:
            return indices, best_info

    if best_indices is None or best_indices.size == 0:
        raise ValueError("No support faces found")
    return best_indices, best_info


def _apply_open_bottom_filter(
    support_indices: np.ndarray,
    centroids: np.ndarray,
    reference_vertices: np.ndarray,
    cfg: SupportFootprintConfig,
    lower_boundary: LowerOpenBoundary,
) -> Tuple[np.ndarray, Dict[str, object]]:
    def filter_indices(indices: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        if indices.size == 0:
            return indices, {
                "height_keep_count": 0,
                "outline_keep_count": 0,
                "combined_keep_count": 0,
            }
        points = centroids[indices]
        height_keep = _not_below_lower_boundary(points, lower_boundary, reference_vertices, cfg)
        if lower_boundary.footprint_mask is not None:
            outline_keep = _inside_open_bottom_footprint(points, lower_boundary, cfg)
        else:
            outline_keep = np.ones_like(height_keep, dtype=bool)
        combined_keep = height_keep & outline_keep
        return indices[combined_keep], {
            "height_keep_count": int(height_keep.sum()),
            "outline_keep_count": int(outline_keep.sum()),
            "combined_keep_count": int(combined_keep.sum()),
        }

    filtered_indices, initial_counts = filter_indices(support_indices)
    broad_indices = _select_broad_support_faces(centroids, cfg)
    broad_filtered_indices, broad_counts = filter_indices(broad_indices)

    low_count_floor = max(8, cfg.min_support_faces // 3)
    selected = support_indices
    selected_source = "unfiltered_initial_support"
    kept = False
    low_count = False
    if filtered_indices.size >= cfg.min_support_faces:
        selected = filtered_indices
        selected_source = "initial_support_filtered_by_open_bottom"
        kept = True
    elif broad_filtered_indices.size >= cfg.min_support_faces:
        selected = broad_filtered_indices
        selected_source = "broad_support_filtered_by_open_bottom"
        kept = True
    elif broad_filtered_indices.size >= low_count_floor:
        selected = broad_filtered_indices
        selected_source = "broad_support_filtered_by_open_bottom_low_count"
        kept = True
        low_count = True
    elif filtered_indices.size >= low_count_floor:
        selected = filtered_indices
        selected_source = "initial_support_filtered_by_open_bottom_low_count"
        kept = True
        low_count = True

    info: Dict[str, object] = {
        "open_boundary_filter_used": True,
        "open_bottom_source": lower_boundary.source,
        "support_face_count_before_boundary_filter": int(support_indices.size),
        "support_face_count_after_boundary_filter": int(filtered_indices.size),
        "broad_support_face_count_before_boundary_filter": int(broad_indices.size),
        "broad_support_face_count_after_boundary_filter": int(broad_filtered_indices.size),
        "open_boundary_filter_kept": bool(kept),
        "open_boundary_filter_low_count": bool(low_count),
        "open_boundary_filter_selected_source": selected_source,
        "initial_filter_counts": initial_counts,
        "broad_filter_counts": broad_counts,
    }
    return selected.astype(np.int64), info


def _select_broad_support_faces(
    centroids: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    bottom_scores = sole_sign * centroids[:, cfg.shoe_up_axis]
    threshold = float(np.quantile(bottom_scores, cfg.broad_support_quantile))
    return np.flatnonzero(bottom_scores >= threshold).astype(np.int64)


def _estimate_floor_profile_from_footprint(
    vertices: np.ndarray,
    faces: np.ndarray,
    centroids: np.ndarray,
    centerline_x: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    footprint_mask: np.ndarray,
    lower_boundary: Optional[LowerOpenBoundary],
    cfg: SupportFootprintConfig,
    *,
    fallback_profile: np.ndarray,
) -> Dict[str, object]:
    """Estimate a smooth floor curve from watertight samples inside the footprint."""

    all_samples = _mesh_sample_points(vertices, faces, centroids)
    inside = _inside_footprint_mask(all_samples, footprint_mask, x_edges, z_edges, cfg, dilation_iterations=1)
    keep = inside
    if lower_boundary is not None:
        keep = keep & _not_below_lower_boundary(all_samples, lower_boundary, vertices, cfg)

    floor_samples = all_samples[keep]
    raw_profile, counts, valid = _raw_axis_profile_from_points(
        floor_samples,
        centerline_x,
        x_edges,
        cfg,
        quantile=cfg.floor_axis_quantile,
        min_samples=cfg.floor_min_samples_per_slice,
    )

    source = "watertight_samples_in_open_bottom_footprint"
    if not np.any(valid):
        raw_profile = np.asarray(fallback_profile, dtype=np.float32)
        counts = np.zeros(centerline_x.shape, dtype=np.int32)
        valid = np.isfinite(raw_profile)
        source = "fallback_support_faces_no_floor_samples"
    else:
        raw_profile = _fill_missing_profile(raw_profile, centerline_x, fallback_profile)

    smooth_profile = _smooth_1d(raw_profile, cfg.floor_smooth_window)
    if lower_boundary is not None:
        smooth_profile = _clamp_profile_to_open_bottom(smooth_profile, centerline_x, x_edges, lower_boundary, cfg)

    heightmap_estimate = _estimate_floor_heightmap_from_points(
        floor_samples,
        footprint_mask,
        x_edges,
        z_edges,
        centerline_x,
        smooth_profile,
        cfg,
    )

    valid_counts = counts[counts > 0]
    confidence = {
        "source": source,
        "sample_count_total": int(floor_samples.shape[0]),
        "inside_footprint_sample_count": int(inside.sum()),
        "kept_after_open_bottom_filter_count": int(keep.sum()),
        "valid_slice_count": int(valid.sum()),
        "profile_point_count": int(centerline_x.size),
        "min_samples_per_valid_slice": int(valid_counts.min()) if valid_counts.size else 0,
        "median_samples_per_valid_slice": float(np.median(valid_counts)) if valid_counts.size else 0.0,
        "max_samples_per_valid_slice": int(valid_counts.max()) if valid_counts.size else 0,
        "floor_axis_quantile": float(cfg.floor_axis_quantile),
        "floor_smooth_window": int(cfg.floor_smooth_window),
        "used_open_bottom_height_filter": bool(lower_boundary is not None),
    }
    return {
        "raw_profile": raw_profile.astype(np.float32),
        "smooth_profile": smooth_profile.astype(np.float32),
        "sample_counts": counts.astype(np.int32),
        "sample_points": floor_samples.astype(np.float32),
        "raw_heightmap": heightmap_estimate["raw_heightmap"].astype(np.float32),
        "smooth_heightmap": heightmap_estimate["smooth_heightmap"].astype(np.float32),
        "heightmap_sample_counts": heightmap_estimate["sample_counts"].astype(np.int32),
        "heightmap_confidence": heightmap_estimate["confidence"],
        "source": source,
        "confidence": confidence,
    }


def _mesh_sample_points(vertices: np.ndarray, faces: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    edge_midpoints = np.stack(
        [
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ],
        axis=1,
    )
    samples = np.concatenate(
        [
            vertices,
            centroids,
            edge_midpoints.reshape(-1, 3),
        ],
        axis=0,
    )
    return samples.astype(np.float32)


def _raw_axis_profile_from_points(
    points: np.ndarray,
    centerline_x: np.ndarray,
    x_edges: np.ndarray,
    cfg: SupportFootprintConfig,
    *,
    quantile: float,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = np.full(centerline_x.shape, np.nan, dtype=np.float32)
    counts = np.zeros(centerline_x.shape, dtype=np.int32)
    valid = np.zeros(centerline_x.shape, dtype=bool)
    if points.size == 0:
        return profile, counts, valid

    x_values = points[:, cfg.shoe_length_axis]
    axis_values = points[:, cfg.shoe_up_axis]
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    signed_axis_values = sole_sign * axis_values
    for i, x in enumerate(centerline_x):
        edge_idx = np.searchsorted(x_edges, x, side="right") - 1
        edge_idx = int(np.clip(edge_idx, 0, len(x_edges) - 2))
        x0 = x_edges[edge_idx]
        x1 = x_edges[edge_idx + 1]
        in_slice = (x_values >= x0) & (x_values <= x1)
        count = int(in_slice.sum())
        counts[i] = count
        if count >= min_samples:
            signed_coord = float(np.quantile(signed_axis_values[in_slice], quantile))
            profile[i] = signed_coord * sole_sign
            valid[i] = True
    return profile, counts, valid


def _estimate_floor_heightmap_from_points(
    points: np.ndarray,
    footprint_mask: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    centerline_x: np.ndarray,
    fallback_profile: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Dict[str, object]:
    """Estimate a real X-Z floor map from watertight samples inside the footprint."""

    raw_heightmap, sample_counts = _raw_heightmap_from_points(points, footprint_mask, x_edges, z_edges, cfg)
    fallback_grid = _profile_grid_from_arrays(
        footprint_mask,
        x_edges,
        centerline_x,
        fallback_profile,
    )
    smooth_heightmap, fill_source = _fill_and_smooth_heightmap(raw_heightmap, fallback_grid, footprint_mask, cfg)

    valid_cells = np.isfinite(raw_heightmap) & footprint_mask
    inside_values = smooth_heightmap[footprint_mask & np.isfinite(smooth_heightmap)]
    valid_counts = sample_counts[valid_cells]
    confidence = {
        "source": "watertight_samples_in_open_bottom_footprint",
        "fill_source": fill_source,
        "shape": [int(v) for v in raw_heightmap.shape],
        "inside_cell_count": int(footprint_mask.sum()),
        "valid_cell_count": int(valid_cells.sum()),
        "valid_cell_fraction": float(valid_cells.sum() / max(int(footprint_mask.sum()), 1)),
        "min_samples_per_valid_cell": int(valid_counts.min()) if valid_counts.size else 0,
        "median_samples_per_valid_cell": float(np.median(valid_counts)) if valid_counts.size else 0.0,
        "max_samples_per_valid_cell": int(valid_counts.max()) if valid_counts.size else 0,
        "smooth_sigma": float(cfg.heightmap_smooth_sigma),
        "profile_clip": float(cfg.heightmap_profile_clip),
        "floor_y_min": float(inside_values.min()) if inside_values.size else None,
        "floor_y_max": float(inside_values.max()) if inside_values.size else None,
    }
    return {
        "raw_heightmap": raw_heightmap.astype(np.float32),
        "smooth_heightmap": smooth_heightmap.astype(np.float32),
        "sample_counts": sample_counts.astype(np.int32),
        "confidence": confidence,
    }


def _raw_heightmap_from_points(
    points: np.ndarray,
    footprint_mask: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(footprint_mask, dtype=bool)
    raw = np.full(mask.shape, np.nan, dtype=np.float32)
    counts = np.zeros(mask.shape, dtype=np.int32)
    if points.size == 0:
        return raw, counts

    x_values = points[:, cfg.shoe_length_axis]
    z_values = points[:, cfg.shoe_width_axis]
    ix = np.searchsorted(x_edges, x_values, side="right") - 1
    iz = np.searchsorted(z_edges, z_values, side="right") - 1
    valid = (ix >= 0) & (ix < mask.shape[0]) & (iz >= 0) & (iz < mask.shape[1])
    if not np.any(valid):
        return raw, counts
    ix = ix[valid].astype(np.int64)
    iz = iz[valid].astype(np.int64)
    inside = mask[ix, iz]
    if not np.any(inside):
        return raw, counts
    ix = ix[inside]
    iz = iz[inside]
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    signed_axis_values = sole_sign * points[valid][inside, cfg.shoe_up_axis]
    flat = ix * mask.shape[1] + iz

    order = np.argsort(flat)
    flat = flat[order]
    signed_axis_values = signed_axis_values[order]
    unique_flat, starts, cell_counts = np.unique(flat, return_index=True, return_counts=True)
    for cell, start, count in zip(unique_flat, starts, cell_counts):
        row = int(cell // mask.shape[1])
        col = int(cell % mask.shape[1])
        counts[row, col] = int(count)
        if count < cfg.heightmap_min_samples_per_cell:
            continue
        values = signed_axis_values[start : start + count]
        signed_coord = float(np.quantile(values, cfg.floor_axis_quantile))
        raw[row, col] = signed_coord * sole_sign
    return raw, counts


def _fill_and_smooth_heightmap(
    raw_heightmap: np.ndarray,
    fallback_grid: np.ndarray,
    footprint_mask: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, str]:
    mask = np.asarray(footprint_mask, dtype=bool)
    fallback = np.asarray(fallback_grid, dtype=np.float32)
    raw = np.asarray(raw_heightmap, dtype=np.float32)
    valid = mask & np.isfinite(raw)
    filled = np.asarray(fallback, dtype=np.float32).copy()
    fill_source = "fallback_profile"

    if np.any(valid):
        filled[valid] = raw[valid]
        if ndimage is not None:
            nearest_indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            nearest = raw[tuple(nearest_indices)]
            missing_inside = mask & ~valid
            filled[missing_inside] = nearest[missing_inside]
            fill_source = "nearest_valid_cell"
        else:
            fill_source = "fallback_profile_no_scipy"

    filled[~mask] = np.nan
    smoothed = filled.copy()
    if ndimage is not None and cfg.heightmap_smooth_sigma > 0.0 and np.any(mask):
        weights = mask.astype(np.float32)
        values = np.nan_to_num(filled, nan=0.0).astype(np.float32)
        sigma = float(cfg.heightmap_smooth_sigma)
        smooth_values = ndimage.gaussian_filter(values * weights, sigma=sigma, mode="nearest")
        smooth_weights = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")
        smoothed = smooth_values / np.maximum(smooth_weights, 1e-8)

    if cfg.heightmap_profile_clip > 0.0:
        sole_sign = -_axis_sign(cfg.shoe_up_sign)
        signed_smooth = sole_sign * smoothed
        signed_fallback = sole_sign * fallback
        signed_smooth = np.clip(
            signed_smooth,
            signed_fallback - cfg.heightmap_profile_clip,
            signed_fallback + cfg.heightmap_profile_clip,
        )
        smoothed = signed_smooth * sole_sign

    smoothed = smoothed.astype(np.float32)
    smoothed[~mask] = np.nan
    return smoothed, fill_source


def _smooth_footbed_profile_from_heightmap(
    footbed_heightmap: np.ndarray,
    footbed_mask: np.ndarray,
    x_edges: np.ndarray,
    centerline_x: np.ndarray,
    fallback_profile: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    """Collapse the noisy footbed map into a stable insole-like length curve."""

    mask = np.asarray(footbed_mask, dtype=bool)
    values = np.asarray(footbed_heightmap, dtype=np.float32)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    profile = np.full((mask.shape[0],), np.nan, dtype=np.float32)
    for row in range(mask.shape[0]):
        row_values = values[row, mask[row]]
        row_values = row_values[np.isfinite(row_values)]
        if row_values.size:
            profile[row] = float(np.quantile(row_values, cfg.smooth_footbed_quantile))

    fallback = np.interp(x_centers, centerline_x, fallback_profile).astype(np.float32)
    valid = np.isfinite(profile)
    if np.any(valid):
        profile[~valid] = np.interp(x_centers[~valid], x_centers[valid], profile[valid])
    else:
        profile[:] = fallback

    window = int(round(max(0.0, cfg.smooth_footbed_window_fraction) * profile.size))
    if window % 2 == 0:
        window += 1
    window = max(window, cfg.floor_smooth_window, 3)
    smooth = _smooth_1d(profile, window)
    smooth = _smooth_1d(smooth, window)

    # Keep the smooth sheet near the original robust curve, but avoid the sharp
    # local spikes that made the OBJ look like torn fabric.
    clip = max(cfg.heightmap_profile_clip * 1.5, 1e-6)
    smooth = np.clip(smooth, fallback - clip, fallback + clip)
    return np.interp(centerline_x, x_centers, smooth).astype(np.float32)


def _smooth_footbed_profile_from_open_boundary(
    lower_boundary: Optional[LowerOpenBoundary],
    vertices: np.ndarray,
    centerline_x: np.ndarray,
    x_edges: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, float, str]:
    """Place the smooth footbed from the open-mesh lower boundary plus offset."""

    _, _, size, _ = mesh_bounds(vertices)
    support_length = max(float(size[cfg.shoe_length_axis]), 1e-8)
    if cfg.open_boundary_footbed_offset is None:
        offset = cfg.open_boundary_footbed_offset_ratio * support_length
        offset = float(np.clip(offset, cfg.open_boundary_footbed_offset_min, cfg.open_boundary_footbed_offset_max))
    else:
        offset = float(cfg.open_boundary_footbed_offset)

    if lower_boundary is not None and lower_boundary.sample_points.size:
        base_profile = _axis_profile_from_points(
            lower_boundary.sample_points,
            centerline_x,
            x_edges,
            cfg,
            quantile=cfg.boundary_axis_quantile,
        )
        profile = base_profile + _axis_sign(cfg.shoe_up_sign) * offset
        window = int(round(max(0.0, cfg.smooth_footbed_window_fraction) * profile.size))
        if window % 2 == 0:
            window += 1
        window = max(window, cfg.floor_smooth_window, 3)
        profile = _smooth_1d(profile, window)
        profile = _smooth_1d(profile, window)
        return profile.astype(np.float32), offset, "open_boundary_offset"

    profile = _smooth_footbed_profile_from_height_fraction(vertices, centerline_x, cfg)
    return profile.astype(np.float32), offset, "height_fraction_fallback"


def _smooth_footbed_profile_from_height_fraction(
    vertices: np.ndarray,
    centerline_x: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    """Fallback: place the footbed at a controlled fraction above shoe bottom."""

    axis_values = np.asarray(vertices, dtype=np.float32)[:, cfg.shoe_up_axis]
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    signed_axis = sole_sign * axis_values
    signed_bottom = float(np.max(signed_axis))
    signed_opening = float(np.min(signed_axis))
    signed_height = max(signed_bottom - signed_opening, 1e-8)
    signed_footbed = signed_bottom - cfg.smooth_footbed_height_fraction_from_bottom * signed_height
    footbed_axis_value = signed_footbed * sole_sign
    return np.full(centerline_x.shape, footbed_axis_value, dtype=np.float32)


def _fill_missing_profile(profile: np.ndarray, centerline_x: np.ndarray, fallback_profile: np.ndarray) -> np.ndarray:
    filled = np.asarray(profile, dtype=np.float32).copy()
    valid = np.isfinite(filled)
    if np.all(valid):
        return filled
    fallback = np.asarray(fallback_profile, dtype=np.float32)
    if np.any(valid):
        filled[~valid] = np.interp(centerline_x[~valid], centerline_x[valid], filled[valid])
    elif fallback.shape == filled.shape and np.any(np.isfinite(fallback)):
        filled[:] = fallback
    else:
        filled[:] = 0.0
    if fallback.shape == filled.shape:
        still_bad = ~np.isfinite(filled)
        filled[still_bad] = fallback[still_bad]
    return filled.astype(np.float32)


def _clamp_profile_to_open_bottom(
    profile: np.ndarray,
    centerline_x: np.ndarray,
    x_edges: np.ndarray,
    lower_boundary: LowerOpenBoundary,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    boundary_axis = _axis_profile_from_points(
        lower_boundary.sample_points,
        centerline_x,
        x_edges,
        cfg,
        quantile=cfg.boundary_axis_quantile,
    )
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    profile_signed = sole_sign * profile
    boundary_signed = sole_sign * boundary_axis
    clamped_signed = np.minimum(profile_signed, boundary_signed + cfg.boundary_filter_margin)
    return (clamped_signed * sole_sign).astype(np.float32)


def _support_sample_points(vertices: np.ndarray, support_faces: np.ndarray, support_centroids: np.ndarray) -> np.ndarray:
    triangles = vertices[support_faces]
    edge_midpoints = np.stack(
        [
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ],
        axis=1,
    )
    samples = np.concatenate(
        [
            triangles.reshape(-1, 3),
            support_centroids,
            edge_midpoints.reshape(-1, 3),
        ],
        axis=0,
    )
    return samples.astype(np.float32)


def _sample_boundary_edges(
    vertices: np.ndarray,
    edge_indices: np.ndarray,
    reference_vertices: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    if edge_indices.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x_edges, z_edges = _footprint_edges(reference_vertices, cfg)
    cell_x = float(np.mean(np.diff(x_edges)))
    cell_z = float(np.mean(np.diff(z_edges)))
    spacing = max(min(cell_x, cell_z) * cfg.boundary_sample_spacing_fraction, 1e-6)
    samples = []
    for a, b in np.asarray(edge_indices, dtype=np.int64):
        p0 = vertices[int(a)]
        p1 = vertices[int(b)]
        delta = p1 - p0
        xz_length = float(
            np.linalg.norm(
                delta[[cfg.shoe_length_axis, cfg.shoe_width_axis]]
            )
        )
        steps = max(int(math.ceil(xz_length / spacing)), 1)
        for t in np.linspace(0.0, 1.0, steps + 1):
            samples.append((1.0 - t) * p0 + t * p1)
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


def _rasterize_footprint(
    vertices: np.ndarray,
    points: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges, z_edges = _footprint_edges(vertices, cfg)
    hist, _, _ = np.histogram2d(
        points[:, cfg.shoe_length_axis],
        points[:, cfg.shoe_width_axis],
        bins=(x_edges, z_edges),
    )
    return hist > 0, x_edges, z_edges


def _rasterize_boundary_footprint(
    vertices: np.ndarray,
    boundary_points: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask, x_edges, z_edges = _rasterize_footprint(vertices, boundary_points, cfg)
    if ndimage is None:
        return mask, x_edges, z_edges
    structure = np.ones((3, 3), dtype=bool)
    grown = ndimage.binary_dilation(mask, structure=structure, iterations=cfg.boundary_morphology_iterations)
    grown = ndimage.binary_closing(grown, structure=structure, iterations=cfg.morphology_iterations)
    grown = ndimage.binary_fill_holes(grown)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    cell_x = float(np.mean(np.diff(x_edges)))
    cell_z = float(np.mean(np.diff(z_edges)))
    x_min = float(boundary_points[:, cfg.shoe_length_axis].min()) - cell_x
    x_max = float(boundary_points[:, cfg.shoe_length_axis].max()) + cell_x
    z_min = float(boundary_points[:, cfg.shoe_width_axis].min()) - cell_z
    z_max = float(boundary_points[:, cfg.shoe_width_axis].max()) + cell_z
    valid_extent = ((x_centers >= x_min) & (x_centers <= x_max))[:, None] & (
        (z_centers >= z_min) & (z_centers <= z_max)
    )[None, :]
    grown = grown & valid_extent
    return _clean_footprint_mask(grown, cfg), x_edges, z_edges


def _rasterize_open_bottom_silhouette(
    reference_vertices: np.ndarray,
    points: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask, x_edges, z_edges = _rasterize_footprint(reference_vertices, points, cfg)
    if ndimage is None:
        return _clean_footprint_mask(mask, cfg), x_edges, z_edges

    structure = np.ones((3, 3), dtype=bool)
    silhouette = np.asarray(mask, dtype=bool)
    if cfg.open_bottom_dilation_iterations > 0:
        silhouette = ndimage.binary_dilation(
            silhouette,
            structure=structure,
            iterations=cfg.open_bottom_dilation_iterations,
        )
    if cfg.open_bottom_morphology_iterations > 0:
        silhouette = ndimage.binary_closing(
            silhouette,
            structure=structure,
            iterations=cfg.open_bottom_morphology_iterations,
        )
    silhouette = ndimage.binary_fill_holes(silhouette)
    return _clean_footprint_mask(silhouette, cfg), x_edges, z_edges


def _outline_points_from_mask(
    mask: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    axis_points: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    if ndimage is not None:
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        outline = mask & ~eroded
    else:
        outline = mask

    rows, cols = np.nonzero(outline)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    max_points = 5000
    if rows.size > max_points:
        stride = int(math.ceil(rows.size / max_points))
        rows = rows[::stride]
        cols = cols[::stride]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    x_values = x_centers[rows]
    z_values = z_centers[cols]
    axis_values = _axis_profile_from_points(axis_points, x_values, x_edges, cfg, quantile=cfg.boundary_axis_quantile)

    outline_points = np.zeros((x_values.shape[0], 3), dtype=np.float32)
    outline_points[:, cfg.shoe_length_axis] = x_values
    outline_points[:, cfg.shoe_up_axis] = axis_values
    outline_points[:, cfg.shoe_width_axis] = z_values
    return outline_points


def _footprint_edges(
    vertices: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    length_values = vertices[:, cfg.shoe_length_axis]
    width_values = vertices[:, cfg.shoe_width_axis]
    length_min = float(length_values.min())
    length_max = float(length_values.max())
    width_min = float(width_values.min())
    width_max = float(width_values.max())
    length_pad = max((length_max - length_min) * cfg.grid_padding_fraction, 1e-5)
    width_pad = max((width_max - width_min) * cfg.grid_padding_fraction, 1e-5)
    x_edges = np.linspace(length_min - length_pad, length_max + length_pad, cfg.grid_resolution + 1)
    z_edges = np.linspace(width_min - width_pad, width_max + width_pad, cfg.grid_resolution + 1)
    return x_edges, z_edges


def _clean_footprint_mask(mask: np.ndarray, cfg: SupportFootprintConfig) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if ndimage is None:
        return mask
    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndimage.binary_closing(mask, structure=structure, iterations=cfg.morphology_iterations)
    cleaned = ndimage.binary_fill_holes(cleaned)
    labeled, count = ndimage.label(cleaned)
    if count <= 1:
        return cleaned.astype(bool)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest = float(component_sizes.max())
    keep_threshold = max(float(cfg.min_component_pixels), largest * cfg.component_keep_ratio)
    keep_labels = np.flatnonzero(component_sizes >= keep_threshold)
    if keep_labels.size == 0:
        keep_labels = np.asarray([int(np.argmax(component_sizes))])
    return np.isin(labeled, keep_labels)


def _make_inner_footbed_mask(mask: np.ndarray, cfg: SupportFootprintConfig) -> np.ndarray:
    """Shrink the full outsole footprint to the usable footbed area."""

    full = np.asarray(mask, dtype=bool)
    if cfg.footbed_inner_margin_cells <= 0 or ndimage is None or not np.any(full):
        return full.copy()

    distance = ndimage.distance_transform_edt(full)
    inner = distance >= float(cfg.footbed_inner_margin_cells)
    min_cells = int(math.ceil(float(full.sum()) * cfg.footbed_inner_min_area_ratio))
    if int(inner.sum()) < max(min_cells, 1):
        # Pick the largest erosion that still leaves enough area. This avoids
        # deleting narrow children's shoes or sandals completely.
        for margin in range(cfg.footbed_inner_margin_cells - 1, -1, -1):
            candidate = distance >= float(margin)
            if int(candidate.sum()) >= max(min_cells, 1):
                inner = candidate
                break
    if not np.any(inner):
        return full.copy()
    return inner.astype(bool)


def _profile_from_mask(
    mask: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    cfg: SupportFootprintConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    centerline_x = []
    centerline_z = []
    left_boundary_z = []
    right_boundary_z = []
    width_profile = []
    for ix in range(mask.shape[0]):
        cols = np.flatnonzero(mask[ix])
        if cols.size < cfg.min_slice_pixels:
            continue
        left = float(z_centers[cols.min()])
        right = float(z_centers[cols.max()])
        centerline_x.append(float(x_centers[ix]))
        left_boundary_z.append(left)
        right_boundary_z.append(right)
        centerline_z.append(0.5 * (left + right))
        width_profile.append(right - left)

    if len(centerline_x) < 3:
        return None

    centerline_x_arr = np.asarray(centerline_x, dtype=np.float32)
    centerline_z_arr = _smooth_1d(np.asarray(centerline_z, dtype=np.float32), cfg.centerline_smooth_window)
    left_arr = _smooth_1d(np.asarray(left_boundary_z, dtype=np.float32), cfg.centerline_smooth_window)
    right_arr = _smooth_1d(np.asarray(right_boundary_z, dtype=np.float32), cfg.centerline_smooth_window)
    width_arr = np.maximum(right_arr - left_arr, 0.0)
    return centerline_x_arr, centerline_z_arr, left_arr, right_arr, width_arr


def _not_below_lower_boundary(
    points: np.ndarray,
    lower_boundary: LowerOpenBoundary,
    reference_vertices: np.ndarray,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x_edges, _ = _footprint_edges(reference_vertices, cfg)
    boundary_axis = _axis_profile_from_points(
        lower_boundary.sample_points,
        points[:, cfg.shoe_length_axis],
        x_edges,
        cfg,
        quantile=cfg.boundary_axis_quantile,
    )
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    point_signed = sole_sign * points[:, cfg.shoe_up_axis]
    boundary_signed = sole_sign * boundary_axis
    return point_signed <= boundary_signed + cfg.boundary_filter_margin


def _inside_open_bottom_footprint(
    points: np.ndarray,
    lower_boundary: LowerOpenBoundary,
    cfg: SupportFootprintConfig,
) -> np.ndarray:
    if lower_boundary.footprint_mask is None or lower_boundary.x_edges is None or lower_boundary.z_edges is None:
        return np.ones((points.shape[0],), dtype=bool)

    return _inside_footprint_mask(
        points,
        np.asarray(lower_boundary.footprint_mask, dtype=bool),
        np.asarray(lower_boundary.x_edges, dtype=np.float32),
        np.asarray(lower_boundary.z_edges, dtype=np.float32),
        cfg,
        dilation_iterations=1,
    )


def _inside_footprint_mask(
    points: np.ndarray,
    footprint_mask: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    cfg: SupportFootprintConfig,
    *,
    dilation_iterations: int = 0,
) -> np.ndarray:
    mask = np.asarray(footprint_mask, dtype=bool)
    if dilation_iterations > 0 and ndimage is not None:
        mask = ndimage.binary_dilation(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=dilation_iterations,
        )
    x_values = points[:, cfg.shoe_length_axis]
    z_values = points[:, cfg.shoe_width_axis]
    ix = np.searchsorted(x_edges, x_values, side="right") - 1
    iz = np.searchsorted(z_edges, z_values, side="right") - 1
    valid = (ix >= 0) & (ix < mask.shape[0]) & (iz >= 0) & (iz < mask.shape[1])
    keep = np.zeros((points.shape[0],), dtype=bool)
    keep[valid] = mask[ix[valid], iz[valid]]
    return keep


def _axis_profile_from_points(
    support_points: np.ndarray,
    centerline_x: np.ndarray,
    x_edges: np.ndarray,
    cfg: SupportFootprintConfig,
    *,
    quantile: float,
) -> np.ndarray:
    x_values = support_points[:, cfg.shoe_length_axis]
    axis_values = support_points[:, cfg.shoe_up_axis]
    sole_sign = -_axis_sign(cfg.shoe_up_sign)
    signed_axis_values = sole_sign * axis_values
    profile = np.full(centerline_x.shape, np.nan, dtype=np.float32)
    for i, x in enumerate(centerline_x):
        edge_idx = np.searchsorted(x_edges, x, side="right") - 1
        edge_idx = int(np.clip(edge_idx, 0, len(x_edges) - 2))
        x0 = x_edges[edge_idx]
        x1 = x_edges[edge_idx + 1]
        in_slice = (x_values >= x0) & (x_values <= x1)
        if np.any(in_slice):
            signed_coord = float(np.quantile(signed_axis_values[in_slice], quantile))
            profile[i] = signed_coord * sole_sign

    valid = np.isfinite(profile)
    if not np.any(valid):
        signed_coord = float(np.quantile(signed_axis_values, quantile))
        profile[:] = signed_coord * sole_sign
    elif not np.all(valid):
        profile[~valid] = np.interp(centerline_x[~valid], centerline_x[valid], profile[valid])
    return _smooth_1d(profile, cfg.centerline_smooth_window)


def _fraction_mask(x_values: np.ndarray, fraction_range: Tuple[float, float]) -> np.ndarray:
    if x_values.size == 0:
        return np.zeros((0,), dtype=bool)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    extent = max(x_max - x_min, 1e-8)
    fraction = (x_values - x_min) / extent
    return (fraction >= fraction_range[0]) & (fraction <= fraction_range[1])


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3 or window <= 1:
        return values
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window < 3:
        return values
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _footprint_extent(footprint: SupportFootprint) -> list[float]:
    return [
        float(footprint.z_edges[0]),
        float(footprint.z_edges[-1]),
        float(footprint.x_edges[0]),
        float(footprint.x_edges[-1]),
    ]


def _draw_footprint_guides(ax, footprint: SupportFootprint, *, show_width: bool) -> None:
    z_centers = 0.5 * (footprint.z_edges[:-1] + footprint.z_edges[1:])
    x_centers = 0.5 * (footprint.x_edges[:-1] + footprint.x_edges[1:])
    try:
        ax.contour(
            z_centers,
            x_centers,
            footprint.footprint_mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.8,
        )
    except Exception:
        pass
    ax.plot(footprint.centerline_z, footprint.centerline_x, color="#d7191c", linewidth=1.8)
    if show_width:
        ax.plot(footprint.left_boundary_z, footprint.centerline_x, color="#2c7bb6", linewidth=0.9)
        ax.plot(footprint.right_boundary_z, footprint.centerline_x, color="#2c7bb6", linewidth=0.9)


def _profile_grid_inside_footprint(footprint: SupportFootprint, profile: np.ndarray) -> np.ndarray:
    return _profile_grid_from_arrays(
        footprint.footprint_mask,
        footprint.x_edges,
        footprint.centerline_x,
        profile,
    )


def _profile_grid_from_arrays(
    footprint_mask: np.ndarray,
    x_edges: np.ndarray,
    centerline_x: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    row_values = np.interp(x_centers, centerline_x, profile)
    mask = np.asarray(footprint_mask, dtype=bool)
    grid = np.repeat(row_values[:, None], mask.shape[1], axis=1).astype(np.float32)
    grid[~mask] = np.nan
    return grid


def _median_step(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return 1.0
    return float(np.median(np.diff(np.sort(values))))


def _shade_regions(
    ax,
    footprint: SupportFootprint,
    *,
    vertical: bool = True,
    alpha: float = 0.16,
    include_labels: bool = True,
) -> None:
    regions = [
        ("heel", footprint.heel_mask, "#fee08b"),
        ("ball", footprint.ball_mask, "#abdda4"),
        ("toe", footprint.toe_mask, "#fdae61"),
    ]
    y_min = float(footprint.centerline_x.min()) if footprint.centerline_x.size else 0.0
    y_max = float(footprint.centerline_x.max()) if footprint.centerline_x.size else 1.0
    for label, mask, color in regions:
        if not np.any(mask):
            continue
        x0 = float(footprint.centerline_x[mask].min())
        x1 = float(footprint.centerline_x[mask].max())
        label_text = f"{label} region" if include_labels else None
        if vertical:
            ax.axhspan(x0, x1, color=color, alpha=alpha, label=label_text)
        else:
            ax.axvspan(x0, x1, color=color, alpha=alpha, label=label_text)
    if vertical:
        ax.set_ylim(y_min, y_max)


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _save_figure(fig, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _finite_range(values: np.ndarray) -> list[float | None]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [None, None]
    return [float(finite.min()), float(finite.max())]


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value
