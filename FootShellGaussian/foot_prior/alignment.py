"""Deterministic rigid SUPR placement in a prepared normalized shoe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import TriangleMesh, sample_triangle_mesh_y


PLANTAR_NORMAL_Y_MIN = float(np.cos(np.deg2rad(45.0)))
MIN_PLANTAR_SUPPORT_COVERAGE = 0.95


@dataclass(frozen=True)
class InitialFootPlacement:
    """Reversible raw-SUPR placement in normalized and original shoe frames."""

    foot_to_normalized_shoe: np.ndarray
    normalized_shoe_to_foot: np.ndarray
    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    foot_to_original_shoe: np.ndarray
    original_shoe_to_foot: np.ndarray
    axis_remap: np.ndarray
    scale: float
    translation: np.ndarray
    requested_plantar_length_ratio: float
    achieved_plantar_length_ratio: float
    plantar_face_indices: np.ndarray
    plantar_vertex_indices: np.ndarray
    heel_vertex_indices: np.ndarray
    heel_reference_remapped: np.ndarray
    heel_reference_normalized: np.ndarray
    remapped_plantar_bounds: np.ndarray
    input_foot_bounds: np.ndarray
    input_foot_extents: np.ndarray
    normalized_shoe_bounds: np.ndarray
    normalized_shoe_extents: np.ndarray
    normalized_support_bounds: np.ndarray
    normalized_support_extents: np.ndarray
    aligned_foot_bounds: np.ndarray
    aligned_foot_extents: np.ndarray
    lateral_face_count: int
    lateral_projected_area: float
    lateral_rms_before: float
    lateral_rms_after: float
    plantar_sample_count: int
    covered_plantar_sample_count: int
    plantar_support_coverage: float
    minimum_support_gap: float
    median_support_gap: float
    maximum_support_gap: float

    def foot_points_to_normalized_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map raw SUPR points into the normalized shoe frame."""

        return transform_points(points, self.foot_to_normalized_shoe)

    def normalized_shoe_points_to_foot(self, points: np.ndarray) -> np.ndarray:
        """Map normalized-shoe points back into the raw SUPR frame."""

        return transform_points(points, self.normalized_shoe_to_foot)

    def foot_points_to_original_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map raw SUPR points directly into the original shoe frame."""

        return transform_points(points, self.foot_to_original_shoe)

    def original_shoe_points_to_foot(self, points: np.ndarray) -> np.ndarray:
        """Map original-shoe points back into the raw SUPR frame."""

        return transform_points(points, self.original_shoe_to_foot)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible placement record without input paths."""

        return {
            "coordinate_conventions": {
                "foot_input": {
                    "x": "foot_width",
                    "y": "anatomical_height",
                    "z": "foot_length_positive_heel_to_toe",
                },
                "normalized_shoe": {
                    "x": "functional_heel_to_toe_with_heel_at_zero",
                    "y": "vertical_positive_down_toward_sole",
                    "z": "shoe_width",
                },
            },
            "transforms": {
                "axis_remap": self.axis_remap.tolist(),
                "foot_to_normalized_shoe": self.foot_to_normalized_shoe.tolist(),
                "normalized_shoe_to_foot": self.normalized_shoe_to_foot.tolist(),
                "shoe_to_normalized": self.shoe_to_normalized.tolist(),
                "normalized_to_shoe": self.normalized_to_shoe.tolist(),
                "foot_to_original_shoe": self.foot_to_original_shoe.tolist(),
                "original_shoe_to_foot": self.original_shoe_to_foot.tolist(),
            },
            "placement": {
                "scale": self.scale,
                "translation": self.translation.tolist(),
                "requested_plantar_length_ratio": (
                    self.requested_plantar_length_ratio
                ),
                "achieved_plantar_length_ratio": (
                    self.achieved_plantar_length_ratio
                ),
                "heel_reference_remapped": self.heel_reference_remapped.tolist(),
                "heel_reference_normalized": (
                    self.heel_reference_normalized.tolist()
                ),
                "remapped_plantar_bounds": self.remapped_plantar_bounds.tolist(),
            },
            "supr_plantar_region": {
                "normal_y_minimum": PLANTAR_NORMAL_Y_MIN,
                "face_indices": self.plantar_face_indices.tolist(),
                "vertex_indices": self.plantar_vertex_indices.tolist(),
                "heel_vertex_indices": self.heel_vertex_indices.tolist(),
            },
            "lateral_centerline_fit": {
                "face_count": self.lateral_face_count,
                "projected_area": self.lateral_projected_area,
                "rms_before_translation": self.lateral_rms_before,
                "rms_after_translation": self.lateral_rms_after,
            },
            "plantar_contact": {
                "minimum_required_coverage": MIN_PLANTAR_SUPPORT_COVERAGE,
                "sample_count": self.plantar_sample_count,
                "covered_sample_count": self.covered_plantar_sample_count,
                "coverage": self.plantar_support_coverage,
                "minimum_gap": self.minimum_support_gap,
                "median_gap": self.median_support_gap,
                "maximum_gap": self.maximum_support_gap,
            },
            "bounds": {
                "input_foot": self.input_foot_bounds.tolist(),
                "normalized_shoe": self.normalized_shoe_bounds.tolist(),
                "normalized_support": self.normalized_support_bounds.tolist(),
                "aligned_foot": self.aligned_foot_bounds.tolist(),
            },
            "extents": {
                "input_foot": self.input_foot_extents.tolist(),
                "normalized_shoe": self.normalized_shoe_extents.tolist(),
                "normalized_support": self.normalized_support_extents.tolist(),
                "aligned_foot": self.aligned_foot_extents.tolist(),
            },
        }


def make_supr_to_shoe_axis_remap() -> np.ndarray:
    """Return the fixed SUPR-width/height/length to shoe-X/Y/Z remap."""

    return np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform an N-by-3 point array with a finite homogeneous matrix."""

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("matrix must be a finite 4x4 array")
    homogeneous = np.column_stack(
        (values, np.ones(len(values), dtype=np.float64))
    )
    transformed = homogeneous @ transform.T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise ValueError(
            "matrix maps at least one point to invalid homogeneous coordinates"
        )
    return transformed[:, :3] / transformed[:, 3, None]


def _uniform_scale_matrix(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    return matrix


def _translation_matrix(translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix


def _validated_inverse_pair(
    forward: np.ndarray, inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(forward, dtype=np.float64)
    second = np.asarray(inverse, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("shoe normalization matrices must have shape (4, 4)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("shoe normalization matrices must be finite")
    if not np.allclose(second @ first, np.eye(4), atol=1e-10, rtol=0.0):
        raise ValueError("shoe normalization matrices are not mutual inverses")
    return first, second


def _validated_centerline(centerline_xz: np.ndarray) -> np.ndarray:
    centerline = np.asarray(centerline_xz, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[1:] != (2,):
        raise ValueError("normalized footbed centerline must have shape (N, 2)")
    if len(centerline) < 2 or not np.isfinite(centerline).all():
        raise ValueError(
            "normalized footbed centerline must contain at least two finite points"
        )
    if np.any(np.diff(centerline[:, 0]) <= 0.0):
        raise ValueError("normalized footbed centerline X must increase strictly")
    return centerline


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    return np.divide(
        crosses,
        lengths[:, None],
        out=np.zeros_like(crosses),
        where=lengths[:, None] > 0.0,
    )


def build_initial_placement(
    foot_mesh: TriangleMesh,
    normalized_shoe_mesh: TriangleMesh,
    normalized_support_mesh: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
    shoe_to_normalized: np.ndarray,
    normalized_to_shoe: np.ndarray,
    foot_length_ratio: float = 0.85,
) -> InitialFootPlacement:
    """Place a neutral SUPR foot at first contact in a normalized normal shoe."""

    ratio = float(foot_length_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("foot_length_ratio must be finite and lie in (0, 1]")
    shoe_to_normalized, normalized_to_shoe = _validated_inverse_pair(
        shoe_to_normalized, normalized_to_shoe
    )
    centerline = _validated_centerline(normalized_centerline_xz)

    axis_remap = make_supr_to_shoe_axis_remap()
    remapped_vertices = transform_points(foot_mesh.vertices, axis_remap)
    normals = _face_normals(remapped_vertices, foot_mesh.faces)
    plantar_face_indices = np.flatnonzero(
        normals[:, 1] >= PLANTAR_NORMAL_Y_MIN
    )
    if len(plantar_face_indices) == 0:
        raise ValueError("SUPR foot has no plantar faces after axis remapping")
    plantar_vertex_indices = np.unique(foot_mesh.faces[plantar_face_indices])
    plantar_vertices = remapped_vertices[plantar_vertex_indices]
    plantar_bounds = np.stack(
        (plantar_vertices.min(axis=0), plantar_vertices.max(axis=0)), axis=0
    )
    plantar_length = float(plantar_bounds[1, 0] - plantar_bounds[0, 0])
    if not np.isfinite(plantar_length) or plantar_length <= 0.0:
        raise ValueError("SUPR plantar footprint must have positive X length")

    heel_x = float(plantar_bounds[0, 0])
    heel_tolerance = max(1.0, abs(heel_x)) * np.finfo(np.float64).eps * 16.0
    heel_local = np.flatnonzero(
        np.abs(plantar_vertices[:, 0] - heel_x) <= heel_tolerance
    )
    heel_vertex_indices = plantar_vertex_indices[heel_local]
    heel_reference_remapped = remapped_vertices[heel_vertex_indices].mean(axis=0)

    scale = ratio / plantar_length
    axis_and_scale = _uniform_scale_matrix(scale) @ axis_remap
    scaled_vertices = transform_points(foot_mesh.vertices, axis_and_scale)
    translation_x = -heel_x * scale
    x_aligned = scaled_vertices.copy()
    x_aligned[:, 0] += translation_x

    plantar_triangles = x_aligned[foot_mesh.faces[plantar_face_indices]]
    centroids = plantar_triangles.mean(axis=1)
    planar = plantar_triangles[:, :, (0, 2)]
    edge_a = planar[:, 1] - planar[:, 0]
    edge_b = planar[:, 2] - planar[:, 0]
    projected_areas = 0.5 * np.abs(
        edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    )
    usable = projected_areas > np.finfo(np.float64).eps
    if not np.any(usable):
        raise ValueError("SUPR plantar faces have zero projected X/Z area")
    centroid_x = centroids[usable, 0]
    tolerance = 1e-12
    if (
        float(np.min(centroid_x)) < centerline[0, 0] - tolerance
        or float(np.max(centroid_x)) > centerline[-1, 0] + tolerance
    ):
        raise ValueError(
            "normalized footbed centerline does not cover the placed plantar faces"
        )
    weights = projected_areas[usable]
    target_z = np.interp(centroid_x, centerline[:, 0], centerline[:, 1])
    residual_before = centroids[usable, 2] - target_z
    translation_z = float(-np.sum(weights * residual_before) / np.sum(weights))
    residual_after = residual_before + translation_z
    lateral_rms_before = float(
        np.sqrt(np.sum(weights * np.square(residual_before)) / np.sum(weights))
    )
    lateral_rms_after = float(
        np.sqrt(np.sum(weights * np.square(residual_after)) / np.sum(weights))
    )

    horizontal_translation = np.asarray(
        [translation_x, 0.0, translation_z], dtype=np.float64
    )
    horizontal_matrix = (
        _translation_matrix(horizontal_translation) @ axis_and_scale
    )
    horizontal_vertices = transform_points(foot_mesh.vertices, horizontal_matrix)
    support_y, valid = sample_triangle_mesh_y(
        normalized_support_mesh,
        horizontal_vertices[plantar_vertex_indices][:, (0, 2)],
    )
    covered_count = int(np.count_nonzero(valid))
    coverage = covered_count / len(plantar_vertex_indices)
    if coverage < MIN_PLANTAR_SUPPORT_COVERAGE:
        raise ValueError(
            "plantar support coverage is below the required threshold: "
            f"{covered_count}/{len(plantar_vertex_indices)} = "
            f"{coverage:.6f} < {MIN_PLANTAR_SUPPORT_COVERAGE:.6f}"
        )

    covered_vertices = plantar_vertex_indices[valid]
    candidate_shifts = support_y[valid] - horizontal_vertices[covered_vertices, 1]
    translation_y = float(np.min(candidate_shifts))
    vertical_matrix = _translation_matrix(
        np.asarray([0.0, translation_y, 0.0], dtype=np.float64)
    )
    foot_to_normalized_shoe = vertical_matrix @ horizontal_matrix
    normalized_shoe_to_foot = np.linalg.inv(foot_to_normalized_shoe)
    foot_to_original_shoe = normalized_to_shoe @ foot_to_normalized_shoe
    original_shoe_to_foot = np.linalg.inv(foot_to_original_shoe)

    aligned_vertices = transform_points(foot_mesh.vertices, foot_to_normalized_shoe)
    aligned_bounds = np.stack(
        (aligned_vertices.min(axis=0), aligned_vertices.max(axis=0)), axis=0
    )
    heel_reference_normalized = transform_points(
        heel_reference_remapped[None, :],
        vertical_matrix
        @ _translation_matrix(horizontal_translation)
        @ _uniform_scale_matrix(scale),
    )[0]
    gaps = support_y[valid] - aligned_vertices[covered_vertices, 1]
    gaps[np.abs(gaps) < 1e-14] = 0.0
    achieved_ratio = float(np.ptp(aligned_vertices[plantar_vertex_indices, 0]))
    translation = np.asarray(
        [translation_x, translation_y, translation_z], dtype=np.float64
    )
    return InitialFootPlacement(
        foot_to_normalized_shoe=foot_to_normalized_shoe,
        normalized_shoe_to_foot=normalized_shoe_to_foot,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        foot_to_original_shoe=foot_to_original_shoe,
        original_shoe_to_foot=original_shoe_to_foot,
        axis_remap=axis_remap,
        scale=scale,
        translation=translation,
        requested_plantar_length_ratio=ratio,
        achieved_plantar_length_ratio=achieved_ratio,
        plantar_face_indices=plantar_face_indices,
        plantar_vertex_indices=plantar_vertex_indices,
        heel_vertex_indices=heel_vertex_indices,
        heel_reference_remapped=heel_reference_remapped,
        heel_reference_normalized=heel_reference_normalized,
        remapped_plantar_bounds=plantar_bounds,
        input_foot_bounds=foot_mesh.bounds,
        input_foot_extents=foot_mesh.extents,
        normalized_shoe_bounds=normalized_shoe_mesh.bounds,
        normalized_shoe_extents=normalized_shoe_mesh.extents,
        normalized_support_bounds=normalized_support_mesh.bounds,
        normalized_support_extents=normalized_support_mesh.extents,
        aligned_foot_bounds=aligned_bounds,
        aligned_foot_extents=aligned_bounds[1] - aligned_bounds[0],
        lateral_face_count=int(np.count_nonzero(usable)),
        lateral_projected_area=float(np.sum(weights)),
        lateral_rms_before=lateral_rms_before,
        lateral_rms_after=lateral_rms_after,
        plantar_sample_count=int(len(plantar_vertex_indices)),
        covered_plantar_sample_count=covered_count,
        plantar_support_coverage=coverage,
        minimum_support_gap=float(np.min(gaps)),
        median_support_gap=float(np.median(gaps)),
        maximum_support_gap=float(np.max(gaps)),
    )
