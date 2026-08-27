"""Fixed-axis neutral SUPR-to-shoe alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .footbed import FootbedSurface, sample_footbed_y
from .mesh import TriangleMesh


PLANTAR_NORMAL_Y_MIN = float(np.cos(np.deg2rad(45.0)))
MIN_PLANTAR_FOOTBED_COVERAGE = 0.95


@dataclass(frozen=True)
class FootAlignment:
    """A complete deterministic mapping between neutral-foot and shoe frames."""

    foot_to_shoe: np.ndarray
    shoe_to_foot: np.ndarray
    axis_remap: np.ndarray
    scale: float
    translation: np.ndarray
    length_ratio: float
    input_foot_bounds: np.ndarray
    input_foot_extents: np.ndarray
    input_shoe_bounds: np.ndarray
    input_shoe_extents: np.ndarray
    aligned_foot_bounds: np.ndarray
    aligned_foot_extents: np.ndarray
    plantar_sample_count: int
    covered_plantar_sample_count: int
    footbed_contact_coverage: float
    minimum_footbed_gap: float
    maximum_footbed_gap: float

    def foot_points_to_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map points from the original SUPR frame into the shoe frame."""

        return transform_points(points, self.foot_to_shoe)

    def shoe_points_to_foot(self, points: np.ndarray) -> np.ndarray:
        """Map points from the shoe frame back into the original SUPR frame."""

        return transform_points(points, self.shoe_to_foot)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible alignment record without input paths."""

        return {
            "coordinate_conventions": {
                "foot_input": {
                    "x": "foot_width",
                    "y": "anatomical_height",
                    "z": "foot_length_positive_heel_to_toe",
                },
                "shoe": {
                    "x": "shoe_length_positive_heel_to_toe",
                    "y": "vertical_positive_down_toward_sole",
                    "z": "shoe_width",
                },
            },
            "axis_remap": self.axis_remap.tolist(),
            "length_ratio": self.length_ratio,
            "scale": self.scale,
            "translation": self.translation.tolist(),
            "foot_to_shoe": self.foot_to_shoe.tolist(),
            "shoe_to_foot": self.shoe_to_foot.tolist(),
            "bounds": {
                "input_foot": self.input_foot_bounds.tolist(),
                "input_shoe": self.input_shoe_bounds.tolist(),
                "aligned_foot": self.aligned_foot_bounds.tolist(),
            },
            "extents": {
                "input_foot": self.input_foot_extents.tolist(),
                "input_shoe": self.input_shoe_extents.tolist(),
                "aligned_foot": self.aligned_foot_extents.tolist(),
            },
            "plantar_contact": {
                "normal_y_minimum": PLANTAR_NORMAL_Y_MIN,
                "sample_count": self.plantar_sample_count,
                "covered_sample_count": self.covered_plantar_sample_count,
                "coverage": self.footbed_contact_coverage,
                "minimum_gap": self.minimum_footbed_gap,
                "maximum_gap": self.maximum_footbed_gap,
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
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
    transformed = homogeneous @ transform.T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise ValueError("matrix maps at least one point to invalid homogeneous coordinates")
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


def build_axis_scale_xz_transform(
    foot_mesh: TriangleMesh,
    shoe_mesh: TriangleMesh,
    length_ratio: float = 0.85,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Build the fixed remap, length scale, and X/Z center translation."""

    ratio = float(length_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("length_ratio must be finite and positive")
    axis_remap = make_supr_to_shoe_axis_remap()
    remapped = transform_points(foot_mesh.vertices, axis_remap)
    remapped_length = float(np.ptp(remapped[:, 0]))
    shoe_length = float(shoe_mesh.extents[0])
    if remapped_length <= 0.0 or shoe_length <= 0.0:
        raise ValueError("foot and shoe must have positive X length extents")
    scale = ratio * shoe_length / remapped_length
    axis_and_scale = _uniform_scale_matrix(scale) @ axis_remap
    scaled = transform_points(foot_mesh.vertices, axis_and_scale)
    scaled_bounds = np.stack((scaled.min(axis=0), scaled.max(axis=0)), axis=0)
    scaled_center = scaled_bounds.mean(axis=0)
    translation = np.asarray(
        [
            shoe_mesh.center[0] - scaled_center[0],
            0.0,
            shoe_mesh.center[2] - scaled_center[2],
        ],
        dtype=np.float64,
    )
    matrix = _translation_matrix(translation) @ axis_and_scale
    return matrix, scale, translation


def build_initial_alignment(
    foot_mesh: TriangleMesh,
    shoe_mesh: TriangleMesh,
    footbed: FootbedSurface,
    length_ratio: float = 0.85,
) -> FootAlignment:
    """Place a neutral SUPR foot at first plantar contact with the footbed."""

    horizontal_matrix, scale, translation = build_axis_scale_xz_transform(
        foot_mesh, shoe_mesh, length_ratio
    )
    horizontal_vertices = transform_points(foot_mesh.vertices, horizontal_matrix)
    triangles = horizontal_vertices[foot_mesh.faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(crosses, axis=1)
    normals = np.divide(
        crosses,
        lengths[:, None],
        out=np.zeros_like(crosses),
        where=lengths[:, None] > 0.0,
    )
    plantar_faces = normals[:, 1] >= PLANTAR_NORMAL_Y_MIN
    plantar_vertices = np.unique(foot_mesh.faces[plantar_faces])
    if len(plantar_vertices) == 0:
        raise ValueError("foot mesh has no plantar faces in the remapped shoe frame")

    footbed_y, valid = sample_footbed_y(
        footbed, horizontal_vertices[plantar_vertices][:, (0, 2)]
    )
    covered_count = int(np.count_nonzero(valid))
    coverage = covered_count / len(plantar_vertices)
    if coverage < MIN_PLANTAR_FOOTBED_COVERAGE:
        raise ValueError(
            "plantar footbed projection coverage is below the required threshold: "
            f"{covered_count}/{len(plantar_vertices)} = {coverage:.6f} < "
            f"{MIN_PLANTAR_FOOTBED_COVERAGE:.6f}"
        )

    covered_vertices = plantar_vertices[valid]
    candidate_shifts = footbed_y[valid] - horizontal_vertices[covered_vertices, 1]
    translation_y = float(np.min(candidate_shifts))
    translation = translation.copy()
    translation[1] = translation_y
    foot_to_shoe = (
        _translation_matrix(np.asarray([0.0, translation_y, 0.0]))
        @ horizontal_matrix
    )
    shoe_to_foot = np.linalg.inv(foot_to_shoe)
    aligned_vertices = transform_points(foot_mesh.vertices, foot_to_shoe)
    aligned_bounds = np.stack(
        (aligned_vertices.min(axis=0), aligned_vertices.max(axis=0)), axis=0
    )
    gaps = footbed_y[valid] - aligned_vertices[covered_vertices, 1]
    gaps[np.abs(gaps) < 1e-14] = 0.0
    return FootAlignment(
        foot_to_shoe=foot_to_shoe,
        shoe_to_foot=shoe_to_foot,
        axis_remap=make_supr_to_shoe_axis_remap(),
        scale=scale,
        translation=translation,
        length_ratio=float(length_ratio),
        input_foot_bounds=foot_mesh.bounds,
        input_foot_extents=foot_mesh.extents,
        input_shoe_bounds=shoe_mesh.bounds,
        input_shoe_extents=shoe_mesh.extents,
        aligned_foot_bounds=aligned_bounds,
        aligned_foot_extents=aligned_bounds[1] - aligned_bounds[0],
        plantar_sample_count=int(len(plantar_vertices)),
        covered_plantar_sample_count=covered_count,
        footbed_contact_coverage=coverage,
        minimum_footbed_gap=float(np.min(gaps)),
        maximum_footbed_gap=float(np.max(gaps)),
    )
