"""Fixed-axis neutral SUPR-to-shoe alignment."""

from __future__ import annotations

import numpy as np

from .mesh import TriangleMesh


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
