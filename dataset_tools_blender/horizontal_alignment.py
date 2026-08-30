"""Deterministic horizontal heading estimation for canonical shoe meshes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


METHOD = "lower_geometry_area_weighted_pca"


@dataclass(frozen=True)
class HorizontalAlignment:
    """Measured shoe heading and the rigid correction that aligns it to +X."""

    method: str
    lower_height_fraction: float
    minimum_axis_ratio: float
    maximum_abs_angle_degrees: float
    lower_height_cutoff: float
    triangle_count: int
    lower_triangle_count: int
    lower_triangle_area: float
    weighted_centroid_xy: np.ndarray
    covariance_xy: np.ndarray
    eigenvalues: np.ndarray
    axis_ratio: float
    direction_xy: np.ndarray
    measured_angle_degrees: float
    correction_angle_degrees: float
    residual_angle_degrees: float
    rotation_3x3: np.ndarray
    rotation_4x4: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lower_height_fraction": self.lower_height_fraction,
            "minimum_axis_ratio": self.minimum_axis_ratio,
            "maximum_abs_angle_degrees": self.maximum_abs_angle_degrees,
            "lower_height_cutoff": self.lower_height_cutoff,
            "triangle_count": self.triangle_count,
            "lower_triangle_count": self.lower_triangle_count,
            "lower_triangle_area": self.lower_triangle_area,
            "weighted_centroid_xy": self.weighted_centroid_xy.tolist(),
            "covariance_xy": self.covariance_xy.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "axis_ratio": self.axis_ratio,
            "direction_xy": self.direction_xy.tolist(),
            "measured_angle_degrees": self.measured_angle_degrees,
            "correction_angle_degrees": self.correction_angle_degrees,
            "residual_angle_degrees": self.residual_angle_degrees,
            "rotation_3x3": self.rotation_3x3.tolist(),
            "rotation_4x4": self.rotation_4x4.tolist(),
        }


def validate_horizontal_alignment_config(config: dict[str, Any]) -> None:
    """Reject unsupported or unsafe heading-estimation configuration."""
    expected = {
        "method",
        "lower_height_fraction",
        "minimum_axis_ratio",
        "maximum_abs_angle_degrees",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(
            "horizontal_alignment must contain exactly "
            f"{sorted(expected)}"
        )
    if config["method"] != METHOD:
        raise ValueError(
            f"Unsupported horizontal alignment method: {config['method']!r}"
        )
    fraction = float(config["lower_height_fraction"])
    minimum_ratio = float(config["minimum_axis_ratio"])
    maximum_angle = float(config["maximum_abs_angle_degrees"])
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("lower_height_fraction must be finite and between 0 and 1")
    if not math.isfinite(minimum_ratio) or minimum_ratio <= 1.0:
        raise ValueError("minimum_axis_ratio must be finite and greater than 1")
    if not math.isfinite(maximum_angle) or not 0.0 < maximum_angle < 90.0:
        raise ValueError(
            "maximum_abs_angle_degrees must be finite and between 0 and 90"
        )


def validate_horizontal_alignment_metadata(
    canonical: dict[str, Any], expected_config: dict[str, Any]
) -> list[str]:
    """Validate recorded heading geometry against its manifest contract."""
    errors: list[str] = []
    alignment = canonical.get("horizontal_alignment")
    if not isinstance(alignment, dict):
        return ["missing horizontal alignment metadata"]

    for key, expected in expected_config.items():
        actual = alignment.get(key)
        if actual != expected:
            errors.append(
                "horizontal alignment metadata mismatch for "
                f"{key}: expected {expected!r}, got {actual!r}"
            )
    try:
        rotation = np.asarray(alignment.get("rotation_3x3"), dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("matrix is not finite 3 x 3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
            raise ValueError("matrix is not orthonormal")
        if not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10
        ):
            raise ValueError("matrix determinant is not one")
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid horizontal rotation: {exc}")

    measured = alignment.get("measured_angle_degrees")
    correction = alignment.get("correction_angle_degrees")
    residual = alignment.get("residual_angle_degrees")
    ratio = alignment.get("axis_ratio")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (measured, correction, residual, ratio)
    ):
        errors.append("horizontal alignment scalars are not finite")
    else:
        if not math.isclose(
            float(measured) + float(correction), 0.0, abs_tol=1e-10
        ):
            errors.append(
                "horizontal correction does not cancel the measured heading"
            )
        if abs(float(residual)) > 0.1:
            errors.append("horizontal residual exceeds 0.1 degrees")
        if float(ratio) < float(expected_config["minimum_axis_ratio"]):
            errors.append(
                "horizontal axis ratio is below the configured minimum"
            )

    source_to_canonical = np.asarray(
        canonical.get("source_to_canonical_matrix"), dtype=np.float64
    )
    if source_to_canonical.shape != (4, 4) or not np.isfinite(
        source_to_canonical
    ).all():
        errors.append("source-to-canonical matrix is missing or invalid")
    else:
        try:
            inverse = np.linalg.inv(source_to_canonical)
        except np.linalg.LinAlgError:
            errors.append("source-to-canonical matrix is not reversible")
        else:
            if not np.allclose(
                source_to_canonical @ inverse, np.eye(4), atol=1e-10
            ):
                errors.append(
                    "source-to-canonical matrix inverse is inaccurate"
                )

    bounds_min = np.asarray(
        canonical.get("canonical_bbox_min"), dtype=np.float64
    )
    bounds_max = np.asarray(
        canonical.get("canonical_bbox_max"), dtype=np.float64
    )
    if (
        bounds_min.shape != (3,)
        or bounds_max.shape != (3,)
        or not np.isfinite(bounds_min).all()
        or not np.isfinite(bounds_max).all()
    ):
        errors.append("canonical bounds are missing or invalid")
    elif not np.allclose(
        0.5 * (bounds_min + bounds_max), np.zeros(3), atol=1e-6
    ):
        errors.append("canonical geometry is not centered")
    return errors


def estimate_horizontal_alignment(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    lower_height_fraction: float = 0.20,
    minimum_axis_ratio: float = 2.0,
    maximum_abs_angle_degrees: float = 45.0,
) -> HorizontalAlignment:
    """Estimate a selected shoe's long direction from its lowest triangles.

    Coordinates are Blender canonical coordinates: X is nominal heel-to-toe,
    Y is width, and Z is physical up. The signed source-axis mapping has
    already established that the positive half of the PCA axis is +X.
    """
    config = {
        "method": METHOD,
        "lower_height_fraction": lower_height_fraction,
        "minimum_axis_ratio": minimum_axis_ratio,
        "maximum_abs_angle_degrees": maximum_abs_angle_degrees,
    }
    validate_horizontal_alignment_config(config)

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("vertices must be a non-empty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("vertices contain non-finite values")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) == 0:
        raise ValueError("faces must be a non-empty M x 3 array")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("faces must contain integer indices")
    triangles = triangles.astype(np.int64, copy=False)
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(points):
        raise ValueError("faces contain out-of-range vertex indices")

    triangle_points = points[triangles]
    cross = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    nondegenerate = areas > np.finfo(np.float64).eps * scale * scale
    if not np.any(nondegenerate):
        raise ValueError("mesh has no nondegenerate triangles")

    centroids = triangle_points.mean(axis=1)
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    height = z_max - z_min
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("mesh has zero physical height")
    cutoff = z_min + float(lower_height_fraction) * height
    lower = nondegenerate & (centroids[:, 2] <= cutoff)
    if np.count_nonzero(lower) < 2:
        raise ValueError(
            "lower geometry contains fewer than two nondegenerate triangles"
        )

    samples = centroids[lower, :2]
    weights = areas[lower]
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("lower geometry has zero triangle area")
    mean = np.sum(samples * weights[:, None], axis=0) / total_weight
    centered = samples - mean
    covariance = (centered * weights[:, None]).T @ centered / total_weight
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if eigenvalues[-1] <= np.finfo(np.float64).eps * scale * scale:
        raise ValueError("lower geometry has no measurable horizontal direction")
    ratio_denominator = max(
        float(eigenvalues[0]),
        float(np.finfo(np.float64).eps * eigenvalues[-1]),
    )
    axis_ratio = float(eigenvalues[-1] / ratio_denominator)
    if axis_ratio < float(minimum_axis_ratio):
        raise ValueError(
            "lower geometry has ambiguous horizontal direction: "
            f"axis ratio {axis_ratio:.6g} is below {minimum_axis_ratio:.6g}"
        )

    direction = eigenvectors[:, -1]
    if direction[0] < 0.0:
        direction = -direction
    measured_radians = math.atan2(float(direction[1]), float(direction[0]))
    measured_degrees = math.degrees(measured_radians)
    if abs(measured_degrees) >= float(maximum_abs_angle_degrees):
        raise ValueError(
            "horizontal correction is too large for the reviewed source axes: "
            f"{measured_degrees:.6g} degrees"
        )

    correction_radians = -measured_radians
    sine = math.sin(correction_radians)
    cosine = math.cos(correction_radians)
    rotation_3x3 = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_4x4 = np.eye(4, dtype=np.float64)
    rotation_4x4[:3, :3] = rotation_3x3
    rotated_direction = rotation_3x3[:2, :2] @ direction
    residual_degrees = math.degrees(
        math.atan2(float(rotated_direction[1]), float(rotated_direction[0]))
    )
    return HorizontalAlignment(
        method=METHOD,
        lower_height_fraction=float(lower_height_fraction),
        minimum_axis_ratio=float(minimum_axis_ratio),
        maximum_abs_angle_degrees=float(maximum_abs_angle_degrees),
        lower_height_cutoff=cutoff,
        triangle_count=int(np.count_nonzero(nondegenerate)),
        lower_triangle_count=int(np.count_nonzero(lower)),
        lower_triangle_area=total_weight,
        weighted_centroid_xy=mean,
        covariance_xy=covariance,
        eigenvalues=eigenvalues,
        axis_ratio=axis_ratio,
        direction_xy=direction,
        measured_angle_degrees=measured_degrees,
        correction_angle_degrees=-measured_degrees,
        residual_angle_degrees=residual_degrees,
        rotation_3x3=rotation_3x3,
        rotation_4x4=rotation_4x4,
    )
