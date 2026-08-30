"""Canonical right-shoe validation and functional-length normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .footbed import FootbedSurface
from .mesh import TriangleMesh, transform_mesh


EXPECTED_SHOE_COORDINATE_SYSTEM = "effective_gshell_x_length_y_down_z_width"
SHOE_SIDE = "right"
SHOE_AXIS_SEMANTICS = {
    "+X": "heel_to_toe",
    "+Y": "down_toward_sole",
    "+Z": "shoe_width",
}
FUNCTIONAL_WIDTH_FRACTION = 0.10
LANDMARK_LENGTH_FRACTION = 0.15
LANDMARK_CENTRAL_WIDTH_FRACTION = 0.50


@dataclass(frozen=True)
class ShoeNormalization:
    """A reversible mapping from one shoe frame to functional unit length."""

    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    origin: np.ndarray
    functional_length: float
    outer_length_ratio: float
    heel_landmark: np.ndarray
    toe_landmark: np.ndarray
    functional_column_range: tuple[int, int]
    median_support_width: float
    footprint_component_count: int
    selected_footprint_cell_count: int
    selected_footprint_x_column_count: int
    centerline_original_xz: np.ndarray
    centerline_normalized_xz: np.ndarray
    input_shoe_bounds: np.ndarray
    input_shoe_extents: np.ndarray
    normalized_shoe_bounds: np.ndarray
    normalized_shoe_extents: np.ndarray

    def shoe_points_to_normalized(self, points: np.ndarray) -> np.ndarray:
        """Map shoe-frame points into functional normalized coordinates."""

        return _transform_points(points, self.shoe_to_normalized)

    def normalized_points_to_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map functional normalized points back into the shoe frame."""

        return _transform_points(points, self.normalized_to_shoe)

    def shoe_mesh_to_normalized(self, mesh: TriangleMesh) -> TriangleMesh:
        """Map a mesh from the shoe frame into normalized coordinates."""

        return transform_mesh(mesh, self.shoe_to_normalized)

    def normalized_mesh_to_shoe(self, mesh: TriangleMesh) -> TriangleMesh:
        """Map a mesh from normalized coordinates back into the shoe frame."""

        return transform_mesh(mesh, self.normalized_to_shoe)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible normalization record."""

        heel_normalized = self.shoe_points_to_normalized(
            self.heel_landmark[None, :]
        )[0]
        toe_normalized = self.shoe_points_to_normalized(
            self.toe_landmark[None, :]
        )[0]
        return {
            "origin": self.origin.tolist(),
            "functional_length": self.functional_length,
            "outer_length_ratio": self.outer_length_ratio,
            "parameters": {
                "footprint_connectivity": 8,
                "functional_width_fraction": FUNCTIONAL_WIDTH_FRACTION,
                "landmark_length_fraction": LANDMARK_LENGTH_FRACTION,
                "landmark_central_width_fraction": (
                    LANDMARK_CENTRAL_WIDTH_FRACTION
                ),
            },
            "shoe_to_normalized": self.shoe_to_normalized.tolist(),
            "normalized_to_shoe": self.normalized_to_shoe.tolist(),
            "landmarks": {
                "heel_original": self.heel_landmark.tolist(),
                "toe_original": self.toe_landmark.tolist(),
                "heel_normalized": heel_normalized.tolist(),
                "toe_normalized": toe_normalized.tolist(),
            },
            "functional_columns": {
                "heel_index": self.functional_column_range[0],
                "toe_index": self.functional_column_range[1],
                "median_support_width": self.median_support_width,
            },
            "footprint": {
                "component_count": self.footprint_component_count,
                "selected_cell_count": self.selected_footprint_cell_count,
                "selected_x_column_count": (
                    self.selected_footprint_x_column_count
                ),
            },
            "centerline": {
                "original_xz": self.centerline_original_xz.tolist(),
                "normalized_xz": self.centerline_normalized_xz.tolist(),
            },
            "bounds": {
                "input_shoe": self.input_shoe_bounds.tolist(),
                "normalized_shoe": self.normalized_shoe_bounds.tolist(),
            },
            "extents": {
                "input_shoe": self.input_shoe_extents.tolist(),
                "normalized_shoe": self.normalized_shoe_extents.tolist(),
            },
        }


def validate_shoe_frame_metadata(path: str | Path) -> None:
    """Require metadata for the project's fixed canonical right-shoe frame."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid shoe-frame metadata JSON: {source}") from error
    if not isinstance(metadata, dict):
        raise ValueError("shoe-frame metadata must contain a JSON object")

    reference_mesh = metadata.get("reference_mesh")
    if not isinstance(reference_mesh, dict):
        raise ValueError("shoe-frame metadata is missing reference_mesh")
    coordinate_system = reference_mesh.get("coordinate_system")
    if coordinate_system != EXPECTED_SHOE_COORDINATE_SYSTEM:
        raise ValueError(
            "unsupported shoe coordinate system: "
            f"expected {EXPECTED_SHOE_COORDINATE_SYSTEM!r}, "
            f"received {coordinate_system!r}"
        )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
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
        raise ValueError("matrix maps a point to invalid homogeneous coordinates")
    return transformed[:, :3] / transformed[:, 3, None]


def _validate_footbed_grid(
    footbed: FootbedSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(footbed.valid_mask, dtype=bool)
    heights = np.asarray(footbed.height_grid, dtype=np.float64)
    x_coordinates = np.asarray(footbed.x_coordinates, dtype=np.float64)
    z_coordinates = np.asarray(footbed.z_coordinates, dtype=np.float64)
    expected_shape = (len(x_coordinates), len(z_coordinates))
    if mask.ndim != 2 or heights.shape != mask.shape or mask.shape != expected_shape:
        raise ValueError(
            "footbed grid arrays must share shape "
            "(len(x_coordinates), len(z_coordinates))"
        )
    if len(x_coordinates) < 2 or len(z_coordinates) == 0:
        raise ValueError("footbed grid must contain at least two X columns")
    if not np.isfinite(x_coordinates).all() or not np.isfinite(z_coordinates).all():
        raise ValueError("footbed grid coordinates must be finite")
    if np.any(np.diff(x_coordinates) <= 0.0) or np.any(np.diff(z_coordinates) <= 0.0):
        raise ValueError("footbed grid coordinates must be strictly increasing")
    if not np.array_equal(mask, np.isfinite(heights)):
        raise ValueError("footbed valid_mask must exactly match finite height samples")
    if not np.any(mask):
        raise ValueError("footbed footprint contains no occupied grid cells")
    return mask, heights, x_coordinates, z_coordinates


def _footprint_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for seed_x, seed_z in np.argwhere(mask):
        x_index = int(seed_x)
        z_index = int(seed_z)
        if seen[x_index, z_index]:
            continue
        seen[x_index, z_index] = True
        stack = [(x_index, z_index)]
        cells: list[tuple[int, int]] = []
        while stack:
            current_x, current_z = stack.pop()
            cells.append((current_x, current_z))
            for offset_x in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    if offset_x == 0 and offset_z == 0:
                        continue
                    neighbour_x = current_x + offset_x
                    neighbour_z = current_z + offset_z
                    if not (
                        0 <= neighbour_x < mask.shape[0]
                        and 0 <= neighbour_z < mask.shape[1]
                    ):
                        continue
                    if (
                        mask[neighbour_x, neighbour_z]
                        and not seen[neighbour_x, neighbour_z]
                    ):
                        seen[neighbour_x, neighbour_z] = True
                        stack.append((neighbour_x, neighbour_z))
        components.append(np.asarray(cells, dtype=np.int64))
    return components


def _component_sort_key(
    component: np.ndarray, z_count: int
) -> tuple[int, int, int]:
    x_span = int(np.ptp(component[:, 0]) + 1)
    first_flat_index = int(np.min(component[:, 0] * z_count + component[:, 1]))
    return (-len(component), -x_span, first_flat_index)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _landmark_from_columns(
    x_value: float,
    column_indices: np.ndarray,
    footprint: np.ndarray,
    heights: np.ndarray,
    z_coordinates: np.ndarray,
    z_cell_size: float,
    centerline_z: np.ndarray,
    label: str,
) -> np.ndarray:
    sampled_heights: list[float] = []
    for x_index in column_indices:
        z_indices = np.flatnonzero(footprint[int(x_index)])
        if len(z_indices) == 0:
            continue
        minimum_z = int(z_indices[0])
        maximum_z = int(z_indices[-1])
        midpoint_z = float(
            (z_coordinates[minimum_z] + z_coordinates[maximum_z]) / 2.0
        )
        local_width = float(
            z_coordinates[maximum_z] - z_coordinates[minimum_z]
        )
        if maximum_z == minimum_z:
            local_width = z_cell_size
        half_central_width = (
            0.5 * LANDMARK_CENTRAL_WIDTH_FRACTION * local_width
        )
        central = z_indices[
            np.abs(z_coordinates[z_indices] - midpoint_z)
            <= half_central_width + np.finfo(np.float64).eps
        ]
        valid_heights = heights[int(x_index), central]
        sampled_heights.extend(valid_heights[np.isfinite(valid_heights)].tolist())
    if not sampled_heights:
        raise ValueError(f"{label} landmark has no valid central support heights")
    return np.asarray(
        [
            x_value,
            float(np.median(sampled_heights)),
            float(np.median(centerline_z[column_indices])),
        ],
        dtype=np.float64,
    )


def build_shoe_normalization(
    shoe_mesh: TriangleMesh,
    footbed: FootbedSurface,
) -> ShoeNormalization:
    """Normalize one canonical right shoe by its functional support length."""

    mask, heights, x_coordinates, z_coordinates = _validate_footbed_grid(footbed)
    if shoe_mesh.extents[0] <= np.finfo(np.float64).eps:
        raise ValueError("shoe mesh must have positive X length extent")

    components = _footprint_components(mask)
    components.sort(key=lambda value: _component_sort_key(value, mask.shape[1]))
    selected_component = components[0]
    footprint = np.zeros_like(mask, dtype=bool)
    footprint[selected_component[:, 0], selected_component[:, 1]] = True

    column_width_cells = np.zeros(mask.shape[0], dtype=np.int64)
    minimum_z_indices = np.full(mask.shape[0], -1, dtype=np.int64)
    maximum_z_indices = np.full(mask.shape[0], -1, dtype=np.int64)
    for x_index in np.unique(selected_component[:, 0]):
        z_indices = np.flatnonzero(footprint[int(x_index)])
        minimum_z_indices[int(x_index)] = int(z_indices[0])
        maximum_z_indices[int(x_index)] = int(z_indices[-1])
        column_width_cells[int(x_index)] = int(z_indices[-1] - z_indices[0] + 1)

    nonempty_widths = column_width_cells[column_width_cells > 0]
    median_width_cells = float(np.median(nonempty_widths))
    functional_columns = (
        column_width_cells >= FUNCTIONAL_WIDTH_FRACTION * median_width_cells
    )
    runs = _contiguous_runs(functional_columns)
    if not runs:
        raise ValueError("footbed footprint has no functional X-column range")
    heel_index, toe_index = max(
        runs, key=lambda value: (value[1] - value[0] + 1, -value[0])
    )
    if toe_index <= heel_index:
        raise ValueError("functional support must span at least two X coordinates")

    heel_x = float(x_coordinates[heel_index])
    toe_x = float(x_coordinates[toe_index])
    functional_length = toe_x - heel_x
    if not np.isfinite(functional_length) or functional_length <= 0.0:
        raise ValueError("functional support length must be finite and positive")

    run_indices = np.arange(heel_index, toe_index + 1, dtype=np.int64)
    centerline_z = np.full(mask.shape[0], np.nan, dtype=np.float64)
    centerline_z[run_indices] = (
        z_coordinates[minimum_z_indices[run_indices]]
        + z_coordinates[maximum_z_indices[run_indices]]
    ) / 2.0
    centerline_original = np.column_stack(
        (x_coordinates[run_indices], centerline_z[run_indices])
    )

    rear_indices = run_indices[
        x_coordinates[run_indices]
        <= heel_x + LANDMARK_LENGTH_FRACTION * functional_length
    ]
    front_indices = run_indices[
        x_coordinates[run_indices]
        >= toe_x - LANDMARK_LENGTH_FRACTION * functional_length
    ]
    if len(z_coordinates) > 1:
        z_cell_size = float(np.median(np.diff(z_coordinates)))
    else:
        z_cell_size = float(np.median(np.diff(x_coordinates)))
    if not np.isfinite(z_cell_size) or z_cell_size <= 0.0:
        raise ValueError("footbed grid cell size must be finite and positive")

    heel_landmark = _landmark_from_columns(
        heel_x,
        rear_indices,
        footprint,
        heights,
        z_coordinates,
        z_cell_size,
        centerline_z,
        "heel",
    )
    toe_landmark = _landmark_from_columns(
        toe_x,
        front_indices,
        footprint,
        heights,
        z_coordinates,
        z_cell_size,
        centerline_z,
        "toe",
    )
    origin = heel_landmark.copy()

    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = -origin
    scale = np.eye(4, dtype=np.float64)
    scale[0, 0] = 1.0 / functional_length
    scale[1, 1] = 1.0 / functional_length
    scale[2, 2] = 1.0 / functional_length
    shoe_to_normalized = scale @ translation
    normalized_to_shoe = np.linalg.inv(shoe_to_normalized)

    normalized_vertices = _transform_points(
        shoe_mesh.vertices, shoe_to_normalized
    )
    normalized_bounds = np.stack(
        (normalized_vertices.min(axis=0), normalized_vertices.max(axis=0)),
        axis=0,
    )
    centerline_normalized = np.column_stack(
        (
            (centerline_original[:, 0] - origin[0]) / functional_length,
            (centerline_original[:, 1] - origin[2]) / functional_length,
        )
    )
    return ShoeNormalization(
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        origin=origin,
        functional_length=float(functional_length),
        outer_length_ratio=float(functional_length / shoe_mesh.extents[0]),
        heel_landmark=heel_landmark,
        toe_landmark=toe_landmark,
        functional_column_range=(heel_index, toe_index),
        median_support_width=float(median_width_cells * z_cell_size),
        footprint_component_count=len(components),
        selected_footprint_cell_count=int(len(selected_component)),
        selected_footprint_x_column_count=int(
            len(np.unique(selected_component[:, 0]))
        ),
        centerline_original_xz=centerline_original,
        centerline_normalized_xz=centerline_normalized,
        input_shoe_bounds=shoe_mesh.bounds,
        input_shoe_extents=shoe_mesh.extents,
        normalized_shoe_bounds=normalized_bounds,
        normalized_shoe_extents=normalized_bounds[1] - normalized_bounds[0],
    )
