"""Topology-independent shoe-footbed identification and sampling."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import TriangleMesh


GRID_LENGTH_RESOLUTION = 256
SUPPORT_NORMAL_ABS_Y_MIN = float(np.cos(np.deg2rad(60.0)))
MAX_LAYER_HEIGHT_STEP_RATIO = 0.02
CENTRAL_BAND_WIDTH_FRACTION = 0.20
MIN_LENGTH_COVERAGE = 0.65
MIN_WIDTH_COVERAGE = 0.40
MIN_CENTRAL_SUPPORT_LENGTH_COVERAGE = 0.65
LOWER_REGION_START_FRACTION = 0.40
MIN_FOOTPRINT_FILL_FRACTION = 0.25
MAX_HEIGHT_RANGE_RATIO = 0.15
MIN_ORIENTATION_COHERENCE = 0.90
MIN_COMPONENT_MERGE_EDGES = 4
MIN_COMPONENT_MERGE_EDGE_FRACTION = 0.01
MAX_COMPONENT_FOOTPRINT_OVERLAP = 0.10
MIN_UNDERLYING_SUPPORT_COLUMN_FRACTION = 0.65
HEIGHT_COMPARISON_EPSILON_RATIO = 1e-6
HEEL_LANDMARK_END_FRACTION = 0.15
FOREFOOT_LANDMARK_START_FRACTION = 0.65
FOREFOOT_LANDMARK_END_FRACTION = 0.85


@dataclass(frozen=True)
class FootbedSurface:
    """An exact source-face footbed mesh plus its sampled support layer."""

    mesh: TriangleMesh
    original_face_indices: np.ndarray
    bounds: np.ndarray
    extents: np.ndarray
    x_coordinates: np.ndarray
    z_coordinates: np.ndarray
    height_grid: np.ndarray
    valid_mask: np.ndarray
    length_coverage: float
    width_coverage: float
    central_support_length_coverage: float
    upward_facing_area_fraction: float
    support_like_area_fraction: float
    area_weighted_median_y: float
    projected_xz_area: float
    diagnostics: tuple[dict[str, Any], ...]
    selection_method: str = "component_layers"
    fallback_reason: str | None = None
    primary_selected_layer_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible selection metrics without the dense grid."""

        return {
            "original_face_indices": self.original_face_indices.tolist(),
            "face_count": int(len(self.original_face_indices)),
            "vertex_count": int(len(self.mesh.vertices)),
            "bounds": self.bounds.tolist(),
            "extents": self.extents.tolist(),
            "grid_shape": [int(value) for value in self.height_grid.shape],
            "valid_grid_cell_count": int(np.count_nonzero(self.valid_mask)),
            "grid_x_bounds": [
                float(self.x_coordinates[0]),
                float(self.x_coordinates[-1]),
            ],
            "grid_z_bounds": [
                float(self.z_coordinates[0]),
                float(self.z_coordinates[-1]),
            ],
            "length_coverage": self.length_coverage,
            "width_coverage": self.width_coverage,
            "central_support_length_coverage": (
                self.central_support_length_coverage
            ),
            "selection_method": self.selection_method,
            "fallback_reason": self.fallback_reason,
            "primary_selected_layer_index": self.primary_selected_layer_index,
            "primary_selection_rejected": (
                self.selection_method == "local_height_trace"
            ),
            "upward_facing_area_fraction": self.upward_facing_area_fraction,
            "support_like_area_fraction": self.support_like_area_fraction,
            "area_weighted_median_y": self.area_weighted_median_y,
            "projected_xz_area": self.projected_xz_area,
            "layers": list(self.diagnostics),
        }


@dataclass(frozen=True)
class HighHeelSupport:
    """A reviewed high-heel support surface and fitting landmarks."""

    surface: FootbedSurface
    orientation_mode: str
    underlying_support_column_fraction: float
    heel_landmark: np.ndarray
    forefoot_landmark: np.ndarray
    heel_elevation: float
    support_angle_degrees: float
    support_x_bounds: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible high-heel measurements."""

        return {
            "orientation_mode": self.orientation_mode,
            "underlying_support_column_fraction": (
                self.underlying_support_column_fraction
            ),
            "landmarks": {
                "heel": self.heel_landmark.tolist(),
                "forefoot": self.forefoot_landmark.tolist(),
                "heel_region": [0.0, HEEL_LANDMARK_END_FRACTION],
                "forefoot_region": [
                    FOREFOOT_LANDMARK_START_FRACTION,
                    FOREFOOT_LANDMARK_END_FRACTION,
                ],
            },
            "heel_elevation": self.heel_elevation,
            "support_angle_degrees": self.support_angle_degrees,
            "support_x_bounds": self.support_x_bounds.tolist(),
        }


@dataclass(frozen=True)
class _Grid:
    x_edges: np.ndarray
    z_edges: np.ndarray
    x_centers: np.ndarray
    z_centers: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.x_centers), len(self.z_centers)

    @property
    def dx(self) -> float:
        return float(self.x_edges[1] - self.x_edges[0])

    @property
    def dz(self) -> float:
        return float(self.z_edges[1] - self.z_edges[0])


@dataclass(frozen=True)
class _LayerNode:
    x_index: int
    z_index: int
    y: float
    face_components: tuple[int, ...]
    face_indices: tuple[int, ...]


def _face_connected_components(faces: np.ndarray) -> list[np.ndarray]:
    """Return edge-adjacent face components for an arbitrary face subset."""

    count = len(faces)
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.uint8)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    edge_owner: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(start, end)), int(max(start, end)))
            owner = edge_owner.setdefault(edge, face_index)
            if owner != face_index:
                union(face_index, owner)

    groups: dict[int, list[int]] = {}
    for face_index in range(count):
        groups.setdefault(find(face_index), []).append(face_index)
    components = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    components.sort(key=lambda indices: int(indices[0]))
    return components


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        raise ValueError("cannot compute an area-weighted median without positive area")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, weights.sum() / 2.0, side="left"))
    return float(values[order[position]])


def _make_grid(shoe_mesh: TriangleMesh) -> _Grid:
    bounds = shoe_mesh.bounds
    length = float(shoe_mesh.extents[0])
    width = float(shoe_mesh.extents[2])
    if length <= 0.0 or width <= 0.0:
        raise ValueError("shoe mesh must have positive X length and Z width extents")
    width_resolution = max(2, int(np.ceil(width / length * GRID_LENGTH_RESOLUTION)))
    x_edges = np.linspace(bounds[0, 0], bounds[1, 0], GRID_LENGTH_RESOLUTION + 1)
    z_edges = np.linspace(bounds[0, 2], bounds[1, 2], width_resolution + 1)
    return _Grid(
        x_edges=x_edges,
        z_edges=z_edges,
        x_centers=(x_edges[:-1] + x_edges[1:]) / 2.0,
        z_centers=(z_edges[:-1] + z_edges[1:]) / 2.0,
    )


def _central_support_length_coverage(
    x_indices: np.ndarray,
    z_indices: np.ndarray,
    grid_shape: tuple[int, int],
) -> float:
    """Measure raw shoe-length coverage inside the centered width band."""

    x_count, z_count = grid_shape
    band_count = max(1, int(np.ceil(CENTRAL_BAND_WIDTH_FRACTION * z_count)))
    band_start = (z_count - band_count) // 2
    inside_band = (z_indices >= band_start) & (
        z_indices < band_start + band_count
    )
    return float(len(np.unique(x_indices[inside_band])) / x_count)


def _center_index_range(
    minimum: float,
    maximum: float,
    first_center: float,
    spacing: float,
    count: int,
) -> range:
    lower = max(0, int(np.ceil((minimum - first_center) / spacing - 1e-12)))
    upper = min(count - 1, int(np.floor((maximum - first_center) / spacing + 1e-12)))
    if upper < lower:
        return range(0)
    return range(lower, upper + 1)


def _build_xz_triangle_bins(triangles: np.ndarray, grid: _Grid) -> list[list[int]]:
    """Index projected triangles by grid cell without an external R-tree."""

    x_count, z_count = grid.shape
    bins: list[list[int]] = [[] for _ in range(x_count * z_count)]
    planar = triangles[:, :, (0, 2)]
    planar_min = planar.min(axis=1)
    planar_max = planar.max(axis=1)
    for triangle_index in range(len(triangles)):
        x_indices = _center_index_range(
            float(planar_min[triangle_index, 0]),
            float(planar_max[triangle_index, 0]),
            float(grid.x_centers[0]),
            grid.dx,
            x_count,
        )
        z_indices = _center_index_range(
            float(planar_min[triangle_index, 1]),
            float(planar_max[triangle_index, 1]),
            float(grid.z_centers[0]),
            grid.dz,
            z_count,
        )
        for x_index in x_indices:
            offset = x_index * z_count
            for z_index in z_indices:
                bins[offset + z_index].append(triangle_index)
    return bins


def _cell_intersections(
    triangles: np.ndarray,
    face_component: np.ndarray,
    grid: _Grid,
    bins: list[list[int]],
    shoe_length: float,
) -> tuple[list[_LayerNode], list[list[int]]]:
    """Find and deduplicate exact vertical intersections in every grid cell."""

    _, z_count = grid.shape
    nodes: list[_LayerNode] = []
    cell_nodes: list[list[int]] = [[] for _ in bins]
    barycentric_tolerance = 1e-10
    merge_tolerance = max(np.finfo(np.float64).eps, shoe_length * 1e-8)

    for flat_index, candidates_list in enumerate(bins):
        if not candidates_list:
            continue
        x_index, z_index = divmod(flat_index, z_count)
        point = np.asarray([grid.x_centers[x_index], grid.z_centers[z_index]])
        candidates = np.asarray(candidates_list, dtype=np.int64)
        candidate_triangles = triangles[candidates]
        planar = candidate_triangles[:, :, (0, 2)]
        first = planar[:, 0]
        edge_a = planar[:, 1] - first
        edge_b = planar[:, 2] - first
        relative = point - first
        determinant = edge_a[:, 0] * edge_b[:, 1] - edge_b[:, 0] * edge_a[:, 1]
        nondegenerate = np.abs(determinant) > np.finfo(np.float64).eps
        weight_a = np.divide(
            relative[:, 0] * edge_b[:, 1] - edge_b[:, 0] * relative[:, 1],
            determinant,
            out=np.zeros_like(determinant),
            where=nondegenerate,
        )
        weight_b = np.divide(
            edge_a[:, 0] * relative[:, 1] - relative[:, 0] * edge_a[:, 1],
            determinant,
            out=np.zeros_like(determinant),
            where=nondegenerate,
        )
        weight_first = 1.0 - weight_a - weight_b
        inside = (
            nondegenerate
            & (weight_first >= -barycentric_tolerance)
            & (weight_a >= -barycentric_tolerance)
            & (weight_b >= -barycentric_tolerance)
        )
        if not np.any(inside):
            continue
        candidates = candidates[inside]
        candidate_triangles = candidate_triangles[inside]
        weight_first = weight_first[inside]
        weight_a = weight_a[inside]
        weight_b = weight_b[inside]
        heights = (
            weight_first * candidate_triangles[:, 0, 1]
            + weight_a * candidate_triangles[:, 1, 1]
            + weight_b * candidate_triangles[:, 2, 1]
        )
        order = np.argsort(heights, kind="stable")
        heights = heights[order]
        candidates = candidates[order]
        components = face_component[candidates]

        group_start = 0
        while group_start < len(heights):
            group_end = group_start + 1
            while (
                group_end < len(heights)
                and heights[group_end] - heights[group_end - 1] <= merge_tolerance
            ):
                group_end += 1
            node = _LayerNode(
                x_index=x_index,
                z_index=z_index,
                y=float(np.mean(heights[group_start:group_end])),
                face_components=tuple(
                    int(value) for value in np.unique(components[group_start:group_end])
                ),
                face_indices=tuple(
                    int(value)
                    for value in np.unique(candidates[group_start:group_end])
                ),
            )
            cell_nodes[flat_index].append(len(nodes))
            nodes.append(node)
            group_start = group_end
    return nodes, cell_nodes


def _group_surface_patches_into_layers(
    nodes: list[_LayerNode],
    cell_nodes: list[list[int]],
    face_component_count: int,
    grid_shape: tuple[int, int],
    grid: _Grid,
    shoe_extents: np.ndarray,
    maximum_adjacent_height_step: float,
    maximum_layer_height_range: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge adjoining non-overlapping face patches into coherent layers."""

    parent = np.arange(face_component_count, dtype=np.int64)
    rank = np.zeros(face_component_count, dtype=np.uint8)
    group_min_y = np.full(face_component_count, np.inf, dtype=np.float64)
    group_max_y = np.full(face_component_count, -np.inf, dtype=np.float64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        combined_minimum = min(group_min_y[root_first], group_min_y[root_second])
        combined_maximum = max(group_max_y[root_first], group_max_y[root_second])
        if combined_maximum - combined_minimum > maximum_layer_height_range:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        group_min_y[root_first] = combined_minimum
        group_max_y[root_first] = combined_maximum
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    component_nodes: list[set[int]] = [set() for _ in range(face_component_count)]
    component_cells: list[set[int]] = [set() for _ in range(face_component_count)]
    for node_index, node in enumerate(nodes):
        flat_index = node.x_index * grid_shape[1] + node.z_index
        for component in node.face_components:
            component_nodes[component].add(node_index)
            component_cells[component].add(flat_index)
            group_min_y[component] = min(group_min_y[component], node.y)
            group_max_y[component] = max(group_max_y[component], node.y)

    anchor_components = np.zeros(face_component_count, dtype=bool)
    for component, indices in enumerate(component_nodes):
        if not indices:
            continue
        component_node_values = [nodes[index] for index in indices]
        x_indices = np.asarray(
            [node.x_index for node in component_node_values], dtype=np.int64
        )
        z_indices = np.asarray(
            [node.z_index for node in component_node_values], dtype=np.int64
        )
        bounding_cells = int(
            (x_indices.max() - x_indices.min() + 1)
            * (z_indices.max() - z_indices.min() + 1)
        )
        length_coverage = (
            (x_indices.max() - x_indices.min() + 1) * grid.dx / shoe_extents[0]
        )
        width_coverage = (
            (z_indices.max() - z_indices.min() + 1) * grid.dz / shoe_extents[2]
        )
        fill_fraction = len(component_cells[component]) / bounding_cells
        anchor_components[component] = bool(
            length_coverage >= MIN_LENGTH_COVERAGE
            and width_coverage >= MIN_WIDTH_COVERAGE
            and fill_fraction >= MIN_FOOTPRINT_FILL_FRACTION
        )

    evidence: dict[tuple[int, int], int] = defaultdict(int)
    for node in nodes:
        components = node.face_components
        for first_position, first in enumerate(components):
            for second in components[first_position + 1 :]:
                if anchor_components[first] or anchor_components[second]:
                    continue
                evidence[(min(first, second), max(first, second))] += 1

    x_count, z_count = grid_shape
    previous_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    for x_index in range(x_count):
        for z_index in range(z_count):
            current = cell_nodes[x_index * z_count + z_index]
            if not current:
                continue
            for x_offset, z_offset in previous_neighbours:
                other_x = x_index + x_offset
                other_z = z_index + z_offset
                if not (0 <= other_x < x_count and 0 <= other_z < z_count):
                    continue
                other = cell_nodes[other_x * z_count + other_z]
                if not other:
                    continue
                for current_node_index in current:
                    current_node = nodes[current_node_index]
                    for other_node_index in other:
                        other_node = nodes[other_node_index]
                        if (
                            abs(current_node.y - other_node.y)
                            > maximum_adjacent_height_step
                        ):
                            continue
                        for current_component in current_node.face_components:
                            for other_component in other_node.face_components:
                                if current_component == other_component:
                                    continue
                                if (
                                    anchor_components[current_component]
                                    or anchor_components[other_component]
                                ):
                                    continue
                                pair = (
                                    min(current_component, other_component),
                                    max(current_component, other_component),
                                )
                                evidence[pair] += 1

    ordered_evidence = sorted(
        evidence.items(), key=lambda item: (-item[1], item[0])
    )
    for (first, second), edge_count in ordered_evidence:
        if anchor_components[first] or anchor_components[second]:
            continue
        first_cells = component_cells[first]
        second_cells = component_cells[second]
        if not first_cells or not second_cells:
            continue
        smaller_count = min(len(first_cells), len(second_cells))
        required_edges = max(
            MIN_COMPONENT_MERGE_EDGES,
            int(np.ceil(MIN_COMPONENT_MERGE_EDGE_FRACTION * smaller_count)),
        )
        overlap_fraction = len(first_cells.intersection(second_cells)) / smaller_count
        if (
            edge_count >= required_edges
            and overlap_fraction <= MAX_COMPONENT_FOOTPRINT_OVERLAP
        ):
            union(first, second)

    groups: dict[int, list[int]] = {}
    for component_index, indices in enumerate(component_nodes):
        if indices:
            groups.setdefault(find(component_index), []).append(component_index)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for component_indices in groups.values():
        node_indices = sorted(
            set().union(*(component_nodes[index] for index in component_indices))
        )
        layers.append(
            (
                np.asarray(node_indices, dtype=np.int64),
                np.asarray(component_indices, dtype=np.int64),
            )
        )
    layers.sort(key=lambda item: int(item[0][0]))
    return layers


def _trace_local_height_surfaces(
    nodes: list[_LayerNode],
    cell_nodes: list[list[int]],
    grid_shape: tuple[int, int],
    maximum_adjacent_height_step: float,
    maximum_surface_height_range: float,
) -> list[np.ndarray]:
    """Trace smooth height paths without treating source components atomically."""

    parent = np.arange(len(nodes), dtype=np.int64)
    rank = np.zeros(len(nodes), dtype=np.uint8)
    group_min_y = np.asarray([node.y for node in nodes], dtype=np.float64)
    group_max_y = group_min_y.copy()

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        combined_minimum = min(group_min_y[first_root], group_min_y[second_root])
        combined_maximum = max(group_max_y[first_root], group_max_y[second_root])
        if combined_maximum - combined_minimum > maximum_surface_height_range:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        group_min_y[first_root] = combined_minimum
        group_max_y[first_root] = combined_maximum
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    x_count, z_count = grid_shape
    previous_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    matches: set[tuple[float, int, int]] = set()
    for x_index in range(x_count):
        for z_index in range(z_count):
            current = cell_nodes[x_index * z_count + z_index]
            if not current:
                continue
            for x_offset, z_offset in previous_neighbours:
                other_x = x_index + x_offset
                other_z = z_index + z_offset
                if not (0 <= other_x < x_count and 0 <= other_z < z_count):
                    continue
                other = cell_nodes[other_x * z_count + other_z]
                if not other:
                    continue

                closest_from_current = {
                    current_index: min(
                        other,
                        key=lambda other_index: (
                            abs(nodes[current_index].y - nodes[other_index].y),
                            other_index,
                        ),
                    )
                    for current_index in current
                }
                closest_from_other = {
                    other_index: min(
                        current,
                        key=lambda current_index: (
                            abs(nodes[other_index].y - nodes[current_index].y),
                            current_index,
                        ),
                    )
                    for other_index in other
                }
                for current_index, other_index in closest_from_current.items():
                    if closest_from_other[other_index] != current_index:
                        continue
                    height_difference = abs(
                        nodes[current_index].y - nodes[other_index].y
                    )
                    if height_difference > maximum_adjacent_height_step:
                        continue
                    first, second = sorted((current_index, other_index))
                    matches.add((float(height_difference), first, second))

    for _, first, second in sorted(matches):
        union(first, second)

    groups: dict[int, list[int]] = {}
    for node_index in range(len(nodes)):
        groups.setdefault(find(node_index), []).append(node_index)
    surfaces = [
        np.asarray(sorted(indices), dtype=np.int64)
        for indices in groups.values()
    ]
    surfaces.sort(key=lambda indices: int(indices[0]))
    return surfaces


def _evaluate_surface_candidate(
    *,
    method: str,
    layer_index: int,
    node_indices: np.ndarray,
    original_face_indices: np.ndarray,
    source_component_count: int,
    nodes: list[_LayerNode],
    areas: np.ndarray,
    normals: np.ndarray,
    grid: _Grid,
    shoe_bounds: np.ndarray,
    shoe_extents: np.ndarray,
) -> dict[str, Any]:
    """Measure one component layer or one locally traced height surface."""

    layer = [nodes[int(index)] for index in node_indices]
    x_indices = np.asarray([node.x_index for node in layer], dtype=np.int64)
    z_indices = np.asarray([node.z_index for node in layer], dtype=np.int64)
    heights = np.asarray([node.y for node in layer], dtype=np.float64)
    original_faces = np.asarray(original_face_indices, dtype=np.int64)
    represented_areas = areas[original_faces]
    represented_area = float(represented_areas.sum())
    upward_fraction = float(
        represented_areas[
            normals[original_faces, 1] <= -SUPPORT_NORMAL_ABS_Y_MIN
        ].sum()
        / represented_area
    )
    length_coverage = float(
        (x_indices.max() - x_indices.min() + 1) * grid.dx / shoe_extents[0]
    )
    width_coverage = float(
        (z_indices.max() - z_indices.min() + 1) * grid.dz / shoe_extents[2]
    )
    central_coverage = _central_support_length_coverage(
        x_indices, z_indices, grid.shape
    )
    median_y = float(np.median(heights))
    unique_cell_count = int(
        len(np.unique(x_indices * grid.shape[1] + z_indices))
    )
    bounding_cell_count = int(
        (x_indices.max() - x_indices.min() + 1)
        * (z_indices.max() - z_indices.min() + 1)
    )
    footprint_fill_fraction = float(unique_cell_count / bounding_cell_count)
    height_range_ratio = float(np.ptp(heights) / shoe_extents[0])
    lower_y = float(shoe_bounds[0, 1]) + LOWER_REGION_START_FRACTION * float(
        shoe_extents[1]
    )
    failures: list[str] = []
    if length_coverage < MIN_LENGTH_COVERAGE:
        failures.append("length_coverage")
    if width_coverage < MIN_WIDTH_COVERAGE:
        failures.append("width_coverage")
    if central_coverage < MIN_CENTRAL_SUPPORT_LENGTH_COVERAGE:
        failures.append("central_support")
    if shoe_extents[1] > np.finfo(np.float64).eps and median_y < lower_y:
        failures.append("lower_region")
    if footprint_fill_fraction < MIN_FOOTPRINT_FILL_FRACTION:
        failures.append("footprint_completeness")
    if height_range_ratio > MAX_HEIGHT_RANGE_RATIO:
        failures.append("height_range")
    if max(upward_fraction, 1.0 - upward_fraction) < MIN_ORIENTATION_COHERENCE:
        failures.append("orientation_coherence")

    return {
        "selection_method": method,
        "layer_index": layer_index,
        "minimum_original_face_index": int(original_faces.min()),
        "sample_count": unique_cell_count,
        "intersection_count": int(len(node_indices)),
        "source_component_count": int(source_component_count),
        "source_face_count": int(len(original_faces)),
        "x_index_bounds": [int(x_indices.min()), int(x_indices.max())],
        "z_index_bounds": [int(z_indices.min()), int(z_indices.max())],
        "height_range": [float(heights.min()), float(heights.max())],
        "height_range_ratio": height_range_ratio,
        "length_coverage": length_coverage,
        "width_coverage": width_coverage,
        "central_support_length_coverage": central_coverage,
        "footprint_fill_fraction": footprint_fill_fraction,
        "upward_facing_area_fraction": upward_fraction,
        "median_y": median_y,
        "normalized_median_y": (
            1.0
            if shoe_extents[1] <= np.finfo(np.float64).eps
            else float((median_y - shoe_bounds[0, 1]) / shoe_extents[1])
        ),
        "projected_xz_area": float(unique_cell_count * grid.dx * grid.dz),
        "qualification_failures": failures,
        "qualifies": not failures,
        "node_indices": node_indices,
        "original_face_indices": original_faces,
    }


def _candidate_sort_key(entry: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(entry["median_y"]),
        -float(entry["projected_xz_area"]),
        int(entry["minimum_original_face_index"]),
    )


def _serializable_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    hidden = {
        "height_grid",
        "node_indices",
        "nodes",
        "original_face_indices",
        "selected_node_indices",
    }
    return tuple(
        {key: value for key, value in entry.items() if key not in hidden}
        for entry in diagnostics
    )


def _rasterize_face_subset(
    triangles: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    grid: _Grid,
    shoe_length: float,
) -> tuple[list[np.ndarray], list[_LayerNode], list[list[int]]]:
    """Rasterize one deterministic subset of source faces."""

    subset_faces = faces[face_indices]
    face_components = _face_connected_components(subset_faces)
    face_component = np.empty(len(subset_faces), dtype=np.int64)
    for component_index, relative_indices in enumerate(face_components):
        face_component[relative_indices] = component_index
    subset_triangles = triangles[face_indices]
    bins = _build_xz_triangle_bins(subset_triangles, grid)
    nodes, cell_nodes = _cell_intersections(
        subset_triangles,
        face_component,
        grid,
        bins,
        shoe_length,
    )
    return face_components, nodes, cell_nodes


def _central_band_bounds(z_count: int) -> tuple[int, int]:
    band_count = max(1, int(np.ceil(CENTRAL_BAND_WIDTH_FRACTION * z_count)))
    band_start = (z_count - band_count) // 2
    return band_start, band_start + band_count


def _underlying_support_column_fraction(
    node_indices: np.ndarray,
    nodes: list[_LayerNode],
    all_cell_heights: tuple[np.ndarray, ...],
    grid_shape: tuple[int, int],
    height_epsilon: float,
) -> float:
    """Measure central columns with a distinct support-like surface below."""

    _, z_count = grid_shape
    band_start, band_end = _central_band_bounds(z_count)
    cell_heights: dict[int, float] = {}
    for node_index in node_indices:
        node = nodes[int(node_index)]
        if not band_start <= node.z_index < band_end:
            continue
        flat_index = node.x_index * z_count + node.z_index
        current = cell_heights.get(flat_index)
        if current is None or node.y < current:
            cell_heights[flat_index] = node.y

    candidate_columns: set[int] = set()
    paired_columns: set[int] = set()
    for flat_index, candidate_height in cell_heights.items():
        x_index, _ = divmod(flat_index, z_count)
        candidate_columns.add(x_index)
        if np.any(all_cell_heights[flat_index] > candidate_height + height_epsilon):
            paired_columns.add(x_index)
    if not candidate_columns:
        return 0.0
    return float(len(paired_columns) / len(candidate_columns))


def _high_heel_failures(
    candidate: dict[str, Any],
    underlying_fraction: float,
) -> list[str]:
    failures: list[str] = []
    if candidate["length_coverage"] < MIN_LENGTH_COVERAGE:
        failures.append("length_coverage")
    if candidate["width_coverage"] < MIN_WIDTH_COVERAGE:
        failures.append("width_coverage")
    if (
        candidate["central_support_length_coverage"]
        < MIN_CENTRAL_SUPPORT_LENGTH_COVERAGE
    ):
        failures.append("central_support")
    if candidate["footprint_fill_fraction"] < MIN_FOOTPRINT_FILL_FRACTION:
        failures.append("footprint_completeness")
    if underlying_fraction < MIN_UNDERLYING_SUPPORT_COLUMN_FRACTION:
        failures.append("underlying_support")
    return failures


def _maximum_adjacent_height_step(height_grid: np.ndarray) -> float:
    maximum = 0.0
    for x_offset, z_offset in ((1, 0), (0, 1), (1, 1), (1, -1)):
        first_x = slice(None, -x_offset) if x_offset else slice(None)
        second_x = slice(x_offset, None) if x_offset else slice(None)
        if z_offset == 1:
            first_z = slice(None, -1)
            second_z = slice(1, None)
        elif z_offset == -1:
            first_z = slice(1, None)
            second_z = slice(None, -1)
        else:
            first_z = slice(None)
            second_z = slice(None)
        first = height_grid[first_x, first_z]
        second = height_grid[second_x, second_z]
        valid = np.isfinite(first) & np.isfinite(second)
        if np.any(valid):
            maximum = max(maximum, float(np.max(np.abs(first[valid] - second[valid]))))
    return maximum


def _evaluate_high_heel_orientation(
    *,
    orientation_mode: str,
    face_mask: np.ndarray,
    triangles: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    normals: np.ndarray,
    grid: _Grid,
    shoe_bounds: np.ndarray,
    shoe_extents: np.ndarray,
    all_cell_heights: tuple[np.ndarray, ...],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build an opening-nearest envelope for one winding hypothesis."""

    eligible_face_indices = np.flatnonzero(face_mask)
    if len(eligible_face_indices) == 0:
        return None, []
    face_components, nodes, cell_nodes = _rasterize_face_subset(
        triangles,
        faces,
        eligible_face_indices,
        grid,
        float(shoe_extents[0]),
    )
    if not nodes:
        return None, []
    layers = _group_surface_patches_into_layers(
        nodes,
        cell_nodes,
        len(face_components),
        grid.shape,
        grid,
        shoe_extents,
        MAX_LAYER_HEIGHT_STEP_RATIO * float(shoe_extents[0]),
        np.inf,
    )
    height_epsilon = max(
        np.finfo(np.float64).eps,
        HEIGHT_COMPARISON_EPSILON_RATIO * float(shoe_extents[0]),
    )
    diagnostics: list[dict[str, Any]] = []
    for layer_index, (node_indices, represented_components) in enumerate(layers):
        represented_faces = np.concatenate(
            [face_components[int(index)] for index in represented_components]
        )
        candidate = _evaluate_surface_candidate(
            method="high_heel_layer",
            layer_index=layer_index,
            node_indices=node_indices,
            original_face_indices=np.sort(
                eligible_face_indices[represented_faces]
            ),
            source_component_count=len(represented_components),
            nodes=nodes,
            areas=areas,
            normals=normals,
            grid=grid,
            shoe_bounds=shoe_bounds,
            shoe_extents=shoe_extents,
        )
        underlying_fraction = _underlying_support_column_fraction(
            node_indices,
            nodes,
            all_cell_heights,
            grid.shape,
            height_epsilon,
        )
        failures = _high_heel_failures(candidate, underlying_fraction)
        candidate.update(
            {
                "orientation_mode": orientation_mode,
                "underlying_support_column_fraction": underlying_fraction,
                "qualification_failures": failures,
                "qualifies": not failures,
                "normal_height_range_limit_applied": False,
                "used_in_envelope": False,
            }
        )
        diagnostics.append(candidate)

    qualifying = [entry for entry in diagnostics if entry["qualifies"]]
    if not qualifying:
        return None, diagnostics
    for entry in qualifying:
        entry["used_in_envelope"] = True

    selected_by_cell: dict[int, int] = {}
    z_count = grid.shape[1]
    for entry in qualifying:
        for node_index in np.asarray(entry["node_indices"], dtype=np.int64):
            node = nodes[int(node_index)]
            flat_index = node.x_index * z_count + node.z_index
            current = selected_by_cell.get(flat_index)
            if current is None or (node.y, int(node_index)) < (
                nodes[current].y,
                current,
            ):
                selected_by_cell[flat_index] = int(node_index)
    selected_nodes = np.asarray(
        [selected_by_cell[index] for index in sorted(selected_by_cell)],
        dtype=np.int64,
    )
    selected_relative_faces = np.unique(
        np.concatenate(
            [nodes[int(index)].face_indices for index in selected_nodes]
        )
    )
    selected_faces = np.sort(eligible_face_indices[selected_relative_faces])
    represented_components = {
        component
        for node_index in selected_nodes
        for component in nodes[int(node_index)].face_components
    }
    envelope = _evaluate_surface_candidate(
        method="high_heel_upper_envelope",
        layer_index=-1,
        node_indices=selected_nodes,
        original_face_indices=selected_faces,
        source_component_count=len(represented_components),
        nodes=nodes,
        areas=areas,
        normals=normals,
        grid=grid,
        shoe_bounds=shoe_bounds,
        shoe_extents=shoe_extents,
    )
    underlying_fraction = _underlying_support_column_fraction(
        selected_nodes,
        nodes,
        all_cell_heights,
        grid.shape,
        height_epsilon,
    )
    failures = _high_heel_failures(envelope, underlying_fraction)
    height_grid = np.full(grid.shape, np.nan, dtype=np.float64)
    for node_index in selected_nodes:
        node = nodes[int(node_index)]
        height_grid[node.x_index, node.z_index] = node.y
    maximum_step = _maximum_adjacent_height_step(height_grid)
    allowed_step = MAX_LAYER_HEIGHT_STEP_RATIO * float(shoe_extents[0])
    envelope.update(
        {
            "orientation_mode": orientation_mode,
            "underlying_support_column_fraction": underlying_fraction,
            "maximum_adjacent_height_step": maximum_step,
            "allowed_adjacent_height_step": allowed_step,
            "qualification_failures": failures,
            "qualifies": not failures,
            "normal_height_range_limit_applied": False,
            "height_grid": height_grid,
            "nodes": nodes,
            "selected_node_indices": selected_nodes,
        }
    )
    diagnostics.append(envelope)
    if failures:
        return None, diagnostics
    return envelope, diagnostics


def _high_heel_landmark(
    height_grid: np.ndarray,
    x_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    start_fraction: float,
    end_fraction: float,
) -> np.ndarray:
    """Return a robust central-width landmark from a longitudinal region."""

    valid_columns = np.flatnonzero(np.any(np.isfinite(height_grid), axis=1))
    if len(valid_columns) < 2:
        raise ValueError("high-heel support has fewer than two occupied X columns")
    first = int(valid_columns[0])
    last = int(valid_columns[-1])
    span = last - first
    start = first + int(np.floor(start_fraction * span))
    end = first + int(np.ceil(end_fraction * span))
    samples: list[tuple[float, float, float]] = []
    for x_index in range(start, end + 1):
        occupied = np.flatnonzero(np.isfinite(height_grid[x_index]))
        if len(occupied) == 0:
            continue
        z_min = int(occupied[0])
        z_max = int(occupied[-1])
        local_span = z_max - z_min
        central_start = z_min + int(np.ceil(0.25 * local_span))
        central_end = z_max - int(np.ceil(0.25 * local_span))
        central = occupied[
            (occupied >= central_start) & (occupied <= central_end)
        ]
        for z_index in central:
            samples.append(
                (
                    float(x_coordinates[x_index]),
                    float(height_grid[x_index, z_index]),
                    float(z_coordinates[z_index]),
                )
            )
    if not samples:
        raise ValueError("high-heel landmark region has no central support samples")
    return np.median(np.asarray(samples, dtype=np.float64), axis=0)


def identify_high_heel_support(shoe_mesh: TriangleMesh) -> HighHeelSupport:
    """Detect a steep interior support without applying normal-shoe limits."""

    vertices = shoe_mesh.vertices
    faces = shoe_mesh.faces
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    areas = double_areas / 2.0
    normals = np.divide(
        crosses,
        double_areas[:, None],
        out=np.zeros_like(crosses),
        where=double_areas[:, None] > 0.0,
    )
    shoe_bounds = shoe_mesh.bounds
    shoe_extents = shoe_bounds[1] - shoe_bounds[0]
    grid = _make_grid(shoe_mesh)
    support_like = (double_areas > np.finfo(np.float64).eps) & (
        np.abs(normals[:, 1]) >= SUPPORT_NORMAL_ABS_Y_MIN
    )
    all_face_indices = np.flatnonzero(support_like)
    if len(all_face_indices) == 0:
        raise ValueError("shoe mesh contains no locally support-like triangles")
    _, all_nodes, all_cell_nodes = _rasterize_face_subset(
        triangles,
        faces,
        all_face_indices,
        grid,
        float(shoe_extents[0]),
    )
    if not all_nodes:
        raise ValueError("support-like triangles do not cover any X/Z grid samples")
    all_cell_heights = tuple(
        np.asarray([all_nodes[index].y for index in indices], dtype=np.float64)
        for indices in all_cell_nodes
    )

    attempted_diagnostics: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_diagnostics: list[dict[str, Any]] = []
    orientation_mode = "canonical_opening_facing"
    for mode, orientation_mask in (
        (
            "canonical_opening_facing",
            support_like & (normals[:, 1] <= -SUPPORT_NORMAL_ABS_Y_MIN),
        ),
        (
            "reversed_winding",
            support_like & (normals[:, 1] >= SUPPORT_NORMAL_ABS_Y_MIN),
        ),
    ):
        candidate, diagnostics = _evaluate_high_heel_orientation(
            orientation_mode=mode,
            face_mask=orientation_mask,
            triangles=triangles,
            faces=faces,
            areas=areas,
            normals=normals,
            grid=grid,
            shoe_bounds=shoe_bounds,
            shoe_extents=shoe_extents,
            all_cell_heights=all_cell_heights,
        )
        attempted_diagnostics.extend(diagnostics)
        if candidate is not None:
            selected = candidate
            selected_diagnostics = attempted_diagnostics
            orientation_mode = mode
            break
    if selected is None:
        raise ValueError(
            "no high-heel support satisfies the profile-specific criteria; layers="
            + json.dumps(
                _serializable_diagnostics(attempted_diagnostics),
                sort_keys=True,
            )
        )

    selected_faces = np.asarray(selected["original_face_indices"], dtype=np.int64)
    original_faces = faces[selected_faces]
    original_vertices = np.unique(original_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[original_vertices] = np.arange(len(original_vertices), dtype=np.int64)
    colors = (
        None
        if shoe_mesh.vertex_colors is None
        else shoe_mesh.vertex_colors[original_vertices]
    )
    compact_mesh = TriangleMesh(
        vertices[original_vertices], remap[original_faces], colors
    )
    height_grid = np.asarray(selected["height_grid"], dtype=np.float64)
    valid_mask = np.isfinite(height_grid)
    selected_areas = areas[selected_faces]
    selected_normals = normals[selected_faces]
    total_area = float(selected_areas.sum())
    upward_area = float(
        selected_areas[
            selected_normals[:, 1] <= -SUPPORT_NORMAL_ABS_Y_MIN
        ].sum()
    )
    support_like_area = float(
        selected_areas[
            np.abs(selected_normals[:, 1]) >= SUPPORT_NORMAL_ABS_Y_MIN
        ].sum()
    )
    centroid_y = triangles[selected_faces, :, 1].mean(axis=1)
    surface = FootbedSurface(
        mesh=compact_mesh,
        original_face_indices=selected_faces.copy(),
        bounds=compact_mesh.bounds,
        extents=compact_mesh.extents,
        x_coordinates=grid.x_centers.copy(),
        z_coordinates=grid.z_centers.copy(),
        height_grid=height_grid.copy(),
        valid_mask=valid_mask,
        length_coverage=float(selected["length_coverage"]),
        width_coverage=float(selected["width_coverage"]),
        central_support_length_coverage=float(
            selected["central_support_length_coverage"]
        ),
        upward_facing_area_fraction=upward_area / total_area,
        support_like_area_fraction=support_like_area / total_area,
        area_weighted_median_y=_weighted_median(centroid_y, selected_areas),
        projected_xz_area=float(selected["projected_xz_area"]),
        diagnostics=_serializable_diagnostics(selected_diagnostics),
        selection_method="high_heel_upper_envelope",
        fallback_reason=(
            None
            if orientation_mode == "canonical_opening_facing"
            else "canonical winding produced no valid high-heel support"
        ),
        primary_selected_layer_index=None,
    )
    heel_landmark = _high_heel_landmark(
        height_grid,
        grid.x_centers,
        grid.z_centers,
        0.0,
        HEEL_LANDMARK_END_FRACTION,
    )
    forefoot_landmark = _high_heel_landmark(
        height_grid,
        grid.x_centers,
        grid.z_centers,
        FOREFOOT_LANDMARK_START_FRACTION,
        FOREFOOT_LANDMARK_END_FRACTION,
    )
    longitudinal_distance = float(forefoot_landmark[0] - heel_landmark[0])
    if longitudinal_distance <= 0.0:
        raise ValueError("high-heel landmarks do not progress from heel to forefoot")
    heel_elevation = float(forefoot_landmark[1] - heel_landmark[1])
    support_angle = float(
        np.rad2deg(np.arctan2(heel_elevation, longitudinal_distance))
    )
    occupied_x = np.flatnonzero(np.any(valid_mask, axis=1))
    support_x_bounds = np.asarray(
        [grid.x_centers[occupied_x[0]], grid.x_centers[occupied_x[-1]]],
        dtype=np.float64,
    )
    return HighHeelSupport(
        surface=surface,
        orientation_mode=orientation_mode,
        underlying_support_column_fraction=float(
            selected["underlying_support_column_fraction"]
        ),
        heel_landmark=heel_landmark,
        forefoot_landmark=forefoot_landmark,
        heel_elevation=heel_elevation,
        support_angle_degrees=support_angle,
        support_x_bounds=support_x_bounds,
    )


def identify_footbed_surface(shoe_mesh: TriangleMesh) -> FootbedSurface:
    """Select the top interior support layer without mesh-component assumptions."""

    vertices = shoe_mesh.vertices
    faces = shoe_mesh.faces
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    areas = double_areas / 2.0
    normals = np.divide(
        crosses,
        double_areas[:, None],
        out=np.zeros_like(crosses),
        where=double_areas[:, None] > 0.0,
    )
    shoe_bounds = shoe_mesh.bounds
    shoe_extents = shoe_bounds[1] - shoe_bounds[0]
    grid = _make_grid(shoe_mesh)
    support_like = (double_areas > np.finfo(np.float64).eps) & (
        np.abs(normals[:, 1]) >= SUPPORT_NORMAL_ABS_Y_MIN
    )
    eligible_face_indices = np.flatnonzero(support_like)
    if len(eligible_face_indices) == 0:
        raise ValueError("shoe mesh contains no locally support-like triangles")

    eligible_faces = faces[eligible_face_indices]
    face_components = _face_connected_components(eligible_faces)
    face_component = np.empty(len(eligible_faces), dtype=np.int64)
    for component_index, relative_indices in enumerate(face_components):
        face_component[relative_indices] = component_index

    eligible_triangles = triangles[eligible_face_indices]
    bins = _build_xz_triangle_bins(eligible_triangles, grid)
    nodes, cell_nodes = _cell_intersections(
        eligible_triangles, face_component, grid, bins, float(shoe_extents[0])
    )
    if not nodes:
        raise ValueError("support-like triangles do not cover any X/Z grid samples")
    layers = _group_surface_patches_into_layers(
        nodes,
        cell_nodes,
        len(face_components),
        grid.shape,
        grid,
        shoe_extents,
        MAX_LAYER_HEIGHT_STEP_RATIO * float(shoe_extents[0]),
        MAX_HEIGHT_RANGE_RATIO * float(shoe_extents[0]),
    )

    diagnostics: list[dict[str, Any]] = []
    for layer_index, (node_indices, represented_components) in enumerate(layers):
        represented_faces = np.concatenate(
            [face_components[int(index)] for index in represented_components]
        )
        diagnostics.append(
            _evaluate_surface_candidate(
                method="component_layers",
                layer_index=layer_index,
                node_indices=node_indices,
                original_face_indices=np.sort(
                    eligible_face_indices[represented_faces]
                ),
                source_component_count=len(represented_components),
                nodes=nodes,
                areas=areas,
                normals=normals,
                grid=grid,
                shoe_bounds=shoe_bounds,
                shoe_extents=shoe_extents,
            )
        )

    qualifying = [entry for entry in diagnostics if entry["qualifies"]]
    if not qualifying:
        raise ValueError(
            "no coherent support layer satisfies the footbed criteria; layers="
            + json.dumps(_serializable_diagnostics(diagnostics), sort_keys=True)
        )
    primary_selected = min(qualifying, key=_candidate_sort_key)
    selected = primary_selected
    fallback_reason: str | None = None

    opening_facing_rejected_above = [
        entry
        for entry in diagnostics
        if entry["upward_facing_area_fraction"] >= MIN_ORIENTATION_COHERENCE
        and entry["median_y"] < primary_selected["median_y"]
        and entry["qualification_failures"] == ["central_support"]
    ]
    suspicious_outsole = bool(
        primary_selected["upward_facing_area_fraction"]
        <= 1.0 - MIN_ORIENTATION_COHERENCE
        and opening_facing_rejected_above
    )
    if suspicious_outsole:
        fallback_reason = (
            "primary selection was a consistently downward-facing layer while "
            "an opening-facing layer above it failed only central coverage"
        )
        traced_surfaces = _trace_local_height_surfaces(
            nodes,
            cell_nodes,
            grid.shape,
            MAX_LAYER_HEIGHT_STEP_RATIO * float(shoe_extents[0]),
            MAX_HEIGHT_RANGE_RATIO * float(shoe_extents[0]),
        )
        minimum_possible_samples = int(
            np.ceil(MIN_CENTRAL_SUPPORT_LENGTH_COVERAGE * grid.shape[0])
        )
        trace_diagnostics: list[dict[str, Any]] = []
        for trace_index, node_indices in enumerate(traced_surfaces):
            if len(node_indices) < minimum_possible_samples:
                continue
            relative_faces = np.unique(
                np.concatenate(
                    [nodes[int(index)].face_indices for index in node_indices]
                )
            )
            represented_components = {
                component
                for index in node_indices
                for component in nodes[int(index)].face_components
            }
            trace_diagnostics.append(
                _evaluate_surface_candidate(
                    method="local_height_trace",
                    layer_index=trace_index,
                    node_indices=node_indices,
                    original_face_indices=np.sort(
                        eligible_face_indices[relative_faces]
                    ),
                    source_component_count=len(represented_components),
                    nodes=nodes,
                    areas=areas,
                    normals=normals,
                    grid=grid,
                    shoe_bounds=shoe_bounds,
                    shoe_extents=shoe_extents,
                )
            )
        diagnostics.extend(trace_diagnostics)
        qualifying_traces = [
            entry
            for entry in trace_diagnostics
            if entry["qualifies"]
            and entry["upward_facing_area_fraction"]
            >= MIN_ORIENTATION_COHERENCE
            and entry["median_y"] < primary_selected["median_y"]
        ]
        if not qualifying_traces:
            raise ValueError(
                "primary support selection appears to be an outsole and local-height "
                "tracing found no unambiguous interior support; fallback_reason="
                + fallback_reason
                + "; layers="
                + json.dumps(_serializable_diagnostics(diagnostics), sort_keys=True)
            )
        selected = min(qualifying_traces, key=_candidate_sort_key)

    serializable_diagnostics = _serializable_diagnostics(diagnostics)
    selected_nodes = np.asarray(selected["node_indices"], dtype=np.int64)
    selected_faces = np.asarray(
        selected["original_face_indices"], dtype=np.int64
    )
    original_faces = faces[selected_faces]
    original_vertices = np.unique(original_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[original_vertices] = np.arange(len(original_vertices), dtype=np.int64)
    colors = (
        None
        if shoe_mesh.vertex_colors is None
        else shoe_mesh.vertex_colors[original_vertices]
    )
    compact_mesh = TriangleMesh(
        vertices[original_vertices], remap[original_faces], colors
    )

    height_grid = np.full(grid.shape, np.nan, dtype=np.float64)
    for node_index in selected_nodes:
        node = nodes[int(node_index)]
        current = height_grid[node.x_index, node.z_index]
        if np.isnan(current) or node.y < current:
            height_grid[node.x_index, node.z_index] = node.y
    valid_mask = np.isfinite(height_grid)
    selected_areas = areas[selected_faces]
    selected_normals = normals[selected_faces]
    total_area = float(selected_areas.sum())
    upward_area = float(
        selected_areas[selected_normals[:, 1] <= -SUPPORT_NORMAL_ABS_Y_MIN].sum()
    )
    support_like_area = float(
        selected_areas[np.abs(selected_normals[:, 1]) >= SUPPORT_NORMAL_ABS_Y_MIN].sum()
    )
    centroid_y = triangles[selected_faces, :, 1].mean(axis=1)
    return FootbedSurface(
        mesh=compact_mesh,
        original_face_indices=selected_faces.copy(),
        bounds=compact_mesh.bounds,
        extents=compact_mesh.extents,
        x_coordinates=grid.x_centers.copy(),
        z_coordinates=grid.z_centers.copy(),
        height_grid=height_grid,
        valid_mask=valid_mask,
        length_coverage=float(selected["length_coverage"]),
        width_coverage=float(selected["width_coverage"]),
        central_support_length_coverage=float(
            selected["central_support_length_coverage"]
        ),
        upward_facing_area_fraction=(upward_area / total_area),
        support_like_area_fraction=(support_like_area / total_area),
        area_weighted_median_y=_weighted_median(centroid_y, selected_areas),
        projected_xz_area=float(selected["projected_xz_area"]),
        diagnostics=serializable_diagnostics,
        selection_method=str(selected["selection_method"]),
        fallback_reason=fallback_reason,
        primary_selected_layer_index=int(primary_selected["layer_index"]),
    )


def sample_footbed_y(
    footbed: FootbedSurface, points_xz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentrically interpolate exact source-surface Y at X/Z queries."""

    points = np.asarray(points_xz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("points_xz must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points_xz must contain only finite values")

    triangles = footbed.mesh.vertices[footbed.mesh.faces]
    planar = triangles[:, :, (0, 2)]
    first = planar[:, 0]
    edge_a = planar[:, 1] - first
    edge_b = planar[:, 2] - first
    determinant = edge_a[:, 0] * edge_b[:, 1] - edge_b[:, 0] * edge_a[:, 1]
    nondegenerate = np.abs(determinant) > np.finfo(np.float64).eps
    heights = np.full(len(points), np.nan, dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    tolerance = 1e-10

    for point_index, point in enumerate(points):
        relative = point - first
        weight_a = np.divide(
            relative[:, 0] * edge_b[:, 1] - edge_b[:, 0] * relative[:, 1],
            determinant,
            out=np.zeros_like(determinant),
            where=nondegenerate,
        )
        weight_b = np.divide(
            edge_a[:, 0] * relative[:, 1] - relative[:, 0] * edge_a[:, 1],
            determinant,
            out=np.zeros_like(determinant),
            where=nondegenerate,
        )
        weight_first = 1.0 - weight_a - weight_b
        inside = (
            nondegenerate
            & (weight_first >= -tolerance)
            & (weight_a >= -tolerance)
            & (weight_b >= -tolerance)
            & (weight_first <= 1.0 + tolerance)
            & (weight_a <= 1.0 + tolerance)
            & (weight_b <= 1.0 + tolerance)
        )
        if not np.any(inside):
            continue
        interpolated = (
            weight_first[inside] * triangles[inside, 0, 1]
            + weight_a[inside] * triangles[inside, 1, 1]
            + weight_b[inside] * triangles[inside, 2, 1]
        )
        heights[point_index] = float(np.min(interpolated))
        valid[point_index] = True
    return heights, valid
