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
MIN_LENGTH_COVERAGE = 0.65
MIN_WIDTH_COVERAGE = 0.40
LOWER_REGION_START_FRACTION = 0.40
MIN_FOOTPRINT_FILL_FRACTION = 0.25
MAX_HEIGHT_RANGE_RATIO = 0.15
MIN_ORIENTATION_COHERENCE = 0.90
MIN_COMPONENT_MERGE_EDGES = 4
MIN_COMPONENT_MERGE_EDGE_FRACTION = 0.01
MAX_COMPONENT_FOOTPRINT_OVERLAP = 0.10


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
    upward_facing_area_fraction: float
    support_like_area_fraction: float
    area_weighted_median_y: float
    projected_xz_area: float
    diagnostics: tuple[dict[str, Any], ...]

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
            "upward_facing_area_fraction": self.upward_facing_area_fraction,
            "support_like_area_fraction": self.support_like_area_fraction,
            "area_weighted_median_y": self.area_weighted_median_y,
            "projected_xz_area": self.projected_xz_area,
            "layers": list(self.diagnostics),
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

    lower_y = float(shoe_bounds[0, 1]) + LOWER_REGION_START_FRACTION * float(
        shoe_extents[1]
    )
    diagnostics: list[dict[str, Any]] = []
    for layer_index, (node_indices, represented_components) in enumerate(layers):
        layer = [nodes[int(index)] for index in node_indices]
        x_indices = np.asarray([node.x_index for node in layer], dtype=np.int64)
        z_indices = np.asarray([node.z_index for node in layer], dtype=np.int64)
        heights = np.asarray([node.y for node in layer], dtype=np.float64)
        represented_faces = np.concatenate(
            [face_components[int(index)] for index in represented_components]
        )
        original_faces = eligible_face_indices[represented_faces]
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
        median_y = float(np.median(heights))
        unique_cell_count = int(
            len(np.unique(x_indices * grid.shape[1] + z_indices))
        )
        projected_area = float(unique_cell_count * grid.dx * grid.dz)
        bounding_cell_count = int(
            (x_indices.max() - x_indices.min() + 1)
            * (z_indices.max() - z_indices.min() + 1)
        )
        qualifies = bool(
            length_coverage >= MIN_LENGTH_COVERAGE
            and width_coverage >= MIN_WIDTH_COVERAGE
            and (shoe_extents[1] <= np.finfo(np.float64).eps or median_y >= lower_y)
            and unique_cell_count / bounding_cell_count >= MIN_FOOTPRINT_FILL_FRACTION
            and np.ptp(heights) / shoe_extents[0] <= MAX_HEIGHT_RANGE_RATIO
            and max(upward_fraction, 1.0 - upward_fraction)
            >= MIN_ORIENTATION_COHERENCE
        )
        diagnostics.append(
            {
                "layer_index": layer_index,
                "minimum_original_face_index": int(original_faces.min()),
                "sample_count": unique_cell_count,
                "intersection_count": int(len(node_indices)),
                "source_component_count": int(len(represented_components)),
                "source_face_count": int(len(original_faces)),
                "x_index_bounds": [int(x_indices.min()), int(x_indices.max())],
                "z_index_bounds": [int(z_indices.min()), int(z_indices.max())],
                "height_range": [float(heights.min()), float(heights.max())],
                "height_range_ratio": float(np.ptp(heights) / shoe_extents[0]),
                "length_coverage": length_coverage,
                "width_coverage": width_coverage,
                "footprint_fill_fraction": float(
                    unique_cell_count / bounding_cell_count
                ),
                "upward_facing_area_fraction": upward_fraction,
                "median_y": median_y,
                "normalized_median_y": (
                    1.0
                    if shoe_extents[1] <= np.finfo(np.float64).eps
                    else float((median_y - shoe_bounds[0, 1]) / shoe_extents[1])
                ),
                "projected_xz_area": projected_area,
                "qualifies": qualifies,
                "node_indices": node_indices,
                "represented_components": represented_components,
            }
        )

    qualifying = [entry for entry in diagnostics if entry["qualifies"]]
    serializable_diagnostics = tuple(
        {
            key: value
            for key, value in entry.items()
            if key not in {"node_indices", "represented_components"}
        }
        for entry in diagnostics
    )
    if not qualifying:
        raise ValueError(
            "no coherent support layer satisfies the footbed criteria; layers="
            + json.dumps(serializable_diagnostics, sort_keys=True)
        )
    selected = min(
        qualifying,
        key=lambda entry: (
            entry["median_y"],
            -entry["projected_xz_area"],
            entry["minimum_original_face_index"],
        ),
    )
    selected_nodes = np.asarray(selected["node_indices"], dtype=np.int64)
    selected_components = np.asarray(
        selected["represented_components"], dtype=np.int64
    )
    selected_relative_faces = np.concatenate(
        [face_components[int(index)] for index in selected_components]
    )
    selected_faces = np.sort(eligible_face_indices[selected_relative_faces])
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
        upward_facing_area_fraction=(upward_area / total_area),
        support_like_area_fraction=(support_like_area / total_area),
        area_weighted_median_y=_weighted_median(centroid_y, selected_areas),
        projected_xz_area=float(selected["projected_xz_area"]),
        diagnostics=serializable_diagnostics,
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
