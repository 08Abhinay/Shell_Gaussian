"""Deterministic shoe-footbed identification and sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import TriangleMesh


MIN_LENGTH_COVERAGE = 0.80
MIN_WIDTH_COVERAGE = 0.60
UPWARD_NORMAL_Y_MAX = -0.85
MIN_UPWARD_AREA_FRACTION = 0.90


@dataclass(frozen=True)
class FootbedSurface:
    """A compact selected footbed sheet and its deterministic diagnostics."""

    mesh: TriangleMesh
    original_face_indices: np.ndarray
    bounds: np.ndarray
    extents: np.ndarray
    length_coverage: float
    width_coverage: float
    upward_facing_area_fraction: float
    area_weighted_median_y: float
    projected_xz_area: float
    diagnostics: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible selection metrics."""

        return {
            "original_face_indices": self.original_face_indices.tolist(),
            "face_count": int(len(self.original_face_indices)),
            "vertex_count": int(len(self.mesh.vertices)),
            "bounds": self.bounds.tolist(),
            "extents": self.extents.tolist(),
            "length_coverage": self.length_coverage,
            "width_coverage": self.width_coverage,
            "upward_facing_area_fraction": self.upward_facing_area_fraction,
            "area_weighted_median_y": self.area_weighted_median_y,
            "projected_xz_area": self.projected_xz_area,
            "candidates": list(self.diagnostics),
        }


def _face_connected_components(faces: np.ndarray) -> list[np.ndarray]:
    """Return edge-adjacent face components without altering the mesh."""

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


def identify_footbed_surface(shoe_mesh: TriangleMesh) -> FootbedSurface:
    """Select the topmost broad upward-facing connected shoe component."""

    vertices = shoe_mesh.vertices
    faces = shoe_mesh.faces
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(crosses, axis=1)
    areas = double_areas / 2.0
    normals = np.divide(
        crosses,
        double_areas[:, None],
        out=np.zeros_like(crosses),
        where=double_areas[:, None] > 0.0,
    )
    centroid_y = triangles[:, :, 1].mean(axis=1)
    shoe_extents = shoe_mesh.extents
    if shoe_extents[0] <= 0.0 or shoe_extents[2] <= 0.0:
        raise ValueError("shoe mesh must have positive X length and Z width extents")

    diagnostics: list[dict[str, Any]] = []
    component_faces: list[np.ndarray] = []
    for component_index, face_indices in enumerate(_face_connected_components(faces)):
        component_faces.append(face_indices)
        vertex_indices = np.unique(faces[face_indices])
        component_vertices = vertices[vertex_indices]
        bounds = np.stack(
            (component_vertices.min(axis=0), component_vertices.max(axis=0)), axis=0
        )
        extents = bounds[1] - bounds[0]
        component_areas = areas[face_indices]
        total_area = float(component_areas.sum())
        upward_area = float(
            component_areas[normals[face_indices, 1] <= UPWARD_NORMAL_Y_MAX].sum()
        )
        upward_fraction = upward_area / total_area if total_area > 0.0 else 0.0
        median_y = (
            _weighted_median(centroid_y[face_indices], component_areas)
            if total_area > 0.0
            else float("inf")
        )
        length_coverage = float(extents[0] / shoe_extents[0])
        width_coverage = float(extents[2] / shoe_extents[2])
        projected_area = float(np.sum(np.abs(crosses[face_indices, 1]) / 2.0))
        qualifies = bool(
            length_coverage >= MIN_LENGTH_COVERAGE
            and width_coverage >= MIN_WIDTH_COVERAGE
            and upward_fraction >= MIN_UPWARD_AREA_FRACTION
        )
        diagnostics.append(
            {
                "component_index": component_index,
                "minimum_original_face_index": int(face_indices[0]),
                "face_count": int(len(face_indices)),
                "vertex_count": int(len(vertex_indices)),
                "bounds": bounds.tolist(),
                "extents": extents.tolist(),
                "length_coverage": length_coverage,
                "width_coverage": width_coverage,
                "total_area": total_area,
                "upward_facing_area_fraction": upward_fraction,
                "area_weighted_median_y": median_y,
                "projected_xz_area": projected_area,
                "qualifies": qualifies,
            }
        )

    qualifying = [entry for entry in diagnostics if entry["qualifies"]]
    if not qualifying:
        raise ValueError(
            "no connected component satisfies the footbed criteria; candidates="
            + json.dumps(diagnostics, sort_keys=True)
        )
    selected = min(
        qualifying,
        key=lambda entry: (
            entry["area_weighted_median_y"],
            -entry["projected_xz_area"],
            entry["minimum_original_face_index"],
        ),
    )
    selected_faces = component_faces[int(selected["component_index"])]
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
    return FootbedSurface(
        mesh=compact_mesh,
        original_face_indices=selected_faces.copy(),
        bounds=compact_mesh.bounds,
        extents=compact_mesh.extents,
        length_coverage=float(selected["length_coverage"]),
        width_coverage=float(selected["width_coverage"]),
        upward_facing_area_fraction=float(selected["upward_facing_area_fraction"]),
        area_weighted_median_y=float(selected["area_weighted_median_y"]),
        projected_xz_area=float(selected["projected_xz_area"]),
        diagnostics=tuple(diagnostics),
    )


def sample_footbed_y(
    footbed: FootbedSurface, points_xz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentrically interpolate footbed Y at query points in the X/Z plane."""

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
