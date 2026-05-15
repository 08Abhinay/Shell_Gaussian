"""Foot-to-shoe alignment and diagnostic region utilities.

The foot SDF is stored in raw SUPR coordinates, while GShell meshes live in a
normalized shoe coordinate system. This module keeps that transform explicit:

    raw SUPR foot point -> aligned GShell shoe point

The inverse transform is then used to query the raw SUPR SDF from shoe-space
points.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from .foot_sdf import FootSDFGrid, find_boundary_loops


RAW_SUPR_WIDTH_AXIS = 0
RAW_SUPR_HEIGHT_AXIS = 1
RAW_SUPR_LENGTH_AXIS = 2


@dataclass(frozen=True)
class MeshData:
    """Simple triangular mesh container."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class FootAlignmentConfig:
    """User-facing knobs for placing the foot inside a shoe mesh."""

    length_ratio: float = 0.78
    scale_multiplier: float = 1.0
    plantar_clearance: float = 0.032
    plantar_band: float = 0.012
    surface_band: float = 0.005
    clearance: float = 0.005
    ankle_radius: float = 0.025
    shoe_length_axis: int = 0
    shoe_up_axis: int = 1
    shoe_width_axis: int = 2
    shoe_length_sign: float = 1.0
    shoe_up_sign: float = -1.0
    shoe_width_sign: float = 1.0
    align_ankle_to_opening: bool = True
    opening_min_vertices: int = 20
    opening_min_width_ratio: float = 0.25
    opening_max_length_ratio: float = 0.58
    opening_min_height_position: float = 0.20
    auto_yaw: bool = True
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    translation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FootAlignment:
    """A rigid+scale transform between raw SUPR foot space and shoe space."""

    foot_to_shoe: np.ndarray
    shoe_to_foot: np.ndarray
    scale: float
    plantar_z: float
    auto_yaw_degrees: float
    config: FootAlignmentConfig
    foot_anchor_remapped: Tuple[float, float, float]
    shoe_anchor: Tuple[float, float, float]
    ankle_center_shoe: Optional[Tuple[float, float, float]] = None
    opening_center: Optional[Tuple[float, float, float]] = None
    opening_component_index: Optional[int] = None
    opening_component_size: Optional[Tuple[float, float, float]] = None

    def transform_foot_to_shoe(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.foot_to_shoe)

    def transform_shoe_to_foot(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.shoe_to_foot)

    def to_dict(self) -> Dict[str, object]:
        return {
            "foot_to_shoe": self.foot_to_shoe.astype(float).tolist(),
            "shoe_to_foot": self.shoe_to_foot.astype(float).tolist(),
            "scale": float(self.scale),
            "plantar_z": float(self.plantar_z),
            "auto_yaw_degrees": float(self.auto_yaw_degrees),
            "config": asdict(self.config),
            "foot_anchor_remapped": list(map(float, self.foot_anchor_remapped)),
            "shoe_anchor": list(map(float, self.shoe_anchor)),
            "ankle_center_shoe": None
            if self.ankle_center_shoe is None
            else list(map(float, self.ankle_center_shoe)),
            "opening_center": None
            if self.opening_center is None
            else list(map(float, self.opening_center)),
            "opening_component_index": self.opening_component_index,
            "opening_component_size": None
            if self.opening_component_size is None
            else list(map(float, self.opening_component_size)),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FootAlignment":
        config_payload = payload.get("config", {})
        config = FootAlignmentConfig(**config_payload)
        return cls(
            foot_to_shoe=np.asarray(payload["foot_to_shoe"], dtype=np.float32),
            shoe_to_foot=np.asarray(payload["shoe_to_foot"], dtype=np.float32),
            scale=float(payload["scale"]),
            plantar_z=float(payload["plantar_z"]),
            auto_yaw_degrees=float(payload.get("auto_yaw_degrees", 0.0)),
            config=config,
            foot_anchor_remapped=tuple(payload["foot_anchor_remapped"]),
            shoe_anchor=tuple(payload["shoe_anchor"]),
            ankle_center_shoe=None
            if payload.get("ankle_center_shoe") is None
            else tuple(payload["ankle_center_shoe"]),
            opening_center=None
            if payload.get("opening_center") is None
            else tuple(payload["opening_center"]),
            opening_component_index=payload.get("opening_component_index"),
            opening_component_size=None
            if payload.get("opening_component_size") is None
            else tuple(payload["opening_component_size"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FootAlignment":
        with Path(path).open("r") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def load_triangle_mesh(path: str | Path, compact: bool = True) -> MeshData:
    """Load vertices and triangular faces from a simple OBJ mesh."""

    mesh_path = Path(path)
    vertices = []
    faces = []
    with mesh_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                indices = []
                for field in line.split()[1:]:
                    index_text = field.split("/")[0]
                    indices.append(int(index_text) - 1)
                if len(indices) < 3:
                    continue
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    if not vertices:
        raise ValueError(f"No OBJ vertices found in {mesh_path}")
    if not faces:
        raise ValueError(f"No OBJ faces found in {mesh_path}")
    mesh = MeshData(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )
    return compact_mesh(mesh) if compact else mesh


def compact_mesh(mesh: MeshData) -> MeshData:
    """Drop vertices that are not referenced by any face."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    used_vertices = np.unique(faces.reshape(-1))
    if used_vertices.shape[0] == mesh.vertices.shape[0]:
        return mesh

    remap = np.full((mesh.vertices.shape[0],), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int64)
    return MeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float32)[used_vertices],
        faces=remap[faces],
    )


def write_obj_mesh(path: str | Path, mesh: MeshData, comments: Optional[Iterable[str]] = None) -> None:
    """Write a geometry-only triangular OBJ."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        if comments is not None:
            for comment in comments:
                f.write(f"# {comment}\n")
        for vertex in np.asarray(mesh.vertices):
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in np.asarray(mesh.faces, dtype=np.int64):
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_colored_ply(path: str | Path, mesh: MeshData, colors: np.ndarray) -> None:
    """Write an ASCII PLY with per-vertex RGB colors."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != (vertices.shape[0], 3):
        raise ValueError("colors must have shape [num_vertices, 3]")

    with output_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def mesh_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    size = bounds_max - bounds_min
    center = (bounds_min + bounds_max) * 0.5
    return bounds_min, bounds_max, size, center


def axis_sign(sign: float) -> float:
    """Return a clean +1/-1 axis sign."""

    return 1.0 if float(sign) >= 0.0 else -1.0


def signed_axis_values(points: np.ndarray, axis: int, sign: float) -> np.ndarray:
    return axis_sign(sign) * np.asarray(points, dtype=np.float32)[..., axis]


def bottom_coordinate(bounds_min: np.ndarray, bounds_max: np.ndarray, axis: int, up_sign: float, clearance: float = 0.0) -> float:
    """Coordinate of the sole-side support plane along a signed up axis."""

    if axis_sign(up_sign) > 0.0:
        return float(bounds_min[axis] + clearance)
    return float(bounds_max[axis] - clearance)


def plantar_coordinate(points: np.ndarray, axis: int, up_sign: float) -> float:
    """Return the sole-side coordinate for points along a signed up axis."""

    values = np.asarray(points, dtype=np.float32)[:, axis]
    if axis_sign(up_sign) > 0.0:
        return float(values.min())
    return float(values.max())


def fit_plane_normal(points: np.ndarray) -> np.ndarray:
    """Least-squares plane normal for a boundary component."""

    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0].astype(np.float32)
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0.0:
        normal = -normal
    return normal


def find_boundary_components(mesh: MeshData) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Return tolerant boundary vertex components and edge lists.

    Reconstructed shoe meshes often have non-manifold boundary vertices. For
    opening detection we only need connected boundary components, not perfectly
    ordered loops.
    """

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for face in np.asarray(mesh.faces, dtype=np.int64):
        for start, end in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            a, b = int(start), int(end)
            if a > b:
                a, b = b, a
            edge_counts[(a, b)] += 1

    adjacency: Dict[int, list[int]] = defaultdict(list)
    for (a, b), count in edge_counts.items():
        if count == 1:
            adjacency[a].append(b)
            adjacency[b].append(a)

    seen: set[int] = set()
    components: list[np.ndarray] = []
    edge_components: list[np.ndarray] = []
    for start in adjacency:
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        vertices = []
        edges = []
        while queue:
            vertex = queue.popleft()
            vertices.append(vertex)
            for neighbor in adjacency[vertex]:
                edges.append((vertex, neighbor))
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        unique_edges = np.unique(np.sort(np.asarray(edges, dtype=np.int64), axis=1), axis=0)
        components.append(np.asarray(vertices, dtype=np.int64))
        edge_components.append(unique_edges)

    return components, edge_components


def detect_shoe_opening_boundary(
    mesh: MeshData,
    config: Optional[FootAlignmentConfig] = None,
) -> Optional[Dict[str, object]]:
    """Pick the most collar-like boundary component from a noisy shoe shell.

    The raw GShell shoe shell can have a very large boundary where the sole is
    missing. That boundary is useful evidence that the reconstruction is open,
    but it is not the ankle/collar opening. We score components in a signed
    anatomical height coordinate and skip broad sole-side components.
    """

    cfg = config or FootAlignmentConfig()
    components, edge_components = find_boundary_components(mesh)
    if not components:
        return None

    shoe_min, shoe_max, shoe_size, _ = mesh_bounds(mesh.vertices)
    length_extent = max(float(shoe_size[cfg.shoe_length_axis]), 1e-8)
    width_extent = max(float(shoe_size[cfg.shoe_width_axis]), 1e-8)
    up_extent = max(float(shoe_size[cfg.shoe_up_axis]), 1e-8)
    signed_up_all = signed_axis_values(mesh.vertices, cfg.shoe_up_axis, cfg.shoe_up_sign)
    signed_up_min = float(signed_up_all.min())
    signed_up_extent = max(float(signed_up_all.max() - signed_up_min), 1e-8)

    best: Optional[Dict[str, object]] = None
    for index, component in enumerate(components):
        vertices = mesh.vertices[component]
        bounds_min, bounds_max, size, center = mesh_bounds(vertices)
        width_ratio = float(size[cfg.shoe_width_axis] / width_extent)
        if component.shape[0] < cfg.opening_min_vertices or width_ratio < cfg.opening_min_width_ratio:
            continue

        length_ratio = float(size[cfg.shoe_length_axis] / length_extent)
        up_ratio = float(size[cfg.shoe_up_axis] / up_extent)
        signed_center_up = float(axis_sign(cfg.shoe_up_sign) * center[cfg.shoe_up_axis])
        height_position = float((signed_center_up - signed_up_min) / signed_up_extent)
        sole_like = (
            length_ratio > cfg.opening_max_length_ratio
            and height_position < cfg.opening_min_height_position
        )
        if sole_like:
            continue

        normal = fit_plane_normal(vertices)
        score = width_ratio + 0.55 * up_ratio + 0.45 * height_position - 0.65 * length_ratio
        candidate: Dict[str, object] = {
            "score": score,
            "index": index,
            "vertices": component,
            "edges": edge_components[index],
            "center": center,
            "size": size,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "width_ratio": width_ratio,
            "length_ratio": length_ratio,
            "up_ratio": up_ratio,
            "height_position": height_position,
            "normal": normal,
            "dominant_normal_axis": int(np.argmax(np.abs(normal))),
            "sole_like": sole_like,
            "shoe_bounds_min": shoe_min,
            "shoe_bounds_max": shoe_max,
        }
        if best is None or score > float(best["score"]):
            best = candidate

    return best


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    original_shape = points.shape
    flat_points = points.reshape(-1, 3)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([flat_points, ones], axis=1)
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].reshape(original_shape)


def make_supr_to_shoe_axis_remap(
    shoe_length_axis: int = 0,
    shoe_up_axis: int = 1,
    shoe_width_axis: int = 2,
    shoe_length_sign: float = 1.0,
    shoe_up_sign: float = -1.0,
    shoe_width_sign: float = 1.0,
) -> np.ndarray:
    """Build an axis remap from raw SUPR coords to GShell shoe coords.

    Raw SUPR-Foot coordinates are interpreted as:
        x = foot width, y = foot height, z = foot length.

    The current shoe diagnostics interpret this GShell shoe as:
        x = shoe length, y = shoe height/opening-to-sole axis, z = shoe width.

    For the raw shell mesh used in the debug notebook, the missing sole/base is
    on the positive y side, so anatomical foot-up maps to negative shoe y.
    """

    axes = [shoe_length_axis, shoe_up_axis, shoe_width_axis]
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("shoe_length_axis, shoe_up_axis, and shoe_width_axis must be a permutation of 0,1,2")

    remap = np.zeros((3, 3), dtype=np.float32)
    remap[shoe_length_axis, RAW_SUPR_LENGTH_AXIS] = axis_sign(shoe_length_sign)
    remap[shoe_up_axis, RAW_SUPR_HEIGHT_AXIS] = axis_sign(shoe_up_sign)
    remap[shoe_width_axis, RAW_SUPR_WIDTH_AXIS] = axis_sign(shoe_width_sign)
    return remap


def remap_supr_to_shoe_axes(
    points: np.ndarray,
    shoe_length_axis: int = 0,
    shoe_up_axis: int = 1,
    shoe_width_axis: int = 2,
    shoe_length_sign: float = 1.0,
    shoe_up_sign: float = -1.0,
    shoe_width_sign: float = 1.0,
) -> np.ndarray:
    remap = make_supr_to_shoe_axis_remap(
        shoe_length_axis,
        shoe_up_axis,
        shoe_width_axis,
        shoe_length_sign,
        shoe_up_sign,
        shoe_width_sign,
    )
    return np.asarray(points, dtype=np.float32) @ remap.T


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(offset, dtype=np.float32)
    return matrix


def _scale_matrix(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    return matrix


def _axis_remap_matrix(config: FootAlignmentConfig) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = make_supr_to_shoe_axis_remap(
        config.shoe_length_axis,
        config.shoe_up_axis,
        config.shoe_width_axis,
        config.shoe_length_sign,
        config.shoe_up_sign,
        config.shoe_width_sign,
    )
    return matrix


def _axis_angle_rotation_matrix(axis: int, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos_value = np.cos(angle)
    sin_value = np.sin(angle)

    matrix = np.eye(4, dtype=np.float32)
    if axis == 0:
        matrix[:3, :3] = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_value, -sin_value],
                [0.0, sin_value, cos_value],
            ],
            dtype=np.float32,
        )
    elif axis == 1:
        matrix[:3, :3] = np.asarray(
            [
                [cos_value, 0.0, sin_value],
                [0.0, 1.0, 0.0],
                [-sin_value, 0.0, cos_value],
            ],
            dtype=np.float32,
        )
    elif axis == 2:
        matrix[:3, :3] = np.asarray(
            [
                [cos_value, -sin_value, 0.0],
                [sin_value, cos_value, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return matrix


def _rotation_matrix(
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    config: FootAlignmentConfig,
) -> np.ndarray:
    return (
        _axis_angle_rotation_matrix(config.shoe_up_axis, yaw_degrees)
        @ _axis_angle_rotation_matrix(config.shoe_width_axis, pitch_degrees)
        @ _axis_angle_rotation_matrix(config.shoe_length_axis, roll_degrees)
    )


def principal_yaw_degrees(
    vertices: np.ndarray,
    length_axis: int = 0,
    width_axis: int = 1,
) -> float:
    """Estimate the horizontal long-axis angle of a mesh in degrees."""

    horizontal = np.asarray(vertices, dtype=np.float32)[:, [length_axis, width_axis]]
    centered = horizontal - horizontal.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[0] < 0.0:
        axis = -axis
    return float(np.rad2deg(np.arctan2(axis[1], axis[0])))


def principal_xy_yaw_degrees(vertices: np.ndarray) -> float:
    """Backward-compatible X-Y-plane yaw helper."""

    return principal_yaw_degrees(vertices, length_axis=0, width_axis=1)


def build_alignment_from_meshes(
    foot_mesh: MeshData,
    shoe_mesh: MeshData,
    opening_mesh: Optional[MeshData] = None,
    config: Optional[FootAlignmentConfig] = None,
) -> FootAlignment:
    """Create an initial SUPR-foot-to-GShell-shoe alignment."""

    cfg = config or FootAlignmentConfig()
    foot_remapped = remap_supr_to_shoe_axes(
        foot_mesh.vertices,
        cfg.shoe_length_axis,
        cfg.shoe_up_axis,
        cfg.shoe_width_axis,
        cfg.shoe_length_sign,
        cfg.shoe_up_sign,
        cfg.shoe_width_sign,
    )
    foot_min, foot_max, foot_size, foot_center = mesh_bounds(foot_remapped)
    shoe_min, shoe_max, shoe_size, shoe_center = mesh_bounds(shoe_mesh.vertices)

    if foot_size[cfg.shoe_length_axis] <= 0.0:
        raise ValueError("Foot length extent is zero after axis remap")
    scale = float(
        cfg.length_ratio
        * cfg.scale_multiplier
        * shoe_size[cfg.shoe_length_axis]
        / foot_size[cfg.shoe_length_axis]
    )

    foot_anchor = foot_center.astype(np.float32)
    foot_anchor[cfg.shoe_up_axis] = plantar_coordinate(
        foot_remapped,
        cfg.shoe_up_axis,
        cfg.shoe_up_sign,
    )
    shoe_anchor = shoe_center.astype(np.float32)
    shoe_anchor[cfg.shoe_up_axis] = bottom_coordinate(
        shoe_min,
        shoe_max,
        cfg.shoe_up_axis,
        cfg.shoe_up_sign,
        cfg.plantar_clearance,
    )

    auto_yaw_degrees = (
        principal_yaw_degrees(
            shoe_mesh.vertices,
            length_axis=cfg.shoe_length_axis,
            width_axis=cfg.shoe_width_axis,
        )
        if cfg.auto_yaw
        else 0.0
    )
    total_yaw_degrees = auto_yaw_degrees + cfg.yaw_degrees

    foot_to_shoe = (
        _translation_matrix(shoe_anchor)
        @ _rotation_matrix(total_yaw_degrees, cfg.pitch_degrees, cfg.roll_degrees, cfg)
        @ _scale_matrix(scale)
        @ _translation_matrix(-foot_anchor)
        @ _axis_remap_matrix(cfg)
    )

    alignment_shift = np.zeros(3, dtype=np.float32)
    opening_center = None
    opening_component_index = None
    opening_component_size = None
    if cfg.align_ankle_to_opening and opening_mesh is not None:
        opening = detect_shoe_opening_boundary(opening_mesh, cfg)
        if opening is not None:
            ankle_loop = get_single_boundary_loop(foot_mesh)
            ankle_center = transform_points(foot_mesh.vertices[ankle_loop], foot_to_shoe).mean(axis=0)
            opening_center_array = np.asarray(opening["center"], dtype=np.float32)
            opening_shift = np.zeros(3, dtype=np.float32)
            for axis in [cfg.shoe_length_axis, cfg.shoe_width_axis]:
                opening_shift[axis] = opening_center_array[axis] - ankle_center[axis]
            foot_to_shoe = _translation_matrix(opening_shift) @ foot_to_shoe
            alignment_shift += opening_shift
            opening_center = tuple(float(v) for v in opening_center_array)
            opening_component_index = int(opening["index"])
            opening_component_size = tuple(float(v) for v in np.asarray(opening["size"], dtype=np.float32))

    user_offset = np.asarray(cfg.translation_offset, dtype=np.float32)
    if np.any(user_offset):
        foot_to_shoe = _translation_matrix(user_offset) @ foot_to_shoe
        alignment_shift += user_offset

    shoe_to_foot = np.linalg.inv(foot_to_shoe).astype(np.float32)

    aligned_foot = transform_points(foot_mesh.vertices, foot_to_shoe)
    plantar_z = plantar_coordinate(aligned_foot, cfg.shoe_up_axis, cfg.shoe_up_sign)
    ankle_center_shoe = transform_points(
        foot_mesh.vertices[get_single_boundary_loop(foot_mesh)],
        foot_to_shoe,
    ).mean(axis=0)
    final_shoe_anchor = shoe_anchor + alignment_shift

    return FootAlignment(
        foot_to_shoe=foot_to_shoe.astype(np.float32),
        shoe_to_foot=shoe_to_foot,
        scale=scale,
        plantar_z=plantar_z,
        auto_yaw_degrees=auto_yaw_degrees,
        config=cfg,
        foot_anchor_remapped=tuple(float(v) for v in foot_anchor),
        shoe_anchor=tuple(float(v) for v in final_shoe_anchor),
        ankle_center_shoe=tuple(float(v) for v in ankle_center_shoe),
        opening_center=opening_center,
        opening_component_index=opening_component_index,
        opening_component_size=opening_component_size,
    )


def get_single_boundary_loop(mesh: MeshData) -> np.ndarray:
    """Return the one open boundary loop expected at the ankle."""

    loops = find_boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    if len(loops) != 1:
        raise ValueError(f"Expected one foot boundary loop, found {len(loops)}")
    return np.asarray(loops[0], dtype=np.int64)


def point_to_polyline_distance(points: np.ndarray, loop_vertices: np.ndarray) -> np.ndarray:
    """Distance from each point to a closed 3D polyline."""

    points = np.asarray(points, dtype=np.float32)
    loop_vertices = np.asarray(loop_vertices, dtype=np.float32)
    if loop_vertices.shape[0] < 2:
        raise ValueError("loop_vertices must contain at least two points")

    starts = loop_vertices
    ends = np.roll(loop_vertices, shift=-1, axis=0)
    segments = ends - starts
    segment_lengths_sq = np.maximum((segments * segments).sum(axis=1), 1e-12)

    best = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for start, segment, length_sq in zip(starts, segments, segment_lengths_sq):
        rel = points - start
        t = np.clip((rel * segment).sum(axis=1) / length_sq, 0.0, 1.0)
        closest = start[None, :] + t[:, None] * segment[None, :]
        dist = np.linalg.norm(points - closest, axis=1)
        best = np.minimum(best, dist)
    return best


def query_foot_sdf_in_shoe_space(
    points_shoe: np.ndarray,
    foot_sdf: FootSDFGrid,
    alignment: FootAlignment,
    chunk_size: int = 200000,
) -> np.ndarray:
    """Query the raw foot SDF using points expressed in shoe coordinates."""

    points_foot = alignment.transform_shoe_to_foot(points_shoe).astype(np.float32)
    device = foot_sdf.sdf.device
    values = []
    with torch.no_grad():
        for start in range(0, points_foot.shape[0], chunk_size):
            chunk = torch.as_tensor(points_foot[start : start + chunk_size], dtype=torch.float32, device=device)
            values.append(foot_sdf.query(chunk).detach().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def classify_shoe_points(
    points_shoe: np.ndarray,
    foot_sdf: FootSDFGrid,
    alignment: FootAlignment,
    ankle_loop_shoe: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Classify shoe-space points into anatomy-aware diagnostic regions."""

    cfg = alignment.config
    sdf_values = query_foot_sdf_in_shoe_space(points_shoe, foot_sdf, alignment)
    point_up_values = signed_axis_values(points_shoe, cfg.shoe_up_axis, cfg.shoe_up_sign)
    plantar_up_value = axis_sign(cfg.shoe_up_sign) * alignment.plantar_z
    below_plantar = point_up_values <= plantar_up_value + cfg.plantar_band
    regions = {
        "sdf": sdf_values,
        "inside_foot": sdf_values < 0.0,
        "near_foot_surface": np.abs(sdf_values) <= cfg.surface_band,
        "clearance_violation": sdf_values < cfg.clearance,
        "below_plantar": below_plantar,
    }

    if ankle_loop_shoe is not None:
        ankle_distance = point_to_polyline_distance(points_shoe, ankle_loop_shoe)
        regions["ankle_distance"] = ankle_distance
        regions["near_ankle"] = ankle_distance <= cfg.ankle_radius
    else:
        regions["ankle_distance"] = np.full((points_shoe.shape[0],), np.inf, dtype=np.float32)
        regions["near_ankle"] = np.zeros((points_shoe.shape[0],), dtype=bool)
    return regions


def colors_from_regions(regions: Dict[str, np.ndarray]) -> np.ndarray:
    """Map diagnostic regions to vertex RGB colors."""

    count = regions["sdf"].shape[0]
    colors = np.full((count, 3), [180, 180, 180], dtype=np.uint8)
    colors[regions["below_plantar"]] = [44, 123, 182]
    colors[regions["near_ankle"]] = [117, 112, 179]
    colors[regions["near_foot_surface"]] = [255, 215, 0]
    colors[regions["clearance_violation"]] = [253, 141, 60]
    colors[regions["inside_foot"]] = [215, 48, 39]
    return colors


def region_summary(regions: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Small JSON-friendly summary of region counts and SDF range."""

    sdf_values = regions["sdf"]
    summary: Dict[str, object] = {
        "num_points": int(sdf_values.shape[0]),
        "sdf_min": float(np.min(sdf_values)),
        "sdf_max": float(np.max(sdf_values)),
        "sdf_mean": float(np.mean(sdf_values)),
    }
    for key in [
        "inside_foot",
        "near_foot_surface",
        "clearance_violation",
        "below_plantar",
        "near_ankle",
    ]:
        mask = np.asarray(regions[key], dtype=bool)
        summary[f"{key}_count"] = int(mask.sum())
        summary[f"{key}_fraction"] = float(mask.mean())
    return summary


def select_faces_below_axis(mesh_data: MeshData, axis: int, max_value: float) -> np.ndarray:
    centroids = mesh_data.vertices[mesh_data.faces].mean(axis=1)
    return centroids[:, axis] <= max_value


def select_faces_below_signed_axis(
    mesh_data: MeshData,
    axis: int,
    axis_direction: float,
    plantar_value: float,
    band: float = 0.0,
) -> np.ndarray:
    """Select faces on the sole/material side of a signed anatomical up axis."""

    centroids = mesh_data.vertices[mesh_data.faces].mean(axis=1)
    signed_centroids = axis_sign(axis_direction) * centroids[:, axis]
    signed_plantar = axis_sign(axis_direction) * float(plantar_value)
    return signed_centroids <= signed_plantar + float(band)


def select_faces_by_centroid_z(mesh_data: MeshData, z_max: float) -> np.ndarray:
    return select_faces_below_axis(mesh_data, axis=2, max_value=z_max)


def make_hybrid_mesh(shell_mesh: MeshData, watertight_mesh: MeshData, watertight_face_mask: np.ndarray) -> MeshData:
    """Append selected watertight triangles to the shell mesh for diagnostics."""

    selected_faces = np.asarray(watertight_mesh.faces, dtype=np.int64)[watertight_face_mask]
    used_vertices = np.unique(selected_faces.reshape(-1))
    remap = np.full((watertight_mesh.vertices.shape[0],), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int64)

    appended_vertices = watertight_mesh.vertices[used_vertices]
    appended_faces = remap[selected_faces] + shell_mesh.vertices.shape[0]

    return MeshData(
        vertices=np.concatenate([shell_mesh.vertices, appended_vertices], axis=0).astype(np.float32),
        faces=np.concatenate([shell_mesh.faces, appended_faces], axis=0).astype(np.int64),
    )
