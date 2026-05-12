"""Foot-to-shoe alignment and diagnostic region utilities.

The foot SDF is stored in raw SUPR coordinates, while GShell meshes live in a
normalized shoe coordinate system. This module keeps that transform explicit:

    raw SUPR foot point -> aligned GShell shoe point

The inverse transform is then used to query the raw SUPR SDF from shoe-space
points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from .foot_sdf import FootSDFGrid, find_boundary_loops


SUPR_TO_SHOE_AXIS_REMAP = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class MeshData:
    """Simple triangular mesh container."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class FootAlignmentConfig:
    """User-facing knobs for placing the foot inside a shoe mesh."""

    length_ratio: float = 0.88
    scale_multiplier: float = 1.0
    plantar_clearance: float = 0.008
    plantar_band: float = 0.012
    surface_band: float = 0.005
    clearance: float = 0.005
    ankle_radius: float = 0.025
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
    config: FootAlignmentConfig
    foot_anchor_remapped: Tuple[float, float, float]
    shoe_anchor: Tuple[float, float, float]

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
            "config": asdict(self.config),
            "foot_anchor_remapped": list(map(float, self.foot_anchor_remapped)),
            "shoe_anchor": list(map(float, self.shoe_anchor)),
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
            config=config,
            foot_anchor_remapped=tuple(payload["foot_anchor_remapped"]),
            shoe_anchor=tuple(payload["shoe_anchor"]),
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


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    original_shape = points.shape
    flat_points = points.reshape(-1, 3)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([flat_points, ones], axis=1)
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].reshape(original_shape)


def remap_supr_to_shoe_axes(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ SUPR_TO_SHOE_AXIS_REMAP.T


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


def _axis_remap_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = SUPR_TO_SHOE_AXIS_REMAP
    return matrix


def _rotation_matrix(yaw_degrees: float, pitch_degrees: float, roll_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    roll = np.deg2rad(roll_degrees)

    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


def build_alignment_from_meshes(
    foot_mesh: MeshData,
    shoe_mesh: MeshData,
    config: Optional[FootAlignmentConfig] = None,
) -> FootAlignment:
    """Create an initial SUPR-foot-to-GShell-shoe alignment from bounding boxes."""

    cfg = config or FootAlignmentConfig()
    foot_remapped = remap_supr_to_shoe_axes(foot_mesh.vertices)
    foot_min, foot_max, foot_size, foot_center = mesh_bounds(foot_remapped)
    shoe_min, shoe_max, shoe_size, shoe_center = mesh_bounds(shoe_mesh.vertices)

    if foot_size[0] <= 0.0:
        raise ValueError("Foot length extent is zero after axis remap")
    scale = float(cfg.length_ratio * cfg.scale_multiplier * shoe_size[0] / foot_size[0])

    foot_anchor = np.asarray([foot_center[0], foot_center[1], foot_min[2]], dtype=np.float32)
    shoe_anchor = np.asarray(
        [
            shoe_center[0],
            shoe_center[1],
            shoe_min[2] + cfg.plantar_clearance,
        ],
        dtype=np.float32,
    )
    shoe_anchor = shoe_anchor + np.asarray(cfg.translation_offset, dtype=np.float32)

    foot_to_shoe = (
        _translation_matrix(shoe_anchor)
        @ _rotation_matrix(cfg.yaw_degrees, cfg.pitch_degrees, cfg.roll_degrees)
        @ _scale_matrix(scale)
        @ _translation_matrix(-foot_anchor)
        @ _axis_remap_matrix()
    )
    shoe_to_foot = np.linalg.inv(foot_to_shoe).astype(np.float32)

    aligned_foot = transform_points(foot_mesh.vertices, foot_to_shoe)
    plantar_z = float(aligned_foot[:, 2].min())

    return FootAlignment(
        foot_to_shoe=foot_to_shoe.astype(np.float32),
        shoe_to_foot=shoe_to_foot,
        scale=scale,
        plantar_z=plantar_z,
        config=cfg,
        foot_anchor_remapped=tuple(float(v) for v in foot_anchor),
        shoe_anchor=tuple(float(v) for v in shoe_anchor),
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
    below_plantar = points_shoe[:, 2] <= alignment.plantar_z + cfg.plantar_band
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


def select_faces_by_centroid_z(mesh_data: MeshData, z_max: float) -> np.ndarray:
    centroids = mesh_data.vertices[mesh_data.faces].mean(axis=1)
    return centroids[:, 2] <= z_max


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
