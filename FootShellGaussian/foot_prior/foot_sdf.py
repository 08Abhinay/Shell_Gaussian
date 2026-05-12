"""Foot SDF generation and differentiable grid queries.

The expected sign convention is positive outside the foot and negative inside.
This matches the clearance loss we want for shoe geometry:

    ReLU(clearance - SDF_foot(x_outer)) ** 2
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FootSDFConfig:
    """Settings for querying the foot SDF prior."""

    clearance: float = 0.005
    align_corners: bool = True
    padding_mode: str = "border"


@dataclass(frozen=True)
class FootMeshForSDF:
    """Triangle mesh container used for SDF generation."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class FootSDFBuildConfig:
    """Settings for building a signed foot SDF grid from a mesh."""

    resolution: int = 128
    padding: float = 0.03
    chunk_size: int = 65536
    cap_boundary: bool = True
    hash_resolution: int = 512
    device: str = "cuda"


def load_obj_mesh(path: str) -> FootMeshForSDF:
    """Load a simple triangular OBJ mesh with ``v`` and ``f`` records."""

    vertices = []
    faces = []
    obj_path = Path(path)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ mesh not found: {obj_path}")

    with obj_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed vertex line: {line.strip()}")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) != 3:
                    raise ValueError("Only triangular OBJ faces are supported")
                face = []
                for field in fields:
                    index_text = field.split("/")[0]
                    face.append(int(index_text) - 1)
                faces.append(face)

    if not vertices:
        raise ValueError(f"OBJ mesh has no vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ mesh has no faces: {obj_path}")

    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    _validate_mesh_indices(vertices_array, faces_array)
    return FootMeshForSDF(vertices=vertices_array, faces=faces_array)


def _validate_mesh_indices(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("faces contain indices outside the vertex array")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("faces contain degenerate triangles")


def find_boundary_loops(faces: np.ndarray) -> List[List[int]]:
    """Return ordered boundary loops from a triangular mesh."""

    edge_counts: Counter[Tuple[int, int]] = Counter()
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, tri)
        for start, end in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted((start, end)))] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return []

    adjacency: Dict[int, List[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[start].append(end)
        adjacency[end].append(start)

    bad_vertices = sorted(vertex for vertex, neighbors in adjacency.items() if len(neighbors) != 2)
    if bad_vertices:
        raise ValueError(
            "Expected manifold boundary loops with degree 2, "
            f"but found bad boundary vertices: {bad_vertices[:16]}"
        )

    loops: List[List[int]] = []
    unused_edges = {tuple(sorted(edge)) for edge in boundary_edges}
    while unused_edges:
        start, next_vertex = min(unused_edges)
        loop = [start]
        previous = start
        current = next_vertex
        unused_edges.remove(tuple(sorted((start, next_vertex))))

        while current != start:
            loop.append(current)
            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            if not candidates:
                raise ValueError("Boundary loop ended before closing")
            following = candidates[0]
            edge = tuple(sorted((current, following)))
            if following != start and edge not in unused_edges:
                raise ValueError("Boundary loop traversal found an already-used edge")
            if edge in unused_edges:
                unused_edges.remove(edge)
            previous, current = current, following

        loops.append(loop)

    return loops


def cap_single_boundary_loop(mesh: FootMeshForSDF) -> FootMeshForSDF:
    """Close a mesh with one boundary loop using a centroid fan cap."""

    _validate_mesh_indices(mesh.vertices, mesh.faces)
    loops = find_boundary_loops(mesh.faces)
    if not loops:
        return mesh
    if len(loops) != 1:
        raise ValueError(f"Expected exactly one boundary loop, found {len(loops)}")

    loop = loops[0]
    centroid = mesh.vertices[loop].mean(axis=0, keepdims=True).astype(np.float32)
    cap_index = mesh.vertices.shape[0]
    cap_faces = []
    for index, start in enumerate(loop):
        end = loop[(index + 1) % len(loop)]
        cap_faces.append([start, end, cap_index])

    vertices = np.concatenate([mesh.vertices, centroid], axis=0)
    faces = np.concatenate([mesh.faces, np.asarray(cap_faces, dtype=np.int64)], axis=0)
    return FootMeshForSDF(vertices=vertices, faces=faces)


def _make_grid_points(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Create points in z/y/x grid order with point coordinates x/y/z."""

    x_axis = torch.linspace(bounds_min[0], bounds_max[0], resolution, device=device)
    y_axis = torch.linspace(bounds_min[1], bounds_max[1], resolution, device=device)
    z_axis = torch.linspace(bounds_min[2], bounds_max[2], resolution, device=device)
    zz, yy, xx = torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


def build_signed_sdf_grid_from_mesh(
    mesh: FootMeshForSDF,
    config: Optional[FootSDFBuildConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a signed SDF grid from a foot mesh.

    Returns:
        ``sdf`` with shape ``[D, H, W]`` ordered as ``[z, y, x]``, plus
        ``bounds_min`` and ``bounds_max`` in xyz world/model coordinates.
    """

    build_config = config or FootSDFBuildConfig()
    if build_config.resolution < 2:
        raise ValueError("resolution must be at least 2")
    if build_config.padding < 0:
        raise ValueError("padding must be non-negative")
    if build_config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    _validate_mesh_indices(mesh.vertices, mesh.faces)
    sdf_mesh = cap_single_boundary_loop(mesh) if build_config.cap_boundary else mesh

    device = torch.device(build_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SDF generation, but it is not available")

    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from kaolin.ops.mesh import check_sign, index_vertices_by_faces

    vertices_np = sdf_mesh.vertices.astype(np.float32)
    faces_np = sdf_mesh.faces.astype(np.int64)
    bounds_min_np = vertices_np.min(axis=0) - np.float32(build_config.padding)
    bounds_max_np = vertices_np.max(axis=0) + np.float32(build_config.padding)

    vertices = torch.as_tensor(vertices_np, dtype=torch.float32, device=device).unsqueeze(0)
    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
    face_vertices = index_vertices_by_faces(vertices, faces)
    bounds_min = torch.as_tensor(bounds_min_np, dtype=torch.float32, device=device)
    bounds_max = torch.as_tensor(bounds_max_np, dtype=torch.float32, device=device)
    points = _make_grid_points(bounds_min, bounds_max, build_config.resolution, device)

    signed_sdf_chunks = []
    for start in range(0, points.shape[0], build_config.chunk_size):
        chunk = points[start : start + build_config.chunk_size].unsqueeze(0)
        squared_distance, _, _ = point_to_mesh_distance(chunk, face_vertices)
        distance = torch.sqrt(torch.clamp(squared_distance, min=0.0))
        inside = check_sign(
            vertices,
            faces,
            chunk,
            hash_resolution=build_config.hash_resolution,
        )
        sign = torch.where(inside, -torch.ones_like(distance), torch.ones_like(distance))
        signed_sdf_chunks.append((distance * sign).squeeze(0).detach().cpu())

    sdf = torch.cat(signed_sdf_chunks, dim=0).reshape(
        build_config.resolution,
        build_config.resolution,
        build_config.resolution,
    )
    return (
        sdf.numpy().astype(np.float32),
        bounds_min_np.astype(np.float32),
        bounds_max_np.astype(np.float32),
    )


def save_signed_sdf_npz(
    output_path: str,
    sdf: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Save a foot SDF grid in the format expected by ``FootSDFGrid.from_npz``."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sdf": np.asarray(sdf, dtype=np.float32),
        "bounds_min": np.asarray(bounds_min, dtype=np.float32),
        "bounds_max": np.asarray(bounds_max, dtype=np.float32),
    }
    if metadata is not None:
        payload["metadata"] = np.asarray(metadata, dtype=object)
    np.savez_compressed(path, **payload)


def build_and_save_signed_sdf_from_obj(
    obj_path: str,
    output_path: str,
    config: Optional[FootSDFBuildConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an OBJ foot mesh, build a signed SDF grid, and save it as ``.npz``."""

    mesh = load_obj_mesh(obj_path)
    build_config = config or FootSDFBuildConfig()
    sdf, bounds_min, bounds_max = build_signed_sdf_grid_from_mesh(mesh, build_config)
    metadata = {
        "source_obj": str(Path(obj_path)),
        "resolution": int(build_config.resolution),
        "padding": float(build_config.padding),
        "cap_boundary": bool(build_config.cap_boundary),
        "sign_convention": "negative_inside_positive_outside",
        "grid_order": "D,H,W = z,y,x",
    }
    save_signed_sdf_npz(output_path, sdf, bounds_min, bounds_max, metadata=metadata)
    return sdf, bounds_min, bounds_max


class FootSDFGrid(nn.Module):
    """Trilinear SDF lookup for a precomputed foot grid.

    The SDF tensor is stored as ``[D, H, W]`` and queried with points shaped
    ``[..., 3]`` in world/model coordinates ordered as ``x, y, z``.
    """

    def __init__(
        self,
        sdf: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        config: Optional[FootSDFConfig] = None,
    ) -> None:
        super().__init__()

        if sdf.ndim != 3:
            raise ValueError("sdf must have shape [D, H, W]")
        if bounds_min.shape != (3,) or bounds_max.shape != (3,):
            raise ValueError("bounds_min and bounds_max must have shape [3]")
        if torch.any(bounds_max <= bounds_min):
            raise ValueError("bounds_max must be greater than bounds_min on all axes")

        self.config = config or FootSDFConfig()
        self.register_buffer("sdf", sdf[None, None].contiguous())
        self.register_buffer("bounds_min", bounds_min.contiguous())
        self.register_buffer("bounds_max", bounds_max.contiguous())

    @classmethod
    def from_npz(
        cls,
        path: str,
        config: Optional[FootSDFConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FootSDFGrid":
        """Load an SDF grid saved with keys ``sdf``, ``bounds_min``, ``bounds_max``."""

        data = np.load(Path(path))
        required_keys = {"sdf", "bounds_min", "bounds_max"}
        missing_keys = sorted(required_keys.difference(data.files))
        if missing_keys:
            raise KeyError(f"Foot SDF file is missing keys: {missing_keys}")

        sdf = torch.as_tensor(data["sdf"], dtype=dtype, device=device)
        bounds_min = torch.as_tensor(data["bounds_min"], dtype=dtype, device=device)
        bounds_max = torch.as_tensor(data["bounds_max"], dtype=dtype, device=device)
        return cls(sdf=sdf, bounds_min=bounds_min, bounds_max=bounds_max, config=config)

    def points_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Convert world/model points to ``grid_sample`` coordinates in ``[-1, 1]``."""

        return 2.0 * (points - self.bounds_min) / (self.bounds_max - self.bounds_min) - 1.0

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Return SDF values at ``points`` using trilinear interpolation."""

        if points.shape[-1] != 3:
            raise ValueError("points must have last dimension 3")

        original_shape = points.shape[:-1]
        flat_points = points.reshape(1, -1, 1, 1, 3)
        grid = self.points_to_grid(flat_points)
        values = F.grid_sample(
            self.sdf,
            grid,
            mode="bilinear",
            padding_mode=self.config.padding_mode,
            align_corners=self.config.align_corners,
        )
        return values.reshape(*original_shape)

    def clearance_loss(
        self,
        points: torch.Tensor,
        clearance: Optional[float] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Penalize points that are closer to the foot than ``clearance``."""

        target_clearance = self.config.clearance if clearance is None else clearance
        sdf_values = self.query(points)
        loss = torch.relu(target_clearance - sdf_values).square()

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction}")
