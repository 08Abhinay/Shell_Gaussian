"""Exact triangle-surface and ray queries backed by Open3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh


@dataclass(frozen=True)
class ClosestSurfaceResult:
    distances: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    primitive_ids: np.ndarray


class TriangleSurface:
    """Reusable Open3D raycasting scene for one triangle mesh."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        if len(vertices) < 3 or len(triangles) < 1:
            raise ValueError("Triangle surface is empty")
        tensor_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32),
        )
        self.scene = o3d.t.geometry.RaycastingScene()
        self.geometry_id = int(self.scene.add_triangles(tensor_mesh))

    def closest_points(
        self,
        points: np.ndarray,
        chunk_size: int = 250_000,
    ) -> ClosestSurfaceResult:
        """Find exact closest triangle points and primitive normals."""
        query = np.ascontiguousarray(points, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("Closest-point queries must have shape Nx3")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        closest_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        primitive_parts: list[np.ndarray] = []
        for start in range(0, len(query), chunk_size):
            chunk = o3d.core.Tensor(query[start : start + chunk_size])
            result = self.scene.compute_closest_points(chunk)
            closest_parts.append(result["points"].numpy())
            normal_parts.append(result["primitive_normals"].numpy())
            primitive_parts.append(result["primitive_ids"].numpy())

        closest = np.concatenate(closest_parts, axis=0).astype(np.float64, copy=False)
        normals = np.concatenate(normal_parts, axis=0).astype(np.float64, copy=False)
        primitive_ids = np.concatenate(primitive_parts, axis=0).astype(np.int64, copy=False)
        distances = np.linalg.norm(np.asarray(points, dtype=np.float64) - closest, axis=1)
        if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(normals)):
            raise RuntimeError("Closest-surface query produced nonfinite values")
        return ClosestSurfaceResult(
            distances=distances,
            points=closest,
            normals=normals,
            primitive_ids=primitive_ids,
        )

    def cast_rays(self, rays: np.ndarray, chunk_size: int = 250_000) -> np.ndarray:
        """Return first-hit ray parameters for Nx6 origin-direction rays."""
        query = np.ascontiguousarray(rays, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 6:
            raise ValueError("Ray queries must have shape Nx6")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        hits: list[np.ndarray] = []
        for start in range(0, len(query), chunk_size):
            chunk = o3d.core.Tensor(query[start : start + chunk_size])
            hits.append(self.scene.cast_rays(chunk)["t_hit"].numpy())
        return np.concatenate(hits, axis=0).astype(np.float64, copy=False)
