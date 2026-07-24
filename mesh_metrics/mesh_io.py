"""Mesh loading, validation, sampling, and transformation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a triangle mesh and bake any transforms from a scene hierarchy."""
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")

    scene = trimesh.load_scene(mesh_path, process=False)
    geometry = scene.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}, got {type(geometry).__name__}")

    mesh = geometry.copy()
    mesh.remove_infinite_values()
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        raise ValueError(f"Mesh has no valid triangle surface: {mesh_path}")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Mesh contains nonfinite vertices after cleanup: {mesh_path}")
    if not np.isfinite(mesh.area) or mesh.area <= 0.0:
        raise ValueError(f"Mesh has invalid surface area: {mesh_path}")
    return mesh


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic area-uniform surface points and face normals."""
    if count < 3:
        raise ValueError("Surface sample count must be at least 3")
    points, face_indices = trimesh.sample.sample_surface(mesh, count=count, seed=seed)
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(mesh.face_normals[face_indices], dtype=np.float64)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError("Surface sampling produced nonfinite values")
    return points, normals


def transform_mesh(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    """Return a transformed copy without modifying the source mesh."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Transform must be a finite 4x4 matrix")
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, object]:
    """Return JSON-serializable geometry information for an alignment report."""
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
    }
