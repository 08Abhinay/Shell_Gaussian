"""Validated triangle-mesh primitives and file I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


@dataclass(frozen=True)
class TriangleMesh:
    """A non-empty indexed triangle mesh with optional per-vertex colors."""

    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices must have shape (N, 3)")
        if len(vertices) == 0:
            raise ValueError("mesh must contain at least one vertex")
        if not np.issubdtype(vertices.dtype, np.number):
            raise TypeError("vertices must be numeric")
        vertices = np.asarray(vertices, dtype=np.float64)
        if not np.isfinite(vertices).all():
            raise ValueError("vertices must contain only finite values")

        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (M, 3)")
        if len(faces) == 0:
            raise ValueError("mesh must contain at least one face")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("faces must use an integer dtype")
        faces = np.asarray(faces, dtype=np.int64)
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("face indices are outside the vertex array")

        colors = self.vertex_colors
        if colors is not None:
            colors = _validate_vertex_colors(colors, len(vertices))

        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())
        object.__setattr__(
            self, "vertex_colors", None if colors is None else colors.copy()
        )

    @property
    def bounds(self) -> np.ndarray:
        """Return axis-aligned minimum and maximum corners."""

        return np.stack(
            (self.vertices.min(axis=0), self.vertices.max(axis=0)), axis=0
        )

    @property
    def extents(self) -> np.ndarray:
        """Return the axis-aligned bounding-box side lengths."""

        return np.ptp(self.vertices, axis=0)

    @property
    def center(self) -> np.ndarray:
        """Return the axis-aligned bounding-box center."""

        return self.bounds.mean(axis=0)


def _validate_vertex_colors(colors: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(colors)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] not in (3, 4):
        raise ValueError("vertex colors must have shape (N, 3) or (N, 4)")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("vertex colors must use an integer dtype")
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("vertex colors must lie in [0, 255]")
    return np.asarray(values, dtype=np.uint8)


def load_triangle_mesh(path: str | Path) -> TriangleMesh:
    """Load one PLY or OBJ triangle mesh without processing its topology."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {".ply", ".obj"}:
        raise ValueError(f"unsupported mesh format: {source.suffix}")

    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        raise TypeError("mesh file contains a scene; exactly one mesh is required")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected trimesh.Trimesh, received {type(loaded).__name__}")

    colors: np.ndarray | None = None
    visual_colors = getattr(loaded.visual, "vertex_colors", None)
    if visual_colors is not None and len(visual_colors) == len(loaded.vertices):
        colors = np.asarray(visual_colors, dtype=np.uint8)
    return TriangleMesh(loaded.vertices, loaded.faces, colors)


def save_triangle_mesh(
    path: str | Path,
    mesh: TriangleMesh,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Save a triangle mesh as binary PLY or OBJ without topology processing."""

    destination = Path(path)
    if destination.suffix.lower() not in {".ply", ".obj"}:
        raise ValueError(f"unsupported mesh format: {destination.suffix}")
    colors = mesh.vertex_colors if vertex_colors is None else vertex_colors
    if colors is not None:
        colors = _validate_vertex_colors(colors, len(mesh.vertices))

    exported = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
        validate=False,
    )
    if colors is not None:
        exported.visual.vertex_colors = colors
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".ply":
        payload = trimesh.exchange.ply.export_ply(
            exported, encoding="binary_little_endian"
        )
        destination.write_bytes(payload)
    else:
        exported.export(destination, file_type="obj")


def transform_mesh(mesh: TriangleMesh, matrix: np.ndarray) -> TriangleMesh:
    """Apply a finite homogeneous 4x4 transform to every vertex."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("matrix must be a finite 4x4 array")
    homogeneous = np.column_stack(
        (mesh.vertices, np.ones(len(mesh.vertices), dtype=np.float64))
    )
    transformed = homogeneous @ transform.T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise ValueError("matrix maps at least one vertex to an invalid homogeneous point")
    vertices = transformed[:, :3] / transformed[:, 3, None]
    return TriangleMesh(vertices, mesh.faces, mesh.vertex_colors)


def combine_colored_meshes(
    *items: tuple[TriangleMesh, Sequence[int]],
) -> TriangleMesh:
    """Combine meshes while assigning one RGB or RGBA color to each input."""

    if not items:
        raise ValueError("at least one colored mesh is required")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for mesh, color in items:
        rgba = np.asarray(color)
        if rgba.shape not in {(3,), (4,)}:
            raise ValueError("each mesh color must contain RGB or RGBA values")
        if not np.issubdtype(rgba.dtype, np.integer):
            raise TypeError("mesh colors must use integer values")
        if np.any(rgba < 0) or np.any(rgba > 255):
            raise ValueError("mesh colors must lie in [0, 255]")
        rgba = np.asarray(rgba, dtype=np.uint8)
        if len(rgba) == 3:
            rgba = np.append(rgba, np.uint8(255))
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        colors.append(np.tile(rgba, (len(mesh.vertices), 1)))
        offset += len(mesh.vertices)
    return TriangleMesh(
        np.concatenate(vertices, axis=0),
        np.concatenate(faces, axis=0),
        np.concatenate(colors, axis=0),
    )
