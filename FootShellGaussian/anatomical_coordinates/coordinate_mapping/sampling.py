"""Deterministic surface sampling, shared by the audit and the address stage.

Both samplers draw by triangle area rather than per vertex, so a statistic over
them is an area fraction. Both take a seed and are reproducible.
"""

from __future__ import annotations

import numpy as np


def area_samples(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0
) -> np.ndarray:
    """Uniform by area over the surface, deterministic for a given seed."""

    generator = np.random.default_rng(seed)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    keep = areas > 0.0
    triangles, areas = triangles[keep], areas[keep]
    picked = generator.choice(len(triangles), size=count, p=areas / areas.sum())
    u = generator.random(count)
    v = generator.random(count)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    return (
        triangles[picked, 0]
        + u[:, None] * (triangles[picked, 1] - triangles[picked, 0])
        + v[:, None] * (triangles[picked, 2] - triangles[picked, 0])
    )


def shell_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    band: tuple[float, float],
    seed: int = 0,
) -> np.ndarray:
    """Points offset from the surface along its normals, inside a band."""

    generator = np.random.default_rng(seed)
    triangles = vertices[faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    areas = np.linalg.norm(normals, axis=1)
    keep = areas > 1e-12
    triangles, normals, areas = triangles[keep], normals[keep], areas[keep]
    normals = normals / areas[:, None]
    probability = areas / areas.sum()
    picked = generator.choice(len(triangles), size=count, p=probability)
    u = generator.random(count)
    v = generator.random(count)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    base = (
        triangles[picked, 0]
        + u[:, None] * (triangles[picked, 1] - triangles[picked, 0])
        + v[:, None] * (triangles[picked, 2] - triangles[picked, 0])
    )
    offset = generator.uniform(band[0], band[1], size=count)
    return base + normals[picked] * offset[:, None]
